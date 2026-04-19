"""
ISAA Host CLI v4 - The Multi-Agent Host System
===============================================

A production-ready CLI that acts as a host system controlled by a "Self Agent".
Features:
- Global Rate Limiter configuration shared across all agents
- Self Agent with exclusive shell access and system management tools
- Multi-Agent registry with background task support
- Audio interface with F4 keybinding for voice input
- Skill sharing and agent binding capabilities
- Professional terminal UI using cli_printing

Author: ISAA Team
Version: 4.0.0
"""

import asyncio
import dataclasses
import logging
import os
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from typing import Any

import requests
from prompt_toolkit.document import Document
from prompt_toolkit.filters import is_done

from toolboxv2.utils.workers import get_registry
from toolboxv2.utils.extras.Style import SpinnerManager, Spinner
from toolboxv2.utils.extras.pt_spinner_patch import apply_prompt_toolkit_patch_safe, get_spinner_toolbar_fragment, register_app

# Suppress noisy loggers
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

from prompt_toolkit import PromptSession, ANSI
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter, PathCompleter, Completer, \
    Completion
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

# ToolBoxV2 Imports
from toolboxv2 import get_app, remove_styles, get_logger

# ISAA Agent Imports
from toolboxv2.mods.isaa.base.Agent.builder import (
    FlowAgentBuilder,
)
from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
from toolboxv2.mods.isaa.base.Agent.instant_data_vis import (
    visualize_data_terminal,
)
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType, VFSFile
from toolboxv2.mods.isaa.base.AgentUtils import detect_shell

from toolboxv2.mods.isaa.extras.zen.zen_renderer import  _esc, C
from toolboxv2.mods.isaa.extras.jobs import JobDefinition, TriggerConfig, JobScheduler

import html
from pathlib import Path
from toolboxv2.utils.extras.mkdocs import DocsSystem
from toolboxv2 import init_cwd, tb_root_dir
from prompt_toolkit import print_formatted_text, HTML
from toolboxv2.mods.isaa.CodingAgent.coder import CoderAgent
from toolboxv2.mods.isaa.base.audio_io.audioIo import (
    AudioStreamPlayer, LocalPlayer, WebPlayer, NullPlayer,
)
from toolboxv2.mods.isaa.base.audio_io.Tts import TTSConfig, TTSBackend, TTSEmotion
from toolboxv2.mods.isaa.base.audio_io.audio_live import (
    LiveModeEngine, LiveModeConfig, EndMode,
    SpeakerProfileStore,
)
import os
os.environ["NARRATOR_CONSOLE_PRINT"] = "false"
def ensure_utf8_stdout():
    if sys.platform == "win32" and "pytest" not in sys.modules:
        try:
            # Reconfigure existiert ab Python 3.7 und ist 100% sicher!
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass


"""
ISAA Live View — A+B combined task dashboard.
A: persistent bottom_toolbar (all tasks always visible)
B: F2 fullscreen overlay (left=task list, right=iteration detail)
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

C = {
    "dim": "#6b7280", "cyan": "#67e8f9", "green": "#4ade80",
    "red": "#f87171", "amber": "#fbbf24", "white": "#e5e7eb",
    "bright": "#ffffff", "blue": "#60a5fa", "purple": "#a78bfa",
}
SYM = {
    "ok": "✓", "fail": "✗", "done": "●", "think": "◎",
    "tool": "◇", "sub": "✦", "bar_fill": "━", "bar_empty": "─",
}
STATUS_SYM = {
    "running":   ("◯", "cyan"),
    "completed": ("●", "green"),
    "done":      ("●", "green"),
    "failed":    ("✗", "red"),
    "error":     ("✗", "red"),
    "cancelled": ("⏸", "dim"),
}


def _short(s: str, n: int) -> str:
    return s[:n] + ".." if len(s) > n + 2 else s


def _bar(cur: int, total: int, w: int = 8) -> str:
    if total <= 0:
        return SYM["bar_empty"] * w
    f = int(w * min(cur, total) / total)
    return SYM["bar_fill"] * f + SYM["bar_empty"] * (w - f)


def _fmt_elapsed(secs: float) -> str:
    s = int(secs)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _tool_result_info(name: str, result_raw: str) -> tuple[bool, str]:
    """Returns (success, short_info). Matches init_session_tools return types."""
    try:
        r = json.loads(result_raw) if isinstance(result_raw, str) and result_raw.strip() else result_raw
    except Exception:
        r = {}
    if isinstance(r, list):
        return True, f"{len(r)} items"
    if not isinstance(r, dict):
        return True, _short(str(result_raw), 35)

    success = r.get("success", True)

    if name == "vfs_shell":
        stdout = r.get("stdout", "") or ""
        return success, _short(stdout.split("\n")[0], 38)
    if name == "vfs_view":
        lines = r.get("lines_shown", r.get("total_lines", ""))
        return success, f"{lines} lines" if lines else ""
    if name == "search_vfs":
        results = r.get("results", r.get("matches", []))
        return success, f"{len(results) if isinstance(results, list) else '?'} results"
    if name in ("fs_copy_to_vfs", "fs_copy_from_vfs", "fs_copy_dir_from_vfs",
                "vfs_mount", "vfs_unmount", "vfs_sync_all", "vfs_refresh_mount"):
        return success, _short(r.get("path", r.get("vfs_path", "")), 30)
    if name == "docker_run":
        stdout = r.get("stdout", "") or ""
        return success, _short(stdout.split("\n")[0], 38)
    if name == "docker_logs":
        logs = r.get("logs", r.get("output", "")) or ""
        return success, _short(logs.split("\n")[-2] if logs else "", 38)
    if name == "check_permissions":
        return r.get("allowed", True), _short(r.get("rule", ""), 30)
    if name == "set_agent_situation":
        return success, _short(r.get("intent", ""), 30)
    if name == "vfs_share_create":
        return success, _short(r.get("share_id", ""), 20)
    for key in ("message", "info", "path", "error"):
        if key in r:
            return success, _short(str(r[key]), 35)

    if not r:
        return True, ""
    return success, _short(str(r), 38)


    # ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class IterView:
    n: int
    thoughts: list[str] = field(default_factory=list)
    # (tool_name, success, elapsed_s, info_str)
    tools: list[tuple[str, bool, float, str]] = field(default_factory=list)
    _tool_start_times: dict[str, float] = field(default_factory=dict)
    _tool_start_inputs: dict[str, str] = field(default_factory=dict)
    pending_tool: str = ""   # currently running tool name
    tools_raw: list[tuple[str, str, str]] = field(default_factory=list)
    _in_reasoning: bool = False


@dataclass
class TaskView:
    task_id: str
    agent_name: str
    query: str
    status: str = "running"
    persona: str = ""
    narrator_msg: str = ""
    status_msg: str = ""
    skills: list[str] = field(default_factory=list)
    iteration: int = 0
    max_iter: int = 0
    tokens_used: int = 0
    tokens_max: int = 0
    phase: str = "running"
    last_tool: str = ""
    last_tool_ok: bool = True
    last_tool_info: str = ""
    last_thought: str = ""
    completed_at: Optional[float] = None
    sub_agents: dict[str, int] = field(default_factory=dict)
    sub_task_ids: dict[str, str] = field(default_factory=dict)
    _sub_color_counter: int = 0
    iterations: list[IterView] = field(default_factory=list)
    _iter_map: dict[int, IterView] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    is_swarm_sub: bool = False
    swarm_parent_id: str = ""
    is_swarm_summary: bool = False  # NEU: Parent eines Swarms
    swarm_phase: str = ""

    final_answer: str = ""

    def _get_iter(self, n: int) -> IterView:
        if n not in self._iter_map:
            iv = IterView(n=n)
            self._iter_map[n] = iv
            self.iterations.append(iv)
        return self._iter_map[n]

    def _sub_color(self, name: str) -> int:
        if name not in self.sub_agents:
            self.sub_agents[name] = self._sub_color_counter % 6
            self._sub_color_counter += 1
        return self.sub_agents[name]


def ingest_chunk(tv: TaskView, chunk: dict) -> None:
    """Fill TaskView from one stream chunk. No side effects outside tv."""
    if chunk.get("agent"):
        tv.agent_name = chunk["agent"]
    if chunk.get("persona"):
        tv.persona = chunk["persona"]
    if chunk.get("skills"):
        tv.skills = chunk["skills"]
    if chunk.get("narrator_msg"):
        tv.narrator_msg = chunk["narrator_msg"]
    if chunk.get("status_msg"):
        tv.status_msg = chunk["status_msg"]
    if chunk.get("iter") is not None:
        tv.iteration = chunk["iter"]
    if chunk.get("max_iter") is not None:
        tv.max_iter = chunk["max_iter"]
    if chunk.get("tokens_used") is not None:
        tv.tokens_used = chunk["tokens_used"]
    if chunk.get("tokens_max") is not None:
        tv.tokens_max = chunk["tokens_max"]

    sub_id = chunk.get("_sub_agent_id", "")
    if sub_id:
        tv._sub_color(sub_id)

    t = chunk.get("type", "")
    iv = tv._get_iter(tv.iteration) if tv.iteration > 0 else None

    if t == "narrator":
        tv.narrator_msg = chunk.get("narrator_msg", "")
        # === INTERFACE REGISTRY HOOK SOFORT AUSFÜHREN ===
        get_registry().publish_sync(
            id=f"icli.task.{tv.task_id}",
            data=dataclasses.asdict(tv)
        )
        return  # Blockiert das restliche Phasen-Handling nicht


    if t == "reasoning":
        thought = chunk.get("chunk", "")
        tv.phase = "thinking"
        tv.last_thought = thought
        if iv and thought:
            if not iv._in_reasoning or not iv.thoughts:
                iv.thoughts.append(thought)
                iv._in_reasoning = True
            else:
                iv.thoughts[-1] += thought

    elif t == "content":
        tv.phase = "content"
        if iv: iv._in_reasoning = False

    elif t == "tool_start":
        name = chunk.get("name", "?")
        raw_args = chunk.get("args", chunk.get("input", chunk.get("arguments", "")))
        if isinstance(raw_args, dict):
            import json as _j
            raw_args = _j.dumps(raw_args, indent=2)
        tv.last_tool = name
        tv.phase = "tool"
        if iv:
            iv._in_reasoning = False
            iv.pending_tool = name
            iv._tool_start_times[name] = time.time()
            if raw_args:
                iv._tool_start_inputs[name] = str(raw_args)

    elif t == "tool_result":
        name = chunk.get("name", tv.last_tool)
        result_raw = chunk.get("result", "")
        success, info = _tool_result_info(name, result_raw)
        elapsed = 0.0
        if iv and name in iv._tool_start_times:
            elapsed = time.time() - iv._tool_start_times.pop(name)
        tv.last_tool = name
        tv.last_tool_ok = success
        tv.last_tool_info = info
        tv.phase = "tool_done"
        if iv:
            iv.tools.append((name, success, elapsed, info))
            iv.pending_tool = ""
            iv._in_reasoning = False
            # Rohdaten für späteren Drill-Down sichern
            raw_result = chunk.get("result", "")
            raw_input = chunk.get("args", chunk.get("input", chunk.get("arguments", "")))
            if not raw_input:
                raw_input = iv._tool_start_inputs.pop(name, "")
            else:
                iv._tool_start_inputs.pop(name, None)
            if isinstance(raw_result, dict):
                import json as _j
                raw_result = _j.dumps(raw_result, indent=2)
            if isinstance(raw_input, dict):
                import json as _j
                raw_input = _j.dumps(raw_input, indent=2)
            iv.tools_raw.append((name, str(raw_result), str(raw_input)))


    elif t == "done":
        if not chunk.get("_sub_agent_id"):  # sub-agent chunks must not overwrite parent status
            tv.status = "completed" if chunk.get("success", True) else "failed"
        tv.phase = "done"
        tv.completed_at = time.time()
        tv.narrator_msg = ""
        if iv: iv._in_reasoning = False


    elif t == "error":
        if not chunk.get("_sub_agent_id"):
            tv.status = "error"
        tv.phase = "error"
        tv.narrator_msg = "error"
        tv.completed_at = time.time()
        if iv: iv._in_reasoning = False

    elif t == "final_answer":
        tv.phase = "done"
        tv.status = "completed"
        tv.narrator_msg = ""
        tv.completed_at = time.time()
        answer = chunk.get("answer", chunk.get("content", chunk.get("chunk", "")))
        if iv: iv._in_reasoning = False
        if answer:
            tv.final_answer = str(answer)

    # === INTERFACE REGISTRY HOOK ===
    get_registry().publish_sync(
        id=f"icli.task.{tv.task_id}",
        data=dataclasses.asdict(tv)
    )
    # === END HOOK ===


# ── A: Footer toolbar renderer ────────────────────────────────────────────────

_was_spinner: list[bool] = [False]

def render_footer_toolbar(
    task_views: dict[str, TaskView],
    focused_id: Optional[str],
    audio_recording: bool = False,
    audio_processing: bool = False,
    overlay_open: bool = False,
    set_interval: Any = None,
    off_set:int = 0
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    bg     = "fg:#111827 "   # konsistent dunkel, kein terminal-default
    fg_dim = "bg:#6b7280"
    import shutil as _shutil
    term_width = _shutil.get_terminal_size().columns
    pad = " " * max(0, term_width)
    if audio_recording:
        if set_interval:
            set_interval(0.1)
        out.append((bg + "fg:#dc2626 bg:#ffffff", " ● REC  F4=stop "))
        out.append((bg + fg_dim, pad))
        return out

    if audio_processing:
        if set_interval:
            set_interval(0.4)
        out.append((bg + "fg:#7c3aed bg:#ffffff", " ⚙ PROC  processing... "))
        out.append((bg + fg_dim, pad))
        return out

    # ── Spinner ──────────────────────────────────────────────────────────────
    spinner_text = get_spinner_toolbar_fragment()
    if spinner_text:

        if set_interval:
            if not _was_spinner[0]:
                set_interval(1.76)
                _was_spinner[0] = True
            else:
                set_interval(0.859)
        out.append(("fg:#fbbf24 bg:#1f2937", f" {spinner_text} "))
        out.append((bg + fg_dim, pad+'\n'))
    else:
        _was_spinner[0] = False


    if not task_views:
        if set_interval:
            set_interval(1)
        out.append((bg + fg_dim, f" ◦ idle   F2=overview  F4=audio  F5=status "))
        out.append((bg + fg_dim, pad))
        return out

    if overlay_open:
        if set_interval:
            set_interval(1)
        out.append((bg + "bg:#67e8f9", " ◎ ZEN+ OPEN "))
        out.append((bg + fg_dim, " Esc=close\n"))

    # Neueste zuerst
    main_tasks = [(tid, tv) for tid, tv in task_views.items()]
    reversed_tasks = main_tasks[::-1]
    total = len(reversed_tasks)

    # Offset clampen
    off_set = max(0, min(off_set, max(0, total - 8)))

    shown = reversed_tasks[off_set:off_set + 8]
    overflow = total - off_set - len(shown)

    for tid, tv in shown:
        _append_task_line(out, tv, tid == focused_id, pad)

        # Finale Antwort / letzte Nachricht für abgeschlossene Tasks
        if tv.status in ("completed", "done") and tv.final_answer:
            # Erste Zeile der Antwort, kurz
            first_line = tv.final_answer.split("\n")[0].strip()
            out.append(("fg:#111827 bg:#4ade80", f"  ↳ {_short(first_line, 90)} "))
            out.append(("fg:#111827 " + fg_dim, pad))

        out.append(("fg:#111827 " + fg_dim, "\n"))

    if overflow:
        out.append(("fg:#111827 " + fg_dim, f"  … +{overflow} more "))
        out.append(("fg:#111827 " + fg_dim, "\n"))

    # Shortcut-Leiste — explizit dunkel hinterlegt
    out.append(("fg:#0f172a bg:#4b5563",
                " F2=detail  F6=focus  F7=cycle  F8=cancel  F9=close-done  F4=audio  F5=status  "+" "*260))
    return out


def _append_task_line(out: list, tv: TaskView, focused: bool, pad: str = " " * 120) -> None:
    bg = "fg:#111827 "
    fg_dim = "bg:#6b7280"
    fg_main = "bg:#e5e7eb"

    # Einrückung für Swarm-Subs
    indent = "    " if tv.is_swarm_sub else " "

    # Fokus-Pfeil
    out.append((bg + ("bg:#67e8f9" if focused else fg_dim),
                f"{indent}▸ " if focused else f"{indent}  "))

    # Status-Symbol
    sym, col = STATUS_SYM.get(tv.status, ("◯", "cyan"))
    out.append((bg + f"bg:{C[col]}", f"{sym} "))

    # Agent-Name (Summary mit 🐝, Subs mit gekürztem Namen)
    name = tv.agent_name
    if tv.is_swarm_summary:
        name = f"🐝 {name}"
    elif tv.is_swarm_sub:
        # Nur Kategorie zeigen (planner_abc123 → planner)
        parts = name.split("_")
        name = parts[0] if len(parts) > 1 else name
    width = 12 if tv.is_swarm_sub else 10
    out.append((bg + fg_main, f"{_short(name, width):<{width}} "))

    # Progress-Bar
    bar = _bar(tv.iteration, tv.max_iter, 8)
    out.append((bg + "bg:#67e8f9", bar))
    out.append((bg + fg_dim, f" {tv.iteration}/{tv.max_iter:<3} "))

    # Persona (nur auf Non-Summary, Non-Sub)
    if tv.persona and tv.persona != "default" and not tv.is_swarm_sub and not tv.is_swarm_summary:
        p_name = tv.persona if tv.status in ("completed", "done", "failed", "error") \
            else _short(tv.persona, 14)
        out.append((bg + "bg:#a78bfa", f"{p_name} "))
    else:
        out.append((bg + fg_dim, " "))

    # Token-Prozent
    if tv.tokens_max > 0:
        pct = min(100, int(100 * tv.tokens_used / tv.tokens_max))
        tc = C["green"] if pct < 50 else (C["amber"] if pct < 80 else C["red"])
        out.append((bg + f"bg:{tc}", f"{pct:3d}% "))
    else:
        out.append((bg + fg_dim, "     "))

    # ═══ SUMMARY-BRANCH: Phase + Sub-Counter ═══
    if tv.is_swarm_summary:
        phase = tv.swarm_phase or "init"
        phase_col = {
            "init":       "#6b7280",
            "planning":   "#fbbf24",
            "coding":     "#60a5fa",
            "validating": "#a78bfa",
            "fixing":     "#f59e0b",
            "done":       "#4ade80",
            "error":      "#f87171",
        }.get(phase, "#6b7280")
        out.append((bg + f"bg:{phase_col}", f" {phase.upper()} "))

        if tv.sub_agents:
            total = len(tv.sub_agents)
            done = sum(1 for s in tv.sub_agents.values() if s == 1)
            running = sum(1 for s in tv.sub_agents.values() if s == 0)
            err = sum(1 for s in tv.sub_agents.values() if s == 2)
            parts = [f"🐝 {done}/{total}"]
            if running:
                parts.append(f"⟳{running}")
            if err:
                parts.append(f"✗{err}")
            out.append((bg + "bg:#f472b6", " " + "  ".join(parts) + " "))

        if tv.narrator_msg:
            msg = tv.narrator_msg
            if len(msg) > len(pad) - 20:
                msg = msg[:len(pad) - 20] + "..."
            out.append((bg + "bg:#3b82f6", f" {msg} "))
        out.append((bg + fg_dim, pad))
        return

    # ═══ STANDARD-BRANCH: für Non-Summary (inkl. einzelne Swarm-Subs) ═══
    if tv.status in ("completed", "done"):
        elapsed = (tv.completed_at or tv.started_at) - tv.started_at
        out.append((bg + f"bg:{C['green']}", f"● {_fmt_elapsed(elapsed)} "))
        if tv.narrator_msg:
            out.append((bg + "bg:#60a5fa", tv.narrator_msg + " "))

    elif tv.status in ("failed", "error"):
        elapsed = (tv.completed_at or tv.started_at) - tv.started_at
        out.append((bg + f"bg:{C['red']}", f"✗ {_fmt_elapsed(elapsed)} "))

    else:
        if tv.phase == "thinking":
            out.append((bg + fg_dim, f"◎ {_short(tv.last_thought.replace(chr(10), ' '), 20)} "))
        elif tv.phase in ("tool", "tool_done") and tv.last_tool:
            ok_col = C["green"] if tv.last_tool_ok else C["red"]
            ok_sym = SYM["ok"] if tv.last_tool_ok else SYM["fail"]
            out.append((bg + "bg:#60a5fa", f"◇ {_short(tv.last_tool, 14)} "))
            out.append((bg + f"bg:{ok_col}", f"{ok_sym} "))
            if tv.last_tool_info:
                out.append((bg + fg_dim, f"{_short(tv.last_tool_info, 15)} "))
        else:
            out.append((bg + fg_dim, "⋯ "))

        if tv.narrator_msg:
            n_col = "bg:#f59e0b " + bg if focused else "bg:#3b82f6 " + bg
            icon = " 🔉 " if focused else " 🔈 "
            narrator_msg = tv.narrator_msg
            if len(narrator_msg) > len(pad) - 15:
                narrator_msg = narrator_msg[:len(pad) - 15] + "..."
            out.append((bg + n_col, f"{icon}{narrator_msg} "))

    # Nested-Sub-Counter (nur für Non-Summary, Non-Swarm-Sub mit nested Kindern)
    if tv.sub_agents and not tv.is_swarm_summary and not tv.is_swarm_sub:
        out.append((bg + "bg:#f472b6", f"✦{len(tv.sub_agents)} "))

    out.append((bg + fg_dim, pad))
# ── B: Fullscreen overlay ─────────────────────────────────────────────────────

class TaskOverlay:
    """
    Fullscreen detail view. Shared reference to task_views — no copy, no queue.
    ingest_chunk() writes directly; invalidate() redraws.
    """

    def __init__(self, task_views: dict[str, TaskView]):
        self._views = task_views
        self._selected: str = ""  # aktuell gewählte task_id
        self._selected_sub: str = ""  # gewählte sub-agent id ("" = task selbst)
        self._focus: str = "left"  # "left" | "right"
        self._left_scroll: int = 0  # scroll in linker Liste (für viele tasks)
        self._right_scroll: int = 0  # scroll im rechten Panel
        self._input_scroll: int = 0  # Scroll-Position für Input
        self._scroll_focus: str = "result"  # Fokus ("input" oder "result")
        self._iter_scroll_focus: str = "iter"
        self._selected_iter: Optional[int] = None  # None = liste; int = drill-down
        self._right_iter_cursor: int = 0  # Index in reversed(tv.iterations) — welche Iter ist im Cursor
        self._selected_tool_idx: int = 0  # Welches Tool ist im Drill-Down ausgewählt
        self._tool_view: str = "list"  # "list" | "detail" — detail zeigt raw args+result
        self._app: Optional[Application] = None
        self._last_content_lines: int = 0
        self._last_final_lines: int = 0
        self._final_scroll: int = 0
        self._visible_height: int = 20

    def _max_scroll(self, total_lines: int, visible: int = 20) -> int:
        """Verhindert Scrollen über das Content-Ende hinaus."""
        return max(0, total_lines - visible)

    def _left_items(self) -> list[tuple[str, str]]:
        """Returns list of (task_id, sub_name). sub_name == "" für direkte TaskViews."""
        items: list[tuple[str, str]] = []
        seen_tids: set[str] = set()

        for tid, tv in self._views.items():
            if tid in seen_tids:
                continue

            # Legacy nested subs (durch _ingest_chunk erzeugt) überspringen
            if "__sub__" in tid and not tv.is_swarm_sub:
                continue

            # Swarm-Subs werden direkt nach ihrem Parent eingefügt — skip hier
            if tv.is_swarm_sub:
                continue

            # Reguläre TaskView oder Swarm-Summary als Top-Level-Eintrag
            items.append((tid, ""))
            seen_tids.add(tid)

            # ── Swarm-Summary: direkt darunter seine Swarm-Subs ──
            if tv.is_swarm_summary:
                for sub_name, sub_tid in tv.sub_task_ids.items():
                    if sub_tid in self._views and sub_tid not in seen_tids:
                        # Als eigenen Eintrag mit sub_name markieren
                        items.append((sub_tid, ""))
                        seen_tids.add(sub_tid)

            # ── Non-Swarm mit nested subs (legacy): als (parent_tid, sub_name) ──
            elif tv.status == "running" and tv.sub_agents:
                for sub in tv.sub_agents:
                    items.append((tid, sub))

        return items

    def _effective_view(self) -> Optional[TaskView]:
        if self._selected_sub:
            parent_tv = self._views.get(self._selected)
            if parent_tv:
                sub_task_id = parent_tv.sub_task_ids.get(self._selected_sub)
                if sub_task_id and sub_task_id in self._views:
                    return self._views[sub_task_id]
            # Fallback: Agent-Name oder Task-ID passt direkt
            for tv in self._views.values():
                if tv.agent_name == self._selected_sub or tv.task_id == self._selected_sub:
                    return tv
        return self._views.get(self._selected)

    async def run(self, on_exit) -> None:
        items = self._left_items()
        # Erstes laufendes Item vorauswählen
        for tid, sub in items:
            tv = self._views.get(tid)
            if tv and tv.status == "running":
                self._selected = tid
                self._selected_sub = sub
                break
        else:
            if items:
                self._selected, self._selected_sub = items[0]

        def _left_style():
            return "bg:ansiblack" if self._focus == "right" else "bg:#0a0f1a"

        def _right_style():
            return "bg:ansiblack" if self._focus == "left" else "bg:#0a0f1a"

        left = Window(
            FormattedTextControl(self._render_left, focusable=False),
            width=Dimension(min=28, max=32),
            style="bg:ansiblack",
        )
        right = Window(
            FormattedTextControl(self._render_right, focusable=False),
            style="bg:ansiblack",
        )
        divider = Window(width=1, char="│", style="fg:#374151 bg:ansiblack")
        footer_win = Window(
            FormattedTextControl(self._render_footer, focusable=False),
            height=1,
            style="bg:#111827",
        )

        self._app = Application(
            layout=Layout(HSplit([VSplit([left, divider, right]), footer_win])),
            key_bindings=self._build_keys(on_exit),
            full_screen=True,
            mouse_support=False,
        )
        await self._app.run_async()

    def invalidate(self) -> None:
        if self._app:
            self._app.invalidate()

    # ── renderers ────────────────────────────────────────────────────────────

    def _render_left(self) -> FormattedText:
        bg = "bg:ansiblack "
        focus_bg = "bg:#1a2035 "
        out: list[tuple[str, str]] = [
            (bg + "fg:#67e8f9 bold", " ◯ Tasks\n"),
            (bg + "fg:#374151", " " + "─" * 28 + "\n"),
        ]

        for tid, tv in self._views.items():
            # Legacy nested subs überspringen
            if "__sub__" in tid and not tv.is_swarm_sub:
                continue
            # Swarm-Subs werden zusammen mit ihrem Parent gerendert
            if tv.is_swarm_sub:
                continue

            task_sel = (tid == self._selected and self._selected_sub == "")
            row_bg = focus_bg if task_sel else bg
            sym, col = STATUS_SYM.get(tv.status, ("◯", "cyan"))
            focus_arrow = "▸ " if (self._focus == "left" and task_sel) else "  "

            # Summary-Entry hat 🐝-Prefix
            name = tv.agent_name
            if tv.is_swarm_summary:
                name = f"🐝 {name}"

            out.append((row_bg + f"fg:{C[col]}", f" {focus_arrow}{sym} "))
            out.append((row_bg + ("fg:#ffffff bold" if task_sel else "fg:#e5e7eb"),
                        _short(name, 18) + "\n"))

            # Phase + Sub-Counter unter Summary
            if tv.is_swarm_summary:
                phase = tv.swarm_phase or "init"
                out.append((row_bg + "fg:#fbbf24",
                            f"     {phase.upper()}"))
                if tv.sub_agents:
                    done = sum(1 for s in tv.sub_agents.values() if s == 1)
                    total = len(tv.sub_agents)
                    out.append((row_bg + "fg:#f472b6", f"  {done}/{total}"))
                out.append((row_bg, "\n"))

                # ── Swarm-Subs direkt darunter als navigierbare Top-Level-Items ──
                for sub_name, sub_tid in tv.sub_task_ids.items():
                    sub_tv = self._views.get(sub_tid)
                    if not sub_tv:
                        continue

                    sub_sel = (sub_tid == self._selected and self._selected_sub == "")
                    sub_bg = focus_bg if sub_sel else bg
                    sub_arrow = "  ▸" if (self._focus == "left" and sub_sel) else "   "

                    sub_sym, sub_col = STATUS_SYM.get(sub_tv.status, ("✦", "purple"))
                    # Kategorie-Name (planner_abc → planner)
                    display_name = sub_name.split("_")[0]

                    out.append((sub_bg + f"fg:{C.get(sub_col, '#f472b6')}",
                                f"{sub_arrow} {sub_sym} "))
                    out.append((sub_bg + ("fg:#ffffff bold" if sub_sel else "fg:#d1a8f0"),
                                _short(display_name, 14) + "\n"))

                    # Iter-Anzeige unter Swarm-Sub
                    if sub_sel:
                        if sub_tv.iteration or sub_tv.max_iter:
                            out.append((sub_bg + "fg:#6b7280",
                                        f"       iter {sub_tv.iteration}/{sub_tv.max_iter}\n"))
                        if sub_tv.last_tool:
                            out.append((sub_bg + "fg:#60a5fa",
                                        f"       ◇ {_short(sub_tv.last_tool, 14)}\n"))
                continue  # ← wichtig: nicht auch noch als normalen Eintrag rendern

            # ── Legacy: expanded nested subs (non-swarm) ──
            for sub in tv.sub_agents:
                sub_sel = (tid == self._selected and self._selected_sub == sub)
                sub_bg = "bg:#1f2937 " if sub_sel else bg
                sub_arrow = "  ▸" if (self._focus == "left" and sub_sel) else "   "

                sub_task_id = tv.sub_task_ids.get(sub)
                sub_tv = self._views.get(sub_task_id) if sub_task_id else None
                if not sub_tv:
                    sub_tv = next(
                        (v for v in self._views.values()
                         if v.agent_name == sub or v.task_id == sub), None
                    )

                sub_sym, sub_col = STATUS_SYM.get(
                    sub_tv.status if sub_tv else "running", ("✦", "purple")
                )
                out.append((sub_bg + f"fg:{C.get(sub_col, '#f472b6')}",
                            f"{sub_arrow} {sub_sym} "))
                out.append((sub_bg + ("fg:#ffffff bold" if sub_sel else "fg:#d1a8f0"),
                            _short(sub, 16) + "\n"))

                if sub_sel and sub_tv:
                    if sub_tv.persona and sub_tv.persona != "default":
                        out.append((sub_bg + "fg:#a78bfa",
                                    f"       ✦ {_short(sub_tv.persona, 14)}\n"))
                    if sub_tv.skills:
                        skills_str = " ".join(sub_tv.skills[:3])
                        out.append((sub_bg + "fg:#60a5fa",
                                    f"       {_short(skills_str, 18)}\n"))
                    out.append((sub_bg + "fg:#6b7280",
                                f"       iter {sub_tv.iteration}/{sub_tv.max_iter}\n"))

        out.append((bg + "fg:#374151", " " + "─" * 28 + "\n"))
        if self._focus == "left":
            out.append((bg + "fg:#6b7280", " ↑↓=nav  →/Enter=right\n"))
        else:
            out.append((bg + "fg:#6b7280", " ←=focus left\n"))
        return FormattedText(out)

    def _render_right(self) -> FormattedText:
        bg = "bg:ansiblack "
        sel_bg = "bg:#1a2035 "
        out: list[tuple[str, str]] = []
        tv = self._effective_view()
        if not tv:
            out.append((bg + "fg:#6b7280", " No task selected\n"))
            return FormattedText(out)

        # ── Header ──────────────────────────────────────────────────────────
        # ── Header ──────────────────────────────────────────────────────────
        sym, col = STATUS_SYM.get(tv.status, ("◯", "cyan"))
        out.append((bg + f"fg:{C[col]} bold", f" {sym} {tv.agent_name}"))

        if tv.is_swarm_sub:
            out.append((bg + "fg:#f472b6", "  🐝 swarm-sub"))
        elif tv.is_swarm_summary:
            out.append((bg + "fg:#fbbf24", f"  🐝 swarm summary ({tv.swarm_phase or 'init'})"))
        elif self._selected_sub:
            out.append((bg + "fg:#f472b6", "  ✦ sub-agent"))

        out.append((bg + "fg:#6b7280", f"  {_short(tv.query, 52)}\n"))
        out.append((bg + "fg:#374151", f"  {_short(tv.narrator_msg, 52)}\n"))

        if tv.persona and tv.persona != "default":
            out.append((bg + "fg:#a78bfa", f"   Persona: {tv.persona}"))
            if tv.skills:
                out.append((bg + "fg:#6b7280", "  Skills: "))
                out.append((bg + "fg:#60a5fa", " ".join(tv.skills[:8])))
            out.append((bg, "\n"))

        if tv.tokens_max > 0:
            pct = min(100, int(100 * tv.tokens_used / tv.tokens_max))
            filled = int(20 * pct / 100)
            bar = "█" * filled + "░" * (20 - filled)
            tc = C["green"] if pct < 50 else (C["amber"] if pct < 80 else C["red"])
            out.append((bg + "fg:#6b7280", "   Tokens: "))
            out.append((bg + f"fg:{tc}", f"[{bar}] {pct}%"))
            out.append((bg + "fg:#6b7280", f"  {tv.tokens_used:,}/{tv.tokens_max:,}\n"))

        bar_s = _bar(tv.iteration, tv.max_iter, 20)
        out.append((bg + "fg:#67e8f9", f"   {bar_s}"))
        out.append((bg + "fg:#6b7280", f"  iter {tv.iteration}/{tv.max_iter}\n"))
        out.append((bg + "fg:#374151", "   " + "─" * 64 + "\n"))

        # ── Drill-Down: eine Iteration im Detail ─────────────────────────────
        if self._selected_iter is not None:
            iv = tv._iter_map.get(self._selected_iter)
            if not iv:
                out.append((bg + "fg:#f87171", f"   Iter {self._selected_iter} not found\n"))
                return FormattedText(out)

            is_cur = iv.n == tv.iteration and tv.status == "running"
            hint = " ▸ running" if is_cur else " ● done"
            out.append((bg + "fg:#fbbf24 bold",
                        f"   ── Iter {iv.n}{hint} " + "─" * 40 + "\n"))

            # ── Tool-Detail-Ansicht ──────────────────────────────────────────
            if self._tool_view == "detail" and iv.tools:
                idx = min(self._selected_tool_idx, len(iv.tools) - 1)
                tname, tok, elapsed, info = iv.tools[idx]
                raw_result = iv.tools_raw[idx][1] if idx < len(iv.tools_raw) else ""
                raw_input  = iv.tools_raw[idx][2] if idx < len(iv.tools_raw) else ""
                ok_col = C["green"] if tok else C["red"]
                ok_sym = SYM["ok"] if tok else SYM["fail"]

                out.append((bg + "fg:#60a5fa bold",
                            f"   ◇ Tool {idx + 1}/{len(iv.tools)}: {tname}\n"))
                out.append((bg + f"fg:{ok_col}", f"   {ok_sym}  {elapsed:.3f}s\n"))
                out.append((bg + "fg:#374151", "   " + "─" * 64 + "\n"))

                if raw_input:
                    import textwrap as _tw
                    marker = " [Fokus]" if self._scroll_focus == "input" else ""
                    out.append((bg + "fg:#a78bfa bold", f"   → Input / Arguments{marker}:\n"))

                    display_lines = []
                    for line in raw_input.split("\n"):
                        display_lines.extend(_tw.wrap(line, width=72) or [""])

                    skip = self._input_scroll
                    # Input auf max. 15 Zeilen begrenzen, damit Result sichtbar bleibt
                    visible_lines = display_lines[skip: skip + 20]

                    # Nach dem Rendern: max_scroll merken für bounded up/down
                    if self._scroll_focus == "input":
                        max_input = max(0, len(display_lines) - 20)
                        self._input_scroll = min(self._input_scroll, max_input)

                    for line in visible_lines:
                        out.append((bg + "fg:#d1d5db", f"     {line}\n"))

                    if len(display_lines) > 15:
                        out.append((bg + "fg:#6b7280",
                                    f"     ... ({min(skip + 15, len(display_lines))}/{len(display_lines)} lines)\n"))

                    out.append((bg + "fg:#374151", "   " + "─" * 64 + "\n"))

                import textwrap as _tw

                if raw_result:
                    marker = " [Fokus]" if self._scroll_focus == "result" else ""
                    out.append((bg + "fg:#4ade80 bold", f"   ← Result{marker}:\n"))

                    display_lines: list[str] = []
                    for line in raw_result.split("\n"):
                        display_lines.extend(_tw.wrap(line, width=72) or [""])

                    skip = self._right_scroll
                    # Result auf max. 20 Zeilen begrenzen
                    visible_lines = display_lines[skip: skip + 20]

                    for line in visible_lines:
                        out.append((bg + "fg:#e5e7eb", f"     {line}\n"))

                    if len(display_lines) > 20:
                        out.append((bg + "fg:#6b7280",
                                    f"     ... ({min(skip + 20, len(display_lines))}/{len(display_lines)} lines)\n"))
                    else:
                        out.append((bg + "fg:#6b7280", "   (no result data)\n"))

                    out.append((bg + "fg:#374151", "\n   " + "─" * 64 + "\n"))
                    out.append((bg + "fg:#6b7280",
                                "   ←/→=prev/next  Backspace=list  j/k=scroll  o/l=focus input/result\n"))
                    return FormattedText(out)
                else:
                    out.append((bg + "fg:#6b7280", "   (no result data)\n"))

                out.append((bg + "fg:#374151", "\n   " + "─" * 64 + "\n"))
                out.append((bg + "fg:#6b7280",
                            "   ←/→ = prev/next tool  Backspace = tool list  j/k = scroll result\n"))
                return FormattedText(out)

            # ── Tool-Liste innerhalb der Iter ────────────────────────────────
            out.append((bg + "fg:#6b7280",
                        "   ← Backspace=iter list   Enter=tool detail   j/k=scroll\n"))
            out.append((bg + "fg:#374151", "   " + "─" * 64 + "\n"))

            if iv.thoughts:
                out.append((bg + "fg:#a78bfa bold", "   ◎ Thoughts:\n"))
                # Flatten: alle Thought-Zeilen + Separator in eine Liste
                all_lines: list[tuple[bool, str]] = []  # (is_sep, text)
                for thought in iv.thoughts:
                    for line in thought.split("\n"):
                        all_lines.append((False, line))
                    all_lines.append((True, "·"))

                skip = self._right_scroll
                visible = all_lines[skip: skip + 20]
                for is_sep, line in visible:
                    if is_sep:
                        out.append((bg + "fg:#374151", "     ·\n"))
                    else:
                        out.append((bg + "fg:#d1d5db", f"     {line}\n"))
                if len(all_lines) > 20:
                    out.append((bg + "fg:#6b7280",
                                f"     ... ({min(skip + 20, len(all_lines))}/{len(all_lines)} lines)\n"))

            if iv.tools:
                out.append((bg + "fg:#60a5fa bold", f"   ◇ Tools ({len(iv.tools)}):\n"))
                for idx, (tname, tok, elapsed, info) in enumerate(iv.tools):
                    is_sel = (self._focus == "right" and
                              idx == self._selected_tool_idx % len(iv.tools))
                    row_bg = sel_bg if is_sel else bg
                    ok_col = C["green"] if tok else C["red"]
                    ok_sym = SYM["ok"] if tok else SYM["fail"]
                    cursor = "▸ " if is_sel else "  "
                    elapsed_s = f"{elapsed:.2f}s" if elapsed > 0 else "—"
                    out.append((row_bg + "fg:#60a5fa",
                                f"   {cursor}◇ "))
                    out.append((row_bg + "fg:#e5e7eb",
                                f"{_short(tname, 20):<20} "))
                    out.append((row_bg + f"fg:{ok_col}", f"{ok_sym}  "))
                    out.append((row_bg + "fg:#6b7280", f"{elapsed_s:>7}  "))
                    if info:
                        out.append((row_bg + "fg:#9ca3af", _short(info, 36)))
                    out.append((row_bg, "\n"))

            if iv.pending_tool:
                out.append((bg + "fg:#60a5fa", "   ◇ "))
                out.append((bg + "fg:#fbbf24",
                            f"{_short(iv.pending_tool, 20):<20} ⋯ running...\n"))

            if not iv.thoughts and not iv.tools and not iv.pending_tool:
                out.append((bg + "fg:#6b7280", "   (no data yet)\n"))

            if tv.final_answer and iv.n == max(
                (i.n for i in tv.iterations), default=0
            ):
                out.append((bg + "fg:#374151", "\n   " + "─" * 64 + "\n"))
                out.append((bg + "fg:#4ade80 bold", "   ● Final Answer:\n\n"))
                for line in tv.final_answer.split("\n"):
                    out.append((bg + "fg:#e5e7eb", f"   {line}\n"))

            return FormattedText(out)

        # ── Iterations-Liste (kein Drill-Down) ──────────────────────────────
        iters = list(reversed(tv.iterations))

        # Zwei unabhängige Scroll-Bereiche: Iterationen + Final Answer
        # Flatten alle Iter-Zeilen in eine Liste für saubere Scroll-Berechnung
        iter_lines_flat: list[tuple[str, str]] = []

        for list_idx, iv in enumerate(iters):
            is_cur = iv.n == tv.iteration and tv.status == "running"
            is_cursor_sel = (self._focus == "right"
                             and self._iter_scroll_focus == "iter"
                             and list_idx == self._right_iter_cursor)
            hint = " ▸ running" if is_cur else ""
            cursor_sym = "▸ " if is_cursor_sel else "  "
            label_col = "#ffffff bold" if is_cursor_sel else "#fbbf24 bold"
            iter_bg = sel_bg if is_cursor_sel else bg

            iter_lines_flat.append((iter_bg + f"fg:{label_col}",
                                    f"   {cursor_sym}── iter {iv.n}{hint} " + "─" * 28 + "\n"))

            for thought in iv.thoughts:
                iter_lines_flat.append((bg + "fg:#6b7280", "      ◎ "))
                iter_lines_flat.append((bg + "fg:#e5e7eb",
                                        _short(thought.replace("\n", " "), 68) + "\n"))

            for tname, tok, elapsed, info in iv.tools:
                ok_col = C["green"] if tok else C["red"]
                ok_sym = SYM["ok"] if tok else SYM["fail"]
                elapsed_s = f"{elapsed:.2f}s" if elapsed > 0 else "     "
                iter_lines_flat.append((bg + "fg:#60a5fa", "      ◇ "))
                iter_lines_flat.append((bg + "fg:#e5e7eb", f"{_short(tname, 16):<16} "))
                iter_lines_flat.append((bg + f"fg:{ok_col}", f"{ok_sym}  "))
                iter_lines_flat.append((bg + "fg:#6b7280", f"{elapsed_s:>7}  "))
                if info:
                    iter_lines_flat.append((bg + "fg:#9ca3af", _short(info, 36)))
                iter_lines_flat.append((bg, "\n"))

            if iv.pending_tool:
                iter_lines_flat.append((bg + "fg:#60a5fa", "      ◇ "))
                iter_lines_flat.append((bg + "fg:#fbbf24",
                                        f"{_short(iv.pending_tool, 16):<16} ⋯ running...\n"))

        # Zähle Zeilen anhand "\n"-Vorkommen in den Text-Fragmenten
        def _count_lines(fragments: list[tuple[str, str]]) -> int:
            return sum(frag[1].count("\n") for frag in fragments)

        iter_total_lines = _count_lines(iter_lines_flat)
        self._last_content_lines = iter_total_lines

        # Auto-scroll: halte ausgewählte Iter im Blick
        # (vereinfachte Heuristik — präzise Cursor-Verfolgung könnte man verfeinern)
        visible = self._visible_height
        max_iter_scroll = self._max_scroll(iter_total_lines, visible)
        self._right_scroll = min(self._right_scroll, max_iter_scroll)

        # Fragmente ausgeben mit Scroll-Offset (zeilenweise überspringen)
        skip = self._right_scroll
        skipped = 0
        emitted_lines = 0
        for style, text in iter_lines_flat:
            if skipped < skip:
                nl_count = text.count("\n")
                if skipped + nl_count <= skip:
                    skipped += nl_count
                    continue
                # Teilweise überspringen: finde die erste noch sichtbare Zeile
                parts = text.split("\n")
                remaining = skip - skipped
                if remaining < len(parts) - 1:
                    text = "\n".join(parts[remaining:])
                    skipped = skip
                else:
                    skipped += nl_count
                    continue
            if emitted_lines >= visible:
                break
            out.append((style, text))
            emitted_lines += text.count("\n")

        # Scroll-Indikator für Iter-Bereich
        if iter_total_lines > visible:
            pos_s = f"{min(self._right_scroll + visible, iter_total_lines)}/{iter_total_lines}"
            focus_marker = " [iter ◂]" if self._iter_scroll_focus == "iter" else ""
            out.append((bg + "fg:#6b7280", f"   ── scroll {pos_s}{focus_marker}\n"))

        # ── Final Answer (eigener Scroll-Bereich) ────────────────────────────
        if tv.final_answer:
            final_lines_all = tv.final_answer.split("\n")
            self._last_final_lines = len(final_lines_all)

            out.append((bg + "fg:#374151", "\n   " + "─" * 64 + "\n"))
            focus_marker = " [final ◂]" if self._iter_scroll_focus == "final" else ""
            out.append((bg + "fg:#4ade80 bold", f"   ● Final Answer:{focus_marker}\n\n"))

            visible_final = 12
            max_final_scroll = self._max_scroll(len(final_lines_all), visible_final)
            # Begrenzen (für Fall dass Nachricht kürzer wird)
            if self._iter_scroll_focus == "final":
                pass  # scroll wird über keys gesetzt
            final_scroll = getattr(self, "_final_scroll", 0)
            final_scroll = min(final_scroll, max_final_scroll)
            self._final_scroll = final_scroll

            visible_slice = final_lines_all[final_scroll: final_scroll + visible_final]
            for line in visible_slice:
                out.append((bg + "fg:#e5e7eb", f"   {line}\n"))

            if len(final_lines_all) > visible_final:
                out.append((bg + "fg:#6b7280",
                            f"   ── final {min(final_scroll + visible_final, len(final_lines_all))}/"
                            f"{len(final_lines_all)}\n"))

        if not tv.iterations and not tv.final_answer:
            out.append((bg + "fg:#6b7280", "   waiting for first iteration...\n"))

        return FormattedText(out)

    def _render_footer(self) -> FormattedText:
        tv = self._effective_view()
        n_run = sum(1 for v in self._views.values() if v.status == "running")
        iter_s = f"iter {tv.iteration}/{tv.max_iter}" if tv else ""
        sub_s = f" ✦ {_short(self._selected_sub, 12)}" if self._selected_sub else ""

        if self._selected_iter is not None and self._tool_view == "detail":
            iv = tv._iter_map.get(self._selected_iter) if tv else None
            n_tools = len(iv.tools) if iv else 0
            cur_tool = min(self._selected_tool_idx, n_tools - 1) + 1 if n_tools else 0
            hint = (f" ↑↓=scroll  o/l=input/result  ^←/^→=prev/next tool ({cur_tool}/{n_tools})"
                    f"  Backspace=back")
        elif self._selected_iter is not None:
            iv = tv._iter_map.get(self._selected_iter) if tv else None
            n_tools = len(iv.tools) if iv else 0
            cur_tool = self._selected_tool_idx % n_tools + 1 if n_tools else 0
            hint = (f" ↑↓=select tool ({cur_tool}/{n_tools})"
                    f"  Enter=detail  Backspace=back")
        elif self._focus == "left":
            hint = " ↑↓=navigate  →/Enter=right  Tab=switch"
        else:
            n_iters = len(tv.iterations) if tv else 0
            cur_iter = self._right_iter_cursor + 1 if n_iters else 0
            region = "iter" if self._iter_scroll_focus == "iter" else "final"
            has_final = " │ o/l=iter/final" if tv and tv.final_answer else ""
            hint = (f" ↑↓=scroll {region} ({cur_iter}/{n_iters}){has_final}"
                    f"  Enter=drill-in  ←=back")

        return FormattedText([(
            "bg:#111827 fg:#6b7280",
            f"{hint} │ {n_run} running │ {iter_s}{sub_s}  Esc/F2=close "
        )])

    # ── keys ────────────────────────────────────────────────────────────────

    def _build_keys(self, on_exit) -> KeyBindings:
        kb = KeyBindings()
        ov = self

        @kb.add("escape")
        @kb.add("f2")
        def _close(event):
            event.app.exit()
            on_exit()

        @kb.add("tab")
        def _toggle_focus(event):
            ov._focus = "right" if ov._focus == "left" else "left"
            ov._selected_iter = None
            ov._tool_view = "list"

        # ── Enter: Drill-In ──────────────────────────────────────────────────
        @kb.add("right")
        @kb.add("enter")
        def _enter_or_drill(event):
            if ov._focus == "left":
                ov._focus = "right"
                return

            if ov._selected_iter is None:
                # Iter-Liste → Drill into cursor-Iter
                tv = ov._effective_view()
                if not tv or not tv.iterations:
                    return
                iters_rev = list(reversed(tv.iterations))
                idx = min(ov._right_iter_cursor, len(iters_rev) - 1)
                ov._selected_iter = iters_rev[idx].n
                ov._selected_tool_idx = 0
                ov._tool_view = "list"
                ov._right_scroll = 0
            elif ov._tool_view == "list":
                # Iter-Drill → Tool-Detail öffnen wenn Tool ausgewählt
                tv = ov._effective_view()
                iv = tv._iter_map.get(ov._selected_iter) if tv else None
                if iv and iv.tools:
                    ov._tool_view = "detail"
                    ov._right_scroll = 0

        # ── ← / Backspace: einen Level zurück ───────────────────────────────
        @kb.add("left")
        @kb.add("backspace")
        def _back(event):
            if ov._tool_view == "detail":
                # Tool-Detail → Tool-Liste
                ov._tool_view = "list"
                ov._right_scroll = 0
            elif ov._selected_iter is not None:
                # Iter-Drill → Iter-Liste
                ov._selected_iter = None
                ov._tool_view = "list"
                ov._right_scroll = 0
            else:
                # Iter-Liste → Fokus nach links
                ov._focus = "left"

        # ── Tool-Navigation in Detail-Ansicht ───────────────────────────────
        @kb.add("c-right")
        def _next_tool(event):
            if ov._selected_iter is not None and ov._tool_view == "detail":
                tv = ov._effective_view()
                iv = tv._iter_map.get(ov._selected_iter) if tv else None
                if iv and iv.tools:
                    ov._selected_tool_idx = (ov._selected_tool_idx + 1) % len(iv.tools)
                    ov._right_scroll = 0
                    ov._input_scroll = 0

        @kb.add("c-left")
        def _prev_tool(event):
            if ov._selected_iter is not None and ov._tool_view == "detail":
                tv = ov._effective_view()
                iv = tv._iter_map.get(ov._selected_iter) if tv else None
                if iv and iv.tools:
                    ov._selected_tool_idx = (ov._selected_tool_idx - 1) % len(iv.tools)
                    ov._right_scroll = 0

        # ── ↑↓ kontextabhängig ──────────────────────────────────────────────
        @kb.add("up")
        def _nav_up(event):
            if ov._focus == "right":
                # Tool-Detail
                if ov._selected_iter is not None and ov._tool_view == "detail":
                    if ov._scroll_focus == "input":
                        ov._input_scroll = max(0, ov._input_scroll - 1)
                    else:
                        ov._right_scroll = max(0, ov._right_scroll - 1)
                    return

                # Tool-Liste innerhalb Iter
                if ov._selected_iter is not None:
                    ov._selected_tool_idx = max(0, ov._selected_tool_idx - 1)
                    return

                # Iter-Liste: zwei Modi
                if ov._iter_scroll_focus == "final":
                    # In Final Answer scrollen
                    ov._final_scroll = max(0, ov._final_scroll - 1)
                else:
                    # Iter-Cursor hoch + Content-Scroll anpassen
                    ov._right_iter_cursor = max(0, ov._right_iter_cursor - 1)
                    ov._right_scroll = max(0, ov._right_scroll - 1)
                return

            # Links: Task-Nav
            items = ov._left_items()
            if not items:
                return
            cur = (ov._selected, ov._selected_sub)
            idx = items.index(cur) if cur in items else 0
            ov._selected, ov._selected_sub = items[(idx - 1) % len(items)]
            ov._selected_iter = None
            ov._tool_view = "list"
            ov._right_scroll = 0
            ov._right_iter_cursor = 0
            ov._final_scroll = 0

        @kb.add("down")
        def _nav_down(event):
            if ov._focus == "right":
                # Tool-Detail
                if ov._selected_iter is not None and ov._tool_view == "detail":
                    if ov._scroll_focus == "input":
                        ov._input_scroll += 1  # Begrenzung via render
                    else:
                        ov._right_scroll += 1
                    return

                # Tool-Liste
                if ov._selected_iter is not None:
                    tv = ov._effective_view()
                    iv = tv._iter_map.get(ov._selected_iter) if tv else None
                    max_idx = len(iv.tools) - 1 if iv and iv.tools else 0
                    ov._selected_tool_idx = min(max_idx, ov._selected_tool_idx + 1)
                    return

                # Iter-Liste
                if ov._iter_scroll_focus == "final":
                    max_fs = ov._max_scroll(ov._last_final_lines, 12)
                    ov._final_scroll = min(max_fs, ov._final_scroll + 1)
                else:
                    tv = ov._effective_view()
                    n = len(tv.iterations) if tv else 0
                    ov._right_iter_cursor = min(max(n - 1, 0), ov._right_iter_cursor + 1)
                    max_is = ov._max_scroll(ov._last_content_lines, ov._visible_height)
                    ov._right_scroll = min(max_is, ov._right_scroll + 1)
                return

            # Links
            items = ov._left_items()
            if not items:
                return
            cur = (ov._selected, ov._selected_sub)
            idx = items.index(cur) if cur in items else -1
            ov._selected, ov._selected_sub = items[(idx + 1) % len(items)]
            ov._selected_iter = None
            ov._tool_view = "list"
            ov._right_scroll = 0
            ov._right_iter_cursor = 0
            ov._final_scroll = 0

        # ── o/l: Scroll-Fokus wechseln ──────────────────────────────────────
        @kb.add("o")
        @kb.add("s-up")
        @kb.add("s-left")
        def _focus_prev_region(event):
            if ov._tool_view == "detail":
                ov._scroll_focus = "input"
            elif ov._selected_iter is None:
                # Iter-Liste: toggle iter ↔ final
                ov._iter_scroll_focus = "iter"

        @kb.add("l")
        @kb.add("s-down")
        @kb.add("s-right")
        def _focus_next_region(event):
            if ov._tool_view == "detail":
                ov._scroll_focus = "result"
            elif ov._selected_iter is None:
                tv = ov._effective_view()
                # Nur in final wechseln wenn's was gibt
                if tv and tv.final_answer:
                    ov._iter_scroll_focus = "final"

        # ── j/k: 5-Zeilen-Jumps (Power-User) ────────────────────────────────
        @kb.add("j")
        @kb.add("pagedown")
        def _sd(event):
            if ov._tool_view == "detail":
                if ov._scroll_focus == "input":
                    ov._input_scroll += 5
                else:
                    ov._right_scroll += 5
            elif ov._selected_iter is None and ov._iter_scroll_focus == "final":
                max_fs = ov._max_scroll(ov._last_final_lines, 12)
                ov._final_scroll = min(max_fs, ov._final_scroll + 5)
            else:
                max_is = ov._max_scroll(ov._last_content_lines, ov._visible_height)
                ov._right_scroll = min(max_is, ov._right_scroll + 5)

        @kb.add("k")
        @kb.add("pageup")
        def _su(event):
            if ov._tool_view == "detail":
                if ov._scroll_focus == "input":
                    ov._input_scroll = max(0, ov._input_scroll - 5)
                else:
                    ov._right_scroll = max(0, ov._right_scroll - 5)
            elif ov._selected_iter is None and ov._iter_scroll_focus == "final":
                ov._final_scroll = max(0, ov._final_scroll - 5)
            else:
                ov._right_scroll = max(0, ov._right_scroll - 5)


        # ── Direkt zu Iter 1-9 springen ─────────────────────────────────────
        for n in range(1, 10):
            def _jump_iter(event, _n=n):
                tv = ov._effective_view()
                if tv and _n in tv._iter_map and ov._selected_iter is None:
                    ov._selected_iter = _n
                    ov._selected_tool_idx = 0
                    ov._tool_view = "list"
                    ov._right_scroll = 0
                    ov._focus = "right"

            kb.add(str(n))(_jump_iter)

        return kb

# =================== Helpers & Setup ===================
MODEL_MAPPING = {

    # -------------------------------------------------
    # OpenAI
    # -------------------------------------------------
    "gpt-oss-120bF": "openrouter/openai/gpt-oss-120b:free",
    "gpt-oss-120b": "openrouter/openai/gpt-oss-120b",
    "gpt-oss-20b": "openrouter/openai/gpt-oss-20b:nitro",
    "gpt-5.2": "openrouter/openai/gpt-5.2",
    "gpt-5.2-c": "openrouter/openai/gpt-5.2-codex",
    "gpt-5.4": "openrouter/openai/gpt-5.4",
    "gpt-oss-safeguard-20b": "openrouter/openai/gpt-oss-safeguard-20b",

    # -------------------------------------------------
    # Google Gemini
    # -------------------------------------------------
    "gemini-3-flash": "openrouter/google/gemini-3-flash-preview",
    "gemini-3.1": "openrouter/google/gemini-3.1-pro-preview",
    "gemini-flash-3": "openrouter/google/gemini-3-flash-preview",
    "gemini-flash-3.1": "openrouter/google/gemini-3.1-flash-lite-preview",
    "gemini-flash-2.5-lite": "openrouter/google/gemini-2.5-flash-lite",

    # -------------------------------------------------
    # DeepSeek
    # -------------------------------------------------
    "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",

    # -------------------------------------------------
    # Moonshot AI
    # -------------------------------------------------
    "kimi-k2.5": "openrouter/moonshotai/kimi-k2.5",
    "kimi-k2-thinking": "openrouter/moonshotai/kimi-k2-thinking",

    # -------------------------------------------------
    # Anthropic
    # -------------------------------------------------
    "sonnet-4.6": "openrouter/anthropic/claude-sonnet-4.6",
    "opus-4.6": "openrouter/anthropic/claude-opus-4.6",

    # -------------------------------------------------
    # Z-AI (GLM Models)
    # -------------------------------------------------
    "glm-4.5f": "openrouter/z-ai/glm-4.5-air:free",
    "glm-4.6": "openrouter/z-ai/glm-4.6",
    "glm-4.6v": "openrouter/z-ai/glm-4.6v",
    "glm-4.7-flash": "openrouter/z-ai/glm-4.7-flash",
    "glm-4.7": "openrouter/z-ai/glm-4.7",
    "glm-5": "openrouter/z-ai/glm-5",

    # -------------------------------------------------
    # MiniMax
    # -------------------------------------------------
    "minimax-m2.1": "openrouter/minimax/minimax-m2.1",
    "minimax-2.5": "openrouter/minimax/minimax-m2.5",

    # -------------------------------------------------
    # NVIDIA
    # -------------------------------------------------
    "nemotron-3": "openrouter/nvidia/nemotron-3-nano-30b-a3b",

    # -------------------------------------------------
    # Mistral
    # -------------------------------------------------
    "mistral-14b": "openrouter/mistralai/ministral-14b-2512",
    "devstral": "openrouter/mistralai/devstral-2512",

    # -------------------------------------------------
    # StepFun
    # -------------------------------------------------
    "step-3.5-flash": "openrouter/stepfun/step-3.5-flash:free",
    "step-3.5": "openrouter/stepfun/step-3.5-flash",

    # -------------------------------------------------
    # Arcee
    # -------------------------------------------------
    "trinity-large": "openrouter/arcee-ai/trinity-large-preview:free",

    # -------------------------------------------------
    # Inception (Custom Provider)
    # -------------------------------------------------
    "mercury-2": "openrouter/inception/mercury-2",

    # -------------------------------------------------
    # Qwen (OpenRouter)
    # -------------------------------------------------
    "qwen3-coder": "openrouter/qwen/qwen3-coder-next",
    "qwen3.5-27b": "openrouter/qwen/qwen3.5-27b",
    "qwen3.5-35b-3a": "openrouter/qwen/qwen3.5-35b-a3b",
    "qwen3.5-122b-a10": "openrouter/qwen/qwen3.5-122b-a10b",
    "qwen3.5-397b-a17": "openrouter/qwen/qwen3.5-397b-a17b",
    "qwen3.5-flash": "openrouter/qwen/qwen3.5-flash-02-23",
    "qwen3.5-plus": "openrouter/qwen/qwen3.5-plus-02-15",

    "lfm2": "ollama/lfm2",
    "lfm2.5-thinking": "ollama/lfm2.5-thinking",

    "qwen2.5_0.5b": "ollama/qwen2.5:0.5b",

    "qwen3_8b": "ollama/qwen3:8b",
    "qwen3_14b": "ollama/qwen3:14b",

    "qwen3.5_0.8b": "ollama/qwen3.5:0.8b",
    "qwen3.5_2b": "ollama/qwen3.5:2b",
    "qwen3.5_27b": "ollama/qwen3.5:27b",

    "deepseek-r1_8b": "ollama/deepseek-r1:8b",
    "deepseek-r1_14b": "ollama/deepseek-r1:14b",
}


def load_gateway_models():
    try:
        from toolboxv2.mods.isaa.base.IntelligentRateLimiter import gateway
        models = gateway.get_available_models()

        for m in models:
            MODEL_MAPPING[m] = f"gateway/{m}"

    except Exception:
        pass
    MODEL_MAPPING.update(fetch_openrouter_models())
def fetch_openrouter_models():

    try:

        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            timeout=10
        )

        data = r.json()

        return {
            m["id"]: f"openrouter/{m['id']}"
            for m in data["data"]
        }

    except Exception:
        return {}


def start_gateway_loader():
    thread = threading.Thread(
        target=load_gateway_models,
        daemon=True
    )
    thread.start()

start_gateway_loader()

class PTColors:
    """Farb-Mapping für Prompt Toolkit HTML"""
    GREY = 'gray'
    WHITE = 'ansiwhite'
    GREEN = 'ansigreen'
    YELLOW = 'ansiyellow'
    CYAN = 'ansicyan'
    BLUE = 'ansiblue'
    RED = 'ansired'
    MAGENTA = 'ansimagenta'
    BRIGHT_WHITE = '#ffffff'
    BRIGHT_CYAN = '#00ffff'


    # Zen Colors
    ZEN_DIM = '#6b7280'
    ZEN_CYAN = '#67e8f9'
    ZEN_AMBER = '#fbbf24'
    ZEN_GREEN = '#4ade80'
    ZEN_RED = '#fb7185'


class Colors:
    """Legacy Support falls anderer Code direkt Colors.RED aufruft"""
    RED = 'ansired'
    GREEN = 'ansigreen'
    YELLOW = 'ansiyellow'
    BLUE = 'ansiblue'
    CYAN = 'ansicyan'
    GREY = 'gray'
    RESET = ''
    BOLD = 'bold'


def esc(text: Any) -> str:
    """Escaped Text für HTML-Tags, verhindert Crash bei < oder > im Text"""
    return html.escape(str(text).encode().decode(encoding="utf-8", errors="replace"), quote=False)

def c_print(*args, **kwargs):
    """Drop-in Replacement für print, nutzt prompt_toolkit"""
    # Konvertiert alles zu Strings und escaped es
    text = " ".join(str(a) for a in args)

    # Wenn bereits HTML Objekt, direkt drucken, sonst wrappen
    if len(text) == len(args) == 0:
        print()
    elif isinstance(args[0], HTML):
        print_formatted_text(*args, **kwargs)
    elif isinstance(args[0], ANSI):
        print_formatted_text(*args, **kwargs)
    else:
        try:
            print_formatted_text(text, **kwargs)
        except:
            print(text)

def ansi_c_print(*a, **kw):
    text = " ".join(str(x) for x in a)
    # Wir übergeben es explizit als ANSI-Objekt an c_print
    c_print(ANSI(text), **kw)

def print_box_header(title: str, icon: str = "ℹ", width: int = 76):
    """1. Header mit Icon und Titel"""
    c_print(HTML(""))  # Leere Zeile
    c_print(HTML(f"<style font-weight='bold'>{icon} {esc(title)}</style>"))
    c_print(HTML(f"<style fg='{PTColors.GREY}'>{'─' * width}</style>"))


def print_box_footer(width: int = 76):
    """2. Footer (einfacher Abschluss)"""
    c_print(HTML(""))


def print_box_content(text: str, style: str = "", width: int = 76, auto_wrap: bool = True):
    """3. Inhalt mit Icon-Mapping für Status"""
    style_config = {
        'success': {'icon': '✓', 'color': PTColors.GREEN},
        'error': {'icon': '✗', 'color': PTColors.RED},
        'warning': {'icon': '⚠', 'color': PTColors.YELLOW},
        'info': {'icon': 'ℹ', 'color': PTColors.BLUE},
    }

    safe_text = esc(text)
    if style in style_config:
        config = style_config[style]
        # Icon farbig, Text normal
        c_print(HTML(f"  <style fg='{config['color']}'>{config['icon']}</style> {safe_text}"))
    else:
        c_print(HTML(f"  {safe_text}"))


def print_status(message: str, status: str = "info"):
    """4. Statusmeldungen (für Logs, Progress, etc.)"""
    status_config = {
        'success': {'icon': '✓', 'color': PTColors.GREEN},
        'error': {'icon': '✗', 'color': PTColors.RED},
        'warning': {'icon': '⚠', 'color': PTColors.YELLOW},
        'info': {'icon': 'ℹ', 'color': PTColors.BLUE},
        'progress': {'icon': '⟳', 'color': PTColors.CYAN},
        'data': {'icon': '💾', 'color': PTColors.YELLOW},
        'configure': {'icon': '🔧', 'color': PTColors.YELLOW},
        'launch': {'icon': '🚀', 'color': PTColors.GREEN},
    }

    config = status_config.get(status, {'icon': '•', 'color': PTColors.WHITE})
    color_attr = f"fg='{config['color']}'" if config['color'] else ""

    c_print(HTML(f"<style {color_attr}>{config['icon']}</style> {esc(message)}"))


def print_separator(char: str = "─", width: int = 76):
    """5. Trennlinie"""
    c_print(HTML(f"<style fg='{PTColors.GREY}'>{char * width}</style>"))


def print_table_header(columns: list, widths: list):
    """6. Tabellenkopf"""
    header_parts = []
    for (name, _), width in zip(columns, widths):
        # Text fett und hellweiß
        header_parts.append(f"<style font-weight='bold' fg='{PTColors.BRIGHT_WHITE}'>{esc(name):<{width}}</style>")

    # Trenner in Cyan
    sep_parts = [f"<style fg='{PTColors.BRIGHT_CYAN}'>{'─' * w}</style>" for w in widths]

    joined_headers = " │ ".join(header_parts)
    joined_seps = f"<style fg='{PTColors.BRIGHT_CYAN}'>─┼─</style>".join(sep_parts)

    c_print(HTML(f"  {joined_headers}"))
    c_print(HTML(f"  {joined_seps}"))


def print_table_row(values: list, widths: list, styles: list = None):
    """7. Tabellenzeile mit spaltenweisen Farben"""
    if styles is None:
        styles = [""] * len(values)

    color_map = {
        'grey': PTColors.GREY,
        'white': PTColors.WHITE,
        'green': PTColors.GREEN,
        'yellow': PTColors.YELLOW,
        'cyan': PTColors.CYAN,
        'blue': PTColors.BLUE,
        'red': PTColors.RED,
    }

    row_parts = []
    for value, width, style in zip(values, widths, styles):
        safe_val = esc(str(value))
        color = color_map.get(style.lower(), '')

        if color:
            # Wir berechnen das Padding manuell, damit die Farbe nicht den Leerraum füllt
            padding = width - len(safe_val)
            padding_str = " " * max(0, padding)
            row_parts.append(f"<style fg='{color}'>{safe_val}</style>{padding_str}")
        else:
            row_parts.append(f"{safe_val:<{width}}")

    # Vertikale Linien in Grau
    sep = f" <style fg='{PTColors.GREY}'>│</style> "
    c_print(HTML(f"  {sep.join(row_parts)}"))

def json_to_md(data: dict):
    def _(d, t=0):
        indent = "  " * t
        md = ""

        if isinstance(d, dict):
            for k, v in d.items():
                if k == "system_message":
                    continue
                if isinstance(v, (dict, list)):
                    md += f"{indent}- {k}:\n"
                    md += _(v, t + 1)
                else:
                    md += f"{indent}- {k}: {v}\n"   # inline

        elif isinstance(d, list):
            for item in d:
                if isinstance(item, (dict, list)):
                    md += _(item, t)
                else:
                    md += f"{indent}- {item}\n"

        else:
            md += f"{indent}{d}\n"

        return md

    final_md = ""
    for k, v in data.items():
        if k == "system_message":
            continue
        if not v:
            continue
        if v is None:
            continue
        final_md += f"# {k}\n" + _(v, 1)  # war 1
    return esc(final_md)

def print_code_block(code: str, language: str = "text", width: int = 76, show_line_numbers: bool = False):
    """8. Code Block mit Basic Syntax Highlighting"""
    lines = []

    # JSON Highlighting Logic
    if language.lower() == 'json':
        try:
            parsed = json.loads(code) if isinstance(code, str) else code
            formatted = json.dumps(parsed, indent=2)
            raw_lines = formatted.split('\n')
            for line in raw_lines:
                # Key-Highlighting (Cyan für Keys, Grün für Strings)
                safe_line = esc(line)
                if ':' in safe_line:
                    k, v = safe_line.split(':', 1)
                    lines.append(f"<style fg='{PTColors.CYAN}'>{k}</style>:{v}")
                else:
                    lines.append(safe_line)
        except:
            lines = [esc(l) for l in code.split('\n')]

    # YAML/ENV Highlighting Logic
    elif language.lower() in ['yaml', 'yml', 'env']:
        for line in code.split('\n'):
            safe_line = esc(line)
            if safe_line.strip().startswith('#'):
                lines.append(f"<style fg='{PTColors.GREY}'>{safe_line}</style>")
            elif ':' in safe_line:
                k, v = safe_line.split(':', 1)
                lines.append(f"<style fg='{PTColors.CYAN}'>{k}</style>:{v}")
            elif '=' in safe_line:
                k, v = safe_line.split('=', 1)
                lines.append(f"<style fg='{PTColors.CYAN}'>{k}</style>={v}")
            else:
                lines.append(safe_line)

    # Fallback / Markdown (improved)
    else:
        in_code_block = False
        import re as _re

        def render_inline(text: str) -> str:
            result = ""
            i = 0
            while i < len(text):
                if text[i:i + 2] == '**':
                    end = text.find('**', i + 2)
                    if end != -1:
                        result += f"<style fg='{PTColors.YELLOW}'><b>{esc(text[i + 2:end])}</b></style>"
                        i = end + 2;
                        continue
                if text[i] == '*' and text[i:i + 2] != '**':
                    end = text.find('*', i + 1)
                    if end != -1:
                        result += f"<style fg='{PTColors.MAGENTA}'><i>{esc(text[i + 1:end])}</i></style>"
                        i = end + 1;
                        continue
                if text[i] == '`':
                    end = text.find('`', i + 1)
                    if end != -1:
                        result += f"<style fg='{PTColors.GREEN}'>{esc(text[i + 1:end])}</style>"
                        i = end + 1;
                        continue
                if text[i] == '[':
                    m = _re.match(r'\[([^\]]+)\]\(([^)]+)\)', text[i:])
                    if m:
                        result += (
                            f"<style fg='{PTColors.CYAN}'>{esc(m.group(1))}</style>"
                            f"<style fg='{PTColors.GREY}'> ↗ {esc(m.group(2))}</style>"
                        )
                        i += m.end();
                        continue
                result += esc(text[i])
                i += 1
            return result

        def _flush_lines():
            """Accumulated lines sofort ausgeben und leeren."""
            for ln in lines:
                if show_line_numbers:
                    c_print(HTML(f"  <style fg='{PTColors.GREY}'>{len(lines):3d}</style> {ln}"))
                else:
                    c_print(HTML(f"  {ln}"))
            lines.clear()

        def _render_table(buf: list):
            if len(buf) < 2:
                return
            parse = lambda row: [c.strip() for c in row.strip('|').split('|') if c.strip() != '']
            headers = parse(buf[0])
            n = len(headers)

            data_rows = []
            for row in buf[2:]:  # buf[1] = Separator-Zeile überspringen
                sep_line = _re.fullmatch(r'[\|\s\-:]+', row)
                if sep_line:
                    continue
                cells = parse(row)
                cells = (cells + [''] * n)[:n]
                data_rows.append(cells)

            all_cells = [headers] + data_rows
            widths = [
                max((len(all_cells[r][c]) for r in range(len(all_cells)) if c < len(all_cells[r])), default=4) + 2
                for c in range(n)
            ]

            _flush_lines()  # Pending-Lines vor Tabelle ausgeben
            print_separator()
            print_table_header([(h, '') for h in headers], widths)
            for row in data_rows:
                print_table_row(row, widths)
            print_separator()

        table_buf: list[str] = []

        for line in code.split('\n'):
            stripped = line.strip()

            # ─── Tabellenerkennung ────────────────────────────────────────────
            is_table_row = (
                stripped.startswith('|') and stripped.endswith('|')
                and stripped.count('|') >= 2
            )
            if is_table_row:
                table_buf.append(stripped)
                continue
            if table_buf:
                _render_table(table_buf)
                table_buf = []

            # ─── Code-Block ───────────────────────────────────────────────────
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                lang = stripped[3:].strip()
                if in_code_block:
                    label = f" {lang} " if lang else ""
                    bar = '─' * max(0, 34 - len(label))
                    lines.append(f"<style fg='{PTColors.YELLOW}'>┌─{label}{bar}┐</style>")
                else:
                    lines.append(f"<style fg='{PTColors.YELLOW}'>└{'─' * 35}┘</style>")
                continue

            if in_code_block:
                lines.append(
                    f"<style fg='{PTColors.YELLOW}'>│ </style><style fg='{PTColors.GREEN}'>{esc(line)}</style>")
                continue

            if not stripped:
                lines.append("")
                continue

            if _re.match(r'^[-*_]{3,}$', stripped):
                lines.append(f"<style fg='{PTColors.GREY}'>{'─' * 50}</style>")
                continue

            # ─── Headers ──────────────────────────────────────────────────────
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                content = stripped[level:].strip()
                colors = [PTColors.MAGENTA, PTColors.CYAN, PTColors.BLUE,
                          PTColors.WHITE, PTColors.WHITE, PTColors.WHITE]
                underlines = ['═', '─', '·', '', '', '']
                color = colors[min(level - 1, 5)]
                ul = underlines[min(level - 1, 5)]
                lines.append(f"<style fg='{color}'><b>{render_inline(content)}</b></style>")
                if ul:
                    lines.append(f"<style fg='{color}'>{ul * (len(content) + 1)}</style>")
                continue

            # ─── Blockquote ───────────────────────────────────────────────────
            if stripped.startswith('>'):
                lines.append(
                    f"<style fg='{PTColors.GREY}'>▌ </style>"
                    f"<style fg='{PTColors.WHITE}'><i>{render_inline(stripped[1:].strip())}</i></style>"
                )
                continue

            # ─── Bullet list ──────────────────────────────────────────────────
            if _re.match(r'^[-*+] ', stripped):
                indent = (len(line) - len(line.lstrip())) // 2
                bullet = ['•', '◦', '▸', '▹'][min(indent, 3)]
                lines.append(
                    f"<style fg='{PTColors.CYAN}'>{'  ' * indent}{bullet} </style>"
                    f"{render_inline(stripped[2:])}"
                )
                continue

            # ─── Numbered list ────────────────────────────────────────────────
            m = _re.match(r'^(\d+)\.\s+(.*)', stripped)
            if m:
                lines.append(
                    f"<style fg='{PTColors.CYAN}'>{m.group(1)}. </style>"
                    f"{render_inline(m.group(2))}"
                )
                continue

            lines.append(render_inline(stripped))

        # Rest-Tabelle + Rest-Lines
        if table_buf:
            _render_table(table_buf)
        for ln in lines:
            if show_line_numbers:
                c_print(HTML(f"<style fg='{PTColors.GREY}'>{lines.index(ln) + 1:3d}</style> {ln}"))
            else:
                c_print(HTML(f"{ln}"))

        return

    # Ausgabe
    for i, line in enumerate(lines, 1):
        if show_line_numbers:
            c_print(HTML(f"  <style fg='{PTColors.GREY}'>{i:3d}</style> {line}"))
        else:
            c_print(HTML(f"  {line}"))

def _pct(part: int, total: int) -> str:
    return f"{part / total * 100:.1f}%" if total else "0.0%"

def _bar_fet(used: int, limit: int, width: int = 44) -> str:
    """Farbiger ASCII-Fortschrittsbalken (Kern-Element, bleibt in allen Varianten)."""
    pct = used / limit if limit else 0
    filled = int(pct * width)
    empty = width - filled
    color = PTColors.GREEN if pct < 0.5 else (PTColors.YELLOW if pct < 0.8 else PTColors.RED)
    return (
        f"<style fg='{color}'>{'█' * filled}</style>"
        f"<style fg='{PTColors.GREY}'>{'░' * empty}</style>"
    )


def show_xray_v3(data: dict):
    sid = data['session_id']
    model = data['model']
    used = data['t_total']
    t_last = data.get('t_last', 0)
    limit = data['limit']
    bd = data['breakdown']
    sys_det = data['system_details']
    meta = data['meta']
    free = limit - used

    BAR_W = 20

    def mini_bar(tokens: int, total: int, width: int = BAR_W) -> str:
        filled = int((tokens / total) * width) if total else 0
        return (
            f"<style fg='{PTColors.CYAN}'>{'▓' * filled}</style>"
            f"<style fg='{PTColors.GREY}'>{'░' * (width - filled)}</style>"
        )

    def sub_row(label: str, tokens: int, color: str = PTColors.GREY, note: str = ""):
        """Eingerückte Sub-Zeile – korrekt an Tabellenspalten ausgerichtet."""
        PREFIX = "  └ "  # 4 Zeichen
        pad = max(0, NAME_W - len(PREFIX) - len(label))
        empty_bar = " " * (BAR_W + 2)
        note_html = (f" <style fg='{PTColors.GREY}'>{esc(note)}</style>" if note else "")
        c_print(HTML(
            f"  <style fg='{PTColors.GREY}'>{PREFIX}{esc(label)}{' ' * pad}</style>{sep}"
            f"{empty_bar}{sep}"
            f"<style fg='{color}'>{tokens:>9,}</style>{sep}"
            f"<style fg='{PTColors.GREY}'>{_pct(tokens, used):>9}</style>"
            f"{note_html}"
        ))

    # ── HEADER ──────────────────────────────────────────────────────────────
    print_box_header("CONTEXT X-RAY", icon="🔍")
    c_print(HTML(
        f"  <style fg='{PTColors.GREY}'>Session:</style> "
        f"<style fg='{PTColors.BRIGHT_WHITE}'><b>{esc(sid)}</b></style>   "
        f"<style fg='{PTColors.GREY}'>Modell:</style> "
        f"<style fg='{PTColors.BRIGHT_CYAN}'>{esc(model)}</style>   "
        f"<style fg='{PTColors.GREY}'>Limit:</style> "
        f"<style fg='{PTColors.WHITE}'>{limit:,}</style>"
    ))
    c_print(HTML(""))

    # ── HAUPT-BAR ───────────────────────────────────────────────────────────
    c_print(HTML(f"  {_bar_fet(used, limit, width=50)}"))
    c_print(HTML(
        f"  <style fg='{PTColors.BRIGHT_WHITE}'><b>{used:,}</b></style>"
        f"<style fg='{PTColors.GREY}'> / {limit:,} Token"
        f"   ·   {_pct(used, limit)} belegt"
        f"   ·   {free:,} frei ({_pct(free, limit)})</style>"
    ))
    c_print(HTML(""))

    # ── TABELLE ─────────────────────────────────────────────────────────────
    NAME_W = 22
    cols = [("KOMPONENTE", ""), ("MINI-BAR", ""), ("TOKENS", ""), ("ANTEIL", "")]
    widths = [NAME_W, BAR_W + 2, 9, 9]
    print_table_header(cols, widths)

    sep = f" <style fg='{PTColors.GREY}'>│</style> "

    def main_row(name: str, tokens: int, name_color: str = PTColors.WHITE,
                 tok_color: str = PTColors.CYAN):
        safe = esc(name)
        pad = max(0, NAME_W - len(name))
        bar_h = mini_bar(tokens, used)
        tok_s = f"{tokens:>9,}"
        pct_s = f"{_pct(tokens, used):>9}"
        c_print(HTML(
            f"  <style fg='{name_color}'>{safe}{' ' * pad}</style>{sep}"
            f"{bar_h}  {sep}"
            f"<style fg='{tok_color}'>{tok_s}</style>{sep}"
            f"<style fg='{PTColors.YELLOW}'>{pct_s}</style>"
        ))

    # System Prompt + Sub-Details
    skills_vol = sys_det["All Skills (Volume)"]
    skills_note = (f"{meta['active_skill_count']} aktiv · vol {skills_vol:,}" if skills_vol else "")
    main_row("System Prompt Total", bd["System Prompt Total"])
    sub_row("Base System Prompt", sys_det["Base System Prompt"])
    sub_row("VFS Content", sys_det["VFS Content"])
    if sys_det["Active Skills"] > 0:
        sub_row("Active Skills", sys_det["Active Skills"],
                color=PTColors.ZEN_CYAN, note=skills_note)
    elif skills_vol > 0:
        sub_row("Skills (inaktiv)", 0, note=f"vol {skills_vol:,}")

    # Active Tools – Info als kompakte Anmerkungszeile
    tools_note = f"{meta['tool_count']} defs · {meta['dynamic_tools_loaded']} dyn"
    main_row("Active Tools", bd["Active Tools"], tok_color=PTColors.ZEN_AMBER)
    c_print(HTML(f"  <style fg='{PTColors.GREY}'>  └ {tools_note}</style>"))

    # History getrennt – msg_count als kompakte Anmerkungszeile
    main_row("History (Perm)", bd["History (Perm)"], tok_color=PTColors.CYAN)
    c_print(HTML(f"  <style fg='{PTColors.GREY}'>  └ {meta['msg_count']} msgs</style>"))
    main_row("History (Work)", bd["History (Work)"], tok_color=PTColors.CYAN)
    c_print(HTML(f"  <style fg='{PTColors.GREY}'>  └ {meta['w_msg_count']} msgs</style>"))

    # Last Input
    main_row("Last Input", bd["Last Input"],
             name_color=PTColors.ZEN_DIM, tok_color=PTColors.GREY)

    print_separator()

    # ── TOTALS + SAVINGS ────────────────────────────────────────────────────
    hist_total = bd["History (Perm)"] + bd["History (Work)"]
    print_status(
        f"Skills: {meta['active_skill_count']} aktiv"
        f"   Tools: {meta['tool_count']} defs · {meta['dynamic_tools_loaded']} dyn"
        f"   Msgs: {meta['msg_count']}",
        "info",
    )

    # Warnings
    usage_pct = used / limit * 100 if limit else 0
    if usage_pct > 85:
        print_status("KRITISCHE AUSLASTUNG – 'shift_focus' ausführen!", "error")
    elif sys_det["VFS Content"] > 4000:
        print_status(
            f"Hohe VFS-Last ({sys_det['VFS Content']:,} tokens) – 'vfs_close' empfohlen",
            "warning",
        )
    elif hist_total > 6000:
        print_status(
            f"Langer Kontext ({hist_total:,} History-Tokens) – Zusammenfassung empfohlen",
            "warning",
        )

    print_box_footer()


# =============================================================================
# CONSTANTS & VERSION
# =============================================================================

VERSION = "4.0.0"
CLI_NAME = "ISAA Host"
NAME = "icli"
# Default Rate Limiter Configuration (shared across all agents)
DEFAULT_RATE_LIMITER_CONFIG = {
    "features": {
        "rate_limiting": True,
        "model_fallback": True,
        "key_rotation": True,
        "key_rotation_mode": "balance",
    },
    "api_keys": {},
    "fallback_chains": {
        "zglm/glm-4.7": [
            "zglm/glm-4.7-flash",
            "zglm/glm-4.7-flashx",
            "zai/glm-4.7-flash",
            "zai/glm-4.7-flashx",
        ],
    },
    "limits": {},
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ExecutionTask:
    """Unified execution record — single source of truth for all agent runs.
    kind  : 'chat' | 'task' | 'job' | 'delegate'
    """

    task_id: str
    agent_name: str
    query: str
    kind: str                                   # 'chat' | 'task' | 'job' | 'delegate'
    async_task: asyncio.Task
    run_id: str = ""
    stream: Any = None
    started_at: float = field(default_factory=time.time)
    status: str = "running"
    is_focused: bool = False
    result_text: str = ""
    _agent_ref: Any = None


# Backward-compat alias — nothing external should need this any more
BackgroundTask = ExecutionTask


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    name: str
    created_at: datetime = field(default_factory=datetime.now)
    persona: str = "default"
    is_self_agent: bool = False
    has_shell_access: bool = False
    mcp_servers: list[str] = field(default_factory=list)
    bound_agents: list[str] = field(default_factory=list)


# =========================== Simple Feature Manager ==========================
class SimpleFeatureManager:
    def __init__(self):
        self.agent_ref = None
        self.features = { }

    def list_features(self):
        return list(self.features.keys())

    async def enable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = True
            if self.features[feature]["activation_f"]:
                res = self.features[feature]["activation_f"](self.agent_ref)
                if asyncio.iscoroutine(res):
                    await res

    def is_enabled(self, feature):
        return self.features.get(feature, {"is_enabled": False})["is_enabled"]

    def disable(self, feature):
        if feature in self.features:
            self.features[feature]["is_enabled"] = False
            if self.features[feature]["deactivation_f"]:
                self.features[feature]["deactivation_f"](self.agent_ref)

    def add_feature(self, feature, activation_f=None, deactivation_f=None):
        self.features[feature] = {
            "is_enabled": False,
            "activation_f": activation_f,
            "deactivation_f": deactivation_f,
        }

    def set_agent(self, agent):
        self.agent_ref = agent


# ============================ Feature Definitions ============================
#
def load_desktop_auto_feature(fm: SimpleFeatureManager):
    def enable_desktop_auto(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.add_tools(tools)
        print_status("Desktop Automation enabled.", "success")

    def disable_desktop_auto(agent):
        from toolboxv2.mods.isaa.extras.destop_auto import register_enhanced_tools
        kit, tools = register_enhanced_tools()
        agent.remove_tools(tools)
        print_status("Desktop Automation enabled.", "success")
    fm.add_feature("desktop_auto", activation_f=enable_desktop_auto, deactivation_f=disable_desktop_auto)

def load_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import PlaywrightProxy
    proxy = PlaywrightProxy(full=False, headless=True)
    tools_set = [None]
    def enable(agent):
        proxy.start()
        tools_set[0] = proxy.build_agent_tools()
        agent.add_tools(tools_set[0])
        print_status("Mini Web Automation enabled.", "success")

    def disable(agent):
        proxy.shutdown()
        agent.remove_tools(tools_set[0])
        print_status("Mini Web Automation disabled.", "success")

    fm.add_feature("mini_web_auto", activation_f=enable, deactivation_f=disable)


def load_full_web_auto_feature(fm):
    from toolboxv2.mods.isaa.extras.web_helper.tooklit import PlaywrightProxy
    proxy = PlaywrightProxy(full=True, headless=True)
    tools_set = [None]
    def enable(agent):
        proxy.start()
        tools_set[0] = proxy.build_agent_tools()
        agent.add_tools(tools_set[0])
        print_status("Full Web Automation enabled.", "success")

    def disable(agent):
        proxy.shutdown()
        agent.remove_tools(tools_set[0])
        print_status("Full Web Automation disabled.", "success")

    fm.add_feature("full_web_auto", activation_f=enable, deactivation_f=disable)


def load_coder_toolkit(fm):
    from toolboxv2.mods.isaa.CodingAgent.coder_toolset import coder_register_flow_tools
    from toolboxv2 import init_cwd
    pool = [None]

    def enable(agent):
        c_print(f"Starting coder from: {str(init_cwd)}")
        _pool, tools = coder_register_flow_tools(agent, str(init_cwd))
        pool[0] = _pool
        agent.add_tools(tools)
        print_status("Coder enabled.", "success")

    def disable(agent):
        _pool, tools = coder_register_flow_tools(agent, init_cwd)
        agent.remove_tools(tools)
        print_status("Coder disabled.", "success")

    fm.add_feature("coder", activation_f=enable, deactivation_f=disable)

def load_chain_toolkit(fm):
    from toolboxv2.mods.isaa.base.chain.chain_tools import create_chain_tools
    tools_set = [None]

    agent_registry = {}
    coder_registry = {}
    format_registry = {}

    def enable(agent):
        tools_set[0] = create_chain_tools(agent, agent_registry=agent_registry, coder_registry=coder_registry, format_registry=format_registry)
        agent.add_tools(tools_set[0])
        print_status("Chains enabled.", "success")

    def disable(agent):
        agent.remove_tools(tools_set[0])
        print_status("Chains disabled.", "success")

    fm.add_feature("chain", activation_f=enable, deactivation_f=disable)

def load_execute(fm):
    from toolboxv2.mods.isaa.base.Agent.executors import register_code_exec_tools
    tools_set = [None]

    def enable(agent):
        tools_set[0] = register_code_exec_tools(agent)[0]
        print_status("exec_code enabled.", "success")

    def disable(agent):
        agent.remove_tools(tools_set[0])
        print_status("exec_code disabled.", "success")

    fm.add_feature("chain", activation_f=enable, deactivation_f=disable)

def load_docs_feature(fm):
    """
    Documentation Feature - Integrates mkdocs-based docs system.

    When active in toolboxv2 directory: Uses tb_root_dir.parent/docs as doc dir.
    When active in another directory: Prompts user to select docs directory.
    """
    docs_system = [None]  # Mutable container for the docs system instance
    docs_tools = [None]   # Mutable container for the tool list
    _TOOL_HEALTH_EXTENSIONS = {
        "docs_reader": {
            "live_test_inputs": [
                {
                    "query": "_probe_health_check_query",
                    "max_results": 1
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": (
                    "Das Tool muss ein Dictionary mit 'count' und 'time_ms' (oder 'sections') zurückgeben."
                )
            },
            "cleanup_func": None
        },
        "docs_writer": {
            "live_test_inputs": [
                {
                    # Negative Testing: Wir prüfen das Routing und Error-Handling,
                    # ohne echte Dateien zu schreiben.
                    "action": "_probe_invalid_action"
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": (
                    "Da die Action ungültig ist, muss ein Dict mit dem Key 'error' "
                    "und einer entsprechenden Fehlermeldung zurückgegeben werden."
                )
            },
            "cleanup_func": None
        },
        "docs_lookup": {
            "live_test_inputs": [
                {
                    "name": "_probe_SystemTest",
                    "max_results": 1
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'results' (Liste) und 'count' zurückgeben."
            },
            "cleanup_func": None
        },
        "docs_sync": {
            "live_test_inputs": [{}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'changes_detected' und 'files_updated' zurückgeben."
            },
            "cleanup_func": None
        },
        "docs_init": {
            "live_test_inputs": [
                {
                    # Wichtig: force_rebuild=False für einen schnellen, sicheren Test!
                    "force_rebuild": False,
                    "show_tqdm": False
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'status' (z.B. 'loaded') und 'time_ms' zurückgeben."
            },
            "cleanup_func": None
        },
        "get_task_context": {
            "live_test_inputs": [
                {
                    "files": [],
                    "intent": "_probe_health_check_intent"
                }
            ],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'result' und 'meta' zurückgeben."
            },
            "cleanup_func": None
        }
    }
    async def enable(agent):
        """Enable documentation system and add tools to agent."""
        try:
            # Determine docs directory based on current working directory
            current_dir = init_cwd
            project_root = tb_root_dir

            # Check if we're in toolboxv2 or its parent
            if current_dir == project_root or current_dir == project_root.parent or project_root in current_dir.parents:
                docs_dir = project_root.parent / "docs"
                c_print(HTML(f"<style fg='ansicyan'>📚 Auto-detected docs dir:</style> <style fg='ansigreen'>{docs_dir}</style>"))
            else:
                # Prompt user for docs directory
                docs_input = tb_root_dir
                docs_dir = Path(docs_input).expanduser().resolve()

            if not docs_dir.exists():
                c_print(HTML(f"<style fg='ansired'>⚠️ Docs directory not found: {docs_dir}</style>"))
                c_print(HTML("<style fg='ansiyellow'>Creating minimal docs structure...</style>"))
                docs_dir.mkdir(parents=True, exist_ok=True)

            # Create docs system instance
            system = DocsSystem(
                project_root=project_root,
                docs_root=docs_dir,
                include_dirs=["toolboxv2", "flows", "mods", "utils", "docs", "src"]
            )

            # Initialize (load existing index or build new one)
            result = await system.initialize()
            c_print(HTML(f"<style fg='ansigreen'>✓ Docs initialized:</style> {result['sections']} sections, {result['elements']} elements"))

            docs_system[0] = system

            # Prepare tool list with proper descriptions
            tools = [
                {
                    "tool_func": system.read,
                    "name": "docs_reader",
                    "description": "Durchsucht die Dokumentation nach relevanten Abschnitten basierend auf einer Suchanfrage. Nützlich um schnell Informationen aus der Dokumentation zu finden ohne manuell zu suchen. Gibt strukturierte Ergebnisse mit Relevanz-Scores zurück.",
                    "category": ["docs", "read", "search"],
                    "flags": {}
                },
                {
                    "tool_func": system.write,
                    "name": "docs_writer",
                    "description": "Schreibt neue Dokumentations-Dateien oder aktualisiert existierende. Unterstützt Markdown-Formatierung und speichert im docs-Verzeichnis. Erstellt automatisch die nötige Verzeichnisstruktur wenn nötig. Ideal um Ergebnisse und Erkenntnisse zu dokumentieren.",
                    "category": ["docs", "write"],
                    "flags": {}
                },
                {
                    "tool_func": system.lookup_code,
                    "name": "docs_lookup",
                    "description": "Sucht nach Code-Elementen (Klassen, Funktionen, Module, etc.) im gesamten Codebase. Gibt Definitionen, Signaturen und Docstrings zurück. Nützlich um Implementierungsdetails schnell zu verstehen ohne durch Dateien navigieren zu müssen.",
                    "category": ["docs", "code", "search"],
                    "flags": {}
                },
                {
                    "tool_func": system.sync,
                    "name": "docs_sync",
                    "description": "Synchronisiert Änderungen aus der VFS zurück zum Dateisystem. Stellt sicher dass Änderungen persistent gespeichert werden. Sollte regelmäßig aufgerufen werden nachdem Änderungen vorgenommen wurden.",
                    "category": ["docs", "sync"],
                    "flags": {}
                },
                {
                    "tool_func": system.initialize,
                    "name": "docs_init",
                    "description": "Baut den Dokumentations-Index neu auf. Nützlich wenn neue Dateien hinzugefügt wurden oder der Index veraltet ist. Indiziert Markdown-, Python- und JavaScript/TypeScript-Dateien aus den konfigurierten include_dirs.",
                    "category": ["docs", "index"],
                    "flags": {}
                },
                {
                    "tool_func": system.get_task_context,
                    "name": "get_task_context",
                    "description": "Generiert Kontext-Informationen für Aufgaben wie relevante Dateien, Klassen und Dokumentation basierend auf einer Aufgabenbeschreibung. Hilft dem Agent die Aufgabe besser zu verstehen und die richtigen Ressourcen zu finden.",
                    "category": ["docs", "context"],
                    "flags": {}
                }
            ]

            for tool in tools:
                if tool["name"] in _TOOL_HEALTH_EXTENSIONS:
                    tool.update(_TOOL_HEALTH_EXTENSIONS[tool["name"]])

            docs_tools[0] = tools
            agent.add_tools(tools)

            print_status("Documentation feature enabled.", "success")

        except Exception as e:
            c_print(HTML(f"<style fg='ansired'>✗ Failed to enable docs: {e}</style>"))
            import traceback
            c_print(traceback.format_exc())

    def disable(agent):
        """Disable documentation system."""
        try:
            if docs_tools[0]:
                agent.remove_tools(docs_tools[0])
            docs_system[0] = None
            docs_tools[0] = None
            print_status("Documentation feature disabled.", "success")
        except Exception as e:
            c_print(HTML(f"<style fg='ansired'>✗ Failed to disable docs: {e}</style>"))

    fm.add_feature("docs", activation_f=enable, deactivation_f=disable)

def load_autodoc_feature(fm):
    """AutoDoc: findet getesteten, undokumentierten Code → schreibt 2-Part Docs."""
    from toolboxv2.mods.isaa.base.chain.chain_tools import ChainStore, StoredChain
    from toolboxv2 import get_app, tb_root_dir
    import ast, re

    _TOOL_HEALTH_EXTENSIONS = {
        # ─── AUTODOC ───
        "tb_doc_attach_system": {
            "live_test_inputs": [{"query": ""}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Gibt ein Dict mit status 'ok' oder 'already_attached' zurück."
            },
            "cleanup_func": None
        },
        "tb_find_tested_symbols": {
            "live_test_inputs": [{"query": "_probe_missing_symbol"}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Sollte Kandidaten (oder Fehler bei fehlendem Init) als Dict zurückgeben."
            },
            "cleanup_func": None
        },
        "tb_fetch_code_for_doc": {
            "live_test_inputs": [{"query": "_probe_missing_element::probe.py"}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss ein Dict mit 'error' (not found) zurückgeben."
            },
            "cleanup_func": None
        },
        "tb_write_doc": {
            "live_test_inputs": [{"query": "invalid_json_payload_to_prevent_disk_write"}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss mit 'error' wegen invalidem JSON fehlschlagen."
            },
            "cleanup_func": None
        },
    }

    tools_set = [None]

    def enable(agent):

        async def tb_find_tested_symbols(query: str = "") -> dict:
            """
            Findet Code-Elemente die getestet sind und noch keine/schlechte Doku haben.
            Filtert: hat test_* Funktion irgendwo im Repo.
            Gibt zurück: {candidates: [{name, file, type, signature, has_doc}]}
            """
            import os
            from toolboxv2.utils.extras.mkdocs import DocsSystem
            try:
                system: DocsSystem = agent._autodoc_system
            except AttributeError:
                return {"error": "docs feature not enabled — run /feature enable docs first"}

            # 1. Suggestions holen (undokumentiert)
            raw = await system.get_suggestions(max_suggestions=60)
            undoc = {s["element"]: s for s in raw.get("suggestions", [])
                     if s.get("type") == "missing_docs"}

            if not undoc:
                return {"candidates": [], "total_undoc": 0, "total_tested": 0}

            # 2. Test-Dateien scannen — mit hartem Pruning + Cap
            EXCLUDE_DIRS = {
                ".venv", "venv", "env", ".env",
                "node_modules", "__pycache__", ".git", ".hg", ".svn",
                ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
                "build", "dist", ".eggs", "site-packages",
                ".idea", ".vscode", "htmlcov", ".coverage",
            }
            MAX_TEST_FILES = 40
            MAX_FILE_BYTES = 200_000  # skip riesige Test-Files

            # Nur an den üblichen Stellen suchen statt im ganzen Repo
            candidate_roots = [
                tb_root_dir / "tests",
                tb_root_dir / "test",
                tb_root_dir / "toolboxv2" / "tests",
            ]
            candidate_roots = [p for p in candidate_roots if p.exists()]
            if not candidate_roots:
                candidate_roots = [tb_root_dir]  # fallback

            test_files: list[Path] = []
            for root in candidate_roots:
                for dirpath, dirnames, filenames in os.walk(root):
                    # Prune in-place — os.walk betritt gelöschte dirs nicht mehr
                    dirnames[:] = [d for d in dirnames
                                   if d not in EXCLUDE_DIRS and not d.startswith(".")]
                    for fn in filenames:
                        if fn.endswith(".py") and (fn.startswith("test_") or fn.endswith("_test.py")):
                            test_files.append(Path(dirpath) / fn)
                            if len(test_files) >= MAX_TEST_FILES:
                                break
                    if len(test_files) >= MAX_TEST_FILES:
                        break
                if len(test_files) >= MAX_TEST_FILES:
                    break

            # Nur Identifier die wir *suchen* — nicht jeden call-Namen sammeln
            undoc_names = set(undoc.keys())
            if not undoc_names:
                return {"candidates": [], "total_undoc": 0, "total_tested": 0}

            # Ein einziges Regex über alle gesuchten Namen → O(bytes) statt O(bytes * names)
            name_pattern = re.compile(
                r'\b(' + '|'.join(re.escape(n) for n in undoc_names) + r')\s*\(',
            )

            test_names: set[str] = set()
            for tf in test_files:
                try:
                    if tf.stat().st_size > MAX_FILE_BYTES:
                        continue
                    src = tf.read_text(encoding="utf-8", errors="ignore")
                    for m in name_pattern.finditer(src):
                        test_names.add(m.group(1))
                except Exception:
                    continue

            # 3. Schnittmenge: undokumentiert UND getestet
            q = query.lower() if query else ""
            candidates = []
            for name, meta in undoc.items():
                if name in test_names or (q and q in name.lower()):
                    candidates.append({
                        "name": name,
                        "file": meta.get("file", ""),
                        "type": meta.get("element_type", ""),
                    })

            if query and not candidates:
                for name, meta in undoc.items():
                    if q in name.lower():
                        candidates.append({
                            "name": name,
                            "file": meta.get("file", ""),
                            "type": meta.get("element_type", ""),
                        })

            return {"candidates": candidates[:20], "total_undoc": len(undoc),
                    "total_tested": len(test_names)}

        async def tb_fetch_code_for_doc(query: str) -> dict:
            """
            Holt Code-Element + Kontext für die Dokumentation.
            Input: name oder 'name::file'.
            Gibt zurück: {name, signature, code, docstring, file, related_elements}
            """
            from toolboxv2.utils.extras.mkdocs import DocsSystem
            try:
                system: DocsSystem = agent._autodoc_system
            except AttributeError:
                return {"error": "docs feature not enabled"}

            # Parse input
            if "::" in query:
                name, file_hint = query.split("::", 1)
            else:
                name, file_hint = query.strip(), None

            result = await system.lookup_code(
                name=name.strip(),
                file_path=file_hint,
                include_code=True,
                max_results=3,
            )
            elements = result.get("results", [])
            if not elements:
                return {"error": f"Element '{name}' not found in index"}

            best = elements[0]
            ctx = await system.get_task_context(
                files=[best["file"]],
                intent=f"document {best['name']} {best['type']}"
            )
            deps = ctx.get("result", {}).get("context_graph", {})

            return {
                "name": best["name"],
                "type": best["type"],
                "signature": best["signature"],
                "code": best.get("code", "")[:3000],
                "docstring": best.get("docstring", ""),
                "file": best["file"],
                "upstream": deps.get("upstream_dependencies", [])[:5],
                "downstream": deps.get("downstream_usages", [])[:5],
            }

        async def tb_write_doc(query: str) -> dict:
            """
            Schreibt eine fertige 2-Part Dokumentation ins Docs-System.
            Input-Format (JSON-String):
            {
              "name": "FunctionName",
              "file_path": "api/my_module.md",  (relativ zu docs_root)
              "part1": "## How to Use\\n...examples...",
              "part2": "## How it Works\\n...internals..."
            }
            Gibt zurück: {status, path}
            """
            from toolboxv2.utils.extras.mkdocs import DocsSystem
            try:
                system: DocsSystem = agent._autodoc_system
            except AttributeError:
                return {"error": "docs feature not enabled"}

            try:
                import json as _json
                # Robustes Parsing: JSON direkt oder aus Markdown-Codeblock
                raw = query.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                data = _json.loads(raw.strip())
            except Exception as e:
                return {"error": f"Invalid JSON input: {e}. Provide {{name, file_path, part1, part2}}"}

            name      = data.get("name", "unknown")
            file_path = data.get("file_path") or f"api/{name.lower()}.md"
            part1     = data.get("part1", "")
            part2     = data.get("part2", "")

            if not part1 or not part2:
                return {"error": "Both part1 (How to Use) and part2 (How it Works) are required"}

            content = (
                f"# {name}\n\n"
                f"{part1.strip()}\n\n"
                "---\n\n"
                f"{part2.strip()}\n"
            )

            result = await system.write(
                action="create_file",
                file_path=file_path,
                content=content,
            )
            if result.get("error"):
                # File exists → update via add_section
                result = await system.write(
                    action="update_section",
                    section_id=f"{Path(file_path).name}#{name}",
                    content=content,
                )
            return {"status": result.get("status", "done"), "path": file_path, "name": name}

        async def tb_doc_attach_system(query: str = "") -> dict:
            """Intern: hängt DocsSystem an den Agent (lazy init). Gibt Status zurück."""
            from toolboxv2.utils.extras.mkdocs import DocsSystem
            from toolboxv2 import tb_root_dir
            if hasattr(agent, "_autodoc_system"):
                return {"status": "already_attached"}
            try:
                docs_dir = tb_root_dir.parent / "docs"
                docs_dir.mkdir(exist_ok=True)
                system = DocsSystem(
                    project_root=tb_root_dir,
                    docs_root=docs_dir,
                    include_dirs=["toolboxv2", "flows", "mods", "utils", "docs", "src"],
                )
                await system.initialize()
                agent._autodoc_system = system
                return {"status": "ok", "docs_root": str(docs_dir)}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        tools = [
            {"tool_func": tb_doc_attach_system,   "name": "tb_doc_attach_system",
             "description": "Initialisiert das DocsSystem für AutoDoc (einmalig nötig). Gibt {status} zurück.",
             "category": ["autodoc", "docs"]},
            {"tool_func": tb_find_tested_symbols,  "name": "tb_find_tested_symbols",
             "description": (
                 "Findet getesteten, undokumentierten Code. "
                 "Input: optionaler Suchbegriff. "
                 "Gibt {candidates:[{name,file,type}], total_undoc, total_tested} zurück."
             ),
             "category": ["autodoc", "docs"]},
            {"tool_func": tb_fetch_code_for_doc,   "name": "tb_fetch_code_for_doc",
             "description": (
                 "Holt Code + Kontext für ein Element. Input: 'Name' oder 'Name::file.py'. "
                 "Gibt {name, signature, code, upstream, downstream, file} zurück."
             ),
             "category": ["autodoc", "docs"]},
            {"tool_func": tb_write_doc,             "name": "tb_write_doc",
             "description": (
                 "Schreibt 2-Part Dokumentation ins Docs-System. "
                 'Input: JSON-String mit {name, file_path, part1, part2}. '
                 "part1 = How to Use (mit Beispielen), part2 = How it Works (Internals). "
                 "Gibt {status, path} zurück."
             ),
             "category": ["autodoc", "docs"]},
        ]
        # --- HEALTH-CHECK INTEGRATION ---
        for tool in tools:
            if tool["name"] in _TOOL_HEALTH_EXTENSIONS:
                tool.update(_TOOL_HEALTH_EXTENSIONS[tool["name"]])
        agent.add_tools(tools)
        tools_set[0] = tools

        # Chains registrieren
        store = ChainStore(Path(get_app().data_dir) / "chains")

        # ── Chain 1: Unguided ─────────────────────────────────────────────
        if not store.get("autodoc_unguided"):
            dsl_unguided = (
                'tool:tb_doc_attach_system() >> '
                'tool:tb_find_tested_symbols() >> '
                '@self("You have a list of undocumented but tested code elements in the \'candidates\' field. '
                'For each candidate (process ALL of them): '
                '1. Call tb_fetch_code_for_doc with the element name to get full code + context. '
                '2. Analyze: signature, code body, upstream dependencies, downstream usages. '
                '3. Write part1 (## How to Use\\n- clear description\\n- 2-3 concrete usage examples with code). '
                '4. Write part2 (## How it Works Internally\\n- data flow\\n- key decisions\\n- dependencies). '
                '5. Call tb_write_doc with JSON: {name, file_path: api/<name_lower>.md, part1, part2}. '
                'Only document code you fully understand from the source. No speculation. '
                'After all candidates: summarize what was documented.") '
            )
            store.save(StoredChain(
                id="autodoc_unguided",
                name="autodoc_unguided",
                dsl=dsl_unguided,
                description="Scannt das Repo, findet getesteten undokumentierten Code, schreibt 2-Part Docs.",
                tags=["autodoc", "docs", "unguided", "batch"],
                accepted=False,
                is_valid=True,
            ))

        # ── Chain 2: Guided ───────────────────────────────────────────────
        if not store.get("autodoc_guided"):
            dsl_guided = (
                'tool:tb_doc_attach_system() >> '
                'tool:tb_fetch_code_for_doc() >> '
                '@self("You have the full code + context for one element. '
                'Write precise 2-part documentation: '
                'part1 = \'## How to Use\\n\' with: what it does (1 sentence), parameters, return value, '
                'and 2-3 runnable code examples that cover the main use cases. '
                'part2 = \'## How it Works Internally\\n\' with: step-by-step data flow, '
                'key algorithms or design decisions, upstream dependencies called, '
                'and any important edge cases in the implementation. '
                'Then call tb_write_doc with JSON {name, file_path, part1, part2}. '
                'The file_path should be api/<module_name>/<element_name_lower>.md. '
                'Be precise and factual — only describe what the code actually does.") '
            )
            store.save(StoredChain(
                id="autodoc_guided",
                name="autodoc_guided",
                dsl=dsl_guided,
                description="Dokumentiert ein spezifisches Element (per Name als input). 2-Part Docs: How to Use + Internals.",
                tags=["autodoc", "docs", "guided", "single"],
                accepted=False,
                is_valid=True,
            ))

        print_status(
            "AutoDoc enabled.\n"
            "  Unguided (batch): /chain accept autodoc_unguided  →  /chain run autodoc_unguided\n"
            "  Guided (single):  /chain accept autodoc_guided   →  /chain run autodoc_guided MyFunctionName",
            "success",
        )

    def disable(agent):
        if tools_set[0]:
            agent.remove_tools(tools_set[0])
        if hasattr(agent, "_autodoc_system"):
            del agent._autodoc_system
        print_status("AutoDoc disabled.", "success")

    fm.add_feature("autodoc", activation_f=enable, deactivation_f=disable)

def load_autotest_feature(fm):
    """
    AutoTest: Versteht Codebase-Semantik (Flows, Side-Effects, Datenfluss)
    → erstellt präzise Unit-Tests ODER TDD-Zukunftstests.
    """
    from toolboxv2.mods.isaa.base.chain.chain_tools import ChainStore, StoredChain
    from toolboxv2 import get_app, tb_root_dir
    import ast, re, textwrap
    _TOOL_HEALTH_EXTENSIONS = {
        # ─── AUTOTEST ───
        "tb_analyze_semantics": {
            "live_test_inputs": [{"query": "_probe_missing_semantics::file.py"}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Gibt Fehler-Dict zurück, da das Element nicht existiert."
            },
            "cleanup_func": None
        },
        "tb_write_tests": {
            "live_test_inputs": [{"query": "invalid_json_payload_to_prevent_test_write"}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss mit 'error' wegen invalidem JSON fehlschlagen."
            },
            "cleanup_func": None
        },
        "tb_run_single_test": {
            "live_test_inputs": [{"query": "tests/test_non_existent_probe_file.py"}],
            "result_contract": {
                "expected_type": dict,
                "semantic_check_hint": "Muss Fehler-Dict (Test file not found) zurückgeben."
            },
            "cleanup_func": None
        },
    }
    tools_set = [None]

    def enable(agent):

        # ── Tool 1: Semantik-Analyse ──────────────────────────────────────
        async def tb_analyze_semantics(query: str) -> dict:
            """
            Tiefe Semantik-Analyse eines Code-Elements oder einer Datei.
            Input: 'Name' oder 'Name::file.py' oder '/pfad/zu/datei.py'
            Gibt zurück:
            {
              name, file, signature, code,
              side_effects: [str],       # I/O, state mutations, network, fs
              data_flow: [str],          # Input → Transformation → Output
              dependencies: [str],       # was wird aufgerufen
              callers: [str],            # wer ruft es auf
              existing_tests: [str],     # gefundene test_* Funktionen
              testable_units: [str],     # was sich gut testen lässt
              edge_cases: [str],         # None, empty, error paths
            }
            """
            # DocsSystem lazy attach
            if not hasattr(agent, "_autotest_system"):
                from toolboxv2.utils.extras.mkdocs import DocsSystem
                docs_dir = tb_root_dir.parent / "docs"
                docs_dir.mkdir(exist_ok=True)
                system = DocsSystem(
                    project_root=tb_root_dir,
                    docs_root=docs_dir,
                    include_dirs=["toolboxv2", "flows", "mods", "utils", "docs", "src"],
                )
                await system.initialize()
                agent._autotest_system = system

            system = agent._autotest_system

            # Parse query
            if "::" in query:
                name, file_hint = query.split("::", 1)
                name = name.strip()
            elif query.strip().endswith(".py"):
                name, file_hint = "", query.strip()
            else:
                name, file_hint = query.strip(), None

            # Code-Element holen
            lookup = await system.lookup_code(
                name=name or None,
                file_path=file_hint,
                include_code=True,
                max_results=1,
            )
            elements = lookup.get("results", [])
            if not elements:
                return {"error": f"Element '{query}' not found. Check name or add ::file.py"}
            elem = elements[0]

            code = elem.get("code", "")[:4000]
            file_path = elem.get("file", "")

            # Kontext-Graph
            ctx = await system.get_task_context(
                files=[file_path],
                intent=f"understand data flow side effects and testable units of {elem['name']}",
            )
            graph = ctx.get("result", {}).get("context_graph", {})
            upstream   = [f"{u['name']} ({u['type']}) in {u['file']}"
                          for u in graph.get("upstream_dependencies", [])[:8]]
            downstream = [f"{d['name']} in {d['file']}"
                          for d in graph.get("downstream_usages", [])[:8]]

            # Statische Analyse: Side-Effects
            side_effects: list[str] = []
            se_patterns = [
                (r'\bopen\s*\(',            "file I/O"),
                (r'\brequests\.\w+\s*\(',   "HTTP request"),
                (r'\baiohttp\.',            "async HTTP"),
                (r'\bsubprocess\.',         "subprocess"),
                (r'\bos\.(remove|rename|mkdir|makedirs|write)', "filesystem mutation"),
                (r'\bself\.\w+\s*=',        "instance state mutation"),
                (r'(?<!\w)print\s*\(',      "stdout side-effect"),
                (r'\blogging\.\w+\(',       "logging"),
                (r'\basyncio\.create_task', "async task spawn"),
                (r'\bsocket\.',             "network socket"),
                (r'\.execute\s*\(',         "DB/shell execute"),
                (r'\bjson\.(dump|dumps)\s*\(', "serialization"),
            ]
            for pattern, label in se_patterns:
                if re.search(pattern, code):
                    side_effects.append(label)

            # Edge-Case Erkennung
            edge_cases: list[str] = []
            if re.search(r'\bif\s+not\b|\bif\s+\w+\s+is\s+None', code):
                edge_cases.append("None / empty guard present")
            if re.search(r'\bexcept\b', code):
                edge_cases.append("exception handling present")
            if re.search(r'\braise\b', code):
                edge_cases.append("explicit raise — test error path")
            if re.search(r'\bfor\b.*\bin\b', code):
                edge_cases.append("iteration — test empty collection")
            if re.search(r':\s*list\[|:\s*List\[|:\s*dict\[|:\s*Dict\[', code):
                edge_cases.append("typed collection param — test with wrong type")
            if re.search(r'\blen\s*\(', code):
                edge_cases.append("length check — test zero-length input")
            if re.search(r'\basync\b', code):
                edge_cases.append("async — needs asyncio.run or pytest-asyncio")

            # Existing tests
            existing_tests: list[str] = []
            for tf in list(tb_root_dir.rglob("test_*.py"))[:30]:
                try:
                    src = tf.read_text(encoding="utf-8", errors="ignore")
                    for m in re.finditer(
                        rf'\bdef\s+(test_\w*{re.escape(elem["name"])}\w*)\s*\(',
                        src, re.IGNORECASE
                    ):
                        existing_tests.append(f"{m.group(1)} in {tf.name}")
                except Exception:
                    pass

            # Testable units (public methods/branches)
            testable_units: list[str] = []
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_"):
                            args = [a.arg for a in node.args.args if a.arg != "self"]
                            testable_units.append(
                                f"{node.name}({', '.join(args[:4])})"
                            )
            except SyntaxError:
                pass
            if not testable_units:
                testable_units.append(elem["name"])

            return {
                "name":           elem["name"],
                "type":           elem["type"],
                "file":           file_path,
                "signature":      elem["signature"],
                "code":           code,
                "side_effects":   side_effects or ["none detected"],
                "data_flow":      [
                    f"Input: {elem['signature']}",
                    f"Dependencies called: {', '.join(upstream[:4]) or 'none'}",
                    f"Used by: {', '.join(downstream[:3]) or 'none known'}",
                ],
                "dependencies":   upstream,
                "callers":        downstream,
                "existing_tests": existing_tests or ["none found"],
                "testable_units": testable_units,
                "edge_cases":     edge_cases or ["no obvious edge cases"],
            }

        # ── Tool 2: Test-Datei schreiben ──────────────────────────────────
        async def tb_write_tests(query: str) -> dict:
            """
            Schreibt generierte Tests in die korrekte test_*.py Datei.
            Input: JSON-String mit:
            {
              "target_file": "toolboxv2/mods/isaa/base/chain/chain_tools.py",
              "test_file":   "toolboxv2/tests/test_mods/test_isaa/test_base/test_chain/test_chain_tools.py",  (optional, wird auto-berechnet)
              "test_code":   "import unittest\\nclass Test...\\n",
              "mode":        "append" | "create"   (default: append wenn Datei existiert)
            }
            Gibt zurück: {status, test_file, tests_added}
            """
            import json as _json
            raw = query.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
                if raw.endswith("```"):
                    raw = raw[:-3]
            try:
                data = _json.loads(raw.strip())
            except Exception as e:
                return {"error": f"Invalid JSON: {e}"}

            target  = data.get("target_file", "")
            test_code = data.get("test_code", "").strip()
            mode    = data.get("mode", "auto")

            if not test_code:
                return {"error": "test_code is required"}

            # Auto-berechne test_file wenn nicht angegeben
            if data.get("test_file"):
                test_path = tb_root_dir / data["test_file"]
            elif target:
                # toolboxv2/mods/x/y.py → toolboxv2/tests/mods/test_x_y.py
                rel = Path(target)
                parts = [p for p in rel.parts if p not in ("toolboxv2",)]
                stem  = "_".join(parts).replace(".py", "").replace("/", "_").replace("\\", "_")
                test_path = tb_root_dir / "tests" / f"test_{stem}.py"
            else:
                return {"error": "Either target_file or test_file required"}

            test_path.parent.mkdir(parents=True, exist_ok=True)

            if test_path.exists() and mode != "create":
                # Append: füge neue Testklassen/Methoden ein, kein Duplikat
                existing = test_path.read_text(encoding="utf-8")
                # Extrahiere nur neue class/def Blöcke
                new_classes = re.findall(
                    r'^(class\s+Test\w+.*?)(?=^class\s+Test|\Z)',
                    test_code, re.MULTILINE | re.DOTALL
                )
                added = 0
                for block in new_classes:
                    class_name = re.match(r'class\s+(\w+)', block)
                    if class_name and class_name.group(1) not in existing:
                        existing += "\n\n" + textwrap.dedent(block).strip()
                        added += 1
                if added:
                    test_path.write_text(existing, encoding="utf-8")
                    status = f"appended {added} new class(es)"
                else:
                    status = "no new classes to add (already exists)"
            else:
                # Create
                header = (
                    f"# Auto-generated by AutoTest Chain\n"
                    f"# Target: {target}\n"
                    f"import unittest\n"
                    f"import asyncio\n\n"
                )
                if not test_code.startswith("import"):
                    test_code = header + test_code
                test_path.write_text(test_code, encoding="utf-8")
                status = "created"

            # Zähle test_ Methoden
            tests_added = len(re.findall(r'^\s+def\s+test_', test_path.read_text(), re.MULTILINE))

            return {
                "status":      status,
                "test_file":   str(test_path.relative_to(tb_root_dir)),
                "tests_added": tests_added,
            }

        # ── Tool 3: Test-Run ──────────────────────────────────────────────
        async def tb_run_single_test(query: str) -> dict:
            """
            Führt eine einzelne Test-Datei aus (unittest, -x).
            Input: relativer Pfad zur test_*.py
            Gibt zurück: {status, passed, failed, output}
            """
            import subprocess
            test_path = tb_root_dir / query.strip()
            if not test_path.exists():
                return {"error": f"Test file not found: {query}"}
            r = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path),
                 "-x", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=120,
                cwd=str(tb_root_dir),
            )
            out = (r.stdout + r.stderr)[:3000]
            passed = r.returncode == 0
            # Zähle passed/failed
            m_pass = re.search(r'(\d+)\s+passed', out)
            m_fail = re.search(r'(\d+)\s+failed', out)
            return {
                "status":  "pass" if passed else "fail",
                "passed":  int(m_pass.group(1)) if m_pass else (1 if passed else 0),
                "failed":  int(m_fail.group(1)) if m_fail else (0 if passed else 1),
                "output":  out,
            }

        tools = [
            {"tool_func": tb_analyze_semantics, "name": "tb_analyze_semantics",
             "description": (
                 "Tiefe Semantik-Analyse: Datenfluss, Side-Effects, Abhängigkeiten, "
                 "Edge-Cases, bestehende Tests, testbare Einheiten. "
                 "Input: 'Name' oder 'Name::file.py' oder '/pfad/datei.py'."
             ),
             "category": ["autotest", "analysis"]},
            {"tool_func": tb_write_tests,       "name": "tb_write_tests",
             "description": (
                 "Schreibt Test-Code in die korrekte test_*.py. "
                 "Input: JSON {target_file, test_file?, test_code, mode?}. "
                 "Appended neue Klassen wenn Datei existiert, erstellt sonst neu."
             ),
             "category": ["autotest", "write"]},
            {"tool_func": tb_run_single_test,   "name": "tb_run_single_test",
             "description": (
                 "Führt eine test_*.py aus (pytest -x --tb=short). "
                 "Input: relativer Pfad zur Testdatei. "
                 "Gibt {status, passed, failed, output} zurück."
             ),
             "category": ["autotest", "run"]},
        ]
        # --- HEALTH-CHECK INTEGRATION ---
        for tool in tools:
            if tool["name"] in _TOOL_HEALTH_EXTENSIONS:
                tool.update(_TOOL_HEALTH_EXTENSIONS[tool["name"]])
        agent.add_tools(tools)
        tools_set[0] = tools

        store = ChainStore(Path(get_app().data_dir) / "chains")

        # ── Chain 1: Logic Tests (bestehender Code) ───────────────────────
        if not store.get("autotest_logic"):
            dsl_logic = (
                'tool:tb_analyze_semantics() >> '
                '@self("You have a deep semantic analysis of a code element. '
                'Your job: write comprehensive unittest-based tests. '
                'Rules: '
                '(1) Use ONLY Python unittest — no pytest fixtures, no pytest.mark. '
                '(2) Test EVERY testable_unit listed. '
                '(3) Cover EVERY edge_case listed. '
                '(4) For each side_effect: use unittest.mock.patch or MagicMock. '
                '(5) Async functions: use asyncio.run() in setUp or a helper. '
                '(6) Tests must be deterministic — no random, no sleep, no network. '
                '(7) Each test method: Arrange → Act → Assert. One assertion focus per test. '
                '(8) Class name: Test<ElementName>. File: auto-calculated from target file. '
                'Output ONLY valid JSON: '
                '{"target_file":"<file>","test_code":"<full python test code>","mode":"auto"} '
                'No markdown, no explanation outside the JSON.") >> '
                'tool:tb_write_tests() >> '
                'tool:tb_run_single_test() >> '
                '(IS(status==pass) >> '
                '@self("Tests pass. Summarize: what was tested, coverage areas, any gaps.") % '
                '@self("Tests FAILED. Analyze: output field. Fix the test code — '
                'common issues: wrong import path, missing mock, async not awaited. '
                'Output fixed JSON {target_file, test_code, mode:create} and call tb_write_tests again.") >> '
                'tool:tb_write_tests() >> '
                'tool:tb_run_single_test())'
            )
            store.save(StoredChain(
                id="autotest_logic",
                name="autotest_logic",
                dsl=dsl_logic,
                description=(
                    "Analysiert Datenfluss + Side-Effects → schreibt präzise unittest-Tests "
                    "→ führt aus → fixt bei Fehler. Input: 'Name' oder 'Name::file.py'."
                ),
                tags=["autotest", "unittest", "logic", "existing-code"],
                accepted=False,
                is_valid=True,
            ))

        # ── Chain 2: TDD Future Tests ─────────────────────────────────────
        if not store.get("autotest_tdd"):
            dsl_tdd = (
                'tool:tb_analyze_semantics() >> '
                '@self("You have a semantic analysis. '
                'Your job: write TDD tests for PLANNED / NOT-YET-IMPLEMENTED behavior. '
                'These tests must FAIL NOW and PASS once the feature is implemented. '
                'Rules: '
                '(1) Use Python unittest only. '
                '(2) Read the existing code carefully — identify what is MISSING or INCOMPLETE. '
                '  Look for: TODOs, NotImplementedError, empty branches, stub returns, '
                '  missing error handling, undocumented behavior. '
                '(3) Write tests that describe the INTENDED contract, not the current (broken) state. '
                '(4) Test method names: test_<feature>_<scenario>_should_<expected>. '
                '(5) Add a docstring to each test: one sentence describing the intended behavior. '
                '(6) Mark tests with: self.skipTest(\"TDD: not implemented yet\") '
                '  ONLY if you cannot even construct the call (missing class/function). '
                '  Otherwise let them fail naturally. '
                '(7) Group by behavior area, not by method. '
                'Output ONLY valid JSON: '
                '{"target_file":"<file>","test_code":"<full python code>","mode":"auto"} '
                'No markdown outside JSON.") >> '
                'tool:tb_write_tests() >> '
                'tool:tb_run_single_test() >> '
                '@self("TDD run complete. Report: '
                'How many tests fail (expected)? '
                'How many accidentally pass (check if they actually test something meaningful)? '
                'What exact behaviors do the failing tests define? '
                'This is the implementation contract.")'
            )
            store.save(StoredChain(
                id="autotest_tdd",
                name="autotest_tdd",
                dsl=dsl_tdd,
                description=(
                    "TDD: schreibt Zukunfts-Tests für geplantes/fehlendes Verhalten. "
                    "Tests sollen JETZT FEHLSCHLAGEN und nach Implementierung grün werden. "
                    "Input: 'Name' oder 'Name::file.py'."
                ),
                tags=["autotest", "tdd", "future", "contract"],
                accepted=False,
                is_valid=True,
            ))

        # ── Chain 3: Full-File Coverage ───────────────────────────────────
        if not store.get("autotest_coverage"):
            dsl_coverage = (
                'tool:tb_analyze_semantics() >> '
                '@self("You have the full semantic picture of a module/file. '
                'Scan ALL testable_units. For EACH one: '
                '1. Call tb_analyze_semantics with \'<unit_name>::<file>\' to get granular detail. '
                '2. Determine: is this unit already tested (check existing_tests)? '
                '   If yes: check if coverage is complete — are edge_cases covered? '
                '   If no: write full unittest class for it. '
                '3. After analyzing all units: '
                '   Write ONE unified test file covering the entire module. '
                '   Include: happy path, all edge_cases, all side_effect mocks. '
                'Output JSON {target_file, test_code, mode:create}.") >> '
                'tool:tb_write_tests() >> '
                'tool:tb_run_single_test() >> '
                '(IS(status==pass) >> '
                '@self("Coverage complete. List: files created, test count, what is covered.") % '
                '@self("Some tests fail. Fix imports and mocks. Output corrected JSON.") >> '
                'tool:tb_write_tests() >> '
                'tool:tb_run_single_test())'
            )
            store.save(StoredChain(
                id="autotest_coverage",
                name="autotest_coverage",
                dsl=dsl_coverage,
                description=(
                    "Full-File Coverage: analysiert jede testbare Einheit einer Datei, "
                    "schreibt ein unified Test-File, führt aus. Input: 'file.py' oder 'Name::file.py'."
                ),
                tags=["autotest", "coverage", "full-file"],
                accepted=False,
                is_valid=True,
            ))

        print_status(
            "AutoTest enabled.\n"
            "  Logic tests:    /chain accept autotest_logic     →  /chain run autotest_logic MyClass::path/file.py\n"
            "  TDD future:     /chain accept autotest_tdd       →  /chain run autotest_tdd MyFunction\n"
            "  Full coverage:  /chain accept autotest_coverage  →  /chain run autotest_coverage path/to/module.py",
            "success",
        )

    def disable(agent):
        if tools_set[0]:
            agent.remove_tools(tools_set[0])
        if hasattr(agent, "_autotest_system"):
            del agent._autotest_system
        print_status("AutoTest disabled.", "success")

    fm.add_feature("autotest", activation_f=enable, deactivation_f=disable)

def load_autofix_feature(fm):
    """AutoFix: tb --test -x → self analysiert → 2x CoderAgent parallel → bester Fix → PR."""
    from toolboxv2.mods.isaa.base.chain.chain_tools import ChainStore, StoredChain
    from toolboxv2.mods.isaa.CodingAgent.coder import CoderAgent
    from toolboxv2 import get_app, init_cwd
    import subprocess, datetime, shutil
    _TOOL_HEALTH_EXTENSIONS = {
        # ─── AUTOFIX ───
        "tb_run_tests": {
            # Das Ausführen der echten Suite via Subprozess ist zu schwer für einen Health-Check.
            "flags": {"guaranteed_healthy": True},
            "result_contract": {"expected_type": dict,
                                "semantic_check_hint": "Sollte {status, output, error_summary} liefern."}
        },
        "tb_coder_fix_a": {
            "flags": {"guaranteed_healthy": True},  # Erfordert LLM und CoderAgent
            "result_contract": {"expected_type": str, "semantic_check_hint": "Sollte Fix-Resultat String liefern."}
        },
        "tb_coder_fix_b": {
            "flags": {"guaranteed_healthy": True},  # Erfordert LLM und CoderAgent
            "result_contract": {"expected_type": str, "semantic_check_hint": "Sollte Fix-Resultat String liefern."}
        },
        "tb_apply_best_fix": {
            "live_test_inputs": [{"query": "APPLY:A"}],
            "result_contract": {
                "expected_type": str,
                "semantic_check_hint": "Muss fehlschlagen ('ERROR: no coder fix available'), da kein Coder im Test existiert."
            },
            "cleanup_func": None
        },
        "tb_create_pr": {
            "flags": {"guaranteed_healthy": True},  # Gefährlich: Führt echte Git-Commits und PRs aus!
            "result_contract": {"expected_type": str, "semantic_check_hint": "Sollte PR-URL String liefern."}
        },
        "tb_report_failure": {
            "live_test_inputs": [{"query": "_probe_failure_reason"}],
            "result_contract": {
                "expected_type": str,
                "semantic_check_hint": "Muss einen String zurückgeben, der den Fehlerbericht enthält."
            },
            "cleanup_func": None
        }
    }
    tools_set = [None]
    _state: dict = {"coder_a": None, "coder_b": None, "project_path": str(init_cwd)}

    def enable(agent):
        project_path = _state["project_path"]

        async def tb_run_tests(query: str = "") -> dict:
            """Run tb --test -x. Returns {status, output, error_summary}."""
            path = query.strip() if query.strip() and Path(query.strip()).is_dir() else project_path
            try:
                r = subprocess.run(
                    [sys.executable, "-m", "toolboxv2", "--test", "--kwargs", "0=-x"],
                    cwd=path, capture_output=True, text=True, timeout=300,
                )
                out = (r.stdout + r.stderr)[:6000]
                passed = r.returncode == 0
                summary = ""
                if not passed:
                    for line in out.splitlines():
                        if any(k in line for k in ("FAILED", "ERROR", "assert", "Traceback", "Exception")):
                            summary += line + "\n"
                            if len(summary) > 2000:
                                break
                return {"status": "pass" if passed else "fail", "output": out, "error_summary": summary}
            except Exception as e:
                return {"status": "fail", "output": str(e), "error_summary": str(e)}

        async def tb_coder_fix_a(query: str) -> str:
            """Fix attempt A: conservative, minimal, targeted change."""
            coder = CoderAgent(agent, project_path, config={"ask_enabled": False})
            coder.print = ansi_c_print
            _state["coder_a"] = coder
            result = await coder.execute(
                f"Fix the failing test. Strategy: CONSERVATIVE — smallest possible change, "
                f"touch only the directly broken code.\n\nAnalysis:\n{query}"
            )
            return (
                f"FIX_A:{'SUCCESS' if result.success else 'FAILED'}\n"
                f"Files: {result.changed_files}\n{result.message[:800]}"
            )

        async def tb_coder_fix_b(query: str) -> str:
            """Fix attempt B: thorough, addresses root cause with full data flow understanding."""
            coder = CoderAgent(agent, project_path, config={"ask_enabled": False})
            coder.print = ansi_c_print
            _state["coder_b"] = coder
            result = await coder.execute(
                f"Fix the failing test. Strategy: THOROUGH — trace the full data flow, "
                f"fix the actual root cause, ensure no regressions.\n\nAnalysis:\n{query}"
            )
            return (
                f"FIX_B:{'SUCCESS' if result.success else 'FAILED'}\n"
                f"Files: {result.changed_files}\n{result.message[:800]}"
            )

        async def tb_apply_best_fix(query: str) -> str:
            """Apply the chosen fix. Input must contain APPLY:A or APPLY:B."""
            choice = "B" if "APPLY:B" in query.upper() else "A"
            winner = _state.get(f"coder_{choice.lower()}")
            loser  = _state.get("coder_b" if choice == "A" else "coder_a")
            if loser:
                try: loser.worktree.cleanup()
                except Exception: pass
            if not winner:
                return "ERROR: no coder fix available"
            try:
                n = await winner.worktree.apply_back()
                winner.worktree.cleanup()
                return f"Applied Fix {choice} ({'git merge' if n == -1 else f'{n} files'}). Repo updated."
            except Exception as e:
                return f"Apply failed: {e}"

        async def tb_create_pr(query: str = "") -> str:
            """Create git branch + commit + PR (gh if available, else push)."""
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            branch = f"autofix/test-fix-{ts}"
            try:
                for cmd in [
                    ["git", "checkout", "-b", branch],
                    ["git", "add", "-A"],
                    ["git", "commit", "-m", f"autofix: fix failing tests ({ts})"],
                ]:
                    subprocess.run(cmd, cwd=project_path, check=True, capture_output=True)
                if shutil.which("gh"):
                    r = subprocess.run(
                        ["gh", "pr", "create", "--repo", "MarkinHaus/ToolBoxV2",
                         "--title", f"AutoFix: failing tests {ts}",
                         "--body", "Automated fix by ISAA AutoFix Chain.",
                         "--head", branch],
                        cwd=project_path, capture_output=True, text=True,
                    )
                    return f"PR created: {(r.stdout + r.stderr).strip()}"
                subprocess.run(["git", "push", "origin", branch], cwd=project_path, capture_output=True)
                return (
                    f"Branch pushed: {branch}\n"
                    f"Create PR: https://github.com/MarkinHaus/ToolBoxV2/compare/{branch}"
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr or e)
                return f"Git error: {stderr}"

        async def tb_report_failure(query: str = "") -> str:
            """Report that autofix failed after retry — manual intervention needed."""
            return f"AutoFix FAILED after retry.\nDetails: {str(query)[:600]}\nManual fix required."

        tools = [
            {"tool_func": tb_run_tests,      "name": "tb_run_tests",
             "description": "Run tb --test -x in project. Returns dict {status, output, error_summary}.",
             "category": ["autofix", "testing"]},
            {"tool_func": tb_coder_fix_a,     "name": "tb_coder_fix_a",
             "description": "Conservative coder fix (minimal change). Input: analysis text. Output: FIX_A result.",
             "category": ["autofix", "coder"]},
            {"tool_func": tb_coder_fix_b,     "name": "tb_coder_fix_b",
             "description": "Thorough coder fix (root cause). Input: analysis text. Output: FIX_B result.",
             "category": ["autofix", "coder"]},
            {"tool_func": tb_apply_best_fix,  "name": "tb_apply_best_fix",
             "description": "Apply best fix. Input must contain APPLY:A or APPLY:B. Merges to repo, cleans worktrees.",
             "category": ["autofix", "coder"]},
            {"tool_func": tb_create_pr,       "name": "tb_create_pr",
             "description": "Create git branch+commit+PR on MarkinHaus/ToolBoxV2.",
             "category": ["autofix", "git"]},
            {"tool_func": tb_report_failure,  "name": "tb_report_failure",
             "description": "Report autofix failure after re-test. Returns human-readable summary.",
             "category": ["autofix"]},
        ]
        # --- HEALTH-CHECK INTEGRATION ---
        for tool in tools:
            if tool["name"] in _TOOL_HEALTH_EXTENSIONS:
                tool.update(_TOOL_HEALTH_EXTENSIONS[tool["name"]])
        agent.add_tools(tools)
        tools_set[0] = tools

        # Chain in ChainStore registrieren (idempotent)
        store = ChainStore(Path(get_app().data_dir) / "chains")
        chain_id = "autofix_test_fixer"
        if not store.get(chain_id):
            dsl = (
                'tool:tb_run_tests() >> '
                '(IS(status==pass) >> tool:tb_create_pr() % '
                '(@self("Analyze this test failure with full depth: examine traceback, '
                'source of failing assertion, all called functions and their implementations, '
                'complete data flow end-to-end. Output: 1) Exact root cause  '
                '2) Affected components  3) Fix strategy A (conservative)  '
                '4) Fix strategy B (thorough root cause fix)") >> '
                '(tool:tb_coder_fix_a() + tool:tb_coder_fix_b()) >> '
                '@self("You have two fix proposals. Evaluate: correctness, invasiveness, '
                'architectural fit, risk of regression. '
                'Output exactly: APPLY:A or APPLY:B — then justify in one sentence.") >> '
                'tool:tb_apply_best_fix() >> '
                'tool:tb_run_tests() >> '
                '(IS(status==pass) >> tool:tb_create_pr() % tool:tb_report_failure())))'
            )
            store.save(StoredChain(
                id=chain_id,
                name="autofix_test_fixer",
                dsl=dsl,
                description=(
                    "Run tb --test -x, deep-analyze failure, generate 2 parallel coder fixes "
                    "(conservative + thorough), self picks best, re-test, create PR."
                ),
                tags=["autofix", "testing", "ci", "coder"],
                accepted=False,
                is_valid=True,
            ))
        print_status(
            "AutoFix enabled. Accept + run: /chain accept autofix_test_fixer  then  /chain run autofix_test_fixer",
            "success",
        )

    def disable(agent):
        if tools_set[0]:
            agent.remove_tools(tools_set[0])
        print_status("AutoFix disabled.", "success")

    fm.add_feature("autofix", activation_f=enable, deactivation_f=disable)

ALL_FEATURES = {
    "desktop_auto": load_desktop_auto_feature,
    "mini_web_auto": load_web_auto_feature,
    "full_web_auto": load_full_web_auto_feature,
    "coder": load_coder_toolkit,
    "chain": load_chain_toolkit,
    "execute": load_execute,
    "docs": load_docs_feature,
    "autodoc": load_autodoc_feature,
    "autotest": load_autotest_feature,
    "autofix": load_autofix_feature,
}


# ─── Subcommand Definitions ─────────────────────────────────────────────────

# Maps subcommand → argument spec
# Spec: list of (arg_type, required) tuples
#   arg_type: "vfs_path" | "vfs_file" | "vfs_dir" | "local_path" | "mount"
#             | "dirty" | "subcmd:<options>" | None (no completion)

SUBCOMMANDS: dict[str, dict] = {
    "mount":       {"args": ["local_path", "vfs_path"], "flags": ["--readonly", "--no-sync"]},
    "unmount":     {"args": ["mount"],                  "flags": ["--no-save"]},
    "sync":        {"args": ["vfs_dirty_or_all"],       "flags": []},
    "refresh":     {"args": ["mount"],                  "flags": []},
    "pull":        {"args": ["vfs_path"],               "flags": []},
    "save":        {"args": ["vfs_path", "local_path"], "flags": []},
    "mounts":      {"args": [],                         "flags": []},
    "dirty":       {"args": [],                         "flags": []},
    "rm":          {"args": ["vfs_path"],               "flags": []},
    "remove":      {"args": ["vfs_path"],               "flags": []},
    "sys-add":     {"args": ["local_path", "vfs_path"], "flags": ["--refresh"]},
    "sys-remove":  {"args": ["vfs_path"],               "flags": []},
    "sys-refresh": {"args": ["vfs_path"],               "flags": []},
    "sys-list":    {"args": [],                         "flags": []},
    "obsidian":    {"args": ["subcmd:mount|unmount|sync", "local_path", "vfs_path"], "flags": []},
}


class VFSCompleter(Completer):
    """
    Hierarchischer VFS-Completer.

    Wird als Sub-Completer eingehängt — empfängt nur den Text nach '/vfs '.
    Beispiel: User tippt '/vfs mount /ho' → dieser Completer sieht 'mount /ho'.
    """

    def __init__(self, vfs: 'VirtualFileSystemV2'):
        self._vfs = vfs

    # ─── Entry Point ─────────────────────────────────────────────────────

    def get_completions(
        self, document: 'Document', complete_event: 'CompleteEvent'
    ):
        text = document.text_before_cursor
        stripped = text.lstrip()

        # ── Case 1: Kein Text oder kein Space → Subcommand + Top-Level VFS Pfade
        if " " not in stripped:
            yield from self._complete_subcommand_or_path(stripped)
            return

        # ── Case 2: Subcommand erkannt → Argumente completieren
        parts = stripped.split(None, 1)  # maxsplit=1
        subcmd = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if subcmd in SUBCOMMANDS:
            yield from self._complete_subcmd_args(subcmd, rest)
        else:
            # Kein bekannter Subcommand → als VFS-Pfad interpretieren
            yield from self._complete_vfs_path(stripped)

    # ─── Subcommand / Top-Level Path Completion ──────────────────────────

    def _complete_subcommand_or_path(self, partial: str):
        """Complete subcommand names AND top-level VFS paths."""
        partial_lower = partial.lower()

        # Subcommands
        for cmd in SUBCOMMANDS:
            if cmd.startswith(partial_lower):
                yield Completion(cmd, start_position=-len(partial), display=cmd)

        # Top-level VFS paths (direct access: /vfs <path>)
        yield from self._complete_vfs_path(partial)

    # ─── Subcommand Argument Completion ──────────────────────────────────

    def _complete_subcmd_args(self, subcmd: str, rest: str):
        """Complete arguments for a known subcommand."""
        spec = SUBCOMMANDS[subcmd]
        arg_defs = spec["args"]
        flags = spec["flags"]

        # Split rest into tokens, but keep track of current partial
        tokens = rest.split()
        # If rest ends with space → completing NEW token, else completing last token
        if rest.endswith(" ") or not rest:
            completed_args = tokens
            current_partial = ""
        else:
            completed_args = tokens[:-1]
            current_partial = tokens[-1]

        # Filter out already-used flags from completed args
        used_flags = {t for t in completed_args if t.startswith("--")}
        non_flag_args = [t for t in completed_args if not t.startswith("--")]

        # ── Flag completion (if partial starts with -)
        if current_partial.startswith("-"):
            for flag in flags:
                if flag not in used_flags and flag.startswith(current_partial):
                    yield Completion(
                        flag, start_position=-len(current_partial), display=flag
                    )
            return

        # ── Determine which positional arg we're on
        arg_index = len(non_flag_args)

        if arg_index < len(arg_defs):
            arg_type = arg_defs[arg_index]
            yield from self._complete_arg_type(arg_type, current_partial)

        # ── Always offer remaining flags
        if not current_partial or current_partial.startswith("-"):
            for flag in flags:
                if flag not in used_flags and flag.startswith(current_partial):
                    yield Completion(
                        flag, start_position=-len(current_partial), display=flag
                    )

    def _complete_arg_type(self, arg_type: str, partial: str):
        """Dispatch completion based on argument type."""
        if arg_type == "vfs_path":
            yield from self._complete_vfs_path(partial)
        elif arg_type == "vfs_dirty_or_all":
            yield from self._complete_vfs_dirty_or_all(partial)
        elif arg_type == "local_path":
            yield from self._complete_local_path(partial)
        elif arg_type == "mount":
            yield from self._complete_mount_points(partial)
        elif arg_type.startswith("subcmd:"):
            options = arg_type.split(":", 1)[1].split("|")
            for opt in options:
                if opt.startswith(partial):
                    yield Completion(opt, start_position=-len(partial), display=opt)

    # ─── VFS Path Completion (hierarchisch!) ─────────────────────────────

    def _complete_vfs_path(self, partial: str):
        """
        Hierarchische VFS-Pfad-Completion.

        Löst nur direkte Kinder des aktuellen Parent-Dirs auf.
        '/project/sr' → listet Kinder von /project die mit 'sr' anfangen.
        """
        # Normalize: ensure starts with /
        if not partial:
            partial_path = "/"
        elif not partial.startswith("/"):
            partial_path = "/" + partial
        else:
            partial_path = partial

        # Determine parent dir and search prefix
        if partial_path.endswith("/") and self._vfs._is_directory(partial_path.rstrip("/") or "/"):
            # User typed full dir with trailing slash → list children
            parent = partial_path.rstrip("/") or "/"
            search = ""
        elif "/" in partial_path[1:]:
            # Has path separator → split into parent + partial name
            parent = partial_path.rsplit("/", 1)[0] or "/"
            search = partial_path.rsplit("/", 1)[1].lower()
        else:
            # Top-level: /something
            parent = "/"
            search = partial_path.lstrip("/").lower()

        # Verify parent exists
        if not self._vfs._is_directory(parent):
            return

        # Get direct children
        contents = self._vfs._list_directory_contents(parent)

        for item in contents:
            name = item["name"]
            if search and not name.lower().startswith(search):
                continue

            is_dir = item["type"] == "directory"
            suffix = "/" if is_dir else ""
            full_path = f"{parent.rstrip('/')}/{name}{suffix}"

            # Display: just the name + type indicator
            if is_dir:
                display = f"📁 {name}/"
            else:
                f = self._vfs.files.get(item["path"])
                icon = f.file_type.icon if f and f.file_type else "📄"
                state = " ●" if f and f.state == "open" else ""
                dirty = " ✱" if f and hasattr(f, "is_dirty") and f.is_dirty else ""
                display = f"{icon} {name}{state}{dirty}"

            yield Completion(
                full_path,
                start_position=-len(partial),
                display=display,
                display_meta=item.get("file_type", "") if not is_dir else "",
            )

    # ─── Dirty / All VFS Paths ───────────────────────────────────────────

    def _complete_vfs_dirty_or_all(self, partial: str):
        """Complete with dirty files first, then fall back to all VFS paths."""
        partial_lower = partial.lower() if partial else ""

        # Dirty files zuerst (Priorität für sync)
        dirty_yielded = set()
        for path, f in self._vfs.files.items():
            if hasattr(f, "is_dirty") and f.is_dirty:
                if not partial or path.lower().startswith(partial_lower) or (
                    not partial.startswith("/") and path.lower().startswith("/" + partial_lower)
                ):
                    dirty_yielded.add(path)
                    yield Completion(
                        path,
                        start_position=-len(partial),
                        display=f"✱ {path}",
                        display_meta="modified",
                    )

        # Dann normale Pfad-Completion (ohne bereits gezeigte)
        for completion in self._complete_vfs_path(partial):
            if completion.text not in dirty_yielded:
                yield completion

    # ─── Mount Point Completion ──────────────────────────────────────────

    def _complete_mount_points(self, partial: str):
        """Complete with active mount points."""
        partial_lower = partial.lower() if partial else ""

        if not hasattr(self._vfs, "mounts"):
            return

        for mount_path, mount in self._vfs.mounts.items():
            if not partial or mount_path.lower().startswith(partial_lower) or (
                not partial.startswith("/") and mount_path.lower().startswith("/" + partial_lower)
            ):
                local = getattr(mount, "local_path", "")
                yield Completion(
                    mount_path,
                    start_position=-len(partial),
                    display=f"📂 {mount_path}",
                    display_meta=local,
                )

    # ─── Local Path Completion ───────────────────────────────────────────

    def _complete_local_path(self, partial: str):
        """
        Complete local filesystem paths.

        Handles ~ expansion and hierarchical directory traversal.
        """
        if not partial:
            partial = "./"

        # Expand user home
        expanded = os.path.expanduser(partial)

        # Determine dir to scan and prefix to match
        if os.path.isdir(expanded):
            scan_dir = expanded
            name_prefix = ""
            # Ensure partial ends with separator for correct start_position
            if not partial.endswith(os.sep) and not partial.endswith("/"):
                partial += os.sep
        else:
            scan_dir = os.path.dirname(expanded) or "."
            name_prefix = os.path.basename(expanded).lower()

        if not os.path.isdir(scan_dir):
            return

        try:
            entries = os.listdir(scan_dir)
        except PermissionError:
            return

        entries.sort()

        for entry in entries:
            # Skip hidden unless user explicitly typed dot
            if entry.startswith(".") and not name_prefix.startswith("."):
                continue

            if name_prefix and not entry.lower().startswith(name_prefix):
                continue

            full = os.path.join(scan_dir, entry)
            is_dir = os.path.isdir(full)

            # Build the completion text preserving user's original format
            if partial.startswith("~"):
                # Keep ~ prefix
                rel_to_home = os.path.relpath(full, os.path.expanduser("~"))
                completion_text = f"~/{rel_to_home}"
            elif os.path.isabs(partial) or os.path.isabs(expanded):
                completion_text = full
            else:
                try:
                    completion_text = os.path.relpath(full)
                except ValueError:
                    completion_text = full

            if is_dir:
                completion_text += os.sep
                display = f"📁 {entry}/"
            else:
                display = f"  {entry}"

            yield Completion(
                completion_text,
                start_position=-len(partial),
                display=display,
            )
class SmartCompleter(Completer):
    """FuzzyCompleter für alles AUSSER /vfs — dort direkt, damit Tab akzeptiert."""

    def __init__(self, nested_dict: dict, vfs_completer: VFSCompleter | None = None):
        self._fuzzy = FuzzyCompleter(NestedCompleter.from_nested_dict(nested_dict))
        self._vfs = vfs_completer

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.lstrip()
        if text.startswith("/vfs ") and self._vfs:
            sub_doc = Document(text[5:], len(text[5:]))
            yield from self._vfs.get_completions(sub_doc, complete_event)
        else:
            yield from self._fuzzy.get_completions(document, complete_event)

# =============================================================================
# HOTKEY POLLER (active during agent streaming, when prompt_toolkit is idle)
# =============================================================================


_MEDIA_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"},
    "pdf":   {".pdf"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
    "audio": {".mp3", ".wav", ".ogg", ".flac", ".m4a"},
}

def _is_pasteable_media_path(text: str) -> bool:
    """Gibt True zurück wenn text ein lokaler Pfad oder eine URL zu einer Mediendatei ist."""
    stripped = text.strip()
    # Kein Whitespace im Pfad (außer quoted) → kein Fließtext
    if "\n" in stripped or (len(stripped.split()) > 1 and not stripped.startswith('"')):
        return False
    p = Path(stripped.strip('"').strip("'"))
    ext = p.suffix.lower()
    for exts in _MEDIA_EXTENSIONS.values():
        if ext in exts:
            return True
    return False


def _safe_decode_paste(raw: str) -> str:
    """Normalisiert eingefügten Text zu sauberem UTF-8."""
    try:
        safe = raw.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        safe = raw.encode("ascii", errors="replace").decode("ascii", errors="replace")
    # Null-Bytes und Windows-Zeilenenden normalisieren
    safe = safe.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return safe

# ------------------------------------------------------------
#  Hilfedaten (aus den von dir angegebenen Quellen)
# ------------------------------------------------------------
_help_text = {
    # Agent Management
    "agent": """/agent list                - List all agents
/agent switch <name>       - Switch active agent
/agent spawn <name> <persona> - Create new agent
/agent stop <name>         - Stop agent tasks
/agent model <fast|complex> <name> - Change LLM model on the fly
/agent checkpoint <save|load> [name] - Manage state persistence
/agent checkpoint help     - List information about additional args like path
/agent load-all            - Initialize all agents from disk
/agent save-all            - Save checkpoints for all active agents
/agent stats [name]        - Show token usage and cost metrics
/agent delete <name>       - Remove agent and its data
/agent config <name>       - View raw JSON configuration
    """,

    # Session Management
    "session": """/session list              - List sessions
/session switch <id>       - Switch session
/session new               - Create new session
/session show [n]          - Show last n messages (default 10)
/session clear             - Clear current session history
/session working             - Show working history
    """,

    # MCP Management (Live)
    "mcp": """/mcp list                 - Zeige aktive MCP Verbindungen
/mcp add <n> <cmd> [args]  - Server hinzufügen & Tools laden
/mcp remove <name>        - Server trennen & Tools löschen
/mcp reload               - Alle MCP Tools neu indizieren
    """,

    # Task Management
    "task": """/task                     - Show all background tasks
/task view [id]           - Live view of task (auto‑selects if 1)
/task cancel <id>         - Cancel a running task
/task clean               - Remove finished tasks
/task log                 - Vew Task internal execution
F6 during execution       - Move agent to background
    """,

    # Job Scheduler
    "job": """/job list                 - List all scheduled jobs
/job add                  - Add a new job (interactive)
/job remove <id>          - Remove a job
/job pause <id>           - Pause a job
/job resume <id>          - Resume a paused job
/job fire <id>            - Manually fire a job now
/job detail <id>          - Show job details
/job autowake <cmd>       - Manage OS auto‑wake (install/remove/status)

    Dreamer Job
/job dream create [agent] - Create nightly dream job (default: self, 03:00)
/job dream status         - Show all configured dream jobs
/job dream live           - Run dream process now with visualization
    """,

    # Advanced
    "bind": "/bind <agent_a> <agent_b> - Bind agents",
    "teach": "/teach <agent> <skill_name> - Teach skill",
    "context": "/context stats - Show context stats",

    # History Management
    "history": """/history show [n]         - Show last n messages (default 10)
/history clear            - Clear current session history
    """,

    # VFS Management
    "vfs": """/vfs                     - Show VFS tree
/vfs <path>              - Show file content or dir listing
/vfs mount <path> [vfs_path] - Mount local folder
/vfs unmount <vfs_path>  - Unmount folder
/vfs sync [path]         - Sync file/dir to disk
/vfs save <vfs_path> <local> - Save file/dir to local path
/vfs refresh <mount>     - Re‑scan mount for changes
/vfs pull <path>         - Reload file/dir from disk
/vfs mounts              - List active mounts
/vfs dirty               - Show modified files
/vfs rm/remove           - Remove Folder or File
    """,

    # System Files (Read‑Only)
    "vfs sys": """/vfs sys-add <local> [path]   - Add file as read‑only system file
/vfs sys-remove <vfs_path>    - Remove a system file
/vfs sys-refresh <vfs_path>   - Reload system file from disk
/vfs sys-list                 - List all system files
    """,

    # Mount Options
    "mount": """
    --readonly   - No write operations
    --no-sync    - Manual sync only
    """,

    # Skill Management
    "skill": """/skill list               - List skills of active agent
/skill list --inactive    - List inactive skills
/skill show <id>          - Show details/instruction
/skill edit <id>          - Edit skill instruction
/skill delete <id>        - Delete a skill
/skill merge <keep_id> <remove_id>
/skill boost <skill_id> 0.3
/skill import <path>      - import skills from directory/skill file
/skill export <id> <path> - Export skill or all skills
    """,

    # Tool Management
    "tools": """/tools list [cat]         - List all tools, optional filter by category
/tools all                - Compact table of every tool (active + disabled) incl. health
/tools info <name>        - Detailed info + health status for one tool
/tools enable <name/cat>  - Enable a tool or entire category
/tools disable <name/cat> - Disable a tool or entire category
/tools enable-all         - Re-enable all disabled tools
/tools disable-all        - Disable all non-system tools
/tools health             - Run health-check on ALL tools (summary)
/tools health <name>      - Run health-check on a single tool
    """,

    # Additional Features
    "feature": """/feature list             - List all features
/feature disable <feature> - Disable a feature
/feature enable <feature>  - Enable a feature
/feature enable desktop    - Enable Desktop Automation
/feature enable web <headless> - Enable Desktop Web Automation
    """,

    # Audio Settings
    # ─── HELP SECTION UPDATE ─────────────────────────────────────────────────────
    "audio": """/audio on                    - Enable verbose audio (all responses spoken)
    /audio off                   - Disable verbose audio
    /audio voice <v>             - Set TTS voice
    /audio backend <b>           - groq_tts / piper / elevenlabs / index_tts
    /audio lang <l>              - de / en / fr / ...
    /audio device                - Interactive output device picker
    /audio device <idx>          - Set output device by index
    /audio device default        - Reset to system default
    /audio devices               - List all output devices
    /audio stop                  - Stop current playback
    /audio restart               - Rebuild player with current settings

    /audio live                  - Start hands-free live mode (VAD + wake word)
    /audio live stop             - Stop live mode
    /audio live status           - Show live mode state
    /audio live keyword <word>   - Set wake word (default: "hey computer")
    /audio live sensitivity <f>  - Wake word sensitivity 0.0–1.0 (default: 0.5)
    /audio live end <mode>       - End-detection: silence / keyword / intent / auto
    /audio live silence <ms>     - Silence timeout before send (default: 800ms)

    /audio speaker               - Speaker profile menu
    /audio speaker list          - List registered speaker profiles
    /audio speaker add <name>    - Register your voice as <name> (5s sample)
    /audio speaker remove <name> - Remove a speaker profile
    /audio speaker who           - Show who is currently detected

    Tip: Append  #audio  to any message for one-time spoken response.
    Tip: In live mode — say wake word, speak, then stop talking or say "fertig".
    """,

    "chain": """/chain list               - Alle gespeicherten Chains auflisten
/chain show <id|name>    - DSL + Metadaten einer Chain anzeigen
/chain accept <id|name>  - Chain als sicher markieren (nötig vor erstem Run)
/chain run <id|name> [input] - Chain direkt ausführen
/chain edit <id|name>    - DSL interaktiv bearbeiten (setzt accepted zurück)
/chain delete <id|name>  - Chain löschen
/chain export <id> <file.json> - Chain als JSON exportieren
/chain import <file.json> - Chain aus JSON importieren

  Aktivierung:  /feature enable chain
  Agent-Tools (für LLM):
    create_validate_chain  - Chain aus DSL erstellen & validieren
    run_chain              - Gespeicherte Chain ausführen
    list_auto_get_fitting  - Chains auflisten & passende für Task finden

  DSL-Kurzreferenz:
    step >> step           - Sequentiell
    (a + b)                - Parallel
    (a | fallback)         - Fehlerbehandlung
    IS(key==val) >> t % f  - Konditional
    tool:name(arg="val")   - Tool-Step
    @agent("fokus")        - Agent-Step
    CF(Model) - "key"      - Pydantic-Format + Extraktion
    def:fn(x) -> expr      - Custom-Funktion
    model:Name(f: type)    - Inline Pydantic-Klasse
    """,

    "autofix": """/feature enable autofix   - AutoFix aktivieren
/chain accept autofix_test_fixer - Einmalig akzeptieren
/chain run autofix_test_fixer    - Ausführen (optional: Pfad als input)

  Flow:
    1. tb --test -x im Projektverzeichnis
    2. Bei Fehler: self analysiert Traceback + Source tief
    3. 2x CoderAgent parallel: Fix A (konservativ) + Fix B (Root-Cause)
    4. self wählt den besseren Fix (APPLY:A / APPLY:B)
    5. Fix anwenden, re-test
    6. Bei Erfolg: git branch + commit + PR (gh oder push)
    7. Bei erneutem Fehler: Report für manuellen Eingriff

  Voraussetzungen:
    /feature enable chain
    /feature enable autofix
    git repo mit remote 'origin'
    optional: gh CLI für direkten PR
    """,

    "autodoc": """/feature enable autodoc           - AutoDoc aktivieren (docs feature NICHT nötig)
/chain accept autodoc_unguided    - Unguided batch einmalig freigeben
/chain accept autodoc_guided      - Guided single einmalig freigeben

  Unguided (batch — scannt ganzes Repo):
    /chain run autodoc_unguided
    → findet getesteten, undokumentierten Code
    → schreibt für jeden: 2-Part Docs

  Guided (single — du gibst das Target vor):
    /chain run autodoc_guided MyClassName
    /chain run autodoc_guided my_function::path/to/file.py

  Docs-Format (immer 2 Parts):
    Part 1 — How to Use:
      - Was macht es (1 Satz)
      - Parameter + Rückgabe
      - 2-3 konkrete Code-Beispiele
    Part 2 — How it Works Internally:
      - Datenfluss Schritt für Schritt
      - Designentscheidungen
      - Upstream-Abhängigkeiten
      - Edge Cases

  Regel: Nur getesteter Code wird dokumentiert.
         Keine Spekulation — nur was der Code tatsächlich tut.
    """,

    "autotest": """/feature enable autotest           - AutoTest aktivieren

  3 Chains — alle brauchen Input: 'Name' oder 'Name::path/to/file.py'

  Logic Tests (bestehender Code):
    /chain accept autotest_logic
    /chain run autotest_logic ChainStore::toolboxv2/mods/isaa/base/chain/chain_tools.py
    → Analysiert Datenfluss, Side-Effects, Edge-Cases
    → Schreibt unittest-Tests, führt aus, fixt bei Fehler

  TDD Future Tests (geplantes Verhalten):
    /chain accept autotest_tdd
    /chain run autotest_tdd MyNewFeature::path/module.py
    → Findet TODOs, Stubs, fehlende Branches
    → Schreibt Tests die JETZT FEHLSCHLAGEN
    → Definiert den Implementierungs-Vertrag

  Full-File Coverage:
    /chain accept autotest_coverage
    /chain run autotest_coverage path/to/module.py
    → Analysiert JEDE testbare Einheit der Datei
    → Schreibt unified Test-File für das gesamte Modul

  Regeln:
    - Nur Python unittest (kein pytest-spezifisch)
    - Mocks für alle Side-Effects (I/O, HTTP, subprocess, state)
    - Async: asyncio.run() oder eigener Event-Loop-Helper
    - Tests: Arrange → Act → Assert, eine Assertion pro Test
    """,
}



def _infer_emotion(text: str) -> TTSEmotion:
    """
    Heuristik: leitet Emotion aus Textinhalt ab.
    Kein LLM-Call — rein regelbasiert für minimale Latenz.
    """
    t = text.lower()
    if any(w in t for w in ("error", "fehler", "achtung", "warning", "kritisch")):
        return TTSEmotion.SERIOUS
    if any(w in t for w in ("!", "super", "excellent", "perfekt", "great")):
        return TTSEmotion.EXCITED
    if any(w in t for w in ("sorry", "entschuldigung", "leider", "unfortunately")):
        return TTSEmotion.EMPATHETIC
    if any(w in t for w in ("dringend", "urgent", "sofort", "immediately")):
        return TTSEmotion.URGENT
    return TTSEmotion.NEUTRAL


# =============================================================================
# ISAA HOST - MAIN CLASS
# =============================================================================

class ISAA_Host:
    """
    The ISAA Host System - A multi-agent host controlled by a Self Agent.

    Features:
    - Global rate limiter configuration shared across all agents
    - Self Agent with exclusive shell access
    - Background task management
    - Agent registry and lifecycle management
    - Audio input via F4 keybinding
    - Skill sharing between agents
    """

    version = VERSION

    def __init__(self, app_instance: Any = None):
        """Initialize the ISAA Host system."""

        self.auto_paste_text = False
        self.dynamic_interval = [1]
        from toolboxv2.mods.isaa.extras.isaa_branding import FlowMatrixAnimation, print_isaa_header

        self.anim = FlowMatrixAnimation(state='initializing', fps=3)

        self.host_id = str(uuid.uuid4())[:8]
        self.idle_hint = print_isaa_header(
            host_id=self.host_id,
            uptime=None,
            version=VERSION,
            state='initializing',
            agent_count=-1,
            task_count=-1,
            show_system_bar=False,
            subtitle='time-random',
        )
        # Startup: animated rain for 2.5s
        # app_instance.run_bg_task_advanced(self.anim.play_startup,duration=1.5)
        self.zen_plus_mode = False
        self.max_iteration = os.getenv("DEFAULT_MAX_ITERATIONS", 30)

        self.app = app_instance or get_app("isaa-host")
        def _(*args, **k):
            text = " ".join(str(a) for a in args)
            try:
                c_print(ANSI(text), **k)
            except:
                c_print(ANSI(text))
        self.app._print = _

        # SSOT: single registry for ALL executions (chat / task / job / delegate)
        self.all_executions: dict[str, ExecutionTask] = {}
        self._focused_task_id: str | None = None

        # Get ISAA Tools module - THE source of truth for agent management
        self.isaa_tools: 'IsaaTools' = self.app.get_mod("isaa")

        # Host state
        self.started_at = datetime.now()

        # Global Rate Limiter Config (shared across all agents)
        self._rate_limiter_config = DEFAULT_RATE_LIMITER_CONFIG.copy()

        # Agent Registry (metadata only - actual instances via isaa_tools)
        self.agent_registry: dict[str, AgentInfo] = {}

        self._task_counter = 0

        # Session state
        self.active_agent_name = "self"
        self.active_session_id = "default"

        # audio
        self.audio_device_index = 0

        # File paths
        self.state_file = Path(self.app.appdata) / "icli" / "isaa_host_state.json"
        self.history_file = Path(self.app.appdata) / "icli" / "isaa_host_history.txt"
        self.rate_limiter_config_file = (
            Path(self.app.appdata) / "icli" / "rate_limiter_config.json"
        )

        if not (Path(self.app.appdata) / "icli" ).exists():
            (Path(self.app.appdata) / "icli").mkdir(parents=True, exist_ok=True)
        if not self.state_file.exists():
            self.state_file.touch(exist_ok=True)
        if not self.history_file.exists():
            self.history_file.touch(exist_ok=True)
        if not self.rate_limiter_config_file.exists():
            self.rate_limiter_config_file.touch(exist_ok=True)

        # Audio state
        self._audio_recording = False
        self._was_recording_is_prossesing_audio = False
        self._audio_buffer: list[bytes] = []
        self._last_transcription: str | None = None
        self._live_engine: Optional[LiveModeEngine] = None
        self._speaker_store: SpeakerProfileStore = SpeakerProfileStore()
        self._live_config: LiveModeConfig = LiveModeConfig()
        self._audio_setup_agents: set[str] = set()

        self._audio_backend = "groq_tts"  # TTSBackend value string
        self._audio_voice = "autumn"
        self._audio_language = "de"
        self._audio_device = None  # None = system default
        self.verbose_audio = False
        self.audio_player = self._build_audio_player()

        self.active_coder: CoderAgent | None = None
        from toolboxv2 import init_cwd
        self.init_dir = init_cwd
        self.active_coder_path: str | None = init_cwd

        # Prompt Toolkit setup
        self.history = FileHistory(str(self.history_file))
        self.key_bindings = self._create_key_bindings()
        self.prompt_session: PromptSession | None = None

        # Self Agent initialization flag
        self._self_agent_initialized = False

        self._task_views: dict[str, TaskView] = {}
        self._overlay: TaskOverlay | None = None

        # Job Scheduler
        self.jobs_file = Path(self.app.appdata) / "icli" / "isaa_host_jobs.json"
        self.job_scheduler: JobScheduler | None = None

        # Load persisted state
        self._load_rate_limiter_config()
        self._load_state()



        self.feature_manager = SimpleFeatureManager()
        for feature in ALL_FEATURES.values():
            feature(self.feature_manager)

        try:
            from toolboxv2.mods.icli_web import IcliWebClient
            c_print("="*20)
            IcliWebClient.get().attach(self)
            c_print("="*20)
        except Exception as e:
            from traceback import format_exc
            c_print(f"WEB ERRRO {format_exc()}")
            self.app.logger.warning(f"icli_web not available: {e}")

    def _build_audio_player(self) -> AudioStreamPlayer:
        """(Re)baut den AudioStreamPlayer aus den aktuellen icli-Einstellungen."""
        cfg = TTSConfig(
            backend=TTSBackend(getattr(self, "_audio_backend", "groq_tts")),
            voice=getattr(self, "_audio_voice", "autumn"),
            language=getattr(self, "_audio_language", "de"),
        )
        device = getattr(self, "_audio_device", None)
        backend = LocalPlayer(device=device)
        return AudioStreamPlayer(
            player_backend=backend,
            tts_config=cfg,
            session_id=getattr(self, "active_session_id", "default"),
        )

    async def _ensure_audio_setup(self, agent_name: str | None = None) -> bool:
        """
        Lazy audio setup: registriert speak-Tool + System-Prompt auf dem Agent.

        Wird aufgerufen:
          - Bei /audio on
          - Bei _handle_agent_interaction wenn should_speak=True
          - Nach _restart_audio_player (invalidiert via _audio_setup_agents.discard)

        Returns True wenn setup erfolgreich, False bei Fehler (Audio bleibt optional).
        """
        name = agent_name or self.active_agent_name

        # Session-ID immer aktuell halten (kein Re-Register nötig)
        self.audio_player.session_id = self.active_session_id

        if name in self._audio_setup_agents:
            return True

        try:
            from toolboxv2.mods.isaa.base.audio_io.audioIo import (
                create_speak_tool, SPEAK_TOOL_SYSTEM_PROMPT
            )
            agent = await self.isaa_tools.get_agent(name)

            # Alten speak-Tool entfernen falls vorhanden (z.B. nach player rebuild)
            try:
                agent.remove_tool("speak")
            except Exception:
                pass

            # Speak-Tool an aktuellen Player binden
            speak_fn = create_speak_tool(self.audio_player)
            agent.add_tool(
                speak_fn,
                name="speak",
                description=(
                    "Speak text aloud to the user. MUST be called for every response "
                    "in audio mode. Call early, call per paragraph. Set emotion."
                ),
                category=["audio", "output"],
            )

            # System-Prompt nur einmal anhängen
            attr = "system_message" if hasattr(agent.amd, "system_message") else "system_prompt"
            existing = getattr(agent.amd, attr, "") or ""
            if "AUDIO MODE" not in existing:
                setattr(agent.amd, attr, existing + "\n\n" + SPEAK_TOOL_SYSTEM_PROMPT)

            self._audio_setup_agents.add(name)
            return True

        except Exception as e:
            print_status(f"Audio setup für '{name}' fehlgeschlagen: {e}", "warning")
            return False

    def _ingest_chunk(self, task_id: str, chunk: dict) -> None:
        """Forward one stream chunk. Sub-agent chunks go to their own TaskView."""
        tv = self._task_views.get(task_id)
        if tv is None:
            return

        sub_id = chunk.get("_sub_agent_id", "")

        if sub_id:
            # Nicht-Swarm-Subagents: wie bisher als geschachtelte TaskView
            # (nur in Detail-View sichtbar, nicht im Footer)
            tv._sub_color(sub_id)
            sub_task_id = tv.sub_task_ids.get(sub_id)
            if sub_task_id is None:
                sub_task_id = f"{task_id}__sub__{sub_id}"
                tv.sub_task_ids[sub_id] = sub_task_id
                self._task_views[sub_task_id] = TaskView(
                    task_id=sub_task_id,
                    agent_name=sub_id,
                    query=f"[sub] {tv.query[:60]}",
                    is_swarm_sub=False,  # kein Swarm — normaler Sub
                    swarm_parent_id=task_id,
                )

            sub_tv = self._task_views[sub_task_id]
            chunk_for_sub = {k: v for k, v in chunk.items() if k != "_sub_agent_id"}
            ingest_chunk(sub_tv, chunk_for_sub)
        else:
            ingest_chunk(tv, chunk)
            if tv.status in ("completed", "failed", "error") and chunk.get("type") in ("done", "error", "final_answer"):
                for sub_task_id in tv.sub_task_ids.values():
                    sub_tv = self._task_views.get(sub_task_id)
                    if sub_tv and sub_tv.status == "running":
                        sub_tv.status = tv.status
                        sub_tv.completed_at = time.time()

        running = any(v.status == "running" for v in self._task_views.values())
        if str(self.dynamic_interval[0]) in ["0", "1"]:
            self.set_dynamic_interval(0.5 if running else 1.5)
        if self.prompt_session and self.prompt_session.app:
            try:
                self.prompt_session.app.invalidate()
            except Exception:
                pass
        if self._overlay:
            self._overlay.invalidate()

    # =========================================================================
    # RATE LIMITER CONFIG MANAGEMENT
    # =========================================================================

    def _load_rate_limiter_config(self):
        """Load rate limiter config from file if exists."""
        data = None
        loaded = None
        if self.rate_limiter_config_file.exists():
            try:
                with open(self.rate_limiter_config_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                    data = f.read()
                    for key in DEFAULT_RATE_LIMITER_CONFIG:
                        if key not in loaded:
                            loaded[key] = DEFAULT_RATE_LIMITER_CONFIG[key]
                    self._rate_limiter_config = loaded
            except Exception as e:
                print_status(f"Failed to load rate limiter config: {e} | {data=} | {loaded=}", "warning")

    def _save_rate_limiter_config(self):
        """Save rate limiter config to file."""
        try:
            self.rate_limiter_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rate_limiter_config_file, "w", encoding="utf-8") as f:
                json.dump(self._rate_limiter_config, f, indent=2)
        except Exception as e:
            print_status(f"Failed to save rate limiter config: {e}", "error")

    def get_rate_limiter_config(self) -> dict:
        """Get the global rate limiter configuration."""
        return self._rate_limiter_config.copy()

    def update_rate_limiter_config(self, updates: dict):
        """Update the global rate limiter configuration."""
        for key, value in updates.items():
            if key in self._rate_limiter_config:
                if isinstance(self._rate_limiter_config[key], dict) and isinstance(
                    value, dict
                ):
                    self._rate_limiter_config[key].update(value)
                else:
                    self._rate_limiter_config[key] = value
        self._save_rate_limiter_config()

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _load_state(self):
        """Load persisted host state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    state = json.load(f)
                    self.active_agent_name = state.get("active_agent", "self")
                    self.active_session_id = state.get("active_session", "default")
                    self.audio_device_index = state.get("audio_device_index", 0)
                    for name, info in state.get("agent_registry", {}).items():
                        self.agent_registry[name] = AgentInfo(
                            name=name,
                            persona=info.get("persona", "default"),
                            is_self_agent=info.get("is_self_agent", False),
                            has_shell_access=info.get("has_shell_access", False),
                            mcp_servers=info.get("mcp_servers", []),
                            bound_agents=info.get("bound_agents", []),
                        )
            except Exception as e:
                print_status(f"Failed to load state: {e}", "warning")

    def _save_state(self):
        """Save host state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "active_agent": self.active_agent_name,
                "active_session": self.active_session_id,
                "audio_device_index": self.audio_device_index,
                "agent_registry": {
                    name: {
                        "persona": info.persona,
                        "is_self_agent": info.is_self_agent,
                        "has_shell_access": info.has_shell_access,
                        "mcp_servers": info.mcp_servers,
                        "bound_agents": info.bound_agents,
                    }
                    for name, info in self.agent_registry.items()
                },
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print_status(f"Failed to save state: {e}", "error")


    @staticmethod
    def _save_to_vfs_and_insert(buf, content: str, vfs, filename: str):
        """Speichert content ins VFS und fügt Referenz in Buffer ein."""
        import time
        ts = int(time.time())
        vfs_path = f"/userpaste/{ts}_{filename}"
        vfs.mkdir("/userpaste", parents=True)  # idempotent
        vfs.create(vfs_path, content)
        buf.insert_text(f"[vfs:{vfs_path}]")

    def _save_to_vfs_async_and_insert(self, buf, content: str, filename: str):
        """Async Variante: holt VFS, speichert, insertet."""

        async def _do():
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                session = await agent.session_manager.get_or_create(self.active_session_id)
                vfs = getattr(session, "vfs", None)
                if vfs:
                    self._save_to_vfs_and_insert(buf, content, vfs, filename)
                else:
                    buf.insert_text(content)  # Fallback
            except Exception:
                buf.insert_text(content)  # Fallback

        asyncio.ensure_future(_do())

    # =========================================================================
    # KEY BINDINGS (AUDIO)
    # =========================================================================

    def _create_key_bindings(self) -> KeyBindings:
        """Create prompt_toolkit key bindings."""
        kb = KeyBindings()

        # ── Safe Paste: Bracketed-Paste (große Texte, Dateipfade, Medien) ──
        @kb.add("<bracketed-paste>")
        def handle_bracketed_paste(event):
            buf = event.app.current_buffer
            raw = event.data or ""
            safe = _safe_decode_paste(raw)
            stripped = safe.strip()

            # ── Mediendatei-Pfad → [media:] Tag ──────────────────────────
            if _is_pasteable_media_path(stripped):
                clean_path = stripped.strip('"').strip("'")
                buf.insert_text(f"[media:{clean_path}]")
                return

            # ── Textdatei-Pfad → ins VFS + Referenz einfügen ─────────────
            if self.auto_paste_text and "\n" not in stripped and len(stripped.split()) == 1:
                candidate = Path(stripped.strip('"').strip("'"))
                if candidate.is_file() and candidate.suffix.lower() in (
                        ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml",
                        ".yml", ".toml", ".csv", ".log", ".sh", ".env",
                ):
                    try:
                        content = candidate.read_text(encoding="utf-8", errors="replace")
                        filename = candidate.name
                        self._save_to_vfs_async_and_insert(buf, content, filename)
                        return
                    except Exception:
                        pass  # Fallthrough

            # ── Normaler großer Text → ins VFS ───────────────────────────
            if "\n" in safe or len(safe) > 500:
                self._save_to_vfs_async_and_insert(buf, safe, "paste.txt")
                return

            # ── Kurzer Text: direkt einfügen ─────────────────────────────
            buf.insert_text(safe)

        @kb.add("c-j")  # Ctrl+J → Newline (einzige echte Option)
        def _newline(event):
            event.app.current_buffer.insert_text("\n")

        @kb.add("c-m")  # Enter → Submit
        def _submit(event):
            event.app.current_buffer.validate_and_handle()

        # ── Ctrl+V Fallback (terminals ohne bracketed-paste support) ──────
        @kb.add("c-v")
        def handle_ctrl_v(event):
            try:
                import pyperclip
                text = pyperclip.paste() or ""
            except Exception:
                return
            if not text:
                return
            safe = _safe_decode_paste(text)
            stripped = safe.strip()
            buf = event.app.current_buffer

            if _is_pasteable_media_path(stripped):
                clean_path = stripped.strip('"').strip("'")
                buf.insert_text(f"[media:{clean_path}]")
                return

            if "\n" in safe or len(safe) > 500:
                self._save_to_vfs_async_and_insert(buf, safe, "paste.txt")
                return

            buf.insert_text(safe)

        @kb.add("f4")
        def _(event):
            """Toggle audio recording with F4."""
            asyncio.create_task(self._toggle_audio_recording())
            self.anim.set_mode('audio' if self._audio_recording else 'dreaming')

        @kb.add("f5")
        def _(event):
            """Show status dashboard with F5."""
            async def __():
                await self._print_status_dashboard()
                await self._cmd_vfs([])
                await self._cmd_session(["show"])
            asyncio.create_task(__())

        @kb.add("f6")
        def _(event):
            running = [t for t in self.all_executions.values() if t.status == "running"]
            if not self._focused_task_id:
                if running:
                    first = running[0]
                    first.is_focused = True
                    self._focused_task_id = first.task_id
                    c_print(HTML(f"<style fg='{PTColors.ZEN_CYAN}'>  ◎ Focus → {first.task_id}</style>"))
                else:
                    c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>  No active tasks</style>"))
                return
            if self._focused_task_id in self.all_executions:
                self.all_executions[self._focused_task_id].is_focused = False
            self._focused_task_id = None
            c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>  ▾ Task unfocused</style>"))

        @kb.add("f7")
        def _(event):
            candidates = [tid for tid, t in self.all_executions.items() if t.status == "running"]
            if not candidates:
                return

            if self._focused_task_id and self._focused_task_id in self.all_executions:
                self.all_executions[self._focused_task_id].is_focused = False

            try:
                idx = candidates.index(self._focused_task_id)
                next_id = candidates[(idx + 1) % len(candidates)]
            except (ValueError, TypeError):
                next_id = candidates[0]

            self.all_executions[next_id].is_focused = True
            self._focused_task_id = next_id

            c_print(HTML(f"<style fg='#67e8f9'>  ◎ Focus → {next_id}</style>"))
            event.app.invalidate()

        @kb.add("f8")
        def _(event):
            """F8: Cancel the focused task — isolated, does not kill the CLI."""
            if not self._focused_task_id:
                return
            exc = self.all_executions.get(self._focused_task_id)
            if exc and exc.status == "running":
                exc.async_task.cancel()
                c_print(HTML(
                    f"<style fg='#fbbf24'>  ⚠ Cancelling {exc.task_id}...</style>"
                ))

        @kb.add("f9")
        def _(event):
            """F9: Abgeschlossene Tasks aus der Ansicht entfernen."""
            done_ids = [
                tid for tid, t in self.all_executions.items()
                if t.status in ("completed", "failed", "error", "cancelled")
            ]
            for tid in done_ids:
                self.all_executions.pop(tid, None)
                self._task_views.pop(tid, None)
                if self._focused_task_id == tid:
                    self._focused_task_id = None
            if done_ids:
                c_print(HTML(
                    f"<style fg='{PTColors.ZEN_DIM}'>  ✓ {len(done_ids)} task(s) closed</style>"
                ))

        @kb.add("f2")
        def _(event):
            if self._overlay:
                # Already open — ignore (Esc closes it from inside)
                return
            self.zen_plus_mode = not self.zen_plus_mode
            mode = "ZEN+" if self.zen_plus_mode else "ZEN"
            c_print(HTML(f"<style fg='#67e8f9'>  ◎ Mode: {mode}</style>"))

            if self.zen_plus_mode:
                overlay = TaskOverlay(self._task_views)
                self._overlay = overlay

                async def _run_overlay():
                    def _on_exit():
                        self.zen_plus_mode = False
                        self._overlay = None
                        if self.prompt_session and self.prompt_session.app:
                            try:
                                self.prompt_session.app.invalidate()
                            except Exception:
                                pass

                    await overlay.run(on_exit=_on_exit)

                asyncio.create_task(_run_overlay())
            else:
                # Toggle off — nothing to close (Esc does it from inside)
                self._overlay = None

        @kb.add("tab")
        def handle_tab(event):
            buf = event.app.current_buffer

            if buf.complete_state:
                completions = list(buf.complete_state.completions)

                if len(completions) == 1:
                    # Eindeutig → sofort akzeptieren
                    buf.apply_completion(completions[0])
                    # Directory drill-down
                    if buf.text.rstrip().endswith("/"):
                        buf.start_completion()
                else:
                    # Mehrere → common prefix einfügen oder cyclen
                    common = _common_prefix([c.text for c in completions])
                    current_partial = buf.document.get_word_before_cursor()

                    if common and len(common) > len(current_partial):
                        # Gemeinsamen Prefix einfügen (ohne Menü zu schließen)
                        buf.insert_text(common[len(current_partial):])
                    else:
                        # Kein weiterer Prefix → nächste Option selecten
                        buf.complete_next()
            else:
                buf.start_completion()
                if buf.complete_state:
                    completions = list(buf.complete_state.completions)
                    if len(completions) == 1:
                        buf.apply_completion(completions[0])
                        if buf.text.rstrip().endswith("/"):
                            buf.start_completion()

        @kb.add("s-tab")
        def handle_shift_tab(event):
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.complete_previous()

        def _common_prefix(strings: list[str]) -> str:
            if not strings:
                return ""
            prefix = strings[0]
            for s in strings[1:]:
                while not s.startswith(prefix):
                    prefix = prefix[:-1]
                    if not prefix:
                        return ""
            return prefix

        return kb

    async def _toggle_audio_recording(self):
        """Toggle audio recording state."""
        if self._audio_recording:
            self._audio_recording = False
            print_status("Processing audio...", "progress")
            self.app.run_bg_task_advanced(self._process_recorded_audio)
        else:
            self._audio_recording = True
            self._audio_buffer = []
            print_status("🎤 Recording... Press F4 to stop", "info")
            asyncio.create_task(self._record_audio())

    def _select_audio_device(self):
        """Select audio input device."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            devices_names = []
            for i, device in enumerate(devices):
                if device["name"] in devices_names:
                    continue
                devices_names.append(device["name"])
                print(f"[{i}] {device['name']}")
            device_index = int(input("Select device index: "))
            self.audio_device_index = device_index
        except ImportError:
            print_status(
                "Audio requires 'sounddevice'. Install: pip install sounddevice", "error"
            )

    async def active_refresher(self):
        while self.app.alive:
            self.prompt_session.app.invalidate()
            await asyncio.sleep(self.dynamic_interval[0])

    def set_dynamic_interval(self, t):
        self.dynamic_interval[0] = t

    async def _record_audio(self):
        """Record audio from microphone."""
        try:
            import numpy as np
            import sounddevice as sd

            sample_rate = 16000
            channels = 1

            def callback(indata, frames, time, status):
                if self._audio_recording:
                    self._audio_buffer.append(indata.copy())

            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                callback=callback,
                device=self.audio_device_index
            ):
                while self._audio_recording:
                    await asyncio.sleep(0.1)

        except ImportError:
            print_status(
                "Audio requires 'sounddevice'. Install: pip install sounddevice", "error"
            )
            self._audio_recording = False
        except Exception as e:
            print_status(f"Audio recording error: {e}", "error")
            self._audio_recording = False

    async def _process_recorded_audio(self):
        """Process recorded audio and transcribe."""
        if not self._audio_buffer:
            print_status("No audio recorded", "warning")
            return
        self._was_recording_is_prossesing_audio = True
        try:
            import numpy as np

            from toolboxv2.mods.isaa.base.audio_io.Stt import STTConfig, transcribe

            audio_data = np.concatenate(self._audio_buffer)
            audio_bytes = audio_data.tobytes()

            result = transcribe(audio_bytes, config=STTConfig(language="de"))

            if result.text:
                print_status(f"Transcribed: {result.text}", "success")
                self._last_transcription = result.text
            else:
                print_status("No speech detected", "warning")

        except ImportError as e:
            print_status(f"Audio processing requires additional packages: {e}", "error")
        except Exception as e:
            print_status(f"Audio processing error: {e}", "error")
            import traceback
            c_print(traceback.format_exc())
        finally:
            self._was_recording_is_prossesing_audio = False
            self._audio_buffer = []

    # =========================================================================
    # SELF AGENT INITIALIZATION
    # =========================================================================

    async def _init_self_agent(self):
        """Initialize the Self Agent with exclusive capabilities."""
        if self._self_agent_initialized:
            return

        print_status("Initializing Self Agent...", "progress")

        builder = self.isaa_tools.get_agent_builder(
            name="self", add_base_tools=True, with_dangerous_shell=True
        )

        self._apply_rate_limiter_to_builder(builder)
        self._register_self_agent_tools(builder)

        await self.isaa_tools.register_agent(builder)

        self.agent_registry["self"] = AgentInfo(
            name="self",
            persona="Host Administrator",
            is_self_agent=True,
            has_shell_access=True,
        )

        self._self_agent_initialized = True
        print_status("Self Agent initialized", "success")

    def _apply_rate_limiter_to_builder(self, builder: FlowAgentBuilder):
        """Apply global rate limiter config to a builder."""
        features = self._rate_limiter_config.get("features", {})

        builder.with_rate_limiter(
            enable_rate_limiting=features.get("rate_limiting", True),
            enable_model_fallback=features.get("model_fallback", True),
            enable_key_rotation=features.get("key_rotation", True),
            key_rotation_mode=features.get("key_rotation_mode", "balance"),
        )

        for provider, keys in self._rate_limiter_config.get("api_keys", {}).items():
            for key in keys:
                builder.add_api_key(provider, key)

        for primary, fallbacks in self._rate_limiter_config.get(
            "fallback_chains", {}
        ).items():
            builder.add_fallback_chain(primary, fallbacks)

        for model, limits in self._rate_limiter_config.get("limits", {}).items():
            builder.set_model_limits(model, **limits)

    def _register_self_agent_tools(self, builder: FlowAgentBuilder):
        """Register exclusive tools for the Self Agent."""

        host_ref = self

        # ===== AGENT MANAGEMENT TOOLS =====

        async def cli_spawn_agent(
            name: str,
            persona: str = "general assistant",
            model: str | None = None,
            background: bool = False,
        ) -> str:
            """
            Spawn a new agent with the given name and persona.

            Args:
                name: Unique name for the agent
                persona: Description of the agent's role/personality
                model: Optional model override
                background: If True, agent runs in background mode

            Returns:
                Status message about agent creation
            """
            return await host_ref._tool_spawn_agent(name, persona, model, background)

        async def cli_list_agents() -> str:
            """
            List all registered agents and their status.

            Returns:
                Formatted list of all agents with their status
            """
            return await host_ref._tool_list_agents()

        # ===== TASK MANAGEMENT TOOLS =====

        async def cli_delegate(
            agent_name: str,
            task: str,
            wait: bool = True,
            session_id: str = "default",
        ) -> str:
            """
            Delegate a task to another agent.

            Args:
                agent_name: Name of the agent to delegate to
                task: The task/query to execute
                wait: If True, wait for result. If False, run in background
                session_id: Session ID for the task

            Returns:
                Result of the task or background task ID
            """
            return await host_ref._tool_delegate(agent_name, task, wait, session_id)

        async def cli_stop_agent(agent_name: str) -> str:
            """
            Stop all running tasks for an agent.

            Args:
                agent_name: Name of the agent to stop

            Returns:
                Status message
            """
            return await host_ref._tool_stop_agent(agent_name)

        async def cli_task_status(task_id: str | None = None) -> str:
            """
            Check status of background tasks.

            Args:
                task_id: Specific task ID or None for all tasks

            Returns:
                Task status information
            """
            return await host_ref._tool_task_status(task_id)

        # ===== SKILL SHARING TOOLS =====

        async def cli_teach_skill(
            target_agent: str,
            skill_name: str,
            instruction: str,
            triggers: list[str],
        ) -> str:
            """
            Teach a skill to an agent.

            Args:
                target_agent: Name of the agent to teach
                skill_name: Name for the skill
                instruction: Step-by-step instructions for the skill
                triggers: Keywords that activate this skill

            Returns:
                Status message
            """
            return await host_ref._tool_teach_skill(
                target_agent, skill_name, instruction, triggers
            )

        async def cli_bind_agents(
            agent_a: str, agent_b: str, mode: str = "public"
        ) -> str:
            """
            Bind two agents for data sharing.

            Args:
                agent_a: First agent name
                agent_b: Second agent name
                mode: Binding mode ('public' or 'private')

            Returns:
                Status message
            """
            return await host_ref._tool_bind_agents(agent_a, agent_b, mode)

        # ===== SYSTEM TOOLS =====

        async def cli_mcp_connect(
            server_name: str,
            command: str,
            args: list[str],
            target_agent: str | None = None,
        ) -> str:
            """
            Connect an MCP server to an agent.

            Args:
                server_name: Name for the MCP server
                command: Command to start the MCP server
                args: Arguments for the MCP server
                target_agent: Agent to add MCP to (None = create new agent)

            Returns:
                Status message
            """
            return await host_ref._tool_mcp_connect(
                server_name, command, args, target_agent
            )

        async def cli_update_agent_config(agent_name: str, config_updates: dict) -> str:
            """
            Update an agent's configuration (saved for next restart).

            Args:
                agent_name: Agent to update
                config_updates: Configuration updates (dict)

            Returns:
                Status message
            """
            return await host_ref._tool_update_agent_config(agent_name, config_updates)

        # ===== REGISTER ALL TOOLS =====
        builder.add_tool(
            cli_spawn_agent,
            "spawnAgent",
            "Create a new agent with specified name and persona",
            category=["agent_management"],
        )
        builder.add_tool(
            cli_list_agents,
            "listAgents",
            "List all registered agents and their status",
            category=["agent_management"],
        )
        builder.add_tool(
            cli_delegate,
            "delegate",
            "Delegate a task to another agent",
            category=["task_management"],
        )
        builder.add_tool(
            cli_stop_agent,
            "stopAgent",
            "Stop running tasks for an agent",
            category=["task_management"],
        )
        builder.add_tool(
            cli_task_status,
            "taskStatus",
            "Check status of background tasks",
            category=["task_management"],
        )
        builder.add_tool(
            cli_teach_skill,
            "teachSkill",
            "Teach a skill to an agent",
            category=["skill_sharing"],
        )
        builder.add_tool(
            cli_bind_agents,
            "bindAgents",
            "Bind two agents for data sharing",
            category=["agent_binding"],
        )
        builder.add_tool(
            cli_mcp_connect,
            "mcpConnect",
            "Connect an MCP server to an agent",
            category=["mcp"],
        )
        builder.add_tool(
            cli_update_agent_config,
            "updateAgentConfig",
            "Update agent configuration for next restart",
            category=["agent_management"],
        )

        # ===== JOB MANAGEMENT TOOLS =====

        async def cli_create_job(
            name: str,
            agent_name: str,
            query: str,
            trigger_type: str,
            trigger_config: dict | None = None,
            timeout_seconds: int = 300,
            session_id: str = "default",
            # Trigger-specific parameters (for backward compatibility)
            cron_expression: str | None = None,
            interval_seconds: int | None = None,
            at_datetime: str | None = None,
            watch_job_id: str | None = None,
            watch_path: str | None = None,
            watch_patterns: list[str] | None = None,
            webhook_path: str | None = None,
            idle_seconds: int | None = None,
            agent_idle_seconds: int | None = None,
            # Allow additional trigger parameters via **kwargs
            **extra_trigger_kwargs
        ) -> str:
            """
            Create a persistent scheduled job that fires an agent query on a trigger.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            ⚠️  IMPORTANT: TRIGGER PARAMETER USAGE
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Trigger-specific parameters (cron_expression, interval_seconds, etc.)
            can be passed in THREE ways:

            1. DIRECTLY as function parameters (recommended):
               createJob(name="my-job", trigger_type="on_cron", cron_expression="0 2 * * 0")

            2. Via trigger_config dict:
               createJob(name="my-job", trigger_type="on_cron",
                        trigger_config={"cron_expression": "0 2 * * 0"})

            3. Mixed - parameters in trigger_config override direct parameters

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            Args:
                name: Human-readable job name
                agent_name: Which agent runs this job (e.g. 'self', 'researcher')
                query: The prompt/query to send to the agent
                trigger_type: Trigger type (on_time, on_interval, on_cron, on_cli_start,
                               on_cli_exit, on_job_completed, on_job_failed, on_file_changed,
                               on_network_available, on_system_idle, on_webhook_received, on_agent_idle, on_dream_start, on_dream_end,
                               on_dream_budget_hit, on_dream_skill_evolved, etc.)
                trigger_config: Optional trigger parameters dict - overrides direct params
                timeout_seconds: Max execution time in seconds (default 300)
                session_id: Session ID for the agent (default 'default')

            Trigger-Specific Parameters (for on_cron, on_interval, etc.):
                cron_expression: Cron schedule string (e.g. "0 2 * * 0" for Sunday 2am)
                interval_seconds: Fire every N seconds
                at_datetime: ISO datetime string for one-time execution
                watch_job_id: Job ID to watch for job_completion/failed/timeout triggers
                watch_path: File/directory path to watch for on_file_changed
                watch_patterns: Glob patterns for file watching
                webhook_path: HTTP path for on_webhook_received
                idle_seconds: Idle threshold for on_system_idle

            Returns:
                Job ID or error message

            Examples:
                # Cron job (weekly Sunday 2am)
                createJob(name="weekly-update", trigger_type="on_cron",
                         agent_name="self", query="run updates",
                         cron_expression="0 2 * * 0")

                # Interval job (every 5 minutes)
                createJob(name="heartbeat", trigger_type="on_interval",
                         agent_name="self", query="ping server",
                         interval_seconds=300)

                # Using trigger_config dict
                createJob(name="daily-backup", trigger_type="on_cron",
                         agent_name="self", query="backup database",
                         trigger_config={"cron_expression": "0 3 * * *"})
            """
            if not host_ref.job_scheduler:
                return "✗ Job scheduler not initialized"

            try:
                import traceback

                # Build trigger config from multiple sources
                # Priority: trigger_config > direct parameters > None
                trigger_params = {}

                # Add direct parameters if provided
                if cron_expression is not None:
                    trigger_params["cron_expression"] = cron_expression
                if interval_seconds is not None:
                    trigger_params["interval_seconds"] = interval_seconds
                if at_datetime is not None:
                    trigger_params["at_datetime"] = at_datetime
                if watch_job_id is not None:
                    trigger_params["watch_job_id"] = watch_job_id
                if watch_path is not None:
                    trigger_params["watch_path"] = watch_path
                if watch_patterns is not None:
                    trigger_params["watch_patterns"] = watch_patterns
                if webhook_path is not None:
                    trigger_params["webhook_path"] = webhook_path
                if idle_seconds is not None:
                    trigger_params["idle_seconds"] = idle_seconds
                if agent_idle_seconds is not None:
                    trigger_params["agent_idle_seconds"] = agent_idle_seconds

                # Add extra kwargs (for extensibility)
                trigger_params.update(extra_trigger_kwargs)

                # Override with trigger_config if provided
                if trigger_config:
                    trigger_params.update(trigger_config)

                # Create TriggerConfig
                tc = TriggerConfig(trigger_type=trigger_type)

                # Apply all trigger parameters
                for k, v in trigger_params.items():
                    if hasattr(tc, k):
                        setattr(tc, k, v)
                    else:
                        # Warn about unknown parameters but don't fail
                        c_print(f"Unknown trigger parameter: {k}={v}")

                # Create job definition
                job = JobDefinition(
                    job_id=JobDefinition.generate_id(),
                    name=name,
                    agent_name=agent_name,
                    query=query,
                    trigger=tc,
                    timeout_seconds=timeout_seconds,
                    session_id=session_id,
                )

                # Add job to scheduler
                job_id = host_ref.job_scheduler.add_job(job)

                return (
                    f"✓ Job created successfully!\n"
                    f"  Job ID: {job_id}\n"
                    f"  Name: {name}\n"
                    f"  Trigger: {trigger_type}\n"
                    f"  Config: {trigger_params or '(default)'}\n"
                    f"  Agent: {agent_name}\n"
                    f"  Verify with: listJobs()"
                )

            except Exception as e:
                # LÖSUNG 2: Bessere Fehlerbehandlung mit Traceback
                import traceback
                error_details = traceback.format_exc()

                get_logger().error(f"Failed to create job '{name}': {e}\n{error_details}")

                return (
                    f"✗ Failed to create job '{name}'\n"
                    f"  Error: {e}\n"
                    f"  Trigger Type: {trigger_type}\n"
                    f"  Parameters: {locals().get('trigger_params', {})}\n\n"
                    f"  Debug Info:\n"
                    f"  {error_details}"
                )

        async def cli_delete_job(job_id: str) -> str:
            """
            Delete a scheduled job by ID.

            Args:
                job_id: The job ID to delete

            Returns:
                Status message
            """
            if not host_ref.job_scheduler:
                return "Job scheduler not initialized"
            if host_ref.job_scheduler.remove_job(job_id):
                return f"✓ Job {job_id} deleted"
            return f"✗ Job {job_id} not found"

        async def cli_list_jobs() -> str:
            """
            List all scheduled jobs with their status.

            Returns:
                Formatted list of all jobs
            """
            if not host_ref.job_scheduler:
                return "Job scheduler not initialized"
            jobs = host_ref.job_scheduler.list_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = ["=== Scheduled Jobs ===\n"]
            for j in jobs:
                lines.append(
                    f"  {j.job_id} | {j.name} | {j.trigger.trigger_type} | "
                    f"{j.status} | runs:{j.run_count} fails:{j.fail_count}"
                )
                if j.last_result:
                    lines.append(f"    Last: {j.last_result} at {j.last_run_at}")
            return "\n".join(lines)

        async def cli_create_dream_job(
            agent_name: str = "self",
            trigger_type: str = "on_cron",
            cron_expression: str = "0 3 * * *",
            agent_idle_seconds: int | None = None,
            max_budget: int = 3000,
            do_skill_split: bool = True,
            do_skill_evolve: bool = True,
            do_persona_evolve: bool = True,
            do_create_new: bool = True,
            hard_stop: bool = False,
        ) -> str:
            """
            Create a dream job (async meta-learning cycle).

            Args:
                agent_name: Agent to dream (default: self)
                trigger_type: on_cron (default), on_agent_idle, on_job_completed, etc.
                cron_expression: Cron schedule (default: nightly 03:00)
                agent_idle_seconds: Idle threshold for on_agent_idle trigger
                max_budget: Max tokens for LLM calls during dream
                do_skill_split: Split bloated skills into sub-skills
                do_skill_evolve: Refine instructions from failure patterns
                do_persona_evolve: Adjust persona profiles
                do_create_new: Allow creation of new skills/personas
                hard_stop: Abort on first error (False = skip & continue)

            Examples:
                createDreamJob()                                          # Nightly at 03:00
                createDreamJob(trigger_type="on_agent_idle", agent_idle_seconds=600)  # After 10min idle
                createDreamJob(trigger_type="on_job_completed")           # After every successful job
            """
            dream_config = {
                "max_budget": max_budget,
                "do_skill_split": do_skill_split,
                "do_skill_evolve": do_skill_evolve,
                "do_persona_evolve": do_persona_evolve,
                "do_create_new": do_create_new,
                "hard_stop": hard_stop,
            }

            return await cli_create_job(
                name=f"dream-{agent_name}",
                agent_name=agent_name,
                query="__dream__",
                trigger_type=trigger_type,
                cron_expression=cron_expression if trigger_type == "on_cron" else None,
                agent_idle_seconds=agent_idle_seconds if trigger_type == "on_agent_idle" else None,
                trigger_config={"extra": {"dream_config": dream_config}},
                timeout_seconds=600,
            )

        builder.add_tool(
            cli_create_dream_job,
            "createDreamJob",
            "Create a dream job (async meta-learning) with configurable triggers",
            category=["job_management"],
        )

        builder.add_tool(
            cli_create_job,
            "createJob",
            "Create a persistent scheduled job that fires an agent on a trigger",
            category=["job_management"],
        )
        builder.add_tool(
            cli_delete_job,
            "deleteJob",
            "Delete a scheduled job",
            category=["job_management"],
        )
        builder.add_tool(
            cli_list_jobs,
            "listJobs",
            "List all scheduled jobs with status",
            category=["job_management"],
        )

    # =========================================================================
    # TOOL IMPLEMENTATIONS
    # =========================================================================

    async def _tool_spawn_agent(
        self, name: str, persona: str, model=None, background=False
    ) -> str:
        try:
            if name in self.agent_registry:
                return f"Agent '{name}' already exists."

            builder = self.isaa_tools.get_agent_builder(
                name=name, add_base_tools=True, with_dangerous_shell=False
            )
            self._apply_rate_limiter_to_builder(builder)
            builder.config.system_message = f"You are {persona}. Act according to this role."
            if model:
                builder.with_models(model, model)

            await self.isaa_tools.register_agent(builder)

            self.agent_registry[name] = AgentInfo(
                name=name, persona=persona, is_self_agent=False, has_shell_access=False
            )
            self._save_state()

            # ── NEU: TaskView für Spawn-Vorgang eintragen ─────────────────────
            self._task_counter += 1
            spawn_id = f"spawn_{self._task_counter}_{name}"
            tv = TaskView(task_id=spawn_id, agent_name=name, query=f"[spawn] {persona[:60]}")
            tv.status = "completed"
            tv.completed_at = time.time()
            self._task_views[spawn_id] = tv

            return f"✓ Agent '{name}' spawned with persona: {persona}"

        except Exception as e:
            return f"✗ Failed to spawn agent: {e}"

    async def _tool_list_agents(self) -> str:
        """Implementation: List all agents."""
        isaa_agents: list[str] = self.isaa_tools.config.get("agents-name-list", [])

        result = ["=== Registered Agents ===\n"]

        for agent_name in isaa_agents:
            info = self.agent_registry.get(agent_name, AgentInfo(name=agent_name))

            instance_key = f"agent-instance-{agent_name}"
            is_active = instance_key in self.isaa_tools.config

            bg_tasks = sum(
                1
                for t in self.all_executions.values()
                if t.agent_name == agent_name and t.status == "running"
            )

            status = "🟢 Active" if is_active else "⚪ Idle"
            shell_icon = "🔓" if info.has_shell_access else ""
            self_icon = "👑" if info.is_self_agent else ""

            result.append(
                f"  {self_icon}{agent_name} {shell_icon}\n"
                f"    Status: {status}\n"
                f"    Persona: {info.persona}\n"
                f"    Background Tasks: {bg_tasks}\n"
                f"    Bound To: {', '.join(info.bound_agents) or 'None'}\n"
            )

        if not isaa_agents:
            result.append("  No agents registered.\n")

        return "\n".join(result)

    async def _tool_delegate(
        self, agent_name: str, task: str, wait: bool, session_id: str
    ) -> str:
        """Delegate task — jetzt MIT Stream + Ingest-Hook für beide Modi."""
        try:
            agent = await self.isaa_tools.get_agent(agent_name)
            run_id = uuid.uuid4().hex[:8]

            # ── STREAM starten (statt a_run) ──────────────────────────────────
            stream = agent.a_stream(query=task, session_id=session_id)

            exc = self._create_execution(
                kind="delegate",
                agent_name=agent_name,
                query=task,
                async_task=None,
                run_id=run_id,
                stream=stream,
                take_focus=False,
            )
            task_id = exc.task_id

            async_task = asyncio.create_task(
                self._drain_agent_stream(task_id, stream, False)
            )
            exc.async_task = async_task

            async_task.add_done_callback(
                lambda fut: self._on_agent_task_done(task_id, fut)
            )

            # ── WAIT=True: Caller blockiert, Stream läuft trotzdem durch ──────
            if wait:
                try:
                    result = await asyncio.shield(async_task)
                    return str(result) if result else ""
                except asyncio.CancelledError:
                    return ""

            # ── WAIT=False: sofort zurück, task läuft im Hintergrund ──────────
            return f"✓ Background task started: {task_id} (RunID: {run_id})"

        except Exception as e:
            return f"✗ Delegation failed: {e}"

    async def _tool_stop_agent(self, agent_name: str) -> str:
        """Implementation: Stop agent tasks."""
        stopped = 0

        for _, bg_task in list(self.all_executions.items()):
            if bg_task.agent_name == agent_name and bg_task.status == "running":
                bg_task.async_task.cancel()
                bg_task.status = "cancelled"
                stopped += 1

        try:
            agent = await self.isaa_tools.get_agent(agent_name)
            # Cancel all active executions for this agent
            for exec_info in agent.list_executions():
                exec_id = exec_info.get("run_id")
                if exec_id:
                    await agent.cancel_execution(exec_id)
        except Exception:
            pass

        return f"✓ Stopped {stopped} task(s) for agent '{agent_name}'"

    async def _tool_task_status(self, task_id: str | None = None) -> str:
        """Implementation: Check task status."""
        import time as _t
        if task_id and task_id in self.all_executions:
            t = self.all_executions[task_id]
            elapsed = _t.time() - t.started_at
            return (
                f"Task: {t.task_id}\n"
                f"Kind: {t.kind}\n"
                f"Agent: {t.agent_name}\n"
                f"Query: {t.query}\n"
                f"Status: {t.status}\n"
                f"Elapsed: {elapsed:.1f}s"
            )

        result = ["=== Executions ===\n"]

        for tid, t in self.all_executions.items():
            result.append(
                f"  [{t.status.upper()}] {tid} ({t.kind})\n"
                f"    Agent: {t.agent_name}\n"
                f"    Query: {t.query[:50]}...\n"
            )

        if not self.all_executions:
            result.append("  No executions.\n")

        return "\n".join(result)

    async def _tool_teach_skill(
        self, target_agent: str, skill_name: str, instruction: str, triggers: list[str]
    ) -> str:
        """Implementation: Teach skill to agent."""
        try:
            agent = await self.isaa_tools.get_agent(target_agent)
            session_id = "default"

            skill_id = f"custom_{skill_name}_{uuid.uuid4().hex[:6]}"
            skill_data = {
                "id": skill_id,
                "name": skill_name,
                "triggers": triggers,
                "instruction": instruction,
                "tools_used": [],
                "tool_groups": [],
                "source": "taught",
                "confidence": 0.8,
                "activation_threshold": 0.6,
                "success_count": 0,
                "failure_count": 0,
                "created_at": datetime.now().isoformat(),
                "last_used": None,
            }

            exec_engine = agent._get_execution_engine()
            success = exec_engine.skills_manager.import_skill(skill_data, overwrite=True)

            if success:
                return f"✓ Skill '{skill_name}' taught to agent '{target_agent}'"
            else:
                return f"✗ Failed to import skill to agent '{target_agent}'"

        except Exception as e:
            return f"✗ Failed to teach skill: {e}"

    async def _tool_bind_agents(
        self, agent_a: str, agent_b: str, mode: str = "public"
    ) -> str:
        """Implementation: Bind two agents."""
        try:
            agent_a_instance = await self.isaa_tools.get_agent(agent_a)
            agent_b_instance = await self.isaa_tools.get_agent(agent_b)

            await agent_a_instance.bind_manager.bind(partner=agent_b_instance, mode=mode)

            if (
                agent_a in self.agent_registry
                and agent_b not in self.agent_registry[agent_a].bound_agents
            ):
                self.agent_registry[agent_a].bound_agents.append(agent_b)
            if (
                agent_b in self.agent_registry
                and agent_a not in self.agent_registry[agent_b].bound_agents
            ):
                self.agent_registry[agent_b].bound_agents.append(agent_a)

            self._save_state()

            return f"✓ Agents '{agent_a}' and '{agent_b}' bound in '{mode}' mode"

        except Exception as e:
            return f"✗ Failed to bind agents: {e}"

    def _tool_shell(self, command: str) -> str:
        """
        Führt einen Shell-Befehl LIVE und INTERAKTIV aus.
        Unterstützt Windows (CMD/PowerShell) und Unix (Bash/Zsh).
        """
        import subprocess

        # Shell-Erkennung (Windows/Unix)
        shell_exe, cmd_flag = detect_shell()

        # Vorbereitung für Windows "Charm" / ANSI Support
        # Wir übergeben stdin/stdout/stderr direkt (None), damit der Prozess
        # das Terminal "besitzt".
        try:
            # Wir nutzen subprocess.run OHNE capture_output,
            # damit das Terminal direkt interagieren kann.
            print_separator(char="═")

            # Ausführung im Vordergrund
            process = subprocess.run(
                [shell_exe, cmd_flag, command],
                stdin=None,  # Direktes Terminal-Input
                stdout=None,  # Direktes Terminal-Output (Live!)
                stderr=None,  # Direktes Terminal-Error
                check=False
            )

            print_separator(char="═")

            result = {
                "success": process.returncode == 0,
                "exit_code": process.returncode
            }
            return json.dumps(result)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    async def _handle_shell(self, command: str):
        """Wrapper für den Shell-Befehl mit Prompt-Toolkit Suspension."""
        from prompt_toolkit.eventloop import run_in_executor_with_context

        # WICHTIG: Wir müssen die prompt_toolkit UI pausieren,
        # damit die Shell das Terminal sauber übernehmen kann.
        try:
            # Wir führen die blockierende Shell-Aktion in einem Thread aus,
            # aber geben ihr vollen Zugriff auf das Terminal.
            result = await run_in_executor_with_context(
                lambda: self._tool_shell(command)
            )
        except Exception as e:
            print_status(f"Shell Error: {e}", "error")
            return
        try:
            data = json.loads(result)
            if data.get("success"):
                if data.get("output"):
                    c_print(data["output"])
                if data.get("error"):
                    print_status(data["error"], "warning")
            else:
                print_status(data.get("error", "Command failed"), "error")
                if data.get("output"):
                    c_print(data["output"])
        except json.JSONDecodeError:
            c_print(result)

    async def _tool_mcp_connect(
        self,
        server_name: str,
        command: str,
        args: list[str],
        target_agent: str | None = None,
    ) -> str:
        """Implementation: Connect MCP server."""
        try:
            mcp_config = {"name": server_name, "command": command, "args": args}

            if target_agent:
                # Update existing agent's config for next restart
                agent_config_path = Path(
                    f"{get_app().data_dir}/Agents/{target_agent}/agent.json"
                )

                if agent_config_path.exists():
                    with open(agent_config_path, encoding="utf-8") as f:
                        config_data = json.load(f)

                    if "mcp" not in config_data:
                        config_data["mcp"] = {"enabled": True, "servers": []}
                    if "servers" not in config_data["mcp"]:
                        config_data["mcp"]["servers"] = []

                    config_data["mcp"]["servers"].append(mcp_config)

                    with open(agent_config_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2)

                    if target_agent in self.agent_registry:
                        self.agent_registry[target_agent].mcp_servers.append(server_name)

                    return f"✓ MCP server '{server_name}' added to '{target_agent}' config. Restart agent to activate."
                else:
                    return f"✗ Agent '{target_agent}' config not found"

            else:
                # Create new agent with MCP
                new_agent_name = f"mcp_{server_name}"

                builder = self.isaa_tools.get_agent_builder(
                    name=new_agent_name, add_base_tools=True, with_dangerous_shell=False
                )

                self._apply_rate_limiter_to_builder(builder)

                # Enable MCP and add server config
                builder.config.mcp.enabled = True
                builder._mcp_config_data = {server_name: mcp_config}
                builder._mcp_needs_loading = True

                await self.isaa_tools.register_agent(builder)

                self.agent_registry[new_agent_name] = AgentInfo(
                    name=new_agent_name,
                    persona=f"MCP Agent: {server_name}",
                    mcp_servers=[server_name],
                )

                self._save_state()

                return (
                    f"✓ Created agent '{new_agent_name}' with MCP server '{server_name}'"
                )

        except Exception as e:
            return f"✗ Failed to connect MCP: {e}"

    async def _tool_update_agent_config(
        self, agent_name: str, config_updates: dict
    ) -> str:
        """Implementation: Update agent config."""
        try:
            agent_config_path = Path(
                f"{get_app().data_dir}/Agents/{agent_name}/agent.json"
            )

            if not agent_config_path.exists():
                return f"✗ Agent '{agent_name}' config not found"

            with open(agent_config_path, encoding="utf-8") as f:
                config_data = json.load(f)

            # Deep merge updates
            def deep_merge(base: dict, updates: dict):
                for key, value in updates.items():
                    if (
                        key in base
                        and isinstance(base[key], dict)
                        and isinstance(value, dict)
                    ):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value

            deep_merge(config_data, config_updates)

            with open(agent_config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2)

            return f"✓ Config updated for '{agent_name}'. Restart agent to apply changes."

        except Exception as e:
            return f"✗ Failed to update config: {e}"

    # =========================================================================
    # CLI INTERFACE
    # =========================================================================

    def _build_completer(self) -> tuple[dict[
        str | Any, None | dict[str, dict[str, None] | None] | dict[str, PathCompleter | None | dict[str, None]] | dict[
            str | Any, dict[Any, None] | None | dict[str, dict] | dict[str, dict[Any, None]] | Any] | dict[
            str, dict[Any, None] | None] | dict[str, dict[Any, Any] | None] | Any], VFSCompleter | dict[str, None]]:
        """Build nested completer dictionary."""
        agents = self.isaa_tools.config.get("agents-name-list", ["self"])

        # Try to get VFS files, dirs, and mounts for autocomplete
        session = None
        is_vfs = False
        model_options: dict = {}
        current_skills: dict = {}
        tool_names: dict = {}
        tool_cats: dict = {}
        checkpoint_structure: dict = {
                    "save": {a: None for a in agents},
                    "load": {a: None for a in agents},
                }
        features: dict = {_:None for _ in self.feature_manager.list_features()}
        try:
            instance_key = f"agent-instance-{self.active_agent_name}"
            if instance_key in self.isaa_tools.config:
                agent = self.isaa_tools.config[instance_key]
                session = agent.session_manager.get(self.active_session_id)
                if session and hasattr(session, "vfs"):
                    is_vfs = True

                engine = agent._get_execution_engine()
                if hasattr(engine, 'skills_manager'):
                    current_skills = {s_id: None for s_id in engine.skills_manager.skills.keys()}

                if hasattr(agent, "tool_manager"):
                    for t_obj in agent.tool_manager.get_all():
                        tool_names[t_obj.name] = None
                        cats = t_obj.category if isinstance(t_obj.category, list) else ["uncategorized"]
                        for c in cats: tool_cats[c] = None

                    if hasattr(agent, "_disabled_tools"):
                        for t_name, t_obj in agent._disabled_tools.items():
                            tool_names[t_name] = None
                            cats = t_obj.category if isinstance(t_obj.category, list) else ["uncategorized"]
                            for c in cats: tool_cats[c] = None
            model_options = {m: None for m in MODEL_MAPPING.keys()}

            # Agent-Unterstruktur für save/load
            checkpoint_structure = {}

            for action in ["save", "load"]:
                checkpoint_structure[action] = {
                    # Agent auswählen
                    **{
                        agent: {
                            # optionaler Checkpoint-Name (frei)
                            # danach Pfad
                            None: PathCompleter(only_directories=True, expanduser=True)
                        }
                        for agent in agents
                    }
                }
        except Exception as e:
            print(e)
            pass

        path_compl = PathCompleter(expanduser=True)

        d = {
            "/agent": None, "/audio": None, "/coder": None, "/job": None,
            "/mcp": None, "/session": None, "/skill": None, "/task": None,
            "/tools": None, "/vfs": None, "/feature": None, "/chain": None,
            "/set_max_iterations": None
        }

        # Die spezifischen Hilfe-Kategorien
        help_sub_commands = {
            "all": None,  # Komplette Referenz
            "min": None,  # Nur das Wichtigste
            "guide": None,  # Nutzungstipps & Prompts
            "discord": None,  # Discord Extension Guide
            "keys": None,  # F-Key Referenz
            "shortcuts": None,
            "chain": None,
            "autofix": None,
            "autodoc": None,
            "autotest": None,
            **{cmd.lstrip("/"): None for cmd in d.keys()}  # Erlaubt /help agent etc.
        }

        return {
            "/help": help_sub_commands,
            "/quit": None,
            "/exit": None,
            "/clear": None,
            "/status": None,
            "/vfs": {"init":None},
            "/tools": {
                "list": tool_cats,
                "all": None,
                "info": tool_names,
                "enable": {**tool_names, **tool_cats},
                "disable": {**tool_names, **tool_cats},
                "enable-all": None,
                "disable-all": None,
                "health": tool_names,
            },
            "/audio": {
                "on": None,
                "off": None,
                "stop": None,
                "voice": None,
                "backend": {
                    "groq": None,
                    "piper": None,
                    "elevenlabs": None,
                },
                "lang": None,
            },
            "/coder": {
                "start": PathCompleter(only_directories=True, expanduser=True),
                "stop": None,
                "task": None,
                "test": None,  # Freitext Befehl
                "accept": None,
                "reject": None,
                "info": None,
                "stream": {"on":None,"off":None},
                "diff": None,
                "files": None,
            },
            "/agent": {
                "switch": {a: None for a in agents},
                "list": None,
                "spawn": None,
                "stop": {a: None for a in agents},
                "model": {
                    "fast": model_options,
                    "complex": model_options
                },
                "checkpoint": checkpoint_structure,
                "load-all": None,
                "save-all": None,
                "stats": {a: None for a in agents},
                "delete": {a: None for a in agents},
                "config": {a: None for a in agents},
            },
            "/mcp": {
                "list": None,
                "add": None,  # /mcp add <name> <cmd> <args>
                "remove": {s: None for s in getattr(self, 'current_mcp_servers', [])},
                "reload": None,  # Re-connectet alle Server
                "info": None,  # Details zu einem Server
            },
            "/session": {
                "switch": {},
                "list": None,
                "new": None,
                "show": {"len":None},
                "working": None,
                "clear": None,
            },
            "/task": {
                "list": None,
                "view": {t: None for t in self.all_executions.keys()},
                "cancel": {t: None for t in self.all_executions.keys()},
                "clean": None,
                "status": {t: None for t in self.all_executions.keys()},
                "log": {t: None for t in self._task_views.keys()},
            },
            "/job": {
                "list": None,
                "add": None,
                "remove": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else [])},
                "pause": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else []) if j.status == "active"},
                "resume": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else []) if j.status == "paused"},
                "fire": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else [])},
                "detail": {j.job_id: None for j in (self.job_scheduler.list_jobs() if self.job_scheduler else [])},
                "autowake": {"install": None, "remove": None, "status": None},
                "dream": {"create": None,"status": None, "live": None},
            },
            "/chain": (lambda _chains: {
                "list": None,
                "show": {c.id: None for c in _chains},
                "accept": {c.id: None for c in _chains if not c.accepted},
                "delete": {c.id: None for c in _chains},
                "edit": {c.id: None for c in _chains},
                "export": {c.id: PathCompleter() for c in _chains},
                "import": PathCompleter(),
                "run": {c.id: None for c in _chains if c.accepted and c.is_valid},
            })((lambda: (lambda store: store.list_all() if store else [])(
                __import__("toolboxv2.mods.isaa.base.chain.chain_tools", fromlist=["ChainStore"])
                .ChainStore(Path(self.app.data_dir) / "chains")
                if (Path(self.app.data_dir) / "chains").exists() else None
            ))()),
            "/context": {
                "stats": None,
            },
            "/skill": {
                "list": None,
                "show": current_skills if current_skills else None,
                "edit": current_skills if current_skills else None,
                "delete": current_skills if current_skills else None,
                "boost": current_skills if current_skills else None,
                "merge": current_skills if current_skills else None,
                "import": PathCompleter(only_directories=True, expanduser=True),
                "export": {
                    "id":
                    {s_id: path_compl for s_id in ["all"] + list(current_skills.keys())} if current_skills else None},
                    "path": PathCompleter(only_directories=True, expanduser=True)
            },
            "/bind": {a: None for a in agents},
            "/set_max_iterations": None,
            "/teach": {a: None for a in agents},
            "/feature": {
                "list": None,
                "enable": features,
                "disable": features,
            },
            "/rate-limiter": {
                "status": None,
                "config": None,
            },

        }, VFSCompleter(session.vfs) if is_vfs else None

    def get_prompt_text(self) -> HTML:
        """Generate prompt text with status indicators."""
        cwd_name = Path.cwd().name
        bg_count = sum(1 for t in self.all_executions.values() if t.status == "running")
        bg_indicator = (
            f" <style fg='ansiyellow'>[{bg_count}bg]</style>" if bg_count > 0 else ""
        )

        audio_indicator = (
            " <style fg='ansired'>REC</style>" if self._audio_recording else ""
        )

        # Coder Mode Indicator
        # Cleanup: Sub-TaskViews + Executions entfernen
        if self.active_coder:
            mode_indicator = f"<style fg='ansimagenta'>[CODER:{Path(self.active_coder_path).name}]</style>"
            agent_indicator = ""
        else:
            mode_indicator = ""
            agent_indicator = f"<style fg='ansiyellow'>({self.active_agent_name})</style>"

        # Active features - compact tag line
        active_feats = [f for f in self.feature_manager.list_features() if self.feature_manager.is_enabled(f)]
        feat_indicator = ""
        if active_feats:
            tags = " ".join(f"<style fg='{PTColors.ZEN_DIM}'>{f}</style>" for f in active_feats)
            feat_indicator = f" {tags}"

        return HTML(
            f"<style fg='ansicyan'>[</style>"
            f"<style fg='ansigreen'>{cwd_name}</style>"
            f"<style fg='ansicyan'>]</style> "
            f"{agent_indicator}"
            f"{mode_indicator}"
            f"<style fg='grey'>@{self.active_session_id}</style>"
            f"{bg_indicator}{audio_indicator}{feat_indicator}"
            f"\n<style fg='ansiblue'>></style> "
        )

    def _get_bottom_toolbar(self):
        if self.zen_plus_mode or self._overlay is not None:
            return []
        off_set = 0
        if self._focused_task_id and self._focused_task_id in self.all_executions:
            off_set = list(self.all_executions.keys()).index(self._focused_task_id)
        return render_footer_toolbar(
            task_views=self._task_views,
            focused_id=self._focused_task_id,
            audio_recording=self._audio_recording,
            audio_processing=self._was_recording_is_prossesing_audio,
            overlay_open=self._overlay is not None,
            set_interval=self.set_dynamic_interval,
            off_set=off_set
        )

    def _get_keybinding_indicator(self) -> str:
        """
        Build compact right-side keybinding hints.
        Only shows high-value toggles.
        """
        items = []

        # F2 – Zen Mode
        mode = "ZEN+" if self.zen_plus_mode else "ZEN"
        items.append(
            f"<style fg='ansimagenta'>F2</style>"
        )

        # F4 – Audio
        if self._audio_recording:
            items.append("<style fg='ansired'>F4:REC</style>")
        else:
            items.append("<style fg='grey'>F4:AUDIO</style>")

        # F5 – Dashboard
        items.append("<style fg='ansicyan'>F5:INFOS</style>")

        width = shutil.get_terminal_size().columns
        return "  ".join(items).rjust(width)

    def _get_keybinding_indicator_ansi(self):
        """
        Build compact right-side keybinding hints.
        ANSI-Version (Formatted Text Tuples) für maximale Stabilität.
        """
        items = []

        # F2 – Zen Mode
        mode = "ZEN+" if self.zen_plus_mode else "ZEN"
        items.append(("fg:ansiblack  bg:ansimagenta", f"F2:{mode}"))
        items.append(("fg:ansiblack", "  "))

        # F4 – Audio
        if hasattr(self, "_audio_recording") and self._audio_recording:
            items.append(("fg:ansiblack bg:ansired", "F4:REC"))
        else:
            items.append(("fg:ansiblack bg:ansigray", "F4:AUDIO"))
        items.append(("fg:ansiblack", "  "))

        # F5 – Dashboard
        items.append(("fg:ansiblack bg:ansicyan", "F5:INFOS"))
        items.append(("fg:ansiblack", "  "))

        return items

    async def _print_status_dashboard(self):
        """Print comprehensive status dashboard."""
        if True:
            from toolboxv2.mods.isaa.extras.isaa_branding import print_status_dashboard_v2
            return await print_status_dashboard_v2(self)
        c_print()
        print_box_header(f"{CLI_NAME} v{VERSION}", "🤖")

        # Host Info
        print_box_content(f"Host ID: {self.host_id}", "info")
        print_box_content(f"Uptime: {datetime.now() - self.started_at}", "info")

        print_separator()

        # Agents
        agents = self.isaa_tools.config.get("agents-name-list", [])
        print_status(f"Agents: {len(agents)} registered", "data")

        max_name_length = max(len(name) for name in agents)
        columns = [("Name", max_name_length), ("Status", 10), ("Persona", 25), ("Tasks", 8)]
        widths = [max_name_length, 10, 25, 8]
        print_table_header(columns, widths)

        for name in agents[:10]:  # Limit display
            info = self.agent_registry.get(name, AgentInfo(name=name))
            instance_key = f"agent-instance-{name}"
            status = "Active" if instance_key in self.isaa_tools.config else "Idle"
            status_style = "green" if status == "Active" else "grey"

            bg_count = sum(
                1
                for t in self.all_executions.values()
                if t.agent_name == name and t.status == "running"
            )

            persona = info.persona[:23] + ".." if len(info.persona) > 25 else info.persona

            name_style = "cyan" if info.is_self_agent else "white"

            print_table_row(
                [name, status, persona, str(bg_count)],
                widths,
                [name_style, status_style, "grey", "yellow"],
            )

        print_separator()

        # Background Tasks - read from engine.live for real-time state
        running_tasks = [t for t in self.all_executions.values() if t.status == "running"]
        print_status(f"Background Tasks: {len(running_tasks)} running", "progress")
        if running_tasks:
            print_table_header(
                [("ID/Agent", 18), ("Progress", 25), ("Phase", 10), ("Thought/Tool", 20)],
                [18, 25, 10, 20]
            )

            for t in running_tasks:
                phase_str = "-"
                bar_str = ""
                focus_str = "-"
                try:
                    agent = await self.isaa_tools.get_agent(t.agent_name)
                    engine = agent._get_execution_engine()
                    live = engine.live

                    # Progress bar from live state
                    it, mx = live.iteration, live.max_iterations
                    if mx > 0:
                        filled = int(20 * it / mx)
                        bar_str = f"{'━' * filled}{'─' * (20 - filled)} {it}/{mx}"
                    else:
                        bar_str = f"{'─' * 20} {it}/?"

                    phase_str = live.phase.value[:10]

                    # Show thought or tool (whichever is most recent)
                    if live.tool.name:
                        focus_str = f"◇ {live.tool.name[:18]}"
                    elif live.thought:
                        focus_str = f"◎ {live.thought[:18]}"
                    elif live.status_msg:
                        focus_str = live.status_msg[:20]

                except Exception:
                    elapsed = (__import__("time").time() - t.started_at)
                    bar_str = f"{'─' * 20} {elapsed:.0f}s"

                print_table_row(
                    [
                        f"{t.task_id[:8]}.. ({t.agent_name})",
                        bar_str,
                        phase_str,
                        focus_str,
                    ],
                    [18, 25, 10, 20],
                    ["cyan", "green", "white", "grey"]
                )

        # Jobs summary
        if self.job_scheduler and self.job_scheduler.total_count > 0:
            aw_indicator = ""
            if self.job_scheduler.has_persistent_jobs():
                try:
                    from toolboxv2.mods.isaa.extras.jobs.os_scheduler import autowake_status
                    _aw_s = autowake_status()
                    aw_indicator = (
                        "  [OS✓]" if "Installed" in _aw_s or "Registered" in _aw_s
                        else "  [OS✗ — run /job autowake install]"
                    )
                except Exception:
                    aw_indicator = ""
            print_status(
                f"Scheduled Jobs: {self.job_scheduler.active_count} active"
                f" / {self.job_scheduler.total_count} total{aw_indicator}",
                "data"
            )

        c_print()


    # ------------------------------------------------------------
    #  Dispatcher‑Funktion
    # ------------------------------------------------------------
    def show_help(self, command: str = "") -> None:
        """
        Gibt den Hilfetext für einen bestimmten Befehl aus.
        Ohne Argument wird eine kurze Übersicht aller verfügbaren Befehle gezeigt.
        """

        # Prüfen, ob der Befehl exakt vorhanden ist


        # Falls nichts gefunden
        print(f"⚠️  Keine Hilfedaten für '{command}' gefunden. "
              "Verfügbare Kategorien: " + ", ".join(sorted(_help_text.keys())))


    def _print_help(self, args):
        """Haupthandler für Hilfe-Ausgaben."""
        sub = args[0].lower() if args else "min"

        if sub == "all":
            self._print_help_all(args)
        elif sub == "guide":
            self._print_help_guide()
        elif sub == "discord":
            self._print_help_discord()
        elif sub == "min":
            self._print_help_min()

        elif sub in ("keys", "shortcuts"):
            self._print_help_keys()

        elif sub in _help_text:
            print_box_header(f"\n=== Hilfe für {sub} ===\n","")
            for line in _help_text[sub].strip().splitlines():
                print_box_content(line, "")
            print_box_footer()
            return

        # Wenn nicht exakt, nach Teil‑Übereinstimmung suchen (z. B. "/agent list")
        for key, txt in _help_text.items():
            if sub.startswith(key):
                print_box_header(f"\n=== Hilfe für {key} (Teil‑Befehl: {sub}) ===\n")
                for line in txt.strip().splitlines():
                    print_box_content(line, "")
                print_box_footer()
                return
        else:
            # Suche nach spezifischer Hilfe für einen Befehl (z.B. /help vfs)
            print_status(f"Detail-Hilfe für {sub} folgt (Nutze /help all für alles)", "info")

            if not args:
                return
            self._print_help_min()

    def _print_help_keys(self):
        """F-Key und Shortcut Referenz."""
        print_box_header("Keyboard Shortcuts", "⌨")
        print_box_content("F2   ZEN/ZEN+ mode toggle", "")
        print_box_content("F4   Start/stop audio recording", "")
        print_box_content("F5   Status dashboard", "")
        print_box_content("F6   Minimize/maximize focused agent", "")
        print_box_content("F7   Cycle focus → next running agent", "")
        print_box_content("F8   Cancel focused agent task", "")
        print_separator()
        print_box_content("TAB     Command autocompletion", "")
        print_box_content("Ctrl+C  Safe stop (continue/quit)", "")
        print_box_content("!cmd    Execute shell command", "")
        print_box_footer()

    def _print_help_min(self):
        """Kompakte Hilfe für den schnellen Überblick."""
        print_box_header("ISAA Quick Help", "🚀")
        print_box_content("Basics:", "bold")
        print_box_content("/status          - Dashboard (F5)", "")
        print_box_content("/agent switch    - Agent wechseln", "")
        print_box_content("/tools list      - Tools verwalten", "")
        print_box_content("/chain list      - Gespeicherte Chains", "")
        print_box_content("/chain run <id>  - Chain ausführen", "")

        print_box_content("F4               - Voice Recording Start/Stop", "")
        print_box_content("F2               - toggel the Live View Overlay.", "")
        print_box_content("F6/F7/F8         - Minimize/Focus/Cancel Agent Tasks", "")
        print_box_content("! <cmd>          - Shell Befehl direkt ausführen", "")
        print_separator()
        print_box_content("Erweiterte Hilfe:", "info")
        print_box_content("/help all        - Alle Befehle (Referenz)", "")
        print_box_content("/help chain      - Chain DSL Referenz", "")  # ← NEU
        print_box_content("/help autofix    - AutoFix CI Pipeline", "")  # ← NEU
        print_box_content("/help guide      - Beispiele & Start-Prompts", "")
        print_box_content("/help discord    - Discord Integration Guide", "")
        print_box_footer()

    def _print_help_guide(self):
        """Guide mit konkreten Beispielen und Prompts."""
        print_box_header("ISAA Usage Guide", "💡")
        print_status("Beispiel-Szenarien:", "info")

        print_box_content("1. Coding Projekt starten:", "bold")
        print_box_content("   > /vfs mount ./my_project /src", "")
        print_box_content("   > /coder start /src", "")
        print_box_content("   Prompt: 'Analysiere die main.py und füge Logging hinzu.'", "")

        print_box_content("2. Web Recherche:", "bold")
        print_box_content("   > /feature enable full_web_auto", "")
        print_box_content("   Prompt: 'Suche nach den neuesten KI-News und fasse sie zusammen.'", "")

        print_box_content("3. Langzeit-Tasks (Jobs):", "bold")
        print_box_content("   > /job add (Interaktiver Modus)", "")
        print_box_content("   Prompt: 'Prüfe jede Stunde ob der Server online ist.'", "")

        print_separator()
        print_status("Tipp: Nutze #audio am Ende einer Nachricht für Sprachausgabe!", "success")
        print_box_footer()

    def _print_help_discord(self):
        """Spezifische Hilfe für die Discord Extension."""
        print_box_header("Discord Integration Guide", "💬")
        print_box_content("Setup:", "bold")
        print_box_content("/discord connect <token>    - Bot starten", "")
        print_box_content("/discord status             - Connection prüfen", "")
        print_separator()
        print_box_content("Voice & Interaction:", "bold")
        print_box_content("/discord voice channels     - Channels auflisten", "")
        print_box_content("/discord voice join <id>    - In Voice Channel gehen", "")
        print_box_content("/discord send <addr> <msg>  - Nachricht an User/Channel", "")
        print_box_content("Hinweis: Der Bot reagiert in Discord auf @Mentions", "info")
        print_box_footer()


    def _print_help_all(self, args):
        """Print help information."""
        print_box_header("ISAA Host Commands", "❓")

        print_status("Navigation", "info")
        print_box_content("/help                                        - Show this help", "")
        print_box_content("/status                                      - Show status dashboard (or F5)", "")
        print_box_content("/clear                                       - Clear screen", "")
        print_box_content("/set_max_iterations", "")
        print_box_content("/quit, /exit - Exit CLI", "")
        print_box_content("F2                                           - Switch between ZEN and ZEN+ interaction mode",
                          "")
        print_box_content("F4                                           - Start / stop audio recording pipeline", "")
        print_box_content("F5                                           - Display status dashboard (VFS, Skills, MCP, Session)", "")
        print_box_content("F6                                           - Minimize/maximize focused agent stream", "")
        print_box_content("F7                                           - Cycle focus to next running agent task", "")
        print_box_content("F8                                           - Cancel focused agent task", "")
        print_box_content("TAB                                          - Activate or confirm command autocompletion",
                          "")
        print_box_content("Ctrl+C                                       - Safe stop agent (continue/fresh/quit)", "")
        print_separator()

        print_status("Agent Management", "info")
        print_box_content("/agent list                                  - List all agents", "")
        print_box_content("/agent switch <name>                         - Switch active agent", "")
        print_box_content("/agent spawn <name> <persona>                - Create new agent", "")
        print_box_content("/agent stop <name>                           - Stop agent tasks", "")
        print_box_content("/agent model <fast|complex> <name>           - Change LLM model on the fly", "")
        print_box_content("/agent checkpoint <save|load> [name]         - Manage state persistence", "")
        print_box_content("/agent checkpoint help                       - list information's about addition args like path", "")
        print_box_content("/agent load-all                              - Initialize all agents from disk", "")
        print_box_content("/agent save-all                              - Save checkpoints for all active agents", "")
        print_box_content("/agent stats [name]                          - Show token usage and cost metrics", "")
        print_box_content("/agent delete <name>                         - Remove agent and its data", "")
        print_box_content("/agent config <name>                         - View raw JSON configuration", "")

        print_separator()

        print_status("Session Management", "info")
        print_box_content("/session list                                 - List sessions", "")
        print_box_content("/session switch <id>                          - Switch session", "")
        print_box_content("/session new                                  - Create new session", "")
        print_box_content("/session show [n]                             - Show last n messages (default 10)", "")
        print_box_content("/session clear                                - Clear current session history", "")
        print_box_content("/session working                              - Show working history", "")

        print_separator()

        print_status("MCP Management (Live)", "info")
        print_box_content("/mcp list                                    - Zeige aktive MCP Verbindungen", "")
        print_box_content("/mcp add <n> <cmd> [args]                    - Server hinzufügen & Tools laden", "")
        print_box_content("/mcp remove <name>                           - Server trennen & Tools löschen", "")
        print_box_content("/mcp reload                                  - Alle MCP Tools neu indizieren", "")

        print_separator()

        print_status("Task Management", "info")
        print_box_content("/task                                        - Show all background tasks", "")
        print_box_content("/task view [id]                              - Live view of task (auto-selects if 1)", "")
        print_box_content("/task cancel <id>                            - Cancel a running task", "")
        print_box_content("/task clean                                  - Remove finished tasks", "")
        print_box_content("/task log                                    - Vew tasks internal execution hisotry", "")
        print_box_content("F6 during execution                          - Move agent to background", "")

        print_separator()

        print_status("Job Scheduler", "info")
        print_box_content("/job list                                    - List all scheduled jobs", "")
        print_box_content("/job add                                     - Add a new job (interactive)", "")
        print_box_content("/job remove <id>                             - Remove a job", "")
        print_box_content("/job pause <id>                              - Pause a job", "")
        print_box_content("/job resume <id>                             - Resume a paused job", "")
        print_box_content("/job fire <id>                               - Manually fire a job now", "")
        print_box_content("/job detail <id>                             - Show job details", "")
        print_box_content("/job autowake <cmd>                          - Manage OS auto-wake (install/remove/status)", "")
        print_status("Dreamer Job", "info")
        print_box_content("/job dream create [agent]                    - Create nightly dream job (default: self, 03:00)", "")
        print_box_content("/job dream status                            - Show all configured dream jobs", "")
        print_box_content("/job dream live                              - Run dream process now with visualization", "")

        print_separator()

        print_status("Advanced", "info")
        print_box_content("/bind <agent_a> <agent_b> - Bind agents", "")
        print_box_content("/teach <agent> <skill_name> - Teach skill", "")
        print_box_content("/context stats - Show context stats", "")
        print_separator()
        print_status("History Management", "info")
        print_box_content("/history show [n]                            - Show last n messages (default 10)", "")
        print_box_content("/history clear                               - Clear current session history", "")
        print_separator()
        print_status("VFS Management", "info")
        print_box_content("/vfs                                         - Show VFS tree", "")
        print_box_content("/vfs <path>                                  - Show file content or dir listing", "")
        print_box_content("/vfs mount <path> [vfs_path]                 - Mount local folder", "")
        print_box_content("/vfs unmount <vfs_path>                      - Unmount folder", "")
        print_box_content("/vfs sync [path]                             - Sync file/dir to disk", "")
        print_box_content("/vfs save <vfs_path> <local>                 - Save file/dir to local path", "")
        print_box_content("/vfs refresh <mount>                         - Re-scan mount for changes", "")
        print_box_content("/vfs pull <path>                             - Reload file/dir from disk", "")
        print_box_content("/vfs mounts                                  - List active mounts", "")
        print_box_content("/vfs dirty                                   - Show modified files", "")
        print_box_content("/vfs rm/remove                               - Remove Folder or File", "")
        print_separator()
        print_status("System Files (Read-Only)", "info")
        print_box_content("/vfs sys-add <local> [path]                  - Add file as read-only system file", "")
        print_box_content("/vfs sys-remove <vfs_path>                   - Remove a system file", "")
        print_box_content("/vfs sys-refresh <vfs_path>                  - Reload system file from disk", "")
        print_box_content("/vfs sys-list                                - List all system files", "")
        print_separator()
        print_status("Mount Options", "info")
        print_box_content("  --readonly                                 - No write operations", "")
        print_box_content("  --no-sync                                  - Manual sync only", "")
        print_separator()
        print_status("Skill Management", "info")
        print_box_content("/skill list                                  - List skills of active agent", "")
        print_box_content("/skill list --inactive                       - List inactive skills ", "")
        print_box_content("/skill show <id>                             - Show details/instruction", "")
        print_box_content("/skill edit <id>                             - Edit skill instruction", "")
        print_box_content("/skill delete <id>                           - Delete a skill", "")
        print_box_content("/skill merge <keep_id> <remove_id>", "")
        print_box_content("/skill boost <skill_id> 0.3                  - Delete a skill", "")
        print_box_content("/skill import <path>                         - import skills from directory/skill file", "")
        print_box_content("/skill export <id> <path>                    - id=all Extprt skill or all skills", "")
        print_separator()

        print_status("Tool Management", "info")
        print_box_content("/tools list [cat]        - List all tools, optional filter by category", "")
        print_box_content("/tools all               - Compact table of every tool (active + disabled)", "")
        print_box_content("/tools info <name>       - Detailed info incl. health status for one tool", "")
        print_box_content("/tools enable <name/cat> - Enable a tool or entire category", "")
        print_box_content("/tools disable <name/cat>- Disable a tool or entire category", "")
        print_box_content("/tools enable-all        - Re-enable all disabled tools", "")
        print_box_content("/tools disable-all       - Disable all non-system tools", "")
        print_box_content("/tools health            - Run health-check on ALL tools (summary)", "")
        print_box_content("/tools health <name>     - Run health-check on a single tool", "")

        print_separator()
        print_status("Chain Management", "info")
        print_box_content("/chain list                                  - Alle Chains auflisten", "")
        print_box_content("/chain show <id|name>                        - DSL + Metadaten anzeigen", "")
        print_box_content("/chain accept <id|name>                      - Chain einmalig akzeptieren", "")
        print_box_content("/chain run <id|name> [input]                 - Chain ausführen", "")
        print_box_content("/chain edit <id|name>                        - DSL interaktiv bearbeiten", "")
        print_box_content("/chain delete <id|name>                      - Chain löschen", "")
        print_box_content("/chain export <id> <file.json>               - Als JSON exportieren", "")
        print_box_content("/chain import <file.json>                    - Aus JSON importieren", "")
        print_separator()
        print_box_content("Agent-Tools (via LLM nutzbar):", "bold")
        print_box_content("  create_validate_chain  name dsl [desc] [tags]", "")
        print_box_content("  run_chain  name_or_id [input] [accept=true]", "")
        print_box_content("  list_auto_get_fitting  [task_description]", "")
        print_separator()

        print_status("AutoFix CI Pipeline", "info")
        print_box_content("/feature enable autofix                      - AutoFix + Tools aktivieren", "")
        print_box_content("/chain accept autofix_test_fixer             - Einmalig freigeben", "")
        print_box_content("/chain run autofix_test_fixer [/pfad]        - Pipeline starten", "")
        print_box_content("  Flow: tb -x → analyse → 2x fix parallel → best pick → re-test → PR", "")
        print_separator()
        print_status("AutoTest — Test Generator", "info")
        print_box_content("/feature enable autotest                     - Tools + Chains aktivieren", "")
        print_box_content("/chain run autotest_logic   <optional_class_Name[::file]>   - Logic Tests (bestehender Code)", "")
        print_box_content("/chain run autotest_tdd     <optional_class_Name[::file]>   - TDD Future Tests (Vertrag)", "")
        print_box_content("/chain run autotest_coverage <file.py>       - Full-File Coverage", "")
        print_box_content("  Analyse: Datenfluss, Side-Effects, Edge-Cases → unittest → run → fix", "")

        print_separator()
        print_status("Additional Features", "info")
        print_box_content("/feature list                                - List all features", "")
        print_box_content("/feature disable <feature>                   - Disable a feature", "")
        print_box_content("/feature enable <feature>                    - Enable a feature", "")
        print_box_content("/feature enable desktop                      - Enable Desktop Automation", "")
        print_box_content("/feature enable web <headless>               - Enable Desktop Web Automation", "")
        print_separator()

        print_separator()
        print_status("AutoDoc — Docs Generator", "info")
        print_box_content("/feature enable autodoc                      - Tools + Chains aktivieren", "")
        print_box_content("/chain run autodoc_unguided                  - Batch: ganzes Repo scannen", "")
        print_box_content("/chain run autodoc_guided <Name[::file]>     - Single: gezielt dokumentieren", "")
        print_box_content("  Regel: nur getesteter Code. Format: Part1=How to Use, Part2=Internals.", "")

        print_status("Audio Commands", "info")
        print_box_content("/audio on                    - All responses spoken", "")
        print_box_content("/audio off                   - Disable verbose audio", "")
        print_box_content("/audio voice <v>             - Set voice", "")
        print_box_content("/audio backend <b>           - groq_tts / piper / elevenlabs / index_tts", "")
        print_box_content("/audio lang <l>              - de / en / ...", "")
        print_box_content("/audio device                - Interactive device picker", "")
        print_box_content("/audio device <idx>          - Set by index", "")
        print_box_content("/audio device default        - Reset to system default", "")
        print_box_content("/audio devices               - List all output devices", "")
        print_box_content("/audio stop                  - Stop current playback", "")
        print_box_content("/audio restart               - Rebuild player with current settings", "")
        print_box_content("", "")
        print_box_content("Tip: append  #audio  to any message for one-time spoken response", "info")
        print_separator()

        print_status("Shortcuts", "info")
        print_box_content("F2 - ZEN/ZEN+ mode toggle", "")
        print_box_content("F4 - Toggle audio recording", "")
        print_box_content("F5 - Show status dashboard", "")
        print_box_content("F6 - Minimize/maximize agent stream", "")
        print_box_content("F7 - Cycle focus between running agents", "")
        print_box_content("F8 - Cancel focused agent task", "")
        print_box_content("!<cmd> - Execute shell command", "")

        print_box_footer()

    async def _handle_command(self, cmd_str: str):
        """Handle slash commands."""
        parts = cmd_str.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/quit", "/exit", "/q", "/x", "/e"):
            running_bg = [t for t in self.all_executions.values() if t.status == "running"]
            active_jobs = self.job_scheduler.active_count if self.job_scheduler else 0

            warnings = []
            if running_bg:
                warnings.append(f"{len(running_bg)} background task(s)")
            if active_jobs:
                warnings.append(f"{active_jobs} scheduled job(s)")

            if warnings:
                warn_text = " + ".join(warnings)
                has_persistent = (
                    self.job_scheduler is not None
                    and self.job_scheduler.has_persistent_jobs()
                )
                bg_label = (
                    "quit + keep jobs via OS scheduler"
                    if has_persistent else "move tasks to background"
                )

                c_print(HTML(
                    f"<style fg='#ef4444'>⚠ Still running: {esc(warn_text)}</style>\n"
                    f"<style fg='{PTColors.ZEN_DIM}'>"
                    f"  [q] quit anyway (cancel all)    "
                    f"  [b] {esc(bg_label)}    "
                    f"  [Enter] abort quit</style>"
                ))

                try:
                    with patch_stdout():
                        choice = await self.prompt_session.prompt_async(
                            HTML(f"<style fg='#ef4444'>▸ </style>")
                        )
                    choice = choice.strip().lower()

                    if choice == "q":
                        for t in self.all_executions.values():
                            if t and t.status == "running":
                                try:
                                    t.async_task.cancel()
                                except:
                                    pass
                        raise EOFError

                    elif choice == "b":
                        if has_persistent:
                            try:
                                from toolboxv2.mods.isaa.extras.jobs.os_scheduler import (
                                    install_autowake,
                                )
                                aw_result = install_autowake(self.jobs_file)
                                c_print(HTML(
                                    f"<style fg='{PTColors.ZEN_DIM}'>"
                                    f"  OS scheduler: {esc(aw_result)}</style>"
                                ))
                            except Exception as _aw_err:
                                c_print(HTML(
                                    f"<style fg='#f59e0b'>"
                                    f"  ⚠ OS scheduler unavailable: {esc(str(_aw_err))}</style>"
                                ))

                        if running_bg:
                            c_print(HTML(
                                f"<style fg='{PTColors.ZEN_DIM}'>"
                                f"  {len(running_bg)} task(s) cancelled.</style>"
                            ))
                            for t in self.all_executions.values():
                                if t.status == "running":
                                    t.async_task.cancel()

                        c_print(HTML(
                            f"<style fg='{PTColors.ZEN_DIM}'>"
                            f"  Scheduled jobs continue via OS scheduler.\n"
                            f"  Reconnect to CLI to see results and manage jobs.</style>"
                        ))
                        raise EOFError

                    else:
                        c_print(HTML(
                            f"<style fg='{PTColors.ZEN_DIM}'>  Quit aborted.</style>"
                        ))
                        return

                except (KeyboardInterrupt, EOFError):
                    raise EOFError
            else:
                raise EOFError

        elif cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            await self._print_status_dashboard()

        elif cmd == "/help":
            self._print_help(args)

        elif cmd == "/set_max_iterations":
            if len(args) < 1:
                print_status(f"is {self.max_iteration}", "success")
            else:
                try:
                    self.max_iteration = int(args[0])
                except ValueError:
                    print_status(f"must be a number u passed {args[0]}", "error")

        elif cmd == "/status":
            await self._print_status_dashboard()

        elif cmd == "/agent":
            await self._cmd_agent(args)

        elif cmd == "/session":
            await self._cmd_session(args)

        elif cmd == "/task":
            await self._cmd_task(args)

        elif cmd == "/mcp":
            await self._cmd_mcp(args)

        elif cmd == "/tools":
            await self._cmd_tools(args)

        elif cmd == "/vfs":
            await self._cmd_vfs(args)

        elif cmd == "/skill":
            await self._cmd_skill(args)

        elif cmd == "/coder":
            await self._cmd_coder(args)

        elif cmd == "/audio":
            await self._handle_audio_command(args)

        elif cmd == "/job":
            await self._cmd_job(args)

        elif cmd == "/context":
            await self._cmd_context(args)
        elif cmd == "/feature":
            await self._cmd_feature(args)

        elif cmd == "/chain":
            await self._cmd_chain(args)

        elif cmd == "/bind":
            if len(args) >= 2:
                result = await self._tool_bind_agents(args[0], args[1])
                print_status(result, "success" if "✓" in result else "error")
            else:
                print_status("Usage: /bind <agent_a> <agent_b>", "warning")

        elif cmd == "/teach":
            if len(args) >= 2:
                agent = args[0]
                skill_name = args[1]
                print_status("Enter skill instruction (end with empty line):", "info")
                lines: list[str] = []
                if self.prompt_session is not None:
                    while True:
                        line = await self.prompt_session.prompt_async(
                            HTML("<style fg='grey'>... </style>")
                        )
                        if not line.strip():
                            break
                        lines.append(line)
                instruction = "\n".join(lines)
                triggers = input("Enter triggers (comma-separated): ").split(",")
                triggers = [t.strip() for t in triggers if t.strip()]
                result = await self._tool_teach_skill(
                    agent, skill_name, instruction, triggers
                )
                print_status(result, "success" if "✓" in result else "error")
            else:
                print_status("Usage: /teach <agent> <skill_name>", "warning")

        elif cmd == "/rate-limiter":
            if args and args[0] == "status":
                print_code_block(json.dumps(self._rate_limiter_config, indent=2), "json")
            else:
                print_status("Usage: /rate-limiter status", "warning")

        else:
            print_status(f"Unknown command: {cmd}. Type /help for help.", "error")


    async def _cmd_agent(self, args: list[str]):
        """Handle /agent commands."""
        if not args:
            print_status("Usage: /agent <list|switch|spawn|stop|model|...> [args]", "warning")
            return

        action = args[0]

        if action == "list":
            result = await self._tool_list_agents()
            c_print(result)

        elif action == "switch":
            if len(args) < 2:
                print_status("Usage: /agent switch <name>", "warning")
                return
            target = args[1]
            if target in self.isaa_tools.config.get("agents-name-list", []):
                self.active_agent_name = target
                self.active_session_id = "default"
                self._save_state()
                print_status(f"Switched to agent: {target}", "success")
                if self.verbose_audio:
                    await self._ensure_audio_setup(target)
            else:
                print_status(f"Agent '{target}' not found", "error")

        elif action == "model":
            if len(args) < 3:
                print_status("Usage: /agent model <fast|complex> <model_name>", "warning")
                return

            target_type = args[1].lower()  # "fast" oder "complex"
            model_alias = args[2]

            if model_alias not in MODEL_MAPPING:
                print_status(f"Model '{model_alias}' not in registry. Using raw name.", "info")
                full_model_name = model_alias
            else:
                full_model_name = MODEL_MAPPING[model_alias]

            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                if target_type == "fast":
                    agent.amd.fast_llm_model = full_model_name
                elif target_type == "complex":
                    agent.amd.complex_llm_model = full_model_name
                else:
                    print_status("Type must be 'fast' or 'complex'", "error")
                    return

                print_status(f"Updated {target_type} model for {self.active_agent_name} to: {full_model_name}",
                             "success")

                # Optional: Config persistent speichern
                await self._tool_update_agent_config(self.active_agent_name, {
                    f"{target_type}_llm_model": full_model_name
                })

            except Exception as e:
                print_status(f"Failed to update model: {e}", "error")

        elif action == "spawn":
            if len(args) < 2:
                print_status("Usage: /agent spawn <name> [persona]", "warning")
                return
            name = args[1]
            persona = " ".join(args[2:]) if len(args) > 2 else "general assistant"
            result = await self._tool_spawn_agent(name, persona)
            print_status(result, "success" if "✓" in result else "error")

        elif action == "stop":
            if len(args) < 2:
                print_status("Usage: /agent stop <name>", "warning")
                return
            result = await self._tool_stop_agent(args[1])
            print_status(result, "success" if "✓" in result else "error")

        elif action == "checkpoint":
            if len(args) < 2:
                w = ( "info" if (args[1] if len(args) > 1 else "warning") == "help" else "warning")
                print_status("Usage: /agent checkpoint <save|load> [name] <path> <tools[t/f]>", w)
                return
            sub = args[1].lower()
            target = args[2] if len(args) > 2 else self.active_agent_name
            path = args[3] if len(args) > 3 else None
            with_tools = args[4] if len(args) > 4 else None
            if path is None:
                try:
                    agent = await self.isaa_tools.get_agent(target)
                    if sub == "save":
                        path = await agent.save()
                        print_status(f"Checkpoint saved: {path}", "success")
                    else:
                        await agent.restore()
                        print_status(f"State restored for {target}", "success")
                except Exception as e:
                    print_status(f"Checkpoint error: {e}", "error")

                return
            data = None
            if sub == "save":
                sucsess, data = await self.isaa_tools.save_agent(
                    agent_name=target,
                    path=path,
                    include_checkpoint=True,
                    include_tools=with_tools,
                    notes="cli-export"
                )
                if not sucsess:
                    print_status(f"Agent saved: {target} Failed {data}", "error")
                    return

            elif path:
                warnings: list[str]
                _, data, warnings = await self.isaa_tools.load_agent(
                    path=path, override_name=target, load_tools=with_tools, register=True
                )
                if _ is None:
                    print_status(f"Agent loading: {target}", "error")

                    if warnings:
                        print_box_header("Load Warnings", icon="⚠")
                        for w in warnings:
                            print_box_content(w, style="warning")
                        print_box_footer()

                    return

            agent_version = data.agent_version if hasattr(data, "agent_version") else None
            has_checkpoint = data.has_checkpoint if hasattr(data, "has_checkpoint") else None
            has_tools = data.has_tools if hasattr(data, "has_tools") else None
            tool_count = data.tool_count if hasattr(data, "tool_count") else None
            serializable_tools = data.serializable_tools if hasattr(data, "serializable_tools") else None
            non_serializable_tools = data.non_serializable_tools if hasattr(data, "non_serializable_tools") else None
            bindings = data.bindings if hasattr(data, "bindings") else None
            # ─────────────────────────────────────────────
            # Header
            # ─────────────────────────────────────────────
            print_box_header("Agent Overview", icon="🤖")

            # ─────────────────────────────────────────────
            # Basisinformationen
            # ─────────────────────────────────────────────
            print_box_content(f"Version: {agent_version or 'N/A'}", style="info")

            if has_checkpoint is True:
                print_box_content("Checkpoint verfügbar", style="success")
            elif has_checkpoint is False:
                print_box_content("Kein Checkpoint vorhanden", style="warning")

            if has_tools is True:
                print_box_content(f"Tools aktiviert ({tool_count or 0})", style="success")
            elif has_tools is False:
                print_box_content("Keine Tools registriert", style="warning")

            print_separator()

            # ─────────────────────────────────────────────
            # Tool-Details (Tabelle)
            # ─────────────────────────────────────────────
            print_table_header(
                columns=[
                    ("Kategorie", None),
                    ("Anzahl", None)
                ],
                widths=[30, 10]
            )

            print_table_row(
                ["Serializable Tools", serializable_tools or 0],
                widths=[30, 10],
                styles=["cyan", "green"]
            )

            print_table_row(
                ["Non-Serializable Tools", non_serializable_tools or 0],
                widths=[30, 10],
                styles=["cyan", "yellow"]
            )

            print_separator()

            # ─────────────────────────────────────────────
            # Bindings
            # ─────────────────────────────────────────────
            if bindings:
                print_box_content("Bindings registriert:", style="info")
                print_code_block(
                    code=str(bindings),
                    language="json",
                    show_line_numbers=False
                )
            else:
                print_box_content("Keine Bindings vorhanden", style="warning")

            # ─────────────────────────────────────────────
            # Footer
            # ─────────────────────────────────────────────
            print_box_footer()
        elif action == "load-all":
            print_status("Scanning agent directory...", "progress")
            agent_dir = Path(self.app.data_dir) / "Agents"
            loaded = 0
            if agent_dir.exists():
                for d in agent_dir.iterdir():
                    if d.is_dir() and (d / "agent.json").exists():
                        try:
                            await self.isaa_tools.get_agent(d.name)
                            loaded += 1
                        except:
                            pass
            print_status(f"Loaded {loaded} agents from disk", "success")

        elif action == "save-all":
            print_status("Saving all active agent states...", "progress")
            for name in self.isaa_tools.config.get("agents-name-list", []):
                instance_key = f"agent-instance-{name}"
                if instance_key in self.isaa_tools.config:
                    agent = self.isaa_tools.config[instance_key]
                    await agent.save()
            print_status("All agents checkpointed", "success")

        elif action == "stats":
            target = args[1] if len(args) > 1 else self.active_agent_name
            try:
                agent = await self.isaa_tools.get_agent(target)
                stats = agent.get_stats()
                print_box_header(f"Stats: {target}", "📊")
                # Modells
                print_table_row(["Fast Model", agent.amd.fast_llm_model], [20, 15], ["white", "blue"])
                print_table_row(["Complex Model", agent.amd.complex_llm_model], [20, 15], ["white", "blue"])
                print_table_row(["Input Tokens", f"{stats['total_tokens_in']:,}"], [20, 15], ["white", "cyan"])
                print_table_row(["Output Tokens", f"{stats['total_tokens_out']:,}"], [20, 15], ["white", "cyan"])
                print_table_row(["Total Cost", f"${stats['total_cost']:.4f}"], [20, 15], ["white", "green"])
                print_table_row(["LLM Calls", str(stats['total_llm_calls'])], [20, 15], ["white", "yellow"])
                # Session data {
                #             'version': 2,
                #             'agent_name': self.agent_name,
                #             'total_sessions': len(self.sessions),
                #             'active_sessions': active_count,
                #             'docker_enabled_sessions': docker_count,
                #             'running_containers': running_containers,
                #             'total_sessions_created': self._total_sessions_created,
                #             'total_history_messages': total_history,
                #             'memory_loaded': self._memory_instance is not None,
                #             'default_lsp_enabled': self.enable_lsp,
                #             'default_docker_enabled': self.enable_docker,
                #             'session_ids': list(self.sessions.keys())
                #         }
                print_table_row(["Total Sessions", str(stats['sessions']['total_sessions'])], [20, 15], ["white", "blue"])
                print_table_row(["Active Sessions", str(stats['sessions']['active_sessions'])], [20, 15], ["white", "blue"])
                print_table_row(["Running Containers", str(stats['sessions']['running_containers'])], [20, 15], ["white", "blue"])
                print_table_row(["Total Sessions", str(stats['sessions']['total_sessions_created'])], [20, 15], ["white", "blue"])
                print_table_row(["Total History", str(stats['sessions']['total_history_messages'])], [20, 15], ["white", "blue"])
                print_table_row(["Memory Loaded", str(stats['sessions']['memory_loaded'])], [20, 15], ["white", "blue"])
                print_table_row(["Default LSP", str(stats['sessions']['default_lsp_enabled'])], [20, 15], ["white", "blue"])
                print_table_row(["Default Docker", str(stats['sessions']['default_docker_enabled'])], [20, 15], ["white", "blue"])
                # Tools section 'total_tools': len(self._registry),
                #             'by_source': {
                #                 source: len(names)
                #                 for source, names in self._source_index.items()
                #             },
                #             'categories': list(self._category_index.keys()),
                #             'total_calls'
                print_table_row(["Total Tools", str(stats['tools']['total_tools'])], [20, 15], ["white", "blue"])
                print_table_row(["Total Calls", str(stats['tools']['total_calls'])], [20, 15], ["white", "blue"])
                # Binding data {
                #             'agent_name': self.agent_name,
                #             'total_bindings': len(self.bindings),
                #             'public_bindings': sum(1 for b in self.bindings.values() if b.mode == 'public'),
                #             'private_bindings': sum(1 for b in self.bindings.values() if b.mode == 'private'),
                #             'total_messages_sent': total_sent,
                #             'total_messages_received': total_received,
                #             'partners': list(self.bindings.keys())
                #         }
                print_table_row(["Total Bindings", str(stats['bindings']['total_bindings'])], [20, 15], ["white", "blue"])
                print_table_row(["Public Bindings", str(stats['bindings']['public_bindings'])], [20, 15], ["white", "blue"])
                print_table_row(["Private Bindings", str(stats['bindings']['private_bindings'])], [20, 15], ["white", "blue"])
                print_table_row(["Total I/O Messages", f"{str(stats['bindings']['total_messages_received'])}/{str(stats['bindings']['total_messages_sent'])}"], [20, 15], ["white", "blue"])
                print_box_footer()
                # tools categories
                print_code_block(json.dumps({"Tools Categories":stats['tools']['categories']}, indent=2), "json")
            except Exception as e:
                print_status(f"Could not get stats: {e}", "error")

        elif action == "delete":
            if len(args) < 2:
                print_status("Usage: /agent delete <name>", "warning")
                return
            target = args[1]
            confirm = input(f"Really delete agent '{target}' and all its data? (y/N): ")
            if confirm.lower() == 'y':
                # Registry cleanup
                self.agent_registry.pop(target, None)
                # Disk cleanup
                import shutil
                agent_path = Path(self.app.data_dir) / "Agents" / target
                if agent_path.exists(): shutil.rmtree(agent_path)
                print_status(f"Agent '{target}' deleted", "success")

        elif action == "config":
            target = args[1] if len(args) > 1 else self.active_agent_name
            agent_path = Path(self.app.data_dir) / "Agents" / target / "agent.json"
            if agent_path.exists():
                with open(agent_path, 'r', encoding='utf-8') as f:
                    data = f.read()

                data_dict = json.loads(data)

                print_code_block(json_to_md(data_dict), "md")
            else:
                print_status("Config not found on disk", "error")

        else:
            print_status(f"Unknown agent action: {action}", "error")

    async def _cmd_session(self, args: list[str]):
        """Handle /session commands."""
        if not args:
            print_status("Usage: /session <list|switch|new|clear|show|working>", "warning")
            return

        def show_history(_history):
            if not _history:
                print_status("History is empty.", "info")
                return

            print_box_header(f"History: {self.active_agent_name}@{self.active_session_id} (Last {len(_history)})", "💬")

            for msg in _history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                try:
                    # Format based on role
                    if role == "user":
                        c_print(HTML(f"  <style font-weight='bold' fg='ansigreen'>User 👤</style>"))
                        print_code_block(f"  {esc(content)}")
                        c_print(HTML(""))  # Spacing

                    elif role == "assistant":
                        c_print(HTML(f"  <style font-weight='bold' fg='ansicyan'>{self.active_agent_name} 🤖</style>"))

                        # Check for Tool Calls
                        if "tool_calls" in msg and msg["tool_calls"]:
                            for tc in msg["tool_calls"]:
                                fn = tc.get("function", {})
                                name = fn.get("name", "unknown")
                                c_print(HTML(f"  <style fg='ansiyellow'>🔧 Calls: {_esc(name)}(...)</style>"))

                        if content:
                            print_code_block(f"  {esc(content)}")
                        c_print(HTML(""))  # Spacing

                    elif role == "tool":
                        # Tool Output - usually verbose, show summary
                        call_id = msg.get("tool_call_id", "unknown")
                        preview = content[:10000] + "..." if len(content) > 10000 else content
                        c_print(HTML(f"  <style fg='ansimagenta'>⚙️ Tool Result ({call_id})</style>"))
                        c_print(HTML(f"  <style fg='gray'>{esc(preview)}</style>"))
                        c_print(HTML(""))

                    elif role == "system":
                        c_print(HTML(f"  <style fg='ansired'>System ⚠️</style>"))
                        print_code_block(f"  {esc(content)}")
                        c_print(HTML(""))
                except:
                    c_print(f"invalid enty {role}-{msg}")
            print_box_footer()

        action = args[0]
        if action == "clear":
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                session = agent.session_manager.get(self.active_session_id)

                if not session:
                    print_status("No active session found.", "error")
                    return
            except Exception as e:
                print_status(f"Error accessing session: {e}", "error")
                return
            session.clear_history()
            # If the session has a persistence layer, ensure it saves
            self._save_state()
            print_status(f"History cleared for session '{self.active_session_id}'.", "success")
        elif action == "working":
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            run_id = None
            if self._focused_task_id and self._focused_task_id is self.all_executions:
                run_id = self.all_executions[self._focused_task_id].run_id

            elif self.all_executions:
                run_id = list(self.all_executions.values())[-1].run_id
            else:
                print_status("No run id found | run agent first", "error")
                return

            engine = agent._execution_engine_cache
            ctx = None
            if not engine:
                print_status("No execution found", "warning")
                return

            ctx = engine.get_execution(run_id)

            if not ctx and engine._active_executions:
                ctx = list(engine._active_executions.values())[-1]

            if not ctx:
                print_status("No working context found", "info")

            history = ctx.working_history
            show_history(history)

        elif action == "show":
            limit = 10
            if len(args) > 1 and args[1].isdigit():
                limit = int(args[1])
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                session = agent.session_manager.get(self.active_session_id)

                if not session:
                    print_status("No active session found.", "error")
                    return
            except Exception as e:
                print_status(f"Error accessing session: {e}", "error")
                return
            history = session.get_history(last_n=limit)
            show_history(history)

        elif action == "list":
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                sessions = list(agent.session_manager.sessions.keys())
                print_box_header("Sessions", "📁")
                for sid in sessions:
                    prefix = "* " if sid == self.active_session_id else "  "
                    print_box_content(f"{prefix}{sid}", "")
                print_box_footer()
            except Exception as e:
                print_status(f"Error: {e}", "error")

        elif action == "switch":
            if len(args) < 2:
                print_status("Usage: /session switch <id>", "warning")
                return
            self.active_session_id = args[1]
            self._save_state()
            print_status(f"Switched to session: {args[1]}", "success")

        elif action == "new":
            new_id = args[-1] if args[-1] != "new" else f"session_{uuid.uuid4().hex[:8]}"
            self.active_session_id = new_id
            self._save_state()
            print_status(f"Created new session: {new_id}", "success")

        else:
            print_status(f"Unknown session action: {action}", "error")

    async def _task_log_detailed(self, task_id: str, show_raw: bool = False):
        """Detaillierte Historie mit optionalem Tool I/O Dump."""
        tv = self._task_views.get(task_id)
        if not tv:
            print_status(f"Task {task_id} nicht gefunden.", "error")
            return

        print_box_header(f"Execution Log: {tv.task_id}", "📜")
        print_box_content(f"Agent: {esc(tv.agent_name)} | Persona: {esc(tv.persona)}", "info")
        print_separator()

        for iv in tv.iterations:
            # Iterations-Header
            c_print(HTML(f"<style fg='{PTColors.ZEN_CYAN}' font-weight='bold'>── Iteration {iv.n} ──</style>"))

            # Thoughts
            if iv.thoughts:
                c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>◎ Thoughts:</style>"))
                for thought in iv.thoughts:
                    c_print(HTML(f"  <style fg='{PTColors.WHITE}'>{thought.strip()}</style>"))

            # Tool History
            if iv.tools:
                c_print(HTML(f"\n<style fg='{PTColors.ZEN_AMBER}'>◇ Tool History:</style>"))

                # Wir gehen durch iv.tools (Summary) UND iv.tools_raw (für die Details)
                for idx, (name, success, elapsed, info) in enumerate(iv.tools):
                    # Zusammenfassung
                    col = PTColors.ZEN_GREEN if success else PTColors.ZEN_RED
                    c_print(HTML(f"  <style fg='{esc(col)}'>● {esc(name)}</style> ({elapsed:.3f}s) - {esc(info)}"))

                    # RAW I/O Dump (wenn -d aktiv)
                    if show_raw and idx < len(iv.tools_raw):
                        tname, raw_res, raw_in = iv.tools_raw[idx]

                        # Versuchen zu parsen für schönes Markdown
                        try:
                            raw_in = json.loads(raw_in) if raw_in.startswith('{') else raw_in
                        except:
                            pass
                        try:
                            raw_res = json.loads(raw_res) if raw_res.startswith('{') else raw_res
                        except:
                            pass
                        io_data = {
                            "Tool": tname,
                            "Input": raw_in,
                            "Result": raw_res
                        }

                        c_print(HTML(f"    <style fg='{PTColors.ZEN_DIM}'>    ↳ Raw I/O:</style>"))
                        # Hier nutzen wir deine json_to_md + print_code_block Logik
                        print_code_block(json_to_md(io_data), "md", show_line_numbers=True)

            c_print()  # Abstand zwischen Iterationen

        print_box_footer()

    async def _cmd_task(self, args: list[str]):
        """Handle /task commands — all from SSOT all_executions."""
        if not args:
            await self._task_show_overview()
            return
        action = args[0]

        if action == "cancel":
            if len(args) < 2:
                print_status("Usage: /task cancel <id>", "warning")
                return
            prefix = args[1]
            matched = [tid for tid in self.all_executions if tid.startswith(prefix) or tid == prefix]
            if not matched:
                print_status(f"Task '{prefix}' not found", "error")
                return
            for tid in matched:
                exc = self.all_executions[tid]
                exc.async_task.cancel()
                exc.status = "cancelled"
                print_status(f"Task {tid} cancelled", "success")

        elif action in ("status", "list"):
            await self._task_show_overview()

        elif action == "view":
            if len(args) < 2:
                running = [t for t in self.all_executions.values() if t.status == "running"]
                if len(running) == 1:
                    await self._task_view_detail(running[0].task_id)
                elif not running:
                    completed = [t for t in self.all_executions.values() if t.status == "completed"]
                    if completed:
                        await self._task_view_detail(completed[-1].task_id)
                    else:
                        print_status("No tasks to view", "info")
                else:
                    print_status(f"{len(running)} tasks running. Specify: /task view <id>", "warning")
                    await self._task_show_overview()
                return
            prefix = args[1]
            matched = [tid for tid in self.all_executions if tid.startswith(prefix) or tid == prefix]
            if matched:
                await self._task_view_detail(matched[0])
            else:
                print_status(f"Task '{prefix}' not found", "error")

        elif action == "clean":
            to_remove = [tid for tid, t in self.all_executions.items()
                         if t.status in ("completed", "failed", "cancelled")]
            for tid in to_remove:
                del self.all_executions[tid]
                if tid in self._task_views:
                    del self._task_views[tid]
            print_status(f"Cleaned {len(to_remove)} finished tasks", "success")

        elif action == "log":
            if len(args) < 2: return print_status("Usage: /task log <id>", "warning")
            if len(args) < 2:
                print_status("Usage: /task log <id> [-d]", "warning")
                return
            task_id = args[1]
            show_raw = "-d" in args
            await self._task_log_detailed(task_id, show_raw=show_raw)

        else:
            print_status(f"Unknown task action: {action}. Use: list, view, cancel, clean", "error")

    async def _task_show_overview(self):
        """Show compact table of all executions (SSOT)."""
        if not self.all_executions:
            print_status("No executions", "info")
            return
        import time as _t
        print_box_header("Executions", "◈")
        columns = [("ID", 26), ("Kind", 8), ("Agent", 12), ("Status", 10), ("Elapsed", 8), ("Query", 22)]
        widths   = [26, 8, 12, 10, 8, 22]
        print_table_header(columns, widths)
        for tid, t in self.all_executions.items():
            elapsed = _t.time() - t.started_at
            elapsed_str = f"{elapsed:.0f}s"
            focused_mark = "▸" if t.is_focused else " "
            status_style = {"running": "green", "completed": "cyan",
                            "failed": "red", "cancelled": "yellow"}.get(t.status, "grey")
            query_short = t.query[:20] + ".." if len(t.query) > 22 else t.query
            print_table_row(
                [f"{focused_mark}{tid[:25]}", t.kind[:8], t.agent_name[:12],
                 t.status, elapsed_str, query_short],
                widths,
                ["cyan", "grey", "white", status_style, "grey", "grey"],
            )
        print_box_footer()

    async def _task_view_detail(self, task_id: str):
        """Show detailed view of a specific execution."""
        import time as _t
        t = self.all_executions.get(task_id)
        if not t:
            print_status(f"Task {task_id} not found", "error")
            return

        elapsed = _t.time() - t.started_at
        print_box_header(f"Execution: {task_id}", "◈")
        print_box_content(
            f"Agent: {t.agent_name}  Kind: {t.kind}  Status: {t.status}  "
            f"Elapsed: {elapsed:.1f}s  Focused: {t.is_focused}", "info"
        )
        print_box_content(f"Query: {t.query}", "")

        try:
            agent = await self.isaa_tools.get_agent(t.agent_name)
            engine = agent._get_execution_engine()
            live = engine.live
            it, mx = live.iteration, live.max_iterations
            bar = f"{'━' * int(20*it/mx) if mx else 0}{'─' * (20 - int(20*it/mx) if mx else 20)} {it}/{'?' if not mx else mx}"
            print_box_content(f"Progress: {bar}", "")
            if live.phase:
                print_box_content(f"Phase: {live.phase.value}", "info")
            if live.thought:
                print_box_content(f"Thought: {live.thought}", "")
            if live.tool.name:
                print_box_content(f"Tool: {live.tool.name}", "")
                print_code_block(json_to_md(json.loads(live.tool.args_summary)))
        except Exception:
            print_box_content("(live state unavailable)", "warning")

        if t.status == "completed":
            if t.result_text:
                result_str = t.result_text
                if len(result_str) > 500:
                    print_box_content(f"Result ({len(result_str)} chars):", "success")
                    print_code_block(result_str[:500] + "\n... (truncated)")
                else:
                    print_box_content("Result:", "success")
                    c_print(result_str)
            else:
                # Fallback: try fut.result() for non-chat tasks
                try:
                    r = t.async_task.result()
                    if r:
                        c_print(str(r)[:500])
                except Exception:
                    pass

        if t.status == "failed":
            try:
                t.async_task.result()
            except Exception as e:
                print_box_content(f"Error: {e}", "error")

        print_box_footer()

    # =========================================================================
    # JOB SCHEDULER INTEGRATION
    # =========================================================================

    async def _fire_job_from_scheduler(self, job: JobDefinition) -> str:
        """Callback for JobScheduler: run an agent query, registered in SSOT."""
        run_id = uuid.uuid4().hex[:8]
        host_ref = self
        _tid_holder = [None]

        async def _run_job():
            try:
                agent = await host_ref.isaa_tools.get_agent(job.agent_name)

                # ── Live state step-callback ──────────────────────────────
                _scheduler = host_ref.job_scheduler
                _iter = [0]

                def _on_step(thought: str = "", tool: str = "",
                             context_used: int = 0, context_max: int = 0):
                    _iter[0] += 1
                    if _scheduler and _scheduler._live:
                        _scheduler._live.update_iteration(
                            job.job_id,
                            iteration=_iter[0],
                            tool=tool or None,
                            thought=thought or None,
                            context_used=context_used or None,
                            context_max=context_max or None,
                        )

                # Inject callback if agent supports it
                if hasattr(agent, "set_step_callback"):
                    agent.set_step_callback(_on_step)
                # ─────────────────────────────────────────────────────────

                result = await agent.a_run(
                    job.query,
                    session_id=job.session_id,
                    execution_id=run_id, max_iterations=self.max_iteration
                )
                exc = host_ref.all_executions.get(_tid_holder[0])
                if exc:
                    exc.status = "completed"
                return result or ""
            except asyncio.CancelledError:
                exc = host_ref.all_executions.get(_tid_holder[0])
                if exc:
                    exc.status = "cancelled"
                return ""
            except Exception as e:
                exc = host_ref.all_executions.get(_tid_holder[0])
                if exc:
                    exc.status = "failed"
                return ""

        async_task = asyncio.create_task(_run_job())

        exc = self._create_execution(
            kind="job",
            agent_name=job.agent_name,
            query=job.query,
            async_task=async_task,
            run_id=run_id,
            take_focus=False,
        )
        _tid_holder[0] = exc.task_id

        def _on_done(fut):
            _tid = _tid_holder[0]
            try:
                r = fut.result() or ""
                preview = (r[:60] + "..") if len(r) > 62 else r
                tv = self._task_views.get(_tid)
                if tv:
                    tv.status = "completed"
                with patch_stdout():
                    c_print(HTML(
                        f"\n<style fg='{PTColors.ZEN_GREEN}'>✓ {_tid}</style>"
                        f"  <style fg='{PTColors.ZEN_DIM}'>{html.escape(preview)}</style>\n"
                    ))
            except (asyncio.CancelledError, Exception):
                tv = self._task_views.get(_tid)
                if tv:
                    tv.status = "failed"

        async_task.add_done_callback(_on_done)
        return await async_task

    async def _cmd_job(self, args: list[str]):
        """Handle /job commands."""
        if not self.job_scheduler:
            print_status("Job scheduler not initialized", "error")
            return

        if not args:
            args = ["list"]

        action = args[0]

        if action == "list":
            jobs = self.job_scheduler.list_jobs()
            if not jobs:
                print_status("No scheduled jobs", "info")
                return
            print_box_header(f"Scheduled Jobs ({len(jobs)})", "◎")
            columns = [("ID", 14), ("Name", 18), ("Trigger", 16), ("Status", 8), ("Runs", 5), ("Last", 12)]
            widths = [14, 18, 16, 8, 5, 12]
            print_table_header(columns, widths)
            for j in jobs:
                status_style = {
                    "active": "green", "paused": "yellow",
                    "expired": "grey", "disabled": "red",
                }.get(j.status, "white")
                last = j.last_run_at[:10] if j.last_run_at else "-"
                print_table_row(
                    [j.job_id[:14], j.name[:18], j.trigger.trigger_type[:16],
                     j.status, str(j.run_count), last],
                    widths,
                    ["cyan", "white", "magenta", status_style, "grey", "grey"],
                )
            print_box_footer()

        elif action == "add":
            await self._job_add_interactive()

        elif action == "remove":
            if len(args) < 2:
                print_status("Usage: /job remove <id>", "warning")
                return
            if self.job_scheduler.remove_job(args[1]):
                print_status(f"Job {args[1]} removed", "success")
            else:
                print_status(f"Job {args[1]} not found", "error")

        elif action == "pause":
            if len(args) < 2:
                print_status("Usage: /job pause <id>", "warning")
                return
            if self.job_scheduler.pause_job(args[1]):
                print_status(f"Job {args[1]} paused", "success")
            else:
                print_status(f"Job {args[1]} not found or not active", "error")

        elif action == "resume":
            if len(args) < 2:
                print_status("Usage: /job resume <id>", "warning")
                return
            if self.job_scheduler.resume_job(args[1]):
                print_status(f"Job {args[1]} resumed", "success")
            else:
                print_status(f"Job {args[1]} not found or not paused", "error")

        elif action == "fire":
            if len(args) < 2:
                print_status("Usage: /job fire <id>", "warning")
                return
            job = self.job_scheduler.get_job(args[1])
            if not job:
                print_status(f"Job {args[1]} not found", "error")
                return
            print_status(f"Firing job {job.job_id} ({job.name})...", "info")
            asyncio.create_task(self.job_scheduler._fire_job(job))

        elif action == "detail":
            if len(args) < 2:
                print_status("Usage: /job detail <id>", "warning")
                return
            job = self.job_scheduler.get_job(args[1])
            if not job:
                # Try partial match
                matches = self.job_scheduler.find_jobs_by_name(args[1])
                if matches:
                    job = matches[0]
                else:
                    print_status(f"Job {args[1]} not found", "error")
                    return
            print_box_header(f"Job: {job.name}", "◎")
            print_box_content(f"ID: {job.job_id}", "")
            print_box_content(f"Agent: {job.agent_name}", "")
            print_box_content(f"Query: {job.query[:180]}", "")
            print_box_content(f"Trigger: {job.trigger.trigger_type}", "info")
            if job.trigger.at_datetime:
                print_box_content(f"  At: {job.trigger.at_datetime}", "")
            if job.trigger.interval_seconds:
                print_box_content(f"  Interval: {job.trigger.interval_seconds}s", "")
            if job.trigger.cron_expression:
                print_box_content(f"  Cron: {job.trigger.cron_expression}", "")
            if job.trigger.watch_job_id:
                print_box_content(f"  Watch Job: {job.trigger.watch_job_id}", "")
            if job.trigger.watch_path:
                print_box_content(f"  Watch Path: {job.trigger.watch_path}", "")
            print_box_content(f"Status: {job.status}", "")
            print_box_content(f"Session: {job.session_id}", "")
            print_box_content(f"Timeout: {job.timeout_seconds}s", "")
            print_box_content(f"Runs: {job.run_count}  Fails: {job.fail_count}", "")
            if job.last_run_at:
                print_box_content(f"Last Run: {job.last_run_at}", "")
            if job.last_result:
                print_box_content(f"Last Result: {job.last_result}", "")
            print_box_content(f"Created: {job.created_at}", "")
            print_box_footer()

        elif action == "autowake":
            await self._job_autowake(args[1:])

        elif action == "dream":
            sub = args[1] if len(args) > 1 else "status"
            if sub == "create":
                agent_name = args[2] if len(args) > 2 else "self"
                if not hasattr(self, '_current_agent') or not self._current_agent:
                    print_status("No active agent", "error")
                    return
                from toolboxv2.mods.isaa.base.Agent.dreamer import a_dream
                agent = self._current_agent
                if not hasattr(agent, 'a_dream'):
                    agent.a_dream = a_dream.__get__(agent, type(agent))
                self.job_scheduler.add_dream_job(agent_name)
                print_status(f"Dream job created for {agent_name} (nightly 03:00)", "success")
            elif sub == "status":
                dream_jobs = [j for j in self.job_scheduler.list_jobs() if j.query == "__dream__"]
                if not dream_jobs:
                    print_status("No dream jobs configured", "info")
                else:
                    for j in dream_jobs:
                        print_status(
                            f"{j.job_id} | {j.name} | {j.trigger.trigger_type} | "
                            f"{j.status} | runs:{j.run_count}",
                            "info"
                        )
            elif sub == "live":
                # Dreamer V3: runs as normal a_stream, rendered by TaskView
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                agent.active_session = self.active_session_id

                stream = agent.a_dream_stream()

                exc = self._create_execution(
                    kind="dream",
                    agent_name=f"dreamer_{self.active_agent_name}",
                    query="Meta-Learning Cycle",
                    async_task=None,
                    stream=stream,
                    take_focus=True,
                )
                task_id = exc.task_id
                async_task = asyncio.create_task(
                    self._drain_agent_stream(task_id, stream, should_speak=False)
                )
                exc.async_task = async_task
                async_task.add_done_callback(
                    lambda fut: self._on_agent_task_done(task_id, fut)
                )

                c_print(HTML(
                    f"<style fg='#67e8f9'>"
                    f"  ◎ dreamer gestartet → {task_id}</style>"
                ))

        else:
            print_status(f"Unknown job action: {action}. Use: list, add, remove, pause, resume, fire, detail, autowake", "error")

    async def _job_add_interactive(self):
        """Interactive job creation."""
        if not self.prompt_session:
            print_status("Prompt session not available", "error")
            return

        agents = self.isaa_tools.config.get("agents-name-list", ["self"])
        available_triggers = self.job_scheduler.trigger_registry.available_types() if self.job_scheduler else []

        print_box_header("Add New Job", "◎")
        print_box_content(f"Agents: {', '.join(agents)}", "info")
        print_box_content(f"Triggers: {', '.join(available_triggers)}", "info")
        print_box_footer()

        try:
            name = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Name: </style>"))
            if not name.strip():
                print_status("Cancelled", "warning")
                return

            agent_name = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Agent: </style>"))
            if not agent_name.strip():
                agent_name = "self"

            query = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Query: </style>"))
            if not query.strip():
                print_status("Query is required", "error")
                return

            trigger_type = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Trigger type: </style>"))
            if not trigger_type.strip():
                print_status("Trigger type is required", "error")
                return

            trigger_cfg = TriggerConfig(trigger_type=trigger_type.strip())

            if trigger_type.strip() == "on_time":
                dt = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Datetime (ISO): </style>"))
                trigger_cfg.at_datetime = dt.strip()
            elif trigger_type.strip() == "on_interval":
                secs = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Interval (seconds): </style>"))
                trigger_cfg.interval_seconds = int(secs.strip())
            elif trigger_type.strip() == "on_cron":
                expr = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Cron expression: </style>"))
                trigger_cfg.cron_expression = expr.strip()
            elif trigger_type.strip() in ("on_job_completed", "on_job_failed", "on_job_timeout"):
                jid = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Watch job ID: </style>"))
                trigger_cfg.watch_job_id = jid.strip()
            elif trigger_type.strip() == "on_file_changed":
                path = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Watch path: </style>"))
                trigger_cfg.watch_path = path.strip()
                pats = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Patterns (comma-sep, empty=all): </style>"))
                if pats.strip():
                    trigger_cfg.watch_patterns = [p.strip() for p in pats.split(",")]
            elif trigger_type.strip() == "on_system_idle":
                idle = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Idle seconds threshold [300]: </style>"))
                trigger_cfg.idle_seconds = int(idle.strip()) if idle.strip() else 300

            timeout = await self.prompt_session.prompt_async(HTML("<style fg='grey'>Timeout seconds [300]: </style>"))
            timeout_s = int(timeout.strip()) if timeout.strip() else 300

            job = JobDefinition(
                job_id=JobDefinition.generate_id(),
                name=name.strip(),
                agent_name=agent_name.strip(),
                query=query.strip(),
                trigger=trigger_cfg,
                timeout_seconds=timeout_s,
            )
            job_id = self.job_scheduler.add_job(job)
            print_status(f"Job created: {job_id} ({name.strip()})", "success")

        except (EOFError, KeyboardInterrupt):
            print_status("Cancelled", "warning")

    async def _job_autowake(self, args: list[str]):
        """Handle /job autowake install/remove/status."""
        if not args:
            print_status("Usage: /job autowake <install|remove|status>", "warning")
            return

        try:
            from toolboxv2.mods.isaa.extras.jobs.os_scheduler import (
                install_autowake, remove_autowake, autowake_status,
            )
        except ImportError:
            print_status("OS scheduler module not available", "error")
            return

        action = args[0]
        if action == "install":
            if self.job_scheduler and not self.job_scheduler.has_persistent_jobs():
                print_status(
                    "No persistent jobs (interval/cron/time/boot) — autowake would do nothing. "
                    "Add a job first.",
                    "warning",
                )
                return
            result = install_autowake(self.jobs_file)
            print_status(result, "success" if "installed" in result.lower() else "error")

        elif action == "remove":
            result = remove_autowake()
            print_status(result, "success" if "removed" in result.lower() else "error")

        elif action == "status":
            result = autowake_status()
            print_status(result, "info")
            if self.job_scheduler:
                persistent = [
                    j for j in self.job_scheduler.list_jobs()
                    if j.status == "active"
                       and j.trigger.trigger_type in self.job_scheduler._PERSISTENT_TRIGGER_TYPES
                ]
                if persistent:
                    print_status(
                        f"{len(persistent)} persistent job(s) will run via OS scheduler "
                        f"while CLI is offline: "
                        + ", ".join(j.name for j in persistent),
                        "info",
                    )
        else:
            print_status("Usage: /job autowake <install|remove|status>", "warning")

    async def _cmd_mcp(self, args: list[str]):
        """Handle live MCP management commands."""
        if not args:
            print_status(
                "Usage: /mcp <list|add|load|remove|reload> [args]", "warning"
            )
            return

        action = args[0].lower()
        agent = await self.isaa_tools.get_agent(self.active_agent_name)
        from toolboxv2.mods.isaa.extras.mcp_session_manager import _find_default_mcp_configs, _load_claude_mcp_configs
        # Ensure MCPSessionManager exists on agent
        if not getattr(agent, "_mcp_session_manager", None):
            from toolboxv2.mods.isaa.extras.mcp_session_manager import MCPSessionManager
            agent._mcp_session_manager = MCPSessionManager()
        mcp_mgr = agent._mcp_session_manager

        # ── LIST ──────────────────────────────────────────────────────────────
        if action == "list":
            print_box_header(f"MCP Servers: {self.active_agent_name}", "🔌")
            if not mcp_mgr.sessions:
                print_box_content("Keine aktiven MCP Server.", "info")
            else:
                columns = [("Name", 20), ("Tools", 6), ("Transport", 12)]
                widths = [20, 6, 12]
                print_table_header(columns, widths)
                for name in mcp_mgr.sessions:
                    caps = mcp_mgr.capabilities_cache.get(name, {})
                    n_tools = len(caps.get("tools", {}))
                    print_table_row(
                        [name, str(n_tools), "stdio"],
                        widths,
                        ["cyan", "green", "grey"],
                    )
            print_box_footer()

        # ── ADD (wizard or inline JSON) ───────────────────────────────────────
        elif action == "add":
            rest = args[1:]

            # ── Inline JSON: /mcp add {"name":"x","command":"npx","args":["y"]}
            if rest and rest[0].strip().startswith("{"):
                try:
                    cfg = json.loads(" ".join(rest))
                    name = cfg.pop("name")
                    cfg.setdefault("transport", "stdio")
                    cfg.setdefault("args", [])
                    cfg.setdefault("env", {})
                except (json.JSONDecodeError, KeyError) as e:
                    print_status(f"Invalid JSON: {e}", "error")
                    return
                servers = {name: cfg}

            # ── One-liner: /mcp add <name> <command> [arg1 arg2 …]
            elif len(rest) >= 2:
                name = rest[0]
                command = rest[1]
                cmd_args = rest[2:]
                servers = {name: {"command": command, "args": cmd_args,
                                  "transport": "stdio", "env": {}}}

            # ── Interactive wizard
            else:
                servers = await self._mcp_add_wizard()
                if not servers:
                    return

            # Connect all collected server configs
            for srv_name, srv_cfg in servers.items():
                await self._mcp_connect_and_register(
                    agent, mcp_mgr, srv_name, srv_cfg
                )

        # ── LOAD (Claude-Code config file) ────────────────────────────────────
        elif action == "load":
            path = args[1] if len(args) > 1 else None

            if not path:
                # Wizard: search for config files
                found = _find_default_mcp_configs()
                if not found:
                    print_status(
                        "No .mcp.json / claude_desktop_config.json found. "
                        "Provide path: /mcp load <path>", "warning"
                    )
                    return
                if len(found) == 1:
                    path = found[0]
                    print_status(f"Auto-detected: {path}", "info")
                else:
                    print_box_header("Found MCP config files", "📄")
                    for i, f in enumerate(found):
                        print_box_content(f"[{i}] {f}", "")
                    try:
                        idx_str = await self.prompt_session.prompt_async(
                            HTML("<style fg='grey'>Select [0]: </style>")
                        )
                        path = found[int(idx_str.strip()) if idx_str.strip().isdigit() else 0]
                    except (EOFError, KeyboardInterrupt):
                        return

            try:
                servers = _load_claude_mcp_configs(path)
            except Exception as e:
                print_status(f"Failed to load config: {e}", "error")
                return

            print_status(f"Loading {len(servers)} server(s) from {path}...", "progress")
            for srv_name, srv_cfg in servers.items():
                await self._mcp_connect_and_register(
                    agent, mcp_mgr, srv_name, srv_cfg
                )

        # ── REMOVE ────────────────────────────────────────────────────────────
        elif action == "remove":
            if len(args) < 2:
                print_status("Usage: /mcp remove <name>", "warning")
                return
            name = args[1]

            await mcp_mgr._cleanup_session(name)

            # Remove tools from tool_manager
            removed = 0
            for t_name in list(agent.tool_manager._registry.keys()):
                if t_name.startswith(f"{name}_"):
                    agent.tool_manager.unregister(t_name)
                    removed += 1

            # Persist to agent.json
            agent_cfg_path = Path(self.app.data_dir) / "Agents" / self.active_agent_name / "agent.json"
            if agent_cfg_path.exists():
                with open(agent_cfg_path, "r+", encoding="utf-8") as f:
                    cfg = json.load(f)
                    if "mcp" in cfg and "servers" in cfg["mcp"]:
                        cfg["mcp"]["servers"] = [
                            s for s in cfg["mcp"]["servers"] if s.get("name") != name
                        ]
                        f.seek(0);
                        json.dump(cfg, f, indent=2);
                        f.truncate()

            print_status(f"Removed '{name}' and {removed} tool(s).", "success")

        # ── RELOAD ───────────────────────────────────────────────────────────
        elif action == "reload":
            agent_cfg_path = Path(self.app.data_dir) / "Agents" / self.active_agent_name / "agent.json"
            if not agent_cfg_path.exists():
                print_status("No agent.json found.", "error")
                return
            print_status("Reloading MCP servers from agent.json...", "progress")
            with open(agent_cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            servers_raw = cfg.get("mcp", {}).get("servers", [])
            for entry in servers_raw:
                name = entry.get("name", "")
                if not name:
                    continue
                srv_cfg = {
                    "command": entry.get("command", ""),
                    "args": entry.get("args", []),
                    "env": entry.get("env", {}),
                    "transport": entry.get("transport", "stdio"),
                }
                await self._mcp_connect_and_register(
                    agent, mcp_mgr, name, srv_cfg
                )

        else:
            print_status(
                f"Unknown MCP action: {action}. "
                "Use: list, add, load, remove, reload", "error"
            )

        # ── Mini-Wizard ────────────────────────────────────────────────────────────

    async def _mcp_add_wizard(self) -> dict[str, dict] | None:
        """
        Interactive wizard for /mcp add.
        Returns {server_name: config_dict} or None on cancel.
        """
        print_box_header("Add MCP Server", "🔌")
        print_box_content(
            "Examples:  npx @modelcontextprotocol/server-filesystem /path\n"
            "           python -m mcp_server_git\n"
            "           node /path/to/server.js",
            "info",
        )
        print_box_footer()

        try:
            ps = self.prompt_session

            name = (await ps.prompt_async(
                HTML("<style fg='ansicyan'>Name: </style>")
            )).strip()
            if not name:
                print_status("Cancelled.", "warning");
                return None

            print_box_content(
                "Transport: [1] stdio (default)  [2] http/streamable-http", ""
            )
            transport_choice = (await ps.prompt_async(
                HTML("<style fg='grey'>Transport [1]: </style>")
            )).strip() or "1"
            transport = "http" if transport_choice == "2" else "stdio"

            if transport == "stdio":
                cmd_line = (await ps.prompt_async(
                    HTML("<style fg='ansicyan'>Command + args (e.g. npx pkg arg): </style>")
                )).strip()
                if not cmd_line:
                    print_status("Cancelled.", "warning");
                    return None
                import shlex as _shlex
                parts = _shlex.split(cmd_line)
                command = parts[0]
                cmd_args = parts[1:]
                url = ""
            else:
                url = (await ps.prompt_async(
                    HTML("<style fg='ansicyan'>URL (e.g. http://localhost:8000/mcp): </style>")
                )).strip()
                command = "";
                cmd_args = []

            env_line = (await ps.prompt_async(
                HTML("<style fg='grey'>Env vars (KEY=VAL KEY2=VAL2, empty=none): </style>")
            )).strip()
            env = {}
            if env_line:
                for pair in env_line.split():
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        env[k.strip()] = v.strip()

            cfg = {
                "command": command,
                "args": cmd_args,
                "transport": transport,
                "env": env,
            }
            if url:
                cfg["url"] = url

            print_box_header("Confirm", "✓")
            print_box_content(f"Name:      {name}", "")
            print_box_content(f"Transport: {transport}", "")
            if command:
                print_box_content(f"Command:   {command} {' '.join(cmd_args)}", "")
            if url:
                print_box_content(f"URL:       {url}", "")
            if env:
                print_box_content(f"Env:       {env}", "")
            print_box_footer()

            ok = (await ps.prompt_async(
                HTML("<style fg='ansicyan'>Connect? [Y/n]: </style>")
            )).strip().lower()
            if ok in ("n", "no"):
                print_status("Cancelled.", "warning");
                return None

            return {name: cfg}

        except (EOFError, KeyboardInterrupt):
            print_status("Cancelled.", "warning")
            return None

        # ── Core connect + register ────────────────────────────────────────────────

    async def _mcp_connect_and_register(
        self,
        agent,
        mcp_mgr,
        srv_name: str,
        srv_cfg: dict,
    ) -> None:
        """
        Connect one MCP server, register its tools on agent, persist to agent.json.
        Full error reporting — never silently returns 0.
        """
        print_status(f"Connecting '{srv_name}'...", "progress")

        # 1. Get session (with timeout + retries handled by MCPSessionManager)
        session = await mcp_mgr.get_session_with_timeout(srv_name, srv_cfg)

        if session is None:
            print_status(
                f"✗ Failed to connect '{srv_name}'. "
                f"Command: {srv_cfg.get('command')} {' '.join(srv_cfg.get('args', []))}\n"
                f"  Check: is the package installed? Try running the command manually.",
                "error",
            )
            return

        # 2. Extract capabilities (with timeout)
        caps = await mcp_mgr.extract_capabilities_with_timeout(session, srv_name)
        tools = caps.get("tools", {})

        if not tools:
            print_status(
                f"⚠ Connected to '{srv_name}' but found 0 tools. "
                f"Resources: {len(caps.get('resources', {}))}, "
                f"Prompts: {len(caps.get('prompts', {}))}. "
                f"The server may need different args or a different transport.",
                "warning",
            )
            # Still continue — maybe it has resources/prompts

        # 3. Register tools directly on agent.tool_manager
        from toolboxv2.mods.isaa.extras.mcp_session_manager import _make_mcp_tool_func
        count = 0
        for t_name, t_info in tools.items():
            wrapper_name = f"{srv_name}_{t_name}"
            func = _make_mcp_tool_func(session, srv_name, t_name, t_info)

            agent.add_tools([{
                "tool_func": func,
                "name": wrapper_name,
                "description": t_info.get("description", f"{srv_name}: {t_name}"),
                "category": [f"mcp_{srv_name}", "mcp"],
                "flags": {"mcp": True, "server": srv_name},
            }])
            count += 1

        print_status(f"✓ '{srv_name}': {count} tool(s) loaded.", "success")

        # 4. Persist to agent.json for /mcp reload
        agent_cfg_path = Path(self.app.data_dir) / "Agents" / self.active_agent_name / "agent.json"
        if agent_cfg_path.exists():
            try:
                with open(agent_cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.setdefault("mcp", {"enabled": True, "servers": []})
                cfg["mcp"].setdefault("servers", [])
                # Replace existing entry or append
                entry = {"name": srv_name, **srv_cfg}
                cfg["mcp"]["servers"] = [
                    s for s in cfg["mcp"]["servers"] if s.get("name") != srv_name
                ]
                cfg["mcp"]["servers"].append(entry)
                with open(agent_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            except Exception as e:
                print_status(f"  (Could not persist to agent.json: {e})", "warning")


    def _print_vfs_tree(self, tree: dict, level: int = 0, max_depth: int = 4):
        """Recursively print VFS directory structure (HTML version)."""
        if level > max_depth:
            c_print(HTML(f"{'  ' * level}<style fg='{PTColors.GREY}'>...</style>"))
            return

        indent = "  " * level
        # Icons als HTML Strings vorbereiten
        folder_icon = f"<style fg='{PTColors.BLUE}'>📂</style>"
        file_icon = f"<style fg='{PTColors.GREY}'>📄</style>"

        # Sort: folders first, then files
        items = sorted(tree.items(), key=lambda x: (not isinstance(x[1], dict), x[0]))

        for name, content in items:
            if name.startswith("."):
                continue  # Skip hidden

            safe_name = html.escape(name)

            if isinstance(content, dict) and content:
                # Directory (non-empty dict) -> Bold Cyan Name
                c_print(HTML(
                    f"{indent}{folder_icon} <style font-weight='bold' fg='{PTColors.CYAN}'>{safe_name}</style>"
                ))
                self._print_vfs_tree(content, level + 1, max_depth)
            else:
                # File -> Grey Size Hint
                size_hint = ""
                if isinstance(content, str) or hasattr(content, "__len__"):
                    size_hint = f" <style fg='{PTColors.GREY}'>({len(content)}b)</style>"

                c_print(HTML(f"{indent}{file_icon} {safe_name}{size_hint}"))

    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from extension."""
        ext = Path(filename).suffix.lower()
        type_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".sh": "bash",
            ".bash": "bash",
            ".sql": "sql",
            ".xml": "xml",
            ".env": "env",
            ".txt": "text",
            ".log": "text",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
        }
        return type_map.get(ext, "markdown")  # Default to markdown

    async def _cmd_vfs(self, args: list[str]):
        """Handle /vfs commands - mount, unmount, sync, tree, file content."""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            session = await agent.session_manager.get_or_create(self.active_session_id)

            if not session or not hasattr(session, "vfs"):
                print_status("No VFS available in current session", "warning")
                return

            if not args:
                # No args: show VFS tree structure
                await self._vfs_show_tree(session)
                return

            cmd = args[0].lower()

            # /vfs mount <local_path> [vfs_path] [--readonly] [--no-sync]
            if cmd == "init":
                print_status("VFS Online")
                return
            elif cmd == "mount":
                if len(args) < 2:
                    print_status("Usage: /vfs mount <local_path> [vfs_path] [--readonly] [--no-sync]", "warning")
                    return

                local_path = args[1]
                vfs_path = "/project"
                readonly = False
                auto_sync = True

                for i, arg in enumerate(args[2:], start=2):
                    if arg == "--readonly":
                        readonly = True
                    elif arg == "--no-sync":
                        auto_sync = False
                    elif not arg.startswith("--") and i == 2:
                        vfs_path = arg

                print_status(f"Mounting {local_path} → {vfs_path}...", "info")
                result = session.vfs.mount(
                    local_path=local_path,
                    vfs_path=vfs_path,
                    readonly=readonly,
                    auto_sync=auto_sync
                )

                if result.get("success"):
                    print_status(f"Mounted: {result['files_indexed']} files, {result['dirs_indexed']} dirs", "success")
                    print_box_content(f"Scan time: {result['scan_time_ms']:.1f}ms", "info")
                else:
                    print_status(f"Mount failed: {result.get('error')}", "error")
            elif cmd == "obsidian":
                if len(args) < 2:
                    print_status("Usage: /vfs obsidian <mount|unmount|sync> <local_path> [vfs_path]", "warning")
                    return
                action = args[1].lower()
                if action == "mount":
                    if len(args) < 3:
                        print_status("Usage: /vfs obsidian mount <local_path> [vfs_path]", "warning")
                        return
                    local_path = args[2]
                    vfs_path = args[3] if len(args) > 3 else "/obsidian"
                    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import sync_obsidian_vault
                    result = sync_obsidian_vault(session.vfs, local_path, vfs_path)
                    if result.get("success"):
                        print_status(f"Obsidian vault mounted: {local_path} → {vfs_path}", "success")
                    else:
                        print_status(f"Mount failed: {result.get('error')}", "error")
                elif action == "unmount":
                    if len(args) < 3:
                        print_status("Usage: /vfs obsidian unmount <vfs_path>", "warning")
                        return
                    vfs_path = args[2]
                    result = session.vfs.unmount(vfs_path, save_changes=False)
                    if result.get("success"):
                        print_status(f"Obsidian vault unmounted: {vfs_path}", "success")
                    else:
                        print_status(f"Unmount failed: {result.get('error')}", "error")
                elif action == "sync":
                    if len(args) < 3:
                        print_status("Usage: /vfs obsidian sync <vfs_path>", "warning")
                        return
                    vfs_path = args[2]
                    result = session.vfs.refresh_mount(vfs_path)
                    if result.get("success"):
                        print_status(f"Obsidian vault synced: {vfs_path}", "success")
                    else:
                        print_status(f"Sync failed: {result.get('error')}", "error")
                else:
                    print_status(f"Unknown obsidian action: {action} available: mount, unmount, sync", "error")

            # /vfs unmount <vfs_path> [--no-save]
            elif cmd == "unmount":
                if len(args) < 2:
                    print_status("Usage: /vfs unmount <vfs_path> [--no-save]", "warning")
                    return

                vfs_path = args[1]
                save_changes = "--no-save" not in args

                result = session.vfs.unmount(vfs_path, save_changes=save_changes)

                if result.get("success"):
                    saved = result.get("files_saved", [])
                    print_status(f"Unmounted: {vfs_path}", "success")
                    if saved:
                        print_box_content(f"Saved {len(saved)} modified files", "info")
                else:
                    print_status(f"Unmount failed: {result.get('error')}", "error")

            # /vfs sync [vfs_path]  - file or directory
            elif cmd == "sync":
                if len(args) > 1:
                    path = session.vfs._normalize_path(args[1])

                    if session.vfs._is_directory(path):
                        # Sync all dirty files under this directory
                        prefix = path if path == "/" else path + "/"
                        synced, errors = [], []
                        for fpath, f in session.vfs.files.items():
                            if fpath.startswith(prefix) and isinstance(f, VFSFile) and f.is_dirty and f.local_path:
                                r = session.vfs._sync_to_local(fpath)
                                if r.get("success"):
                                    synced.append(fpath)
                                else:
                                    errors.append(f"{fpath}: {r.get('error')}")
                        print_status(f"Synced {len(synced)} files in {path}", "success")
                        for err in errors:
                            print_status(err, "error")
                    elif session.vfs._is_file(path):
                        result = session.vfs._sync_to_local(path)
                        if result.get("success"):
                            print_status(f"Synced: {path} → {result['synced_to']}", "success")
                        else:
                            print_status(f"Sync failed: {result.get('error')}", "error")
                    else:
                        print_status(f"Not found: {path}", "error")
                else:
                    # Sync all dirty files
                    result = session.vfs.sync_all()
                    if result.get("success"):
                        print_status(f"Synced {len(result['synced'])} files", "success")
                    else:
                        for err in result.get("errors", []):
                            print_status(err, "error")

            # /vfs save <vfs_path> <local_path>  - file or directory
            elif cmd == "save":
                if len(args) < 3:
                    print_status("Usage: /vfs save <vfs_path> <local_path>", "warning")
                    return
                vfs_path = session.vfs._normalize_path(args[1])
                local_path = args[2]

                if session.vfs._is_directory(vfs_path):
                    # Save entire directory to local path
                    prefix = vfs_path if vfs_path == "/" else vfs_path + "/"
                    saved, errors = 0, 0
                    local_base = os.path.abspath(os.path.expanduser(local_path))
                    os.makedirs(local_base, exist_ok=True)

                    for fpath, f in session.vfs.files.items():
                        if fpath.startswith(prefix):
                            relative = fpath[len(prefix):]
                            target = os.path.join(local_base, relative.replace("/", os.sep))
                            result = session.vfs.save_to_local(
                                fpath, target, overwrite=True, create_dirs=True
                            )
                            if result.get("success"):
                                saved += 1
                            else:
                                errors += 1
                    print_status(f"Saved {saved} files from {vfs_path} → {local_path}", "success")
                    if errors:
                        print_status(f"{errors} files failed", "warning")
                else:
                    result = session.vfs.save_to_local(vfs_path, local_path, overwrite=True, create_dirs=True)
                    if result.get("success"):
                        print_status(f"Saved: {vfs_path} → {local_path}", "success")
                    else:
                        print_status(f"Save failed: {result.get('error')}", "error")

            # /vfs refresh <vfs_path>
            elif cmd == "refresh":
                if len(args) < 2:
                    print_status("Usage: /vfs refresh <mount_path>", "warning")
                    return

                vfs_path = args[1]
                result = session.vfs.refresh_mount(vfs_path)

                if result.get("success"):
                    print_status(f"Refreshed: {result['files_indexed']} files", "success")
                    if result.get("modified_preserved", 0) > 0:
                        print_box_content(f"Preserved {result['modified_preserved']} modified files", "info")
                else:
                    print_status(f"Refresh failed: {result.get('error')}", "error")

            # /vfs pull <vfs_path> - reload from disk (file or directory)
            elif cmd == "pull":
                if len(args) < 2:
                    print_status("Usage: /vfs pull <path>", "warning")
                    return

                path = session.vfs._normalize_path(args[1])

                if session.vfs._is_directory(path):
                    # Pull all shadow files under this directory
                    prefix = path if path == "/" else path + "/"
                    pulled, skipped = 0, 0
                    for fpath, f in session.vfs.files.items():
                        if fpath.startswith(prefix) and hasattr(f, 'local_path') and f.local_path:
                            result = session.vfs._load_shadow_content(fpath)
                            if result.get("success"):
                                f.is_dirty = False
                                f.backing_type = FileBackingType.SHADOW
                                pulled += 1
                            else:
                                skipped += 1
                    print_status(f"Pulled {pulled} files in {path}", "success")
                    if skipped:
                        print_status(f"{skipped} files skipped/failed", "warning")
                elif session.vfs._is_file(path):
                    f = session.vfs.files.get(path)
                    if f and hasattr(f, 'local_path') and f.local_path:
                        result = session.vfs._load_shadow_content(path)
                        if result.get("success"):
                            f.is_dirty = False
                            f.backing_type = FileBackingType.SHADOW
                            print_status(f"Pulled: {path} ({result['loaded_bytes']} bytes)", "success")
                        else:
                            print_status(f"Pull failed: {result.get('error')}", "error")
                    else:
                        print_status("Not a shadow file", "warning")
                else:
                    print_status(f"Not found: {path}", "error")

            # /vfs mounts - list all mounts
            elif cmd == "mounts":
                if not session.vfs.mounts:
                    print_status("No active mounts", "info")
                    return

                print_box_header("Active Mounts", "📂")
                for vfs_path, mount in session.vfs.mounts.items():
                    flags = []
                    if mount.readonly:
                        flags.append("readonly")
                    if mount.auto_sync:
                        flags.append("auto-sync")
                    flags_str = f" [{', '.join(flags)}]" if flags else ""
                    print_box_content(f"{vfs_path} → {mount.local_path}{flags_str}", "")
                print_box_footer()

            # /vfs dirty - show modified files
            elif cmd == "dirty":
                dirty_files = [
                    (path, f) for path, f in session.vfs.files.items()
                    if hasattr(f, 'is_dirty') and f.is_dirty
                ]

                if not dirty_files:
                    print_status("No modified files", "info")
                    return

                print_box_header("Modified Files", "✏️")
                for path, f in dirty_files:
                    local = f.local_path if hasattr(f, 'local_path') else "memory"
                    print_box_content(f"{path} → {local}", "")
                print_box_footer()

            # /vfs sys-add <local_path> [vfs_path] [--refresh]
            elif cmd == "sys-add":
                if len(args) < 2:
                    print_status("Usage: /vfs sys-add <local_path> [vfs_path] [--refresh]", "warning")
                    return

                local_path = args[1]
                vfs_path = args[2] if len(args) > 2 and not args[2].startswith("--") else None
                auto_refresh = "--refresh" in args

                print_status(f"Adding system file: {local_path}...", "info")
                result = session.vfs.add_system_file(
                    local_path=local_path,
                    vfs_path=vfs_path,
                    auto_refresh=auto_refresh
                )

                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                    print_box_content(f"Size: {result['size_bytes']} bytes, Lines: {result['lines']}", "info")
                    if result.get('auto_refresh'):
                        print_box_content("Auto-refresh: enabled", "info")
                else:
                    print_status(f"✗ {result.get('error')}", "error")

            # /vfs sys-remove <vfs_path>
            elif cmd == "sys-remove":
                if len(args) < 2:
                    print_status("Usage: /vfs sys-remove <vfs_path>", "warning")
                    return

                vfs_path = args[1]
                result = session.vfs.remove_system_file(vfs_path)

                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                else:
                    print_status(f"✗ {result.get('error')}", "error")

            elif cmd in ["remove", "rm"]:
                if len(args) < 2:
                    print_status("Usage: /vfs remove <vfs_path>", "warning")
                    return

                # Unterstützt Pfade mit Leerzeichen
                vfs_path = " ".join(args[1:])
                norm_path = session.vfs._normalize_path(vfs_path)

                # 1. Existenz und Typ prüfen
                is_file = session.vfs._is_file(norm_path)
                is_dir = session.vfs._is_directory(norm_path)

                if not is_file and not is_dir:
                    print_status(f"✗ Path not found: {norm_path}", "error")
                    return

                # Systemdateien (Read-Only) vorher abfangen (Verhindert Absturz beim Bestätigen)
                if is_file and session.vfs.files[norm_path].readonly:
                    print_status(f"✗ Cannot delete system file: {norm_path} (Use sys-remove instead)", "error")
                    return
                if is_dir and session.vfs.directories[norm_path].readonly:
                    print_status(f"✗ Cannot delete readonly directory: {norm_path}", "error")
                    return

                # 2. Statistiken sammeln
                total_size = 0
                file_count = 0
                dir_count = 0

                if is_file:
                    f = session.vfs.files[norm_path]
                    file_count = 1
                    total_size = f.size
                    target_desc = f"File: 📄 {f.filename}"
                else:
                    # Ordner rekursiv analysieren
                    ls_result = session.vfs.ls(norm_path, recursive=True)
                    if ls_result.get("success"):
                        for item in ls_result["contents"]:
                            if item["type"] == "directory":
                                dir_count += 1
                            else:
                                file_count += 1
                                total_size += item.get("size", 0)
                    dir_count += 1  # Den Hauptordner selbst mitzählen
                    target_desc = f"Directory: 📁 {norm_path}"

                # Größe formatieren
                size_str = f"{total_size} bytes" if total_size < 1024 else f"{total_size / 1024:.2f} KB"

                # 3. Bestätigungs-Dialog anzeigen
                print_box_header("Confirm Deletion", "⚠️")
                print_box_content(target_desc, "")

                if is_dir:
                    print_box_content(f"  Includes: {file_count} files, {dir_count - 1} subdirectories", "warning")

                print_box_content(f"  Total size: {size_str}", "warning")
                print_box_footer()

                # Nutzer um Erlaubnis fragen
                try:
                    confirm = input("\nAre you sure you want to permanently delete this? (y/N): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    confirm = "n"

                if confirm not in ['y', 'yes']:
                    print_status("Deletion cancelled.", "info")
                    return

                # 4. Löschen ausführen
                if is_file:
                    result = session.vfs.delete(norm_path)
                else:
                    # force=True zwingt das VFS, auch gefüllte Ordner rekursiv zu löschen
                    result = session.vfs.rmdir(norm_path, force=True)

                # 5. Ergebnis ausgeben
                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                else:
                    print_status(f"✗ Failed to delete: {result.get('error')}", "error")

            # /vfs sys-refresh <vfs_path>
            elif cmd == "sys-refresh":
                if len(args) < 2:
                    print_status("Usage: /vfs sys-refresh <vfs_path>", "warning")
                    return

                vfs_path = args[1]
                result = session.vfs.refresh_system_file(vfs_path)

                if result.get("success"):
                    print_status(f"✓ {result['message']}", "success")
                    print_box_content(f"Size: {result['size_bytes']} bytes, Lines: {result['lines']}", "info")
                else:
                    print_status(f"✗ {result.get('error')}", "error")

            # /vfs sys-list - list all system files
            elif cmd == "sys-list":
                result = session.vfs.list_system_files()

                if not result.get("system_files"):
                    print_status("No system files", "info")
                    return

                print_box_header("System Files (Read-Only)", "📄")
                for info in result["system_files"]:
                    path = info["path"]
                    local = info.get("local_path") or "memory"
                    refresh = " [auto-refresh]" if info.get("auto_refresh") else ""
                    print_box_content(f"{path} ← {local}{refresh}", "")
                    print_box_content(f"  {info['lines']} lines, {info['file_type']}", "dim")
                print_box_footer()

            # /vfs <path> - show file content or directory listing
            else:
                path_str = " ".join(args)
                norm = session.vfs._normalize_path(path_str)

                if session.vfs._is_directory(norm):
                    # Directory: show listing
                    contents = session.vfs._list_directory_contents(norm)
                    print_box_header(f"VFS: {norm}", "📂")
                    if not contents:
                        print_box_content("(empty directory)", "")
                    else:
                        for item in contents:
                            if item["type"] == "directory":
                                print_box_content(f"  📁 {item['name']}/", "")
                            else:
                                size = item.get("size", 0)
                                state = item.get("state", "")
                                ftype = item.get("file_type", "")
                                meta = f"{size}b" if size < 1024 else f"{size / 1024:.1f}kb"
                                dirty = " ●" if session.vfs.files.get(item["path"], None) and getattr(session.vfs.files[item["path"]], 'is_dirty', False) else ""
                                print_box_content(f"  {item['name']:<30} {meta:>8}  {ftype}{dirty}", "")
                    print_box_footer()
                else:
                    await self._vfs_show_file(session, path_str)

        except Exception as e:
            print_status(f"Error: {e}", "error")

    async def _vfs_show_tree(self, session):
        """Show VFS tree structure."""
        print_box_header(
            f"VFS Structure: {self.active_agent_name}@{self.active_session_id}", "📂"
        )
        print_code_block(session.vfs.file_tree_string(max_depth=4), "markdown")


    async def _vfs_show_file(self, session, filename: str):
        """Show file content."""
        try:
            result = session.vfs.read(filename)

            if isinstance(result, dict):
                if result.get("success"):
                    content = result.get("content", "")
                else:
                    print_status(f"File not found: {filename}", "error")
                    return
            else:
                content = str(result)

            file_type = self._detect_file_type(filename)

            # Check if shadow/dirty
            f = session.vfs.files.get(session.vfs._normalize_path(filename))
            status_parts = [f"Type: {file_type}", f"Size: {len(content)} bytes"]
            if f and hasattr(f, 'is_dirty') and f.is_dirty:
                status_parts.append("MODIFIED")
            if f and hasattr(f, 'local_path') and f.local_path:
                status_parts.append(f"→ {f.local_path}")

            print_box_header(f"📄 {filename}", "")
            print_box_content(" | ".join(status_parts), "info")
            print_separator()

            # Format based on type
            if file_type == "json":
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
                print_code_block(content, "json", show_line_numbers=True)
            elif file_type in ("yaml", "yml"):
                print_code_block(content, "yaml", show_line_numbers=True)
            elif file_type == "toml":
                print_code_block(content, "toml", show_line_numbers=True)
            elif file_type == "env":
                print_code_block(content, "env", show_line_numbers=False)
            elif file_type == "markdown":
                print_code_block(content, "md", show_line_numbers=False)
            else:
                print_code_block(content, "text", show_line_numbers=True)

            print_box_footer()

        except Exception as e:
            print_status(f"Error reading file '{filename}': {e}", "error")

    def _enqueue_speech(self, text: str):
        """
        Bereinigt Text und stellt ihn in die TTS-Queue.
        Non-blocking — gibt sofort zurück.
        """
        clean = remove_styles(text.strip())
        if not clean:
            return
        # Emotion aus Content-Hinweisen ableiten (optional, heuristisch)
        emotion = _infer_emotion(clean)
        get_app("ci.audio.bg.task").run_bg_task_advanced(
            self.audio_player.queue_text, clean, emotion
        )

    # ─── /audio COMMAND ──────────────────────────────────────────────────────────
    async def _handle_audio_command(self, args: list[str]):
        """Handle /audio commands."""
        import sounddevice as sd

        if not args:
            ap = self.audio_player
            cfg = ap.tts_config
            dev = getattr(self, "_audio_device", None)
            dev_name = "system default" if dev is None else str(dev)
            playing = "▶ playing" if ap.is_busy else "⏹ idle"

            print_box_header("Audio Settings", "🔊")
            print_box_content(f"Status:   {playing}  |  queue: {ap.pending_texts}", "")
            print_box_content(f"Verbose:  {'ON' if self.verbose_audio else 'OFF'}", "")
            print_box_content(f"Backend:  {cfg.backend.value}", "")
            print_box_content(f"Voice:    {cfg.voice}", "")
            print_box_content(f"Language: {cfg.language}", "")
            print_box_content(f"Device:   {dev_name}", "")
            print_separator()
            print_box_content("Commands:", "bold")
            print_box_content("/audio on                    - All responses spoken", "")
            print_box_content("/audio off                   - Disable verbose audio", "")
            print_box_content("/audio voice <v>             - Set voice", "")
            print_box_content("/audio backend <b>           - groq_tts / piper / elevenlabs / index_tts", "")
            print_box_content("/audio lang <l>              - de / en / ...", "")
            print_box_content("/audio device                - Interactive device picker", "")
            print_box_content("/audio device <idx>          - Set by index", "")
            print_box_content("/audio device default        - Reset to system default", "")
            print_box_content("/audio devices               - List all output devices", "")
            print_box_content("/audio stop                  - Stop current playback", "")
            print_box_content("/audio restart               - Rebuild player with current settings", "")
            print_box_content("", "")
            print_box_content("Tip: append  #audio  to any message for one-time spoken response", "info")
            print_box_footer()
            return

        cmd = args[0].lower()

        # ── on / off ──────────────────────────────────────────────────────────────
        if cmd == "on":
            self.verbose_audio = True
            if not self.audio_player._task or self.audio_player._task.done():
                await self.audio_player.start()
            await self._ensure_audio_setup()
            print_status("Verbose audio ON — all responses will be spoken", "success")

        elif cmd == "off":
            self.verbose_audio = False
            print_status("Verbose audio OFF", "success")

        # ── stop ──────────────────────────────────────────────────────────────────
        elif cmd == "stop":
            await self.audio_player.stop()
            # Rebuild so it can be started again later
            self.audio_player = self._build_audio_player()
            print_status("Audio stopped and player reset", "success")

        # ── voice ─────────────────────────────────────────────────────────────────
        elif cmd == "voice" and len(args) > 1:
            self._audio_voice = args[1]
            await self._restart_audio_player()
            print_status(f"Voice → {args[1]}", "success")

        # ── backend ───────────────────────────────────────────────────────────────
        elif cmd == "backend" and len(args) > 1:
            b = args[1].lower()
            valid = [e.value for e in TTSBackend]
            if b not in valid:
                print_status(f"Valid backends: {', '.join(valid)}", "error")
                return
            self._audio_backend = b
            await self._restart_audio_player()
            print_status(f"Backend → {b}", "success")

        # ── lang ──────────────────────────────────────────────────────────────────
        elif cmd == "lang" and len(args) > 1:
            self._audio_language = args[1]
            await self._restart_audio_player()
            print_status(f"Language → {args[1]}", "success")

        # ── devices (list) ────────────────────────────────────────────────────────
        elif cmd == "devices":
            print_box_header("Output Devices", "🔊")
            current = getattr(self, "_audio_device", None)
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_output_channels"] > 0:
                    marker = " ◀ current" if i == current else ""
                    sr = int(dev["default_samplerate"])
                    print_box_content(
                        f"[{i:2d}] {dev['name'][:50]:<50}  {sr}Hz{marker}", ""
                    )
            print_box_footer()

        # ── device (set) ──────────────────────────────────────────────────────────
        elif cmd == "device":
            if len(args) == 1:
                await self._select_audio_device_interactive()
            elif args[1].lower() == "default":
                self._audio_device = None
                await self._restart_audio_player()
                print_status("Audio device → system default", "success")
            else:
                try:
                    idx = int(args[1])
                    devs = sd.query_devices()
                    if idx < 0 or idx >= len(devs):
                        print_status(f"Device index out of range (0–{len(devs) - 1})", "error")
                        return
                    dev = devs[idx]
                    if dev["max_output_channels"] == 0:
                        print_status(f"Device [{idx}] has no output channels", "error")
                        return
                    self._audio_device = idx
                    await self._restart_audio_player()
                    print_status(f"Audio device → [{idx}] {dev['name']}", "success")
                except ValueError:
                    print_status("Usage: /audio device <index>  or  /audio device default", "error")

        # ── restart ───────────────────────────────────────────────────────────────
        elif cmd == "restart":
            await self._restart_audio_player()
            print_status("Audio player rebuilt with current settings", "success")


        elif cmd == "live":
            sub = args[1].lower() if len(args) > 1 else ""

            if sub == "stop":
                if self._live_engine:
                    await self._live_engine.stop()
                    self._live_engine = None
                    print_status("Live mode stopped", "success")
                else:
                    print_status("Live mode not running", "error")

            elif sub == "status":
                if self._live_engine:
                    print_status(self._live_engine.status_line(), "info")
                else:
                    print_status("Live mode not running", "error")

            elif sub == "keyword" and len(args) > 2:
                self._live_config.wake_word_model = args[2]
                print_status(f"Wake word model → {args[2]}", "success")

            elif sub == "sensitivity" and len(args) > 2:
                try:
                    v = float(args[2])
                    self._live_config.wake_sensitivity = max(0.0, min(1.0, v))
                    print_status(f"Wake sensitivity → {self._live_config.wake_sensitivity:.2f}", "success")
                except ValueError:
                    print_status("Usage: /audio live sensitivity 0.0–1.0", "error")

            elif sub == "end" and len(args) > 2:
                mode_map = {
                    "silence": EndMode.SILENCE,
                    "keyword": EndMode.KEYWORD,
                    "intent": EndMode.INTENT,
                    "auto": EndMode.AUTO,
                }
                m = mode_map.get(args[2].lower())
                if m:
                    self._live_config.end_mode = m
                    print_status(f"End mode → {m.value}", "success")
                else:
                    print_status(f"Valid modes: {', '.join(mode_map)}", "error")

            elif sub == "silence" and len(args) > 2:
                try:
                    self._live_config.silence_ms = int(args[2])
                    print_status(f"Silence timeout → {self._live_config.silence_ms}ms", "success")
                except ValueError:
                    print_status("Usage: /audio live silence <ms>", "error")

            else:
                # No sub-arg or unknown → START live mode
                if self._live_engine is not None:
                    print_status("Live mode already running. /audio live stop to stop.", "info")
                    return

                # Sicherstellen dass player läuft
                if self.audio_player._task is None or self.audio_player._task.done():
                    await self.audio_player.start()

                async def on_utterance(wav_bytes: bytes, speaker: Optional[str]):
                    """Called when a complete utterance is captured."""
                    # Speaker-Tag für den Agent
                    speaker_tag = f"[{speaker}]: " if speaker else ""

                    # STT → agent (reuse existing pipeline)
                    try:
                        from toolboxv2.mods.isaa.base.audio_io.Stt import (
                            transcribe, STTConfig, STTBackend,
                        )
                        stt_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: transcribe(
                                wav_bytes,
                                config=STTConfig(
                                    backend=STTBackend.FASTER_WHISPER,
                                    model="small",
                                    device="cpu",
                                    compute_type="int8",
                                    language=None,
                                ),
                            )
                        )
                        text = stt_result.text.strip()
                        if not text:
                            return

                        # Stop keyword check
                        if self._live_engine and self._live_engine._end_detector.check_keyword(text):
                            # Keyword was the entire utterance → ignore, just acknowledged
                            return

                        full_query = speaker_tag + text
                        print(f"\n🎤 {full_query}")

                        # In die normale Agent-Pipeline einspeisen
                        await self._handle_agent_interaction(full_query)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        print_status(f"Live STT error: {e}", "error")

                self._live_engine = LiveModeEngine(
                    config=self._live_config,
                    on_utterance=on_utterance,
                    speaker_store=self._speaker_store,
                )
                await self._live_engine.start()
                cfg = self._live_config
                print_status(
                    f"Live mode active | wake: {cfg.wake_word_model} "
                    f"| end: {cfg.end_mode.value} "
                    f"| silence: {cfg.silence_ms}ms",
                    "success",
                )
                print_status("Say the wake word to start speaking.", "info")

        # ── /audio speaker ────────────────────────────────────────────────────────────

        elif cmd == "speaker":
            sub = args[1].lower() if len(args) > 1 else ""

            if sub == "list" or not sub:
                names = self._speaker_store.list_names()
                if names:
                    print_box_header("Speaker Profiles", "🎙")
                    for n in names:
                        marker = " ◀ you" if n == getattr(self, "_my_speaker_name", None) else ""
                        print_box_content(f"  {n}{marker}", "")
                    print_box_footer()
                else:
                    print_status("No speaker profiles registered yet.", "info")
                    print_status("  /audio speaker add <name>  to register your voice.", "info")

            elif sub == "add" and len(args) > 2:
                name = args[2]
                await self._record_speaker_profile(name)

            elif sub == "remove" and len(args) > 2:
                name = args[2]
                if self._speaker_store.remove(name):
                    print_status(f"Removed profile: {name}", "success")
                else:
                    print_status(f"Profile not found: {name}", "error")

            elif sub == "who":
                if self._live_engine:
                    spk = self._live_engine.current_speaker
                    print_status(f"Current speaker: {spk or 'unknown'}", "info")
                else:
                    print_status("Live mode not running", "error")

            else:
                print_status("Usage: /audio speaker [list|add <name>|remove <name>|who]", "error")

        else:
            print_status(f"Unknown audio command: {cmd}", "error")

    async def _record_speaker_profile(self, name: str):
        """Record 5s of mic audio, extract embedding, store as speaker profile."""
        try:
            import sounddevice as sd
            import numpy as np
            import wave
            import io
        except ImportError:
            print_status("sounddevice required: pip install sounddevice", "error")
            return

        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
            import torch, torchaudio
        except ImportError:
            print_status(
                "pyannote.audio required for speaker profiles:\n"
                "  pip install pyannote.audio\n"
                "  Also needs a HuggingFace token: export HF_TOKEN=hf_...",
                "error",
            )
            return

        DURATION = 5
        SR = 16000
        print_status(f"Recording {DURATION}s for '{name}' — speak now...", "info")
        await asyncio.sleep(0.3)

        loop = asyncio.get_event_loop()
        recording = await loop.run_in_executor(
            None,
            lambda: sd.rec(
                int(DURATION * SR), samplerate=SR, channels=1,
                dtype="float32",
                device=getattr(self, "_audio_device", None),
            )
        )
        await loop.run_in_executor(None, sd.wait)
        print_status("Recording done. Extracting embedding...", "info")

        # Build WAV
        pcm = (recording[:, 0] * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1);
            w.setsampwidth(2);
            w.setframerate(SR)
            w.writeframes(pcm)
        wav_bytes = buf.getvalue()

        # Extract embedding
        try:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            embed_model = PretrainedSpeakerEmbedding(
                "pyannote/embedding",
                use_auth_token=hf_token,
                device=torch.device("cpu"),
            )
            waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
            with torch.no_grad():
                emb = embed_model({"waveform": waveform, "sample_rate": sr})
            embedding = emb.squeeze().numpy()

            self._speaker_store.add(name, embedding)
            print_status(f"Speaker profile '{name}' saved.", "success")

        except Exception as e:
            print_status(f"Embedding extraction failed: {e}", "error")

    async def _restart_audio_player(self):
        """Stop current player, rebuild from settings, restart if was active."""
        was_running = (
            self.audio_player._task is not None
            and not self.audio_player._task.done()
            and self.audio_player._stop_event is not None
            and not self.audio_player._stop_event.is_set()
        )
        await self.audio_player.stop()
        self.audio_player = self._build_audio_player()

        # Speak-Tool-Bindings sind jetzt stale → alle Agents re-registrieren lassen
        self._audio_setup_agents.clear()

        if was_running or self.verbose_audio:
            await self.audio_player.start()
            # Sofort re-setup für aktiven Agent
            await self._ensure_audio_setup()

    async def _select_audio_device_interactive(self):
        """Print output devices, prompt for index."""
        import sounddevice as sd
        output_devs = [
            (i, dev) for i, dev in enumerate(sd.query_devices())
            if dev["max_output_channels"] > 0
        ]
        print_box_header("Select Output Device", "🔊")
        for i, dev in output_devs:
            print_box_content(f"[{i:2d}] {dev['name']}", "")
        print_box_footer()

        raw = await self._async_input("Device index (Enter = default): ")
        raw = raw.strip()
        if not raw:
            self._audio_device = None
            await self._restart_audio_player()
            print_status("Audio device → system default", "success")
            return

        try:
            idx = int(raw)
            valid_ids = [i for i, _ in output_devs]
            if idx not in valid_ids:
                print_status(f"Invalid index", "error")
                return
            self._audio_device = idx
            await self._restart_audio_player()
            dev_name = sd.query_devices()[idx]["name"]
            print_status(f"Audio device → [{idx}] {dev_name}", "success")
        except ValueError:
            print_status("Not a number", "error")

    async def _cmd_coder(self, args: list[str]):
        """Handle /coder commands for native code generation."""
        if not args:
            print_status("Usage: /coder <start|stop|stream|info|task|diff|accept|reject|test|files> [args]", "warning")
            return

        action = args[0].lower()

        # --- START ---
        if action == "start":
            import html

            # Import from coder.py
            from toolboxv2.mods.isaa.CodingAgent.coder import ProjectDetector, ProjectScaffolder

            # Step 1: target path (arg or cwd)
            target_raw = args[1] if len(args) >= 2 else self.init_dir
            target_path = Path(os.path.abspath(os.path.expanduser(target_raw)))

            # Step 2: detect
            try:
                ctx = ProjectDetector.detect(str(target_path))
            except Exception as e:
                print_status(f"Detection failed: {e}", "error")
                return

            # Step 3: show summary
            print_box_header("Coder Setup", "👨‍💻")
            for line in ctx.to_summary().splitlines():
                print_box_content(line, "info")
            for note in ctx.notes:
                print_box_content(f"• {note}", "")
            print_box_footer()

            # Step 4: resolve mode-specific decisions
            config_extras = {}
            scaffold_kind = None
            actual_root = ctx.root

            if ctx.mode == "new":
                # Ask: scaffold yes/no + which kind
                with patch_stdout():
                    c_print(HTML("\n<style fg='ansicyan'>📁 Leeres/neues Verzeichnis erkannt.</style>"))
                    kind_choice = await self.prompt_session.prompt_async(
                        HTML(
                            "<style fg='ansicyan'>Projekttyp? [1=python-uv  2=python-pip  3=node-npm  4=node-bun  5=web-static  6=skip]</style>\n❯ ")
                    )
                    kind_map = {"1": "python-uv", "2": "python-pip", "3": "node-npm",
                                "4": "node-bun", "5": "web-static", "6": None}
                    scaffold_kind = kind_map.get(kind_choice.strip(), None)

                    if scaffold_kind:
                        confirm = await self.prompt_session.prompt_async(
                            HTML(
                                f"<style fg='ansiyellow'>Scaffold '{scaffold_kind}' in {target_path}? [y/N]</style> ❯ ")
                        )
                        if confirm.strip().lower() in ("y", "yes", "j", "ja"):
                            created = ProjectScaffolder.scaffold(target_path, scaffold_kind)
                            print_status(f"Created {len(created)} files: {', '.join(created)}", "success")
                            # Re-detect after scaffold
                            ctx = ProjectDetector.detect(str(target_path))
                        else:
                            print_status("Skipping scaffold — starting on empty dir", "info")

            elif ctx.mode == "existing_file":
                # Single file — ask whether to also pull sibling dir or just this file
                with patch_stdout():
                    c_print(HTML(
                        f"\n<style fg='ansicyan'>📄 Einzelne Datei: {html.escape(str(target_path))}</style>\n"
                        f"<style fg='ansigray'>Scope wählen:</style>\n"
                        f"  1 = Nur diese Datei (isoliert)\n"
                        f"  2 = Datei + sibling tests/ Ordner\n"
                        f"  3 = Ganzer Parent-Ordner\n"
                    ))
                    scope = await self.prompt_session.prompt_async(HTML("<style fg='ansicyan'>❯ Scope [1]: </style>"))
                    scope = scope.strip() or "1"
                    if scope == "3":
                        actual_root = target_path.parent
                    elif scope == "2":
                        actual_root = target_path.parent
                        config_extras["scope_hint"] = f"focus on {target_path.name} and its tests only"
                    else:
                        actual_root = target_path.parent
                        config_extras["scope_hint"] = f"ONLY modify {target_path.name}"

            elif ctx.mode == "file_pair":
                # Source + test file auto-paired
                with patch_stdout():
                    c_print(HTML(
                        f"\n<style fg='ansicyan'>🔗 Datei-Paar erkannt:</style>\n"
                        f"  Source: {ctx.entry_files[0] if ctx.entry_files else '?'}\n"
                        f"  Test:   {ctx.test_paths[0] if ctx.test_paths else '?'}\n"
                    ))
                actual_root = ctx.root
                config_extras["scope_hint"] = f"work on pair: {', '.join(ctx.entry_files)}"

            elif ctx.mode == "subfolder":
                with patch_stdout():
                    c_print(HTML(
                        f"\n<style fg='ansiyellow'>⚠ Subfolder of git repo.</style>\n"
                        f"<style fg='ansigray'>Worktree-Modus:</style>\n"
                        f"  1 = Nur subfolder kopieren (schneller, isoliert)\n"
                        f"  2 = Git worktree auf repo root (volle history)\n"
                    ))
                    wt_mode = await self.prompt_session.prompt_async(HTML("<style fg='ansicyan'>❯ Modus [1]: </style>"))
                    wt_mode = wt_mode.strip() or "1"
                    # mode "1" is already the default in GitWorktree when root != git_root — handled automatically

            # Step 5: isolation / env question for project kinds that need it
            if ctx.kind in ("python-uv", "python-pip") and ctx.mode != "existing_file":
                with patch_stdout():
                    c_print(HTML("\n<style fg='ansicyan'>🐍 Python-Env:</style>"))
                    env_choice = await self.prompt_session.prompt_async(
                        HTML("<style fg='ansicyan'>Root-env (.venv in origin) oder worktree-env? [R/w]</style> ❯ ")
                    )
                    config_extras["use_root_env"] = env_choice.strip().lower() not in ("w", "worktree")

            # Step 6: enable run_tests?
            if ctx.has_tests:
                with patch_stdout():
                    test_enable = await self.prompt_session.prompt_async(
                        HTML("<style fg='ansicyan'>🧪 run_tests nach jedem Edit aktivieren? [y/N]</style> ❯ ")
                    )
                    config_extras["run_tests"] = test_enable.strip().lower() in ("y", "yes", "j", "ja")

            # Step 7: agent + instantiate
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                print_status(f"Initializing Coder on {actual_root}...", "progress")
                agent.verbose = True

                async def custom_ask_callback(question: str) -> str:
                    with patch_stdout():
                        if self.zen_plus_mode:
                            self._overlay = None
                            self.zen_plus_mode = False
                            await asyncio.sleep(1)
                        c_print(HTML(
                            f"\n<style fg='ansicyan'>🤖 Coder fragt:</style> <style fg='ansiyellow'>{html.escape(question)}</style>"))
                        answer = await self.prompt_session.prompt_async(
                            HTML("<style fg='ansicyan'>❯ Antwort: </style>"))
                        return answer

                config = {"ask_callback": custom_ask_callback, **config_extras}

                self.active_coder = CoderAgent(agent, str(actual_root), config=config)
                self.active_coder.print = ansi_c_print
                self.active_coder_path = str(actual_root)
                # Attach detected context for later commands (/coder info uses this)
                self.active_coder._project_ctx = ctx

                print_box_header("Coder Mode Activated", "👨‍💻")
                print_box_content(f"Target:   {actual_root}", "info")
                print_box_content(f"Mode:     {ctx.mode}", "info")
                print_box_content(f"Kind:     {ctx.kind}", "info")
                print_box_content(f"Agent:    {self.active_agent_name}", "info")
                if ctx.run_commands:
                    print_box_content(f"Run:      {ctx.run_commands[0]}", "")
                print_box_content("Commands:", "bold")
                for line in [
                    "  /coder task <instruction>     - Auto-implement",
                    "  <instruction>                 - Auto-implement (shortcut)",
                    "  @<instruction>                - Normal agent / next line",
                    "  /coder diff [file]            - Show changes",
                    "  /coder accept [file ...]      - Apply all or cherry-pick files",
                    "  /coder reject                 - Discard all changes",
                    "  /coder rollback [file ...]    - Reset all or specific files",
                    "  /coder test [cmd]             - Run tests in worktree",
                    "  /coder files                  - List worktree contents",
                    "  /coder info                   - Show paths + status",
                    "  /coder stream [on/off]        - Toggle streaming",
                    "  /coder stop                   - Exit (accept changes first!)",
                ]:
                    print_box_content(line, "")
                print_box_footer()
            except Exception as e:
                import traceback
                print_status(f"Failed to start coder: {e} {traceback.format_exc()}", "error")
            return
        elif action == "stream":
            if len(args) < 2:
                c_print("/coder stream on or off")
                return
            do = args[1]
            if do == "on":
                self.active_coder.stream_enabled = True
            else:
                self.active_coder.stream_enabled = False
            c_print(f"Coder streaming {'enabled' if self.active_coder.stream_enabled else 'disabled'}")

        elif action == "info":
            if not self.active_coder:
                print_status("No coder active. Use /coder start first.", "warning")
                return

            wt = self.active_coder.worktree
            wt_path = wt.worktree_path
            ctx = getattr(self.active_coder, "_project_ctx", None)

            print_box_header("Coder Info", "ℹ")

            # ── Paths & Worktree ──
            print_box_content(f"Origin (dein Repo):  {wt.origin_root}", "info")
            print_box_content(f"Worktree (Coder):    {wt_path}", "info")
            print_box_content(
                f"Git-Modus:           {'Ja (Branch: ' + wt._branch + ')' if wt._is_git else 'Nein (Kopie)'}", "info")

            # ── Project Context (wenn detected) ──
            if ctx:
                print_box_content(f"Projekt-Modus:       {ctx.mode}", "info")
                print_box_content(f"Projekt-Kind:        {ctx.kind}", "info")
                if ctx.entry_files:
                    print_box_content(f"Entries:             {', '.join(ctx.entry_files[:5])}", "")
                if ctx.package_files:
                    print_box_content(f"Packages:            {', '.join(ctx.package_files)}", "")
                if ctx.test_paths:
                    print_box_content(f"Tests:               {', '.join(ctx.test_paths[:3])}", "")
                if ctx.run_commands:
                    print_box_content(f"Run-Commands:        {' | '.join(ctx.run_commands[:2])}", "")

            # ── Agent & Model ──
            print_box_content(f"Agent:               {self.active_agent_name}", "info")
            print_box_content(f"Model:               {self.active_coder.model}", "info")
            print_box_content(f"Stream:              {self.active_coder.stream_enabled}", "info")

            # ── Runtime Config Flags ──
            flags = []
            if getattr(self.active_coder, "run_tests", False):
                flags.append("run_tests")
            if getattr(self.active_coder, "use_root_env", True):
                flags.append("root-env")
            else:
                flags.append("worktree-env")
            if getattr(self.active_coder, "ask_enabled", False):
                flags.append("ask")
            if getattr(self.active_coder, "sync_enabled", True):
                flags.append(f"sync({self.active_coder.sync_interval:.0f}s)")
            scope_hint = self.active_coder.config.get("scope_hint") if self.active_coder.config else None
            if scope_hint:
                flags.append(f"scoped")
            if flags:
                print_box_content(f"Flags:               {', '.join(flags)}", "")
            if scope_hint:
                print_box_content(f"Scope-Hint:          {scope_hint[:80]}", "")

            # ── Sub-Agents (Multi-Agent v5) ──
            if getattr(self.active_coder, "_sub_agents_ready", False):
                coders = self.active_coder._coder_names
                print_box_content(f"Sub-Agents:          planner + {len(coders)} coder + validator", "info")

            # ── Token Tracker ──
            tracker = getattr(self.active_coder, "tracker", None)
            if tracker:
                limit = tracker.limit
                used = tracker.total_tokens
                pct = (used / limit * 100) if limit else 0
                print_box_content(
                    f"Tokens:              {used:,} / {limit:,} ({pct:.1f}%)  "
                    f"compressions: {tracker.compressions_done}", "info")

            print_separator()

            # ── Changed Files ──
            try:
                changed = await wt.changed_files()
                if changed:
                    print_box_content(f"Geänderte Dateien ({len(changed)}):", "bold")
                    for f in changed:
                        src = wt_path / f
                        size = src.stat().st_size if src.exists() else 0
                        # Check if file also exists in origin to mark NEW vs MODIFIED
                        dst = wt.origin_root / f
                        marker = "●" if dst.exists() else "+"
                        c_print(f"  {marker} {f}  ({size:,} bytes)")
                else:
                    print_box_content("Keine Änderungen im Worktree.", "info")
            except Exception as e:
                print_box_content(f"Fehler beim Lesen: {e}", "error")

            # ── Last Tasks from Memory ──
            if self.active_coder.memory.reports:
                print_separator()
                print_box_content("Letzte Tasks:", "bold")
                for r in self.active_coder.memory.reports[-3:]:
                    status = "✓" if r.get("success") else "✗"
                    files = ", ".join(r.get("changed_files", [])[:5])
                    task_str = r.get("task", "?")
                    c_print(f"  {status} {task_str[:60]}{'...' if len(task_str) > 60 else ''}")
                    if files:
                        c_print(f"    → {files}")

            print_box_footer()

        # --- STOP ---
        elif action == "stop":
            if not self.active_coder:
                print_status("Coder Mode is not active.", "warning")
                return

            # Warn about pending changes (bestehender Code)
            try:
                pending = await self.active_coder.worktree.changed_files()
                if pending:
                    print_status(f"⚠ {len(pending)} uncommitted file(s) will be lost:", "warning")
                    for f in pending[:10]:
                        c_print(f"  - {f}")
                    if len(pending) > 10:
                        c_print(f"  ... and {len(pending) - 10} more")
            except Exception:
                pass

            try:
                await self.active_coder.cleanup_agents()
            except Exception:
                pass

            # ── NEU: Swarm-Sub-TaskViews bereinigen ──
            parent_tv = None
            for tid, tv in list(self._task_views.items()):
                if tv.is_swarm_summary and not tv.is_swarm_sub:
                    parent_tv = tv
                    break

            if parent_tv:
                for sub_tid in list(parent_tv.sub_task_ids.values()):
                    self._task_views.pop(sub_tid, None)
                    exc = self.all_executions.get(sub_tid)
                    if exc and exc.status == "running":
                        exc.status = "cancelled"

            # Bestehender Cleanup
            parent_prefix = None
            for tid, exc in list(self.all_executions.items()):
                if exc.kind == "coder_sub":
                    if exc.status == "running":
                        exc.status = "cancelled"
                    tv = self._task_views.get(tid)
                    if tv:
                        tv.status = "cancelled"
            try:
                self.active_coder.worktree.cleanup()
            except Exception:
                pass
            self.active_coder = None
            self.active_coder_path = self.init_dir
            print_status("Coder Mode deactivated.", "success")

        # --- ACTIONS (require active coder) ---
        else:
            if not self.active_coder:
                print_status("Coder not active. Use '/coder start <path>' first.", "error")
                return

            wt = self.active_coder.worktree

            if action == "task":
                if len(args) < 2:
                    print_status("Usage: /coder task <instruction>", "warning")
                    return
                task_prompt = " ".join(args[1:])
                try:
                    print_status(f"Coder working on: {task_prompt}", "progress")
                except UnicodeEncodeError:
                    task_prompt = task_prompt.encode('utf-8').decode('utf-8', errors="replace")

                # State-Halter für die Task ID
                _tid_holder = [None]

                # 1. Eigener Log-Handler: Übersetzt Coder-Logs in UI-Chunks für Zen+
                coder_state = {"current_tool": "unknown"}

                # 1. Eigener Log-Handler: Übersetzt Coder-Logs in UI-Chunks für Zen+
                def coder_log_handler(section: str, content: str):
                    tid = _tid_holder[0]
                    if not tid:
                        return
                    # Nur CoderAgent-eigene Sections weiterleiten
                    if section in ("PHASE", "EDIT", "EDIT-ERR", "SYNC", "AGENTS", "CLEANUP", "VFS"):
                        self._ingest_chunk(tid, {"type": "content", "chunk": f"\n[{section}] {content}\n"})
                    elif section in ("ERROR",):
                        self._ingest_chunk(tid, {"type": "error", "chunk": content})

                    # Stream-Callback: DEAKTIVIERT — row_chunk_fun übernimmt alles
                    # (stream_callback würde Duplikate erzeugen)

                self.active_coder.log_handler = coder_log_handler
                self.active_coder.stream_callback = None  # ← KEY: kein Doppel-Stream
                self.active_coder.stream_enabled = False  # ← irrelevant wenn callback=None

                # Sub-Agent Task-View Management
                _sub_task_ids: dict[str, str] = {}  # agent_name → child_task_id

                # ── Summary-Flag auf Parent-TaskView setzen ──
                # (passiert NACHDEM _create_execution die TaskView angelegt hat)

                def _coder_chunk_router(chunk: dict):
                    parent_tid = _tid_holder[0]
                    if not parent_tid:
                        return

                    parent_tv = self._task_views.get(parent_tid)
                    if parent_tv is None:
                        return

                    # Summary-Flag lazy setzen (beim ersten Chunk ist die TaskView garantiert da)
                    if not parent_tv.is_swarm_summary:
                        parent_tv.is_swarm_summary = True
                        parent_tv.agent_name = "coder_swarm"
                        parent_tv.swarm_phase = "init"

                    ctype = chunk.get("type", "")
                    sub_agent = chunk.get("_sub_agent_id", "")

                    # ═══ Swarm-Phase auf Summary ═══
                    if ctype == "swarm_phase":
                        parent_tv.swarm_phase = chunk.get("swarm_phase", "")
                        info = chunk.get("swarm_info", "")
                        parent_tv.narrator_msg = (f"🐝 {parent_tv.swarm_phase}: {info}"
                                                  if info else f"🐝 {parent_tv.swarm_phase}")
                        # No-op ingest auf Parent → triggert invalidate
                        self._ingest_chunk(parent_tid, {"type": "status", "status": parent_tv.status})
                        return

                    # ═══ Sub-Agent Start: eigene Top-Level-TaskView ═══
                    if ctype == "swarm_sub_start" and sub_agent:
                        sub_tid = f"swarm_sub::{sub_agent}"
                        if sub_tid not in self._task_views:
                            self._task_views[sub_tid] = TaskView(
                                task_id=sub_tid,
                                agent_name=sub_agent,
                                query=chunk.get("query", "")[:80],
                                is_swarm_sub=True,
                                swarm_parent_id=parent_tid,
                                max_iter=chunk.get("max_iter", 0),
                            )
                            parent_tv.sub_task_ids[sub_agent] = sub_tid
                            parent_tv.sub_agents[sub_agent] = 0  # 0 = running
                            parent_tv._sub_color(sub_agent)

                            # Lightweight execution (kein async_task — Parent kontrolliert Lifecycle)
                            self.all_executions[sub_tid] = ExecutionTask(
                                task_id=sub_tid,
                                agent_name=sub_agent,
                                query=chunk.get("query", "")[:80],
                                kind="coder_sub",
                                async_task=None,
                                is_focused=False,
                            )
                        self._ingest_chunk(sub_tid, {"type": "status", "status": "running"})
                        return

                    # ═══ Sub-Agent Done ═══
                    if ctype == "swarm_sub_done" and sub_agent:
                        sub_tid = parent_tv.sub_task_ids.get(sub_agent)
                        if sub_tid:
                            parent_tv.sub_agents[sub_agent] = 1  # 1 = done
                            sub_tv = self._task_views.get(sub_tid)
                            if sub_tv and sub_tv.status == "running":
                                sub_tv.status = "completed"
                                sub_tv.completed_at = time.time()
                            exc = self.all_executions.get(sub_tid)
                            if exc:
                                exc.status = "completed"
                            self._ingest_chunk(sub_tid, {"type": "status", "status": "completed"})
                        return

                    # ═══ Reguläre Chunks mit Sub-Agent-Tag ═══
                    if sub_agent:
                        sub_tid = parent_tv.sub_task_ids.get(sub_agent)
                        if sub_tid:
                            clean_chunk = {k: v for k, v in chunk.items()
                                           if k not in ("_sub_agent_id", "_swarm_phase")}
                            self._ingest_chunk(sub_tid, clean_chunk)

                            # ── Aggregation: Parent-Summary aus allen Sub-TaskViews ableiten ──
                            sub_tvs = [
                                self._task_views[stid]
                                for stid in parent_tv.sub_task_ids.values()
                                if stid in self._task_views
                            ]
                            if sub_tvs:
                                parent_tv.iteration = sum(s.iteration for s in sub_tvs)
                                parent_tv.max_iter = sum(s.max_iter for s in sub_tvs)
                                parent_tv.tokens_used = sum(s.tokens_used for s in sub_tvs)
                                parent_tv.tokens_max = sum(s.tokens_max for s in sub_tvs)

                                # Phase/Tool vom aktivsten Sub übernehmen
                                active = next((s for s in sub_tvs if s.status == "running"), None)
                                if active:
                                    parent_tv.phase = active.phase
                                    parent_tv.last_tool = active.last_tool
                                    parent_tv.last_tool_ok = active.last_tool_ok
                                    parent_tv.last_tool_info = active.last_tool_info

                            # Sub-Status-Flag pflegen
                            if ctype in ("done", "final_answer"):
                                parent_tv.sub_agents[sub_agent] = 1
                            elif ctype == "error":
                                parent_tv.sub_agents[sub_agent] = 2
                        return

                    # ═══ Chunks ohne Sub-Tag → Parent direkt ═══
                    self._ingest_chunk(parent_tid, chunk)

                self.active_coder.row_chunk_fun = _coder_chunk_router
                # 3. Hintergrund-Task Wrapper für den Coder
                async def _run_coder_bg():
                    result = await self.active_coder.execute(task_prompt)

                    # Task abschließen
                    self._ingest_chunk(_tid_holder[0], {"type": "done", "success": result.success})
                    self._ingest_chunk(_tid_holder[0], {"type": "final_answer", "answer": result.message})

                    # Kleine Zusammenfassung fürs Terminal
                    with patch_stdout():
                        if result.success and result.changed_files:
                            c_print(HTML(
                                f"<style fg='ansicyan'>Modified files: {', '.join(result.changed_files)}</style>"))
                            c_print(HTML(
                                "<style fg='ansiyellow'>Use '/coder diff' to review, '/coder accept' to apply.</style>"))

                    if not result.success:
                        raise Exception(result.message)
                    return result.message
                c_print(HTML("<style fg='ansimagenta'>⚡ Starting Code Generation Task...</style>"))

                # 4. Als offiziellen Background-Task registrieren
                async_task = asyncio.create_task(_run_coder_bg())
                exc = self._create_execution(
                    kind="coder",
                    agent_name=f"{self.active_agent_name}_coder",
                    query=task_prompt,
                    async_task=async_task,
                    take_focus=True
                )
                _tid_holder[0] = exc.task_id

                parent_tv = self._task_views.get(exc.task_id)
                if parent_tv:
                    parent_tv.is_swarm_summary = True
                    parent_tv.agent_name = "coder_swarm"

                    # 5. Standard ISAA Task-Lifecycle anhängen (verarbeitet Focus & UI Updates bei Abschluss)
                async_task.add_done_callback(
                    lambda fut: self._on_agent_task_done(exc.task_id, fut)
                )

                # Sofort zurückkehren, damit der Prompt sofort wieder frei ist
                return

            elif action == "diff":
                try:
                    wt_path = wt.worktree_path
                    if wt._is_git:
                        # Windows: asyncio.create_subprocess_shell wirft NotImplementedError
                        # → subprocess.run stattdessen
                        subprocess.run(["git", "add", "-A"], cwd=str(wt_path),
                                       capture_output=True, encoding="utf-8", errors="replace")
                        target = args[1] if len(args) > 1 else ""
                        cmd = ["git", "diff", "--cached", "--color"]
                        if target:
                            cmd.append(target)
                        result = subprocess.run(cmd, cwd=str(wt_path),
                                                capture_output=True, encoding="utf-8", errors="replace")
                        if result.stdout.strip():
                            print(result.stdout)
                        else:
                            print_status("No changes.", "info")
                    else:
                        changed = await wt.changed_files()
                        if not changed:
                            print_status("No changes.", "info")
                        else:
                            import difflib
                            filter_file = args[1] if len(args) > 1 else None
                            for rel in changed:
                                if filter_file and rel != filter_file: continue
                                orig = wt.origin_root / rel
                                curr = wt.path / rel
                                old = orig.read_text(encoding="utf-8",
                                                     errors="replace").splitlines() if orig.exists() else []
                                new = curr.read_text(encoding="utf-8",
                                                     errors="replace").splitlines() if curr.exists() else []
                                diff = difflib.unified_diff(old, new, fromfile=f"a/{rel}", tofile=f"b/{rel}",
                                                            lineterm="")
                                for line in diff:
                                    c_print(line)
                except Exception as e:
                    print_status(f"Diff error: {e}", "error")
                    import traceback
                    c_print(traceback.format_exc())

            elif action == "accept":
                # /coder accept              → apply all (git merge or copy)
                # /coder accept f1.py f2.py  → cherry-pick specific files
                try:
                    if len(args) > 1:
                        # Cherry-pick mode
                        files = args[1:]
                        available = await wt.changed_files()
                        invalid = [f for f in files if f not in available]
                        if invalid:
                            print_status(f"Not changed in worktree: {', '.join(invalid)}", "error")
                            if available:
                                print_status("Available:", "info")
                                for f in available: c_print(f"  {f}")
                            return

                        applied = await wt.apply_files(files)
                        for f in applied:
                            c_print(f"  ✓ {f}")
                        print_status(f"Cherry-picked {len(applied)} file(s).", "success")
                    else:
                        # Full apply
                        n = await wt.apply_back()
                        if n == -1:
                            print_status("Merged via git.", "success")
                        else:
                            print_status(f"Applied {n} file(s).", "success")

                    # Reset worktree for next task
                    wt.cleanup()
                    wt.setup()
                    print_status("Worktree reset for next task.", "info")
                except subprocess.CalledProcessError as e:
                    print_status(f"Merge conflict! Resolve manually in {wt.origin_root}", "error")
                    if e.stderr:
                        for line in e.stderr.splitlines()[:10]:
                            c_print(f"  {line}")
                except Exception as e:
                    print_status(f"Accept failed: {e}", "error")

            elif action == "reject":
                print_status("Discarding all changes...", "warning")
                wt.cleanup()
                wt.setup()
                print_status("Worktree reset.", "success")

            elif action == "rollback":
                # /coder rollback              → reset entire worktree
                # /coder rollback f1.py f2.py  → reset specific files
                try:
                    if len(args) > 1:
                        files = args[1:]
                        await wt.rollback(files)
                        for f in files:
                            c_print(f"  ↩ {f}")
                        print_status(f"Rolled back {len(files)} file(s).", "success")
                    else:
                        changed = await wt.changed_files()
                        if not changed:
                            print_status("Nothing to rollback.", "info")
                            return
                        print_status(f"Rolling back {len(changed)} file(s)...", "warning")
                        await wt.rollback()
                        print_status("Full rollback complete.", "success")
                except Exception as e:
                    print_status(f"Rollback failed: {e}", "error")

            elif action == "test":
                cmd = " ".join(args[1:]) if len(args) > 1 else "pytest"
                print_status(f"Running in worktree: {cmd}", "progress")
                try:
                    print_separator(char=".")
                    proc = await asyncio.create_subprocess_shell(
                        cmd, cwd=str(wt.worktree_path), stdout=None, stderr=None)
                    await proc.wait()
                    print_separator(char=".")
                    if proc.returncode == 0:
                        print_status("Tests passed.", "success")
                    else:
                        print_status(f"Tests failed (exit {proc.returncode})", "error")
                except Exception as e:
                    print_status(f"Test error: {e}", "error")

            elif action == "files":
                wt_path = wt.worktree_path
                changed = set(await wt.changed_files())
                for root, dirs, files in os.walk(wt_path):
                    if ".git" in root: continue
                    level = root.replace(str(wt_path), "").count(os.sep)
                    indent = "  " * level
                    c_print(f"{indent}{os.path.basename(root)}/")
                    for f in files:
                        rel = str((Path(root) / f).relative_to(wt_path))
                        marker = " ●" if rel in changed else ""
                        c_print(f"{indent}  {f}{marker}")

            else:
                print_status(f"Unknown: {action}. Commands: task|diff|accept|reject|rollback|test|files|stop", "error")

    async def _cmd_skill(self, args: list[str]):
        """Handle /skill commands."""
        if not args:
            print_status("Usage: /skill <list|show|edit|delete|boost|merge|import|export> [id]", "warning")
            return

        action = args[0].lower()

        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            # Access the SkillsManager via the ExecutionEngine
            engine = agent._get_execution_engine()
            sm = engine.skills_manager
            if sm.export_skills is None:
                from toolboxv2.mods.isaa.base.Agent.skills import add_anthropic_skill_io
                add_anthropic_skill_io(sm)
        except Exception as e:
            print_status(f"Could not access skills for agent '{self.active_agent_name}'", "error")
            import traceback
            c_print(traceback.format_exc())
            return


        if action == "show":
            if len(args) < 2:
                print_status("Usage: /skill show <skill_id>", "warning")
                return

            skill_id = args[1]
            skill = sm.skills.get(skill_id)

            if not skill:
                print_status(f"Skill '{skill_id}' not found.", "error")
                return

            print_box_header(f"Skill: {skill.name}", "🧠")
            print_box_content(f"ID: {skill.id}", "info")
            print_box_content(f"Source: {skill.source} | Confidence: {skill.confidence:.2f}", "info")
            print_box_content(f"Triggers: {', '.join(skill.triggers)}", "info")
            print_separator()
            print_status("Instruction:", "info")
            print_code_block(skill.instruction, "markdown")
            print_box_footer()

        elif action == "export":
            if len(args) < 3:
                print_status("Usage: /skill export <skill_id/all> <output_path>", "warning")
                return
            skill_id = args[1]
            output_path = args[2]
            if skill_id == "all":
                sm.export_skills(output_path)
            else:
                sm.export_to_skill_file(skill_id, output_path)
            print_status(f"Skill '{skill_id}' exported to '{output_path}'", "success")

        elif action == "import":
            if len(args) < 2:
                print_status("Usage: /skill import <input_path>", "warning")
                return
            input_path = args[1]
            results = sm.import_skills(input_path, True)
            for name, success in results.items():
                print(f"{'✅' if success else '❌'} {name}")
            print_status(f"Skill '{input_path}' imported", "success")

        elif action == "delete":
            if len(args) < 2:
                print_status("Usage: /skill delete <skill_id>", "warning")
                return

            skill_id = args[1]
            if skill_id in sm.skills:
                # Prevent deleting predefined skills unless forced (optional safety)
                if sm.skills[skill_id].source == "predefined":
                    print_status("Cannot delete predefined skills.", "warning")
                    return

                del sm.skills[skill_id]
                # Trigger save via checkpoint manager implicitly later or set dirty flag
                sm._skill_embeddings_dirty = True
                print_status(f"Skill '{skill_id}' deleted.", "success")
            else:
                print_status(f"Skill '{skill_id}' not found.", "error")
        elif action == "list":
            show_inactive_only = "--inactive" in args

            print_box_header(f"Skills: {self.active_agent_name}", "🧠")

            max_id_length = max(len(skill.id) for skill in sm.skills.values())
            max_name_length = max(len(skill.name) for skill in sm.skills.values())
            columns = [("ID", max_id_length), ("Name", max_name_length), ("Src", 10), ("Conf", 6), ("Active", 8)]
            widths = [max_id_length, max_name_length, 10, 6, 8]
            print_table_header(columns, widths)

            skills = sm.skills.values()

            if show_inactive_only:
                skills = [
                    s for s in skills
                    if s.source == "learned" and not s.is_active()
                ]

            sorted_skills = sorted(
                skills,
                key=lambda s: (s.source != "learned", s.confidence),
                reverse=False
            )

            for skill in sorted_skills:
                disp_id = skill.id if len(skill.id) < 24 else skill.id[:22] + ".."

                source_style = "green" if skill.source == "learned" else "grey"
                conf_style = "green" if skill.confidence > 0.8 else "yellow"
                active_style = "green" if skill.is_active() else "grey"

                print_table_row(
                    [
                        disp_id,
                        skill.name,
                        skill.source,
                        f"{skill.confidence:.2f}",
                        "YES" if skill.is_active() else "NO"
                    ],
                    widths,
                    ["cyan", "white", source_style, conf_style, active_style]
                )

            print_box_footer()

        elif action == "merge":
            if len(args) < 3:
                print_status("Usage: /skill merge <keep_id> <remove_id>", "warning")
                return

            keep_id, remove_id = args[1], args[2]

            keep_skill = sm.skills.get(keep_id)
            remove_skill = sm.skills.get(remove_id)

            if not keep_skill or not remove_skill:
                print_status("One or both skills not found.", "error")
                return

            if keep_id == remove_id:
                print_status("Cannot merge a skill into itself.", "warning")
                return

            # Merge logic
            keep_skill.merge_with(remove_skill)

            del sm.skills[remove_id]
            sm._skill_embeddings_dirty = True

            print_status(
                f"Merged skill '{remove_skill.name}' into '{keep_skill.name}'.",
                "success"
            )

        elif action == "boost":
            if len(args) < 3:
                print_status("Usage: /skill boost <skill_id> <amount>", "warning")
                return

            skill_id = args[1]

            try:
                amount = float(args[2])
            except ValueError:
                print_status("Boost amount must be a float (e.g. 0.3).", "error")
                return

            skill = sm.skills.get(skill_id)
            if not skill:
                print_status(f"Skill '{skill_id}' not found.", "error")
                return

            old_conf = skill.confidence
            skill.confidence = min(1.0, skill.confidence + amount)

            sm._skill_embeddings_dirty = True

            print_status(
                f"Skill '{skill.name}' boosted: {old_conf:.2f} → {skill.confidence:.2f}",
                "success"
            )


        elif action == "edit":
            if len(args) < 2:
                print_status("Usage: /skill edit <skill_id>", "warning")
                return

            skill_id = args[1]
            skill = sm.skills.get(skill_id)

            if not skill:
                print_status(f"Skill '{skill_id}' not found.", "error")
                return

            print_box_header(f"Editing: {skill.name}", "✏️")
            print_status("Current Instruction:", "info")
            print_code_block(skill.instruction, "markdown")
            print_separator()

            print_status("Enter NEW instruction (end with empty line) or type 'CANCEL':", "configure")

            lines: list[str] = []
            if self.prompt_session is not None:
                while True:
                    line = await self.prompt_session.prompt_async(
                        HTML("<style fg='grey'>... </style>")
                    )
                    if not line.strip():
                        break
                    if line.strip().upper() == "CANCEL":
                        print_status("Edit cancelled.", "warning")
                        return
                    lines.append(line)

            new_instruction = "\n".join(lines)
            if new_instruction:
                skill.instruction = new_instruction
                sm._skill_embeddings_dirty = True  # Force re-embedding
                print_status(f"Skill '{skill.name}' updated.", "success")

                # Try to save agent state
                try:
                    await agent.save()
                    print_status("Agent state saved.", "data")
                except Exception as e:
                    print_status(f"Warning: Could not save to disk immediately: {e}", "warning")

        else:
            print_status(f"Unknown skill action: {action}", "error")

    async def _cmd_feature(self, args: list[str]):
        """Handle /feature commands."""
        if len(args) < 2:
            print_status("Usage: /feature <action> [options]", "warning")
            return

        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
        except Exception as e:
            print_status(f"Could not access agent '{self.active_agent_name}'", "error")
            import traceback
            c_print(traceback.format_exc())
            return

        self.feature_manager.set_agent(agent)

        action = args[0].lower()
        if action == "list":
            print_box_header("Available Features", "📦")
            for feature in self.feature_manager.list_features():
                print_status(f"{feature}", "info")
        elif action == "enable":
            if len(args) < 2:
                print_status("Usage: /feature enable <feature> [options]", "warning")
                return
            feature = args[1].lower()
            if feature not in self.feature_manager.list_features():
                print_status(f"Feature '{feature}' not found.", "error")
                return
            await self.feature_manager.enable(feature)
            print_status(f"Feature '{feature}' enabled.", "success")
        elif action == "disable":
            if len(args) < 2:
                print_status("Usage: /feature disable <feature> [options]", "warning")
                return
            feature = args[1].lower()
            if feature not in self.feature_manager.list_features():
                print_status(f"Feature '{feature}' not found.", "error")
                return
            # test active
            if self.feature_manager.is_enabled(feature):
                print_status(f"Feature '{feature}' is active. Please disable it first.", "warning")
                return
            self.feature_manager.disable(feature)
            print_status(f"Feature '{feature}' disabled.", "success")
        else:
            print_status(f"Unknown feature action: {action}", "error")

    async def _cmd_chain(self, args: list[str]):
        """Handle /chain commands — human-facing chain store management."""
        from toolboxv2.mods.isaa.base.chain.chain_tools import (
            ChainStore, StoredChain, ChainDSLParser, generate_chain_id,
        )
        store = ChainStore(Path(self.app.data_dir) / "chains")

        if not args or args[0] == "list":
            chains = store.list_all()
            if not chains:
                print_status("No chains. Enable 'chain' feature, then use create_validate_chain tool.", "info")
                return
            print_box_header(f"Chains ({len(chains)})", "⛓")
            widths = [14, 22, 6, 8, 5]
            print_table_header([("ID", 14), ("Name", 22), ("Valid", 6), ("Accepted", 8), ("Runs", 5)], widths)
            for c in chains:
                print_table_row(
                    [c.id[:14], c.name[:22],
                     "YES" if c.is_valid else "NO",
                     "YES" if c.accepted else "NO",
                     str(c.run_count)],
                    widths,
                    ["cyan", "white",
                     "green" if c.is_valid else "red",
                     "green" if c.accepted else "amber",
                     "grey"],
                )
            print_box_footer()
            return

        action = args[0].lower()

        def _resolve(key: str):
            return store.get(key) or store.get_by_name(key)

        if action == "show":
            if len(args) < 2:
                print_status("Usage: /chain show <id|name>", "warning");
                return
            c = _resolve(args[1])
            if not c:
                print_status(f"Not found: {args[1]}", "error");
                return
            print_box_header(f"Chain: {c.name}", "⛓")
            print_box_content(
                f"ID: {c.id}  Valid: {c.is_valid}  Accepted: {c.accepted}  Runs: {c.run_count}", "info"
            )
            if c.description:
                print_box_content(c.description, "")
            if c.tags:
                print_box_content(f"Tags: {', '.join(c.tags)}", "")
            if c.validation_errors:
                print_box_content(f"Errors: {'; '.join(c.validation_errors)}", "error")
            print_separator()
            print_code_block(c.dsl, "text")
            print_box_footer()

        elif action == "accept":
            if len(args) < 2:
                print_status("Usage: /chain accept <id|name>", "warning");
                return
            c = _resolve(args[1])
            if not c:
                print_status(f"Not found: {args[1]}", "error");
                return
            store.accept(c.id)
            print_status(f"Chain '{c.name}' accepted — ready to run.", "success")

        elif action == "delete":
            if len(args) < 2:
                print_status("Usage: /chain delete <id|name>", "warning");
                return
            c = _resolve(args[1])
            if not c:
                print_status(f"Not found: {args[1]}", "error");
                return
            confirm = input(f"Delete '{c.name}'? (y/N): ").strip().lower()
            if confirm == "y":
                store.delete(c.id)
                print_status(f"Chain '{c.name}' deleted.", "success")

        elif action == "edit":
            if len(args) < 2:
                print_status("Usage: /chain edit <id|name>", "warning");
                return
            c = _resolve(args[1])
            if not c:
                print_status(f"Not found: {args[1]}", "error");
                return
            print_box_content("Current DSL:", "info")
            print_code_block(c.dsl, "text")
            print_status("Enter new DSL (empty line = done, CANCEL = abort):", "configure")
            lines: list[str] = []
            if self.prompt_session:
                while True:
                    line = await self.prompt_session.prompt_async(
                        HTML("<style fg='grey'>dsl> </style>")
                    )
                    if not line.strip():
                        break
                    if line.strip().upper() == "CANCEL":
                        print_status("Edit cancelled.", "warning");
                        return
                    lines.append(line)
            new_dsl = "\n".join(lines).strip()
            if not new_dsl:
                print_status("Nothing entered — edit cancelled.", "warning");
                return
            parser = ChainDSLParser()
            _, errors = parser.parse(new_dsl)
            c.dsl = new_dsl
            c.is_valid = not errors
            c.validation_errors = errors
            c.accepted = False  # nach Edit neu akzeptieren
            c.id = generate_chain_id(c.name, new_dsl)
            store.save(c)
            status_txt = "valid" if c.is_valid else f"INVALID: {'; '.join(errors)}"
            print_status(f"Chain updated ({status_txt}). New ID: {c.id}", "success" if c.is_valid else "warning")

        elif action == "export":
            if len(args) < 3:
                print_status("Usage: /chain export <id|name> <file.json>", "warning");
                return
            c = _resolve(args[1])
            if not c:
                print_status(f"Not found: {args[1]}", "error");
                return
            out_path = Path(args[2])
            out_path.write_text(json.dumps(c.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            print_status(f"Exported '{c.name}' → {out_path}", "success")

        elif action == "import":
            if len(args) < 2:
                print_status("Usage: /chain import <file.json>", "warning");
                return
            in_path = Path(args[1])
            if not in_path.exists():
                print_status(f"File not found: {in_path}", "error");
                return
            data = json.loads(in_path.read_text(encoding="utf-8"))
            c = StoredChain.from_dict(data)
            c.accepted = False  # Sicherheit: nach Import neu akzeptieren
            store.save(c)
            print_status(
                f"Imported '{c.name}' (ID: {c.id}). Use '/chain accept {c.id}' before running.",
                "success",
            )

        elif action == "run":
            if len(args) < 2:
                print_status("Usage: /chain run <id|name> [input_data]", "warning");
                return
            name_or_id = args[1]
            input_data = " ".join(args[2:]) if len(args) > 2 else ""
            try:
                agent = await self.isaa_tools.get_agent(self.active_agent_name)
                tool = agent.tool_manager.get("run_chain")
                if not tool:
                    print_status("Chain feature not enabled. Run: /feature enable chain", "error");
                    return
                print_status(f"Running chain '{name_or_id}'...", "progress")
                result = await tool.func(name_or_id, input_data, False)
                c_print(result)
            except Exception as e:
                print_status(f"Error: {e}", "error")

        else:
            print_status(
                f"Unknown: {action}. Commands: list show accept delete edit export import run",
                "error",
            )

    async def _cmd_tools(self, args: list[str]):
        """Intelligentes Tool-Management: Erkennt automatisch Namen oder Kategorien."""
        if not args:
            print_status("Usage: /tools <list|all|info|health|enable|disable|enable-all|disable-all> [name/category]", "warning")
            return

        action = args[0].lower()

        # Helper zum Laden des Agenten
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
        except:
            return print_status(f"Agent '{self.active_agent_name}' nicht erreichbar.", "error")

        if not hasattr(agent, "tool_manager"): return
        if not hasattr(agent, "_disabled_tools"): agent._disabled_tools = {}

        tm = agent.tool_manager
        disabled_map = agent._disabled_tools
        sys_cats = ["sys", "system", "core", "base", "agent_management", "task_management"]

        # Interne Helper
        def get_all_cats():
            cats = set(tm._category_index.keys())
            for t in disabled_map.values():
                for c in (t.category if isinstance(t.category, list) else []): cats.add(str(c).lower())
            return cats

        def get_tool_cats(t_obj):
            return [str(c).lower() for c in t_obj.category] if isinstance(t_obj.category, list) else ["uncategorized"]

        async def perform_action(t_name, mode):
            """Führt die eigentliche Verschiebung durch."""
            if mode == "enable":
                if t_name in disabled_map:
                    entry = disabled_map.pop(t_name)
                    tm._registry[t_name] = entry
                    tm._update_indexes(entry)
                    if tm._rule_set: tm._sync_tool_to_ruleset(entry)
                    return True, t_name
            else:  # disable
                entry = tm.get(t_name)
                if entry:
                    # System-Schutz
                    if any(c in sys_cats for c in get_tool_cats(entry)):
                        return False, f"{t_name} (System-Schutz)"
                    disabled_map[t_name] = entry
                    tm.unregister(t_name)
                    return True, t_name
            return False, None

        # --- LOGIK ---

        C = {
                "dim": "#6b7280", "cyan": "#67e8f9", "green": "#4ade80",
                "red": "#f87171", "amber": "#fbbf24", "white": "#e5e7eb",
                "bright": "#ffffff", "blue": "#60a5fa", "purple": "#a78bfa",
                "deep": "#374151", "pink": "#f472b6", "teal": "#2dd4bf",
                "orange": "#fb923c",
            }

        if action == "info":
            if len(args) < 2: return print_status("Usage: /tools info <tool_name>", "warning")
            t_name = args[1]
            t_obj = tm.get(t_name) or disabled_map.get(t_name)

            if not t_obj: return print_status(f"Tool '{t_name}' nicht gefunden.", "error")

            print_box_header(f"Tool Info: {t_obj.name}", "ℹ")
            status = "ENABLED" if tm.exists(t_obj.name) else "DISABLED"
            col = C["green"] if status == "ENABLED" else C["dim"]

            print_box_content(f"Status: {status} | Source: {t_obj.source} | Server: {t_obj.server_name or 'N/A'}",
                              "info")

            # Flags kompakt
            f_list = [f"<style fg='{C['cyan']}'>{k}</style>" for k, v in t_obj.flags.items() if v]
            print_box_content(f"Flags: {' '.join(f_list) if f_list else 'none'}", "info")

            print_separator()
            hs = getattr(t_obj, "health_status", None) or "UNKNOWN"
            he = getattr(t_obj, "health_error", None)
            lhc = getattr(t_obj, "last_health_check", None)
            hs_col = {"HEALTHY": C["green"], "GUARANTEED": C["teal"],
                      "DEGRADED": C["amber"], "FAILED": C["red"]}.get(hs, C["dim"])
            lhc_str = lhc.strftime("%Y-%m-%d %H:%M:%S") if lhc else "never"
            print_box_content(
                f"Health: <style fg='{hs_col}'>{hs}</style>  |  Last check: {lhc_str}"
                + (f"  |  Error: {he}" if he else ""),
                "info"
            )
            print_status("Description:", "info")
            print_box_content(t_obj.description, "")

            print_status("Arguments Schema:", "configure")
            print_code_block(t_obj.args_schema, "python")

            if t_obj.metadata:
                print_status("Metadata:", "data")
                print_code_block(json.dumps(t_obj.metadata, indent=2), "json")
            print_box_footer()

            # --- NEU: ALL (Super-Kompakt Tabelle für 500+ Tools) ---
        elif action == "all":
            print_box_header(f"All Tools Registry ({self.active_agent_name})", "🗃")

            # Wir nutzen schmale Spalten für maximale Dichte
            # Spalten: Index, Name (gekürzt), Quelle, Flags (R/W/D)
            widths = [4, 30, 4, 6, 8]
            print_table_header([("#", 4), ("Tool Name", 30), ("Src", 4), ("Flags", 6), ("Health", 8)], widths)
            all_tools = []
            for t in tm.get_all(): all_tools.append((t, True))
            for n, t in disabled_map.items(): all_tools.append((t, False))
            all_tools.sort(key=lambda x: x[0].name)


            for i, (t, enabled) in enumerate(all_tools, 1):
                # Kompakte Flags: R=Read, W=Write, D=Dangerous
                f = ""
                if t.flags.get('read'): f += "R"
                if t.flags.get('write'): f += "W"
                if t.flags.get('dangerous'): f += "D"

                name_col = C["bright"] if enabled else C["dim"]
                src_map = {"local": "LCL", "mcp": "MCP", "a2a": "A2A"}
                src_code = src_map.get(t.source, "???")

                # Tabellenzeile drucken
                hs = getattr(t, "health_status", None) or "?"
                hs_short = {"HEALTHY": "OK", "GUARANTEED": "GTD",
                            "DEGRADED": "DEG", "FAILED": "FAIL",
                            "SKIPPED": "SKP"}.get(hs, "?")
                hs_col = {"OK": C["green"], "GTD": C["teal"],
                          "DEG": C["amber"], "FAIL": C["red"]}.get(hs_short, C["dim"])
                print_table_row(
                    [str(i).zfill(3), t.name[:29], src_code, f, hs_short],
                    widths,
                    ["grey", name_col, "cyan", "amber", hs_col]
                )

                # Alle 50 Zeilen ein kleiner Separator zur Orientierung bei Massen
                if i % 50 == 0:
                    c_print(HTML(f"<style fg='{C['dim']}'>{'─' * 50}</style>"))

            print_box_footer()
            print_status(f"Total: {len(all_tools)} tools listed ({len(tm.get_all())} active).", "success")
        elif action == "health":
            # /tools health          → alle testen
            # /tools health <name>   → einzelnes Tool testen
            if len(args) >= 2:
                t_name = args[1]
                if not (tm.exists(t_name) or t_name in disabled_map):
                    return print_status(f"Tool '{t_name}' nicht gefunden.", "error")
                result = await tm.health_check_single(t_name)
                col = {"HEALTHY": C["green"], "GUARANTEED": C["teal"],
                       "DEGRADED": C["amber"], "FAILED": C["red"],
                       "SKIPPED": C["dim"]}.get(result.status, C["dim"])
                print_box_header(f"Health: {t_name}", "🩺")
                print_box_content(
                    f"<style fg='{col}'>{result.status}</style>"
                    + (f"  ({result.execution_time_ms:.0f} ms)" if result.execution_time_ms else ""),
                    "info"
                )
                if result.error:
                    print_box_content(f"Error: {result.error}", "error")
                if result.contract_violations:
                    print_box_content(f"Violations: {', '.join(result.contract_violations)}", "warning")
                if result.result_preview:
                    print_box_content(f"Result: {result.result_preview}", "data")
                print_box_footer()
            else:
                # Alle Tools testen
                total = tm.count()
                print_status(f"Starte Health-Check für {total} Tools...", "progress")
                results = await tm.health_check_all()

                stats = {"HEALTHY": 0, "GUARANTEED": 0, "DEGRADED": 0,
                         "FAILED": 0, "SKIPPED": 0}
                failed_list, degraded_list = [], []
                skipped_list, healthy_list = [], []

                for name, r in results.items():
                    s = r.status
                    stats[s] = stats.get(s, 0) + 1
                    if s == "FAILED":   failed_list.append((name, r.error))
                    if s == "SKIPPED":   skipped_list.append(name)
                    if s == "HEALTHY":   healthy_list.append(name)
                    if s == "DEGRADED": degraded_list.append((name, r.contract_violations))

                print_box_header("Health-Check Results", "🩺")
                print_box_content(f"{stats['HEALTHY']} healthy", style="success")
                print_box_content(f"{stats['GUARANTEED']} guaranteed",
                                  style="success")  # Mapping checkmark to success style
                print_box_content(f"{stats['DEGRADED']} degraded", style="warning")
                print_box_content(f"{stats['FAILED']} failed", style="error")
                print_box_content(f"{stats['SKIPPED']} skipped", style="")

                if healthy_list:
                    print_separator()
                    for name in healthy_list:
                        print_box_content(f"{name}  →  HEALTHY", "error")

                if skipped_list:
                    print_separator()
                    for name in skipped_list:
                        print_box_content(f"{name}  →  SKIPPED", "error")

                if failed_list:
                    print_separator()
                    print_status("Failed:", "error")
                    for name, err in failed_list:
                        print_box_content(f"{name}  →  {(err or '')[:80]}", "error")

                if degraded_list:
                    print_separator()
                    print_status("Degraded:", "warning")
                    for name, violations in degraded_list:
                        v_str = ", ".join(violations or [])[:80]
                        print_box_content(f"{name}  →  {v_str}", "warning")

                print_box_footer()

        elif action == "list":
            # (Bleibt ähnlich wie vorher, zeigt aber alle Kategorien)
            cat_filter = args[1].lower() if len(args) > 1 else None
            print_box_header(f"Tools Overview: {self.active_agent_name}", "🔧")
            print_table_header([("Name", 35), ("Status", 10), ("Category", 25)], [35, 10, 25])

            all_entries = []
            for t in tm.get_all(): all_entries.append((t.name, "enabled", "green", get_tool_cats(t)))
            for n, t in disabled_map.items(): all_entries.append((n, "disabled", "grey", get_tool_cats(t)))

            for name, stat, col, cats in sorted(all_entries):
                if cat_filter and cat_filter not in cats: continue
                print_table_row([name[:33], stat, ", ".join(cats)[:23]], [35, 10, 25], ["cyan", col, "grey"])
            print_box_footer()

        elif action in ["enable", "disable"]:
            if len(args) < 2: return print_status(
                f"Geben Sie einen Namen oder eine Kategorie an: /tools {action} <...>", "warning")

            target = args[1].lower()
            affected = []
            errors = []

            # 1. Identifikation
            is_tool = tm.exists(target) or target in disabled_map
            is_cat = target in get_all_cats()

            # 2. Konfliktlösung
            final_mode = None  # "tool" or "category"
            if is_tool and is_cat:
                print_status(f"'{target}' ist sowohl ein Tool-Name als auch eine Kategorie.", "warning")
                choice = await self.prompt_session.prompt_async(HTML(
                    f"Möchten Sie das <style fg='cyan'>[t]</style>ool oder die ganze <style fg='yellow'>[k]</style>ategorie {action}? (t/k): "))
                final_mode = "category" if choice.strip().lower() in ["k", "c", "cat"] else "tool"
            elif is_tool:
                final_mode = "tool"
            elif is_cat:
                final_mode = "category"
            else:
                return print_status(f"'{target}' wurde weder als Tool noch als Kategorie gefunden.", "error")

            # 3. Ausführung
            if final_mode == "tool":
                success, name = await perform_action(target, action)
                if success:
                    affected.append(name)
                elif name:
                    errors.append(name)
            else:
                # Kategorie-Modus: Alle Tools der Kategorie finden
                source_list = tm.get_all() if action == "disable" else disabled_map.values()
                # Wir müssen Namen sammeln, da die Liste sich beim Loop ändert
                targets = [t.name for t in source_list if target in get_tool_cats(t)]

                print_status(f"Verarbeite Kategorie '{target}' ({len(targets)} Tools)...", "progress")
                for t_name in targets:
                    success, name = await perform_action(t_name, action)
                    if success:
                        affected.append(name)
                    elif name:
                        errors.append(name)

            # 4. Bericht
            if affected:
                print_box_header(f"Erfolgreich {action}d", "✅")
                # Hint anzeigen
                hint = "Einzelnes Tool" if final_mode == "tool" else f"Kategorie: {target}"
                print_box_content(f"Typ: {hint}", "info")
                print_separator()
                for name in affected:
                    print_box_content(name, "success")
                print_box_footer()

            if errors:
                print_status(f"Fehlgeschlagen/Übersprungen: {', '.join(errors)}", "warning")

        elif action == "disable-all":
            # Schützt system-relevante Tools
            targets = [t.name for t in tm.get_all() if not any(c in sys_cats for c in get_tool_cats(t))]
            for t_name in targets: await perform_action(t_name, "disable")
            print_status(f"Alle Nicht-System-Tools ({len(targets)}) wurden deaktiviert.", "success")

        elif action == "enable-all":
            count = len(disabled_map)
            targets = list(disabled_map.keys())
            for t_name in targets: await perform_action(t_name, "enable")
            print_status(f"Alle Tools ({count}) wurden reaktiviert.", "success")

    async def _cmd_context(self, args: list[str]):
        """Handle /context commands."""
        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            run_id = None
            if self._focused_task_id and self._focused_task_id is self.all_executions:
                run_id = self.all_executions[self._focused_task_id].run_id
            data = await agent.context_overview(self.active_session_id, execution_id=run_id, print_visual=False)
            show_xray_v3(data)
        except Exception as e:
            print_status(f"Error: {e}", "error")
            import traceback
            c_print(traceback.format_exc())

    # ─── Add to ICli class ────────────────────────────────────────────────────────

    async def run_agent_for_web(self, agent_name: str, query: str):
        """
        Web-facing agent entry point.

        Uses the same execution pipeline as _handle_agent_interaction but:
          - kind="web" (Zen+ / monitor filtering)
          - take_focus=False (web queries don't steal terminal focus)
          - should_speak=False (TTS is done web-side via AudioStreamPlayer
                                in icli_web.client, not self.audio_player)

        Yields raw stream chunks to the caller (icli_web.client.IcliWebClient)
        while simultaneously feeding _drain_agent_stream which updates
        _task_views and publishes to the registry.

        Cancellation: if the caller cancels (browser disconnect), we cancel
        the drain task too so the agent stops burning tokens.
        """
        try:
            agent = await self.isaa_tools.get_agent(agent_name)
        except Exception as e:
            # Surface the error through the stream protocol
            yield {"type": "error", "message": f"agent '{agent_name}': {e}"}
            return

        try:
            from toolboxv2.mods.icli_web._icli_web_tool import register_icli_web_tools
            register_icli_web_tools(agent)
        except Exception as e:
            c_print(e)

        WEB_TALK_PROMPT = f"""Du bist im Web-TALK Modus. Alle deine Ausgaben werden dem Nutzer vorgelesen — außer Tool-Calls, die laufen still im Hintergrund.

## Verhalten

- Antworte kurz und direkt. Der Nutzer hört dich, er liest dich nicht.
- Sprache des Nutzers spiegeln: Deutsch → Deutsch, Englisch → Englisch.
- Keine Markdown-Formatierung, keine Bullet-Points, keine Codeblöcke — das wird 1:1 gesprochen und klingt schlimm.
- Keine URLs oder lange IDs vorlesen. Wenn nötig, sage "ich habe dir einen Link geschickt" und öffne stattdessen ein Panel.
- DU MUST in der selben sprache antworten wie der nutzer standart DE, denn EN, sonst muss der nutzer es einmal sagen.
- Wider hole dich nicht in eime run ! sage nichs wenn es nichts neues gitb oder das du hier etwas läger benötigst.

## Tools

- Für strukturierte Infos, Formulare, Auswahl oder Visualisierung: nutze `show_template(key)` oder `show_interactive_panel`. Das Panel erscheint rechts, du sprichst parallel dazu.
- Für Status aus dem Panel (was der Nutzer eingegeben/ausgewählt hat): `get_interactive_panel_state()`.
- Für Stimm-Steuerung: nutze `custom_speek` um Tonfall, Emotion, oder Tempo deiner Sprache zu ändern. Das ist wichtig für natürliches Sprechen.
- Erst antworten (= sprechen), dann Tools nutzen wenn du weitere Infos brauchst.

## User-Query

{query}"""

        # Original agent stream
        original_stream = agent.a_stream(
            query=WEB_TALK_PROMPT,
            session_id=self.active_session_id,
            max_iterations=self.max_iteration,
            user_lightning_model=True
        )

        # Tee: both _drain_agent_stream AND our caller see every chunk.
        import asyncio
        web_queue: asyncio.Queue = asyncio.Queue(maxsize=512)

        async def tee_stream():
            try:
                async for chunk in original_stream:
                    try:
                        web_queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        # Drop oldest to make room — web view is best-effort,
                        # monitor is authoritative
                        try:
                            web_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            web_queue.put_nowait(chunk)
                        except Exception:
                            pass
                    yield chunk
            finally:
                await web_queue.put(None)  # sentinel

        stream = tee_stream()

        # Register execution via the canonical factory — same as chat path.
        exc = self._create_execution(
            kind="web",
            agent_name=agent_name,
            query=query,
            async_task=None,
            stream=stream,
            take_focus=False,
        )
        task_id = exc.task_id

        # Drain consumer (handles _ingest_chunk → registry → monitor SSE)
        async_task = asyncio.create_task(
            self._drain_agent_stream(task_id, stream, should_speak=False)
        )
        exc.async_task = async_task
        async_task.add_done_callback(
            lambda fut: self._on_agent_task_done(task_id, fut)
        )

        # Yield chunks to the web client until sentinel or cancellation
        try:
            while True:
                chunk = await web_queue.get()
                if chunk is None:
                    break
                yield chunk
        except asyncio.CancelledError:
            # Browser disconnected or orb explicitly cancelled
            if not async_task.done():
                async_task.cancel()
            raise
        except Exception:
            if not async_task.done():
                async_task.cancel()
            raise

    # =========================================================================
    # UNIFIED EXECUTION FACTORY
    # =========================================================================

    def _create_execution(
        self,
        *,
        kind: str,
        agent_name: str,
        query: str,
        async_task: asyncio.Task,
        run_id: str = "",
        stream=None,
        agent_ref=None,
        take_focus: bool = False,
    ) -> ExecutionTask:
        """Single factory: build an ExecutionTask, register it, notify ZenPlus.

        All three former entry-points (chat / delegate / scheduler-job) call
        this method.  ZenPlus.inject_job() is always called so Zen+ stays in
        sync regardless of how the execution was created.
        """
        self._task_counter += 1
        task_id_final = f"{kind}_{self._task_counter}_{agent_name}"

        exc = ExecutionTask(
            task_id=task_id_final,
            agent_name=agent_name,
            query=query[:100],
            kind=kind,
            async_task=async_task,
            run_id=run_id,
            stream=stream,
            _agent_ref=agent_ref,
            is_focused=take_focus,
        )
        self.all_executions[task_id_final] = exc

        # Focus management
        if take_focus:
            # Mute the previously focused task
            old_id = self._focused_task_id
            if old_id and old_id in self.all_executions:
                self.all_executions[old_id].is_focused = False
            self._focused_task_id = task_id_final

        # Always notify ZenPlus — regardless of active state
        # Register TaskView — single source of truth for the live dashboard
        self._task_views[task_id_final] = TaskView(
            task_id=task_id_final,
            agent_name=agent_name,
            query=query[:80],
        )

        return exc

    async def _handle_agent_interaction(self, user_input: str):
        if self.active_coder and not user_input.startswith("@"):
            await self._cmd_coder(["task", user_input])
            return
        elif self.active_coder and user_input.startswith("@"):
            user_input = user_input[1:]

        try:
            agent = await self.isaa_tools.get_agent(self.active_agent_name)
            engine = agent._get_execution_engine()

            # Audio setup
            wants_audio = user_input.strip().endswith("#audio")
            if wants_audio:
                user_input = user_input.rsplit("#audio", 1)[0].strip()
            should_speak = wants_audio or getattr(self, "verbose_audio", False)

            # Player starten falls noch nicht aktiv
            if should_speak:
                player_running = (
                    self.audio_player._task is not None
                    and not self.audio_player._task.done()
                )
                await self._ensure_audio_setup(self.active_agent_name)
                if not player_running:
                    # Nur neu bauen/starten wenn session gewechselt hat
                    if self.audio_player.session_id != self.active_session_id:
                        await self._restart_audio_player()
                        self.audio_player.session_id = self.active_session_id
                    await self.audio_player.start()

            agent_name = self.active_agent_name

            # Stream starten
            stream = agent.a_stream(
                query=user_input,
                session_id=self.active_session_id,
                max_iterations=self.max_iteration
            )

            # Consumer-Task (task_id assigned by _create_execution below)


            # _create_execution: registers, sets focus, notifies ZenPlus
            exc = self._create_execution(
                kind="chat",
                agent_name=agent_name,
                query=user_input,
                async_task=None,
                stream=stream,
                take_focus=True,
            )
            task_id = exc.task_id  # now the canonical ID
            async_task = asyncio.create_task(
                self._drain_agent_stream(task_id, stream, should_speak)
            )
            exc.async_task = async_task
            # Notification bei Completion
            async_task.add_done_callback(
                lambda fut: self._on_agent_task_done(task_id, fut)
            )

            c_print(HTML(
                f"<style fg='{PTColors.ZEN_CYAN}'>"
                f"  ◯ {agent_name} gestartet → {task_id}</style>"
            ))

            # SOFORT ZURÜCK → Prompt ist frei
            return

        except Exception as e:
            print_status(f"System Error: {e}", "error")
            import traceback
            print_status(f"System Error: {traceback.format_exc()}", "error")

    def _on_agent_task_done(self, task_id: str, fut: asyncio.Future):
        """Called when a chat-mode agent task finishes. Updates SSOT + focus."""
        exc = self.all_executions.get(task_id)
        if exc and exc.kind == "coder_sub":
            return

        # ── Determine final status ONCE, correctly ──
        # cancelled → asyncio.CancelledError or fut.cancelled()
        # failed    → fut.exception() returned a non-None exception
        # completed → clean result
        if fut.cancelled():
            final_status = "cancelled"
        else:
            try:
                fut_exc = fut.exception()
                final_status = "failed" if fut_exc is not None else "completed"
            except asyncio.CancelledError:
                final_status = "cancelled"
            except Exception:
                final_status = "failed"

        # ── Propagate status to child sub-agent tasks ──
        child_prefix = f"{task_id}::"
        for child_tid, child_exc in self.all_executions.items():
            if child_tid.startswith(child_prefix) and child_exc.status == "running":
                child_exc.status = final_status
                tv = self._task_views.get(child_tid)
                if tv:
                    tv.status = final_status

        # ── Swarm cleanup: if this was a coder parent task, mark all swarm
        # sub-TaskViews as finished (status only — do NOT pop them yet; user
        # should still see them until they explicitly clear via F9).
        parent_tv = self._task_views.get(task_id)
        if parent_tv and getattr(parent_tv, "is_swarm_summary", False) \
            and not getattr(parent_tv, "is_swarm_sub", False):
            for sub_tid in list(parent_tv.sub_task_ids.values()):
                sub_exc = self.all_executions.get(sub_tid)
                if sub_exc and sub_exc.status == "running":
                    sub_exc.status = final_status
                sub_tv = self._task_views.get(sub_tid)
                if sub_tv and sub_tv.status == "running":
                    sub_tv.status = final_status

        # _drain_agent_stream already swallows exceptions; fut.result() is safe here
        # but we still guard so a programming error in the callback never kills the CLI.
        try:
            result = fut.result()
            if exc and exc.status not in ("cancelled", "failed"):
                exc.status = final_status
            tv = self._task_views.get(task_id)
            if tv:
                tv.status = final_status
            with patch_stdout():
                try:
                    c_print(HTML(
                        f"\n<style fg='{PTColors.ZEN_GREEN}'>"
                        f"  [ok] {task_id} complete</style>\n"  # ← ASCII statt ✓
                    ))
                except UnicodeEncodeError:
                    c_print(f"\n  [ok] {task_id} complete\n")
                if not result:
                    result = tv.final_answer
                print_code_block(result, "text", show_line_numbers=False)
        except asyncio.CancelledError:
            if exc:
                exc.status = "cancelled"
            tv = self._task_views.get(task_id)
            if tv:
                tv.status = "cancelled"
            with patch_stdout():
                c_print(HTML(
                    f"\n<style fg='{PTColors.ZEN_DIM}'>"
                    f"  ⏸ {task_id} cancelled</style>\n"
                ))
        except Exception as e:
            if exc:
                exc.status = "failed"
            tv = self._task_views.get(task_id)
            if tv:
                tv.status = "failed"
            with patch_stdout():
                c_print(HTML(
                    f"\n<style fg='#f87171'>"
                    f"  ✗ {task_id} failed: {_esc(str(e)[:60])}</style>\n"
                ))

        # Focus hand-off — find next running task
        if self._focused_task_id == task_id:
            self._focused_task_id = None

            next_task = next(
                (t for t in self.all_executions.values() if t.status == "running"),
                None,
            )
            if next_task:
                next_task.is_focused = True
                self._focused_task_id = next_task.task_id

    async def _drain_agent_stream(
        self, task_id: str, stream, should_speak: bool = False
    ):
        result_text = ""
        sentence_buffer = ""  # ← der einzige Buffer
        stop_for_speech = False

        if should_speak:
            if self.audio_player._task is None or self.audio_player._task.done():
                await self.audio_player.start()

        def _get_exc():
            return self.all_executions.get(task_id) or next(
                (t for t in self.all_executions.values() if t.stream is stream), None
            )

        try:
            with patch_stdout():
                while True:
                    try:
                        raw = await stream.__anext__()
                    except StopAsyncIteration:
                        break
                    chunk = dict(raw)

                    exc = _get_exc()
                    real_id = exc.task_id if exc else task_id
                    self._ingest_chunk(real_id, chunk)
                    chunk_type = chunk.get("type", "")

                    if chunk_type == "content":
                        text = chunk.get("chunk", "")
                        result_text += text

                        if should_speak:
                            if "```" in text:
                                stop_for_speech = not stop_for_speech
                                # Flush buffer wenn wir in Code-Block wechseln
                                if stop_for_speech and sentence_buffer.strip():
                                    self._enqueue_speech(sentence_buffer)
                                    sentence_buffer = ""
                                continue

                            if not stop_for_speech:
                                sentence_buffer += text
                                if (
                                    any(sentence_buffer.rstrip().endswith(p)
                                        for p in (".", "!", "?", ":", "\n\n"))
                                    and len(sentence_buffer.strip()) >= 30
                                ):
                                    self._enqueue_speech(sentence_buffer)
                                    sentence_buffer = ""

                    elif chunk_type == "final_answer":
                        result_text = chunk.get("answer", result_text)

            # Restlichen Buffer nach Stream-Ende sprechen
            if should_speak and sentence_buffer.strip():
                self._enqueue_speech(sentence_buffer)

            try:
                await stream.aclose()
            except BaseException:
                pass

            exc = _get_exc()
            if exc:
                exc.status = "completed"
                exc.result_text = result_text
            return result_text

        except asyncio.CancelledError:
            try:
                await asyncio.shield(stream.aclose())
            except BaseException:
                pass
            exc = _get_exc()
            if exc:
                exc.status = "cancelled"
            return ""

        except Exception as e:
            try:
                await asyncio.shield(stream.aclose())
            except BaseException:
                pass
            exc = _get_exc()
            if exc:
                exc.status = "failed"
            with patch_stdout():
                c_print(HTML(
                    f"<style fg='{PTColors.ZEN_RED}'>  ✗ stream error ({type(e).__name__}): "
                    f"{_esc(str(e)[:80])}</style>"
                ))
            return ""


    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================

    async def run(self):
        """Main CLI execution loop."""
        # Print banner
        from prompt_toolkit.styles import Style as PtStyle
        # Switch to audio recording mode (changes animation)

        c_print()
        c_print(HTML(f"<style fg='{PTColors.ZEN_CYAN}'>{CLI_NAME}</style> <style fg='{PTColors.ZEN_DIM}'>v{VERSION}</style>"))
        c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>/help  F4 voice  F5 status  F6 minimize  Ctrl+C safe stop</style>"))
        c_print()
        # Initialize Self Agent
        await self._init_self_agent()

        # Start Job Scheduler
        # Start Job Scheduler
        self.job_scheduler = JobScheduler(self.jobs_file, self._fire_job_from_scheduler)
        await self.job_scheduler.start()
        await self.job_scheduler.fire_lifecycle("on_cli_start")

        # ── Catch up jobs that fired while CLI was offline ────────────────────
        missed_count = await self.job_scheduler.fire_missed_jobs()
        if missed_count:
            c_print(HTML(
                f"<style fg='#fbbf24'>↺ {missed_count} job(s) were due while offline"
                f" — catching up now.</style>"
            ))

        # ── Auto-install OS autowake if persistent jobs exist ─────────────────
        if self.job_scheduler.has_persistent_jobs():
            try:
                from toolboxv2.mods.isaa.extras.jobs.os_scheduler import (
                    autowake_status, install_autowake,
                )
                status_str = autowake_status()
                if "Not installed" in status_str:
                    result = install_autowake(self.jobs_file)
                    #info("Auto-installed OS autowake on startup: %s", result)
            except Exception as _aw_err:
                c_print("OS autowake auto-install skipped: %s", _aw_err)

        # Show active features
        all_feats = self.feature_manager.list_features()
        if all_feats:
            active = [f for f in all_feats if self.feature_manager.is_enabled(f)]
            inactive = [f for f in all_feats if not self.feature_manager.is_enabled(f)]
            parts = []
            for f in active:
                parts.append(f"<style fg='{PTColors.ZEN_GREEN}'>{f}</style>")
            for f in inactive:
                parts.append(f"<style fg='{PTColors.ZEN_DIM}'>{f}</style>")
            c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>features:</style> {' '.join(parts)}"+f"<style fg='{PTColors.ZEN_GREEN}'>{self._get_keybinding_indicator()}</style>"))
            c_print()

        # Print status
        total = await self._print_status_dashboard()

        # Create prompt session
        dict_coplet, vfs_cplet = self._build_completer()
        self.prompt_session = PromptSession(
            history=self.history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=SmartCompleter(
                nested_dict=dict_coplet, vfs_completer=vfs_cplet
            ),
            complete_while_typing=True,
            multiline=False,
            key_bindings=self.key_bindings,
            bottom_toolbar=self._get_bottom_toolbar,
            style=PtStyle.from_dict({
                'bottom-toolbar': 'bg:ansiblack fg:ansigray',
                'bottom-toolbar.text': 'fg:ansigray',
            })
        )
        self.app.run_bg_task_advanced(self.active_refresher)

        apply_prompt_toolkit_patch_safe()
        register_app(self.prompt_session.app)

        # Main loop
        while True:
            try:
                # Update completer

                dict_coplet, vfs_cplet = self._build_completer()
                self.prompt_session.completer = SmartCompleter(
                    nested_dict=dict_coplet, vfs_completer=vfs_cplet
                )

                # Get input
                with patch_stdout():
                    user_input = await self.prompt_session.prompt_async(
                        self.get_prompt_text()
                    )

                # Check for transcription
                if hasattr(self, "_last_transcription") and self._last_transcription is not None:
                    user_input = self._last_transcription + " " + user_input
                    self._last_transcription = None
                    print_status(f"Using transcription: {user_input}", "info")

                user_input = user_input.strip()
                if not user_input:
                    await self._handle_interrupt()
                    continue

                # Route input
                with Spinner("Starting command"):
                    if user_input.startswith("!"):
                        await self._handle_shell(user_input[1:])
                    elif user_input.startswith("/"):
                        await self._handle_command(user_input)
                    else:
                        await self._handle_agent_interaction(user_input)

            except KeyboardInterrupt:
                continue
            except asyncio.CancelledError:
                # CancelledError is BaseException — never let it kill the CLI loop.
                # All stop/resume paths should have swallowed it, but this is
                # the final safety net.
                continue
            except EOFError:
                break
            except Exception as e:
                import traceback
                print_status(f"Unexpected Error: {traceback.format_exc()}", "error")
                import traceback
                c_print(traceback.format_exc())

        # Cleanup
        self._save_state()

        # Stop job scheduler
        if self.job_scheduler:
            await self.job_scheduler.fire_lifecycle("on_cli_exit")
            # Remove OS autowake only when no persistent jobs remain
            if not self.job_scheduler.has_persistent_jobs():
                try:
                    from toolboxv2.mods.isaa.extras.jobs.os_scheduler import remove_autowake
                    remove_autowake()
                    c_print("Removed OS autowake (no persistent jobs left)")
                except Exception:
                    pass
            await self.job_scheduler.stop()

        # Cancel in-flight background tasks
        for _, bg_task in self.all_executions.items():
            if bg_task.status == "running":
                bg_task.async_task.cancel()

        await self.app.a_exit()
        print_status("Goodbye!", "success")

    # ─── Interrupt-Menü ───────────────────────────────────────────────

    async def _handle_interrupt(self):
        task_id = self._focused_task_id

        if not task_id:
            running = next(
                (t for t in self.all_executions.values() if t.status == "running"), None
            )
            if running:
                running.is_focused = True
                self._focused_task_id = running.task_id
                task_id = running.task_id
            else:
                return True

        if task_id not in self.all_executions:
            return True

        task = self.all_executions[task_id]
        if task.status != "running":
            return True

        agent_name = task.agent_name

        # ── [i] kann mehrfach gedrückt werden → Loop ─────────────────────
        while True:
            with patch_stdout():
                c_print(HTML(
                    f"\n<style fg='{PTColors.ZEN_CYAN}'>─── Interrupt: {_esc(agent_name)} ({task_id}) ───</style>\n"
                    f"<style fg='{PTColors.ZEN_DIM}'>"
                    f"  [i] Info: letzter Gedanke / aktives Tool / Antwort\n"
                    f"  [b] In den Hintergrund verschieben\n"
                    f"  [s] Stoppen (cancel)\n"
                    f"  [r] mit weiterem Context fortsetzen (resume)\n"
                    f"  [n] Nichts tun (weiter warten)\n"
                    f"</style>"
                ))

            try:
                with patch_stdout():
                    choice = await self.prompt_session.prompt_async(
                        HTML(f"<style fg='{PTColors.ZEN_CYAN}'>  Auswahl [i/b/s/r/n]: </style>"),
                    )
                choice = choice.strip().lower()
            except (KeyboardInterrupt, EOFError):
                choice = "n"

            if choice == "i":
                self._interrupt_show_info(task_id)
                # Menü erneut anzeigen (Loop weiterführen)
                continue
            elif choice == "b":
                await self._interrupt_move_to_background(task_id)
            elif choice == "s":
                await self._interrupt_stop_task(task_id)
            elif choice == "r":
                await self._interrupt_resume_with_context(task_id)
            else:
                with patch_stdout():
                    c_print(HTML(
                        f"<style fg='{PTColors.ZEN_DIM}'>  → Weiter warten auf {task_id}</style>\n"
                    ))
            break  # alle anderen Optionen verlassen den Loop

        return True

    def _interrupt_show_info(self, task_id: str):
        """[i] — zeigt Snapshot aus TaskView: Gedanken, Tool, Antwort."""
        tv = self._task_views.get(task_id)
        if not tv:
            with patch_stdout():
                c_print(HTML(f"<style fg='{PTColors.ZEN_DIM}'>  (keine TaskView für {task_id})</style>\n"))
            return

        lines = []

        # ── Phase / Status ────────────────────────────────────────────────
        lines.append(
            f"<style fg='{PTColors.ZEN_CYAN}'>  ── Snapshot: {_esc(tv.agent_name)} "
            f"iter {tv.iteration}/{tv.max_iter} [{tv.phase}] ──</style>"
        )

        # ── Letzter Gedanke ───────────────────────────────────────────────
        if tv.last_thought:
            thought = tv.last_thought.replace("\n", " ").strip()
            lines.append(
                f"<style fg='{PTColors.ZEN_DIM}'>  ◎ Gedanke:  </style>"
                f"<style fg='#e5e7eb'>{_esc(thought[:200])}"
                f"{'…' if len(thought) > 200 else ''}</style>"
            )

        # ── Aktives / letztes Tool ────────────────────────────────────────
        if tv.last_tool:
            ok_col = PTColors.ZEN_GREEN if tv.last_tool_ok else PTColors.ZEN_RED
            ok_sym = "✓" if tv.last_tool_ok else "✗"
            info_part = (
                f"  <style fg='{PTColors.ZEN_DIM}'>{_esc(tv.last_tool_info[:60])}</style>"
                if tv.last_tool_info else ""
            )
            lines.append(
                f"<style fg='{PTColors.ZEN_DIM}'>  ◇ Tool:     </style>"
                f"<style fg='#60a5fa'>{_esc(tv.last_tool)}</style> "
                f"<style fg='{ok_col}'>{ok_sym}</style>"
                f"{info_part}"
            )

        # ── Letzte Iteration: Tools-Liste ─────────────────────────────────
        if tv.iterations:
            last_iv = tv.iterations[-1]
            if last_iv.tools:
                tool_summary = "  ".join(
                    f"{'✓' if ok else '✗'}{name[:14]}"
                    for name, ok, _, _ in last_iv.tools[-4:]  # max 4 zeigen
                )
                lines.append(
                    f"<style fg='{PTColors.ZEN_DIM}'>  ◈ Tools[iter {last_iv.n}]: "
                    f"{_esc(tool_summary)}</style>"
                )
            if last_iv.pending_tool:
                lines.append(
                    f"<style fg='#fbbf24'>  ⟳ Läuft:   {_esc(last_iv.pending_tool)}</style>"
                )

        if tv.narrator_msg:
            lines.append(
                f"<style fg='{PTColors.ZEN_DIM}'>  ◎ narr:  </style>"
                f"<style fg='#e5e7eb'>{_esc(tv.narrator_msg)}"
                f"{'…' if len(tv.narrator_msg) > 200 else ''}</style>"
            )

        # ── Finale Antwort (falls schon vorhanden) ────────────────────────
        if tv.final_answer:
            answer = tv.final_answer.replace("\n", " ").strip()
            lines.append(
                f"<style fg='{PTColors.ZEN_DIM}'>  ✦ Antwort: </style>"
                f"<style fg='#4ade80'>{_esc(answer[:300])}"
                f"{'…' if len(answer) > 300 else ''}</style>"
            )
        elif not tv.last_thought and not tv.last_tool:
            lines.append(
                f"<style fg='{PTColors.ZEN_DIM}'>  (wartet noch auf erste Iteration…)</style>"
            )

        with patch_stdout():
            c_print(HTML("\n".join(lines) + "\n"))

    async def _interrupt_move_to_background(self, task_id: str):
        """Task minimieren und Fokus freigeben."""
        task = self.all_executions.get(task_id)
        if not task:
            return

        self._focused_task_id = None


        with patch_stdout():
            c_print(HTML(
                f"<style fg='{PTColors.ZEN_DIM}'>"
                f"  ▾ {task_id} → Hintergrund (Prompt frei)</style>\n"
            ))

    async def _interrupt_stop_task(self, task_id: str):
        task = self.all_executions.get(task_id)
        if not task:
            return

        with patch_stdout():
            c_print(HTML(f"<style fg='#fbbf24'>  Signalisiere sauberen Stopp für {task_id}...</style>"))

        try:
            agent = await self.isaa_tools.get_agent(task.agent_name)
            engine = agent._get_execution_engine()

            # Find execution_id (search all statuses — session_id check dropped:
            # if task is focused, it IS the active one regardless of session)
            execution_id = None
            for eid, ctx in engine._active_executions.items():
                if ctx.status in ("running", "paused"):
                    execution_id = eid
                    break

            if execution_id:
                try:
                    await agent.cancel_execution(execution_id)
                    # cancel() pops ctx from _active_executions and cancels
                    # sub-agent tasks (state._task.cancel() — non-blocking)
                except BaseException:
                    pass

            # Cancel the CLI drain task
            task.async_task.cancel()

            # asyncio.wait() returns (done, pending) — never raises.
            # asyncio.wait_for() would re-raise CancelledError from the task,
            # even with except (CancelledError, TimeoutError) in some edge cases.
            try:
                done, pending = await asyncio.wait(
                    {task.async_task}, timeout=2.0
                )
                # If still pending after timeout: cancel and move on (no await)
                for t in pending:
                    t.cancel()
            except BaseException:
                pass

        except BaseException:  # ← FIX B: was "except Exception"
            pass

        if self.active_coder:
            parent_tv = None
            for tid, tv in list(self._task_views.items()):
                if tv.is_swarm_summary and not tv.is_swarm_sub:
                    parent_tv = tv
                    break

            if parent_tv:
                for sub_tid in list(parent_tv.sub_task_ids.values()):
                    # Aus TaskViews + Executions entfernen
                    self._task_views.pop(sub_tid, None)
                    exc = self.all_executions.get(sub_tid)
                    if exc:
                        if exc.status == "running":
                            exc.status = "cancelled"

        if self._focused_task_id == task_id:
            if task:
                task.is_focused = False
            self._focused_task_id = None

        with patch_stdout():
            c_print(HTML(
                f"<style fg='#f87171'>"
                f"  ✗ {task_id} gestoppt</style>\n"
            ))

    async def _interrupt_resume_with_context(self, task_id: str):
        task = self.all_executions.get(task_id)
        if not task:
            return

        if task.kind == "coder":
            with patch_stdout():
                c_print(HTML(f"<style fg='#fbbf24'>  Stoppe aktuellen Coder-Loop...</style>"))

            task.async_task.cancel()
            try:
                done, pending = await asyncio.wait({task.async_task}, timeout=2.0)
                for t in pending: t.cancel()
            except BaseException:
                pass

            with patch_stdout():
                c_print(HTML(f"<style fg='{PTColors.ZEN_CYAN}'>  Korrektur / Neuer Context für Coder:</style>"))
            try:
                with patch_stdout():
                    new_context = await self.prompt_session.prompt_async(
                        HTML(f"<style fg='{PTColors.ZEN_CYAN}'>  > </style>"))
                new_context = new_context.strip()
            except (KeyboardInterrupt, EOFError):
                if self._focused_task_id == task_id: self._focused_task_id = None
                return

            if new_context:
                self.all_executions.pop(task_id, None)
                self._task_views.pop(task_id, None)
                await self._cmd_coder(["task", f"[Korrektur/Resume]: {new_context}"])
            else:
                if self._focused_task_id == task_id: self._focused_task_id = None
            return

        with patch_stdout():
            c_print(HTML(f"<style fg='#fbbf24'>  Pausiere {task_id}...</style>"))

        # ── Step 1: Memo execution_id BEFORE any cancel ───────────────────────
        execution_id = None
        agent = None
        try:
            agent = await self.isaa_tools.get_agent(task.agent_name)
            engine = agent._get_execution_engine()

            for eid, ctx in engine._active_executions.items():
                if ctx.status == "running":
                    execution_id = eid
                    break

            if not execution_id:
                with patch_stdout():
                    c_print(HTML(
                        "<style fg='#f87171'>  Keine laufende Execution gefunden.</style>\n"
                    ))
                if self._focused_task_id == task_id:
                    self._focused_task_id = None
                return

            # ── Step 2: Set status="paused" (ctx stays in _active_executions) ─
            # pause() is: ctx.status="paused", live.enter(PAUSED,...) — no awaits
            try:
                await agent.pause_execution(execution_id)
            except BaseException:
                pass

        except BaseException as e:
            with patch_stdout():
                c_print(HTML(
                    f"<style fg='#f87171'>  Engine-Fehler: {_esc(str(e)[:60])}</style>\n"
                ))
            if self._focused_task_id == task_id:
                self._focused_task_id = None
            return

        # ── Step 3: Cancel drain task, wait for it to finish ─────────────────
        # stream_generator checks ctx.status=="paused" at iteration start,
        # but current LLM call might take seconds. Cancel immediately.
        # With FIX A, the drain task handles CancelledError cleanly.
        task.async_task.cancel()
        try:
            done, pending = await asyncio.wait({task.async_task}, timeout=3.0)
            for t in pending:
                t.cancel()  # timeout: force cancel, no await needed
        except BaseException:
            pass

        # ── Step 4: Get new context from user ────────────────────────────────
        with patch_stdout():
            c_print(HTML(
                f"<style fg='{PTColors.ZEN_CYAN}'>  Neuer Context für {task.agent_name}:</style>"
            ))

        new_context = ""
        try:
            with patch_stdout():
                new_context = await self.prompt_session.prompt_async(
                    HTML(f"<style fg='{PTColors.ZEN_CYAN}'>  > </style>"),
                )
            new_context = new_context.strip()
        except (KeyboardInterrupt, EOFError):
            with patch_stdout():
                c_print(HTML(
                    f"<style fg='{PTColors.ZEN_DIM}'>  → Resume abgebrochen.</style>\n"
                ))
            if self._focused_task_id == task_id:
                self._focused_task_id = None
            return

        # ── Step 5: Resume via engine ─────────────────────────────────────────
        # engine.resume() finds ctx in _active_executions (pause left it there)
        # verifies ctx.status=="paused", adds content, calls execute_stream(ctx=ctx)
        # returns tuple[Callable, ExecutionContext]
        try:
            resume_result = await agent.resume_execution(
                execution_id=execution_id,
                content=new_context,
                stream=True,
            )
        except BaseException as e:
            with patch_stdout():
                c_print(HTML(
                    f"<style fg='#f87171'>  ✗ resume_execution: {_esc(str(e)[:80])}</style>\n"
                ))
            if self._focused_task_id == task_id:
                self._focused_task_id = None
            return

        # Unpack (stream_func, ctx) — execute_stream always returns this tuple
        if isinstance(resume_result, tuple):
            stream_func, ctx_obj = resume_result
            stream = stream_func(ctx_obj)
        elif isinstance(resume_result, str) and resume_result.startswith("Error:"):
            with patch_stdout():
                c_print(HTML(
                    f"<style fg='#f87171'>  ✗ {_esc(resume_result)}</style>\n"
                ))
            if self._focused_task_id == task_id:
                self._focused_task_id = None
            return
        else:
            stream = resume_result  # unexpected but handle gracefully

        # ── Step 6: Register new drain task ──────────────────────────────────


        query_text = (
            f"[resumed] {new_context[:80]}"
            if new_context
            else f"[resumed] {task.query[:80]}"
        )

        exc = self._create_execution(
            kind="chat",
            agent_name=task.agent_name,
            query=query_text,
            async_task=None,
            stream=stream,
            take_focus=True,
        )

        new_task_id = exc.task_id
        async_task = asyncio.create_task(
            self._drain_agent_stream(new_task_id, stream, False)
        )
        exc.async_task = async_task


        # Clean up old task from SSOT
        self.all_executions.pop(task_id, None)
        self._task_views.pop(task_id, None)

        async_task.add_done_callback(
            lambda fut: self._on_agent_task_done(new_task_id, fut)
        )

        with patch_stdout():
            c_print(HTML(
                f"<style fg='{PTColors.ZEN_GREEN}'>"
                f"  ↻ {task.agent_name} resumed → {new_task_id}</style>\n"
            ))


# =============================================================================
# ENTRY POINT
# =============================================================================


async def run(app=None, *args):
    """Entry point for ISAA Host CLI."""
    app = app or get_app("isaa-host")
    host = ISAA_Host(app)
    try:
        from toolboxv2.mods.isaa.extras.discord_interface.integration_example import patch_cli_for_discord
        patch_cli_for_discord(host)
        print("Discord integration enabled.")
    except ImportError as e:
        import traceback
        c_print(traceback.format_exc())
        print(e)
        print("⚠️ Discord integration not available.")
        pass
    await host.run()


def main():
    """Synchronous entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    import argparse
    import asyncio
    import sys

    async def main_cli():
        parser = argparse.ArgumentParser(description="ISAA iCLI")
        parser.add_argument("query", nargs="?", help="Anfrage an den Agenten")
        parser.add_argument("--agent", default="self")
        parser.add_argument("--session", default="direct_run")
        parser.add_argument("--remember", default="direct_run")
        parser.add_argument("--feature", action="append")
        parser.add_argument("--mcp", action="append")
        parser.add_argument("--model")
        # --- NEU ---
        parser.add_argument(
            "--gui",
            action="store_true",
            help="GUI mode: kein TUI, ZMQ I/O, wartet auf follow-up queries"
        )
        parser.add_argument(
            "--gui-session",
            default=None,
            help="Session ID für ZMQ input channel (gui.input.<id>)"
        )

        args = parser.parse_args()
        app = get_app("isaa-host")
        host = ISAA_Host(app)

        # ── Normaler interaktiver Modus ────────────────────────────────
        if not args.query and not args.gui:
            await host.run()
            return

        # ── Shared setup für single-run UND gui-mode ──────────────────
        await host._init_self_agent()
        agent = await host.isaa_tools.get_agent(args.agent)

        if args.feature:
            host.feature_manager.set_agent(agent)
            for feat in args.feature:
                await host.feature_manager.enable(feat)

        if args.model and args.model in MODEL_MAPPING:
            agent.amd.complex_llm_model = MODEL_MAPPING[args.model]

        if args.mcp:
            for mcp_json in args.mcp:
                import json
                m_cfg = json.loads(mcp_json)
                await host._tool_mcp_connect(
                    m_cfg['name'], m_cfg['command'],
                    m_cfg.get('args', []), args.agent
                )

        # ── GUI Mode ───────────────────────────────────────────────────
        if args.gui:
            from toolboxv2.utils.workers.interface_registry import get_registry

            gui_session = args.gui_session or args.session
            reg = get_registry()
            connected = await reg.start()

            if not connected:
                print("[icli --gui] ZMQ offline — running without stream", flush=True)
                # Kein return — einfach single-run oder interaktiv ohne ZMQ
                if args.query:
                    result = await agent.a_run(args.query, session_id=gui_session)
                    print(f"\n[RESULT]:\n{result}", flush=True)
                else:
                    await host.run()  # ← normaler interaktiver modus als fallback
                await app.a_exit()
                return

            # Initial query ausführen falls mitgegeben
            if args.query:
                await agent.a_run(args.query, session_id=gui_session)

            # Follow-up loop via ZMQ
            input_channel = f"gui.input.{gui_session}"
            follow_up_queue: asyncio.Queue = asyncio.Queue()

            def _on_gui_input(payload: dict) -> None:
                follow_up_queue.put_nowait(payload)

            reg.register_sub(
                id=input_channel,
                callback=_on_gui_input,
                filter_prefix=False  # exact match auf diese session
            )

            print(f"[icli --gui] Listening on {input_channel}", flush=True)

            # Warte auf follow-up oder exit signal
            while True:
                try:
                    msg = await asyncio.wait_for(follow_up_queue.get(), timeout=300.0)
                except asyncio.TimeoutError:
                    print("[icli --gui] Timeout — keine GUI activity, exit", flush=True)
                    break

                action = msg.get("action", "query")

                if action == "exit":
                    break

                if action == "query":
                    query_text = msg.get("query", "").strip()
                    if not query_text:
                        continue
                    override_agent = msg.get("agent")
                    if override_agent and override_agent != args.agent:
                        agent = await host.isaa_tools.get_agent(override_agent)
                    await agent.a_run(query_text, session_id=gui_session)

            await reg.stop()
            if not args.remember:
                agent.clear_session_history(gui_session)
            await app.a_exit()
            return

        # ── Single Run Modus (unveränderter bestehender Pfad) ──────────
        print(f"[*] Agent {args.agent} denkt nach...", flush=True)
        result = await agent.a_run(args.query, session_id=args.session)

        if not args.remember:
            agent.clear_session_history(args.session)
        await app.a_exit()
        print(f"\n[RESULT]:\n{result}", flush=True)
    # ZMQ benötigt SelectorEventLoop auf Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


    asyncio.run(main_cli())
