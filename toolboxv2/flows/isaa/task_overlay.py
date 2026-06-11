# task_overlay.py — ZEN+ TaskOverlay v2 (from scratch)
# ─────────────────────────────────────────────────────────────────────────────
# Drei Ebenen, ein Gesetz: Breite zuerst, Tiefe auf Verlangen.
#
#   DEPTH 0  FLEET   — alle Agents + Swarm-Hierarchie, Health auf einen Blick
#   DEPTH 1  AGENT   — ein Agent: Stats-Block, Iterations-Timeline (kompakt)
#   DEPTH 2  STEP    — eine Iteration: Gedanken-Dump + volles Tool-I/O
#
# Navigation (read-only Overlay → WASD kollidiert mit nichts):
#   W/S, ↑/↓        cursor / scroll
#   D/→/Enter       eine Ebene tiefer
#   A/←/Backspace   eine Ebene hoch; auf Ebene 0 → close
#   PgUp/PgDn       ±10 Zeilen scroll (Ebene 1+2)
#   g / G           jump top / bottom (Ebene 1+2)
#   q / Esc / F2    close
#
# TB dark-term palette (konsistent mit footer/host):
#   bg #111827 · sel #1a2035 · dim #6b7280 · text #e5e7eb
#   cyan #67e8f9 · green #4ade80 · amber #fbbf24 · violet #a78bfa
#   blue #60a5fa · red #f87171 · pink #f472b6
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import ast
import asyncio
import json
import textwrap
import time
from typing import TYPE_CHECKING, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    Dimension,
    FormattedTextControl,
    HSplit,
    Layout,
    VSplit,
    Window,
)

# ── Palette ──────────────────────────────────────────────────────────────────
P = {
    "bg":     "bg:#111827",
    "sel":    "bg:#1a2035",
    "dim":    "fg:#6b7280",
    "text":   "fg:#e5e7eb",
    "cyan":   "fg:#67e8f9",
    "green":  "fg:#4ade80",
    "amber":  "fg:#fbbf24",
    "violet": "fg:#a78bfa",
    "blue":   "fg:#60a5fa",
    "red":    "fg:#f87171",
    "pink":   "fg:#f472b6",
}

_STATUS = {
    "running":   ("◉", P["cyan"]),
    "thinking":  ("◎", P["amber"]),
    "completed": ("●", P["green"]),
    "done":      ("●", P["green"]),
    "failed":    ("✗", P["red"]),
    "error":     ("✗", P["red"]),
    "cancelled": ("⏸", P["dim"]),
    "paused":    ("⏸", P["amber"]),
    "max_iterations": ("◍", P["amber"]),
}

_PHASE_COL = {
    "init": "#6b7280", "planning": "#fbbf24", "coding": "#60a5fa",
    "validating": "#a78bfa", "fixing": "#f59e0b",
    "done": "#4ade80", "error": "#f87171",
}

FT = list[tuple[str, str]]


# ── kleine Helfer ────────────────────────────────────────────────────────────
def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_elapsed(sec: float) -> str:
    sec = max(0, sec)
    if sec < 60:
        return f"{sec:4.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _bar(cur: int, total: int, width: int = 10) -> str:
    total = max(1, total)
    fill = min(width, int(width * cur / total))
    return "▰" * fill + "▱" * (width - fill)


def _spark(values: list[bool]) -> str:
    """Tool-Erfolgs-Sparkline der letzten Iterationen: ▪ ok, ▫ fail."""
    return "".join("▪" if v else "▫" for v in values[-12:])


def _try_parse_struct(raw):
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _format_struct(obj, indent: int = 0, out: Optional[FT] = None) -> FT:
    """Rekursiver Pretty-Printer: dict/list → farbige FormattedText-Zeilen."""
    if out is None:
        out = []
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((P["dim"], pad))
            out.append((P["cyan"], f"{k}"))
            out.append((P["dim"], ": "))
            if isinstance(v, (dict, list)):
                out.append(("", "\n"))
                _format_struct(v, indent + 1, out)
            else:
                _format_scalar(v, out)
                out.append(("", "\n"))
    elif isinstance(obj, list):
        for v in obj:
            out.append((P["dim"], pad + "• "))
            if isinstance(v, (dict, list)):
                out.append(("", "\n"))
                _format_struct(v, indent + 1, out)
            else:
                _format_scalar(v, out)
                out.append(("", "\n"))
    else:
        out.append((P["dim"], pad))
        _format_scalar(obj, out)
        out.append(("", "\n"))
    return out


def _format_scalar(v, out: FT) -> None:
    if isinstance(v, bool) or v is None:
        out.append((P["violet"], str(v)))
    elif isinstance(v, (int, float)):
        out.append((P["amber"], str(v)))
    elif isinstance(v, str):
        if "\n" in v:
            out.append(("", "\n"))
            for ln in v.split("\n"):
                out.append((P["green"], "      " + ln + "\n"))
            out.pop()  # letztes \n landet vom Caller
        else:
            out.append((P["green"], v))
    else:
        out.append((P["text"], str(v)))


def _wrap_block(text: str, width: int, style: str, prefix: str = "  ") -> FT:
    frags: FT = []
    for para in (text or "").split("\n"):
        lines = textwrap.wrap(para, width=width) or [""]
        for ln in lines:
            frags.append((style, prefix + ln + "\n"))
    return frags


def _apply_scroll(frags: FT, scroll: int) -> FT:
    """Zeilenbasiertes Scrolling über flache Fragment-Liste (newline-zählend)."""
    if scroll <= 0:
        return frags
    out: FT = []
    skipped = 0
    for style, text in frags:
        if skipped >= scroll:
            out.append((style, text))
            continue
        nl = text.count("\n")
        if skipped + nl <= scroll:
            skipped += nl
            continue
        parts = text.split("\n")
        need = scroll - skipped
        out.append((style, "\n".join(parts[need:])))
        skipped = scroll
    return out


def _count_lines(frags: FT) -> int:
    return sum(t.count("\n") for _, t in frags)


# ═════════════════════════════════════════════════════════════════════════════
class TaskOverlay:
    """ZEN+ Fullscreen-Overlay (F2). Read-only Monitor über task_views (SSOT)."""

    LEFT_W = Dimension(min=30, max=38)

    def __init__(self, task_views: dict):
        self._task_views = task_views         # SSOT-Referenz (Host-dict, live)
        # ── Zustands-Vektor ─────────────────────────────────────────────
        self._depth = 0                       # 0 fleet · 1 agent · 2 step
        self._selected: Optional[str] = None  # task_id
        self._iter_cursor = 0                 # Index in Iterations-Liste (Ebene 1)
        self._selected_iter_n: Optional[int] = None  # Iteration-Nr (Ebene 2)
        self._agent_scroll = 0
        self._step_scroll = 0
        self._app: Optional[Application] = None

    # ── Lifecycle ───────────────────────────────────────────────────────────
    async def show(self, on_exit):
        tvs = self._tvs()
        if self._selected not in tvs:
            self._selected = next(iter(self._flat_order()), None)

        left = Window(FormattedTextControl(self._render_left), width=self.LEFT_W,
                      style=P["bg"])
        divider = Window(width=1, char="│", style=P["bg"] + " fg:#374151")
        right = Window(FormattedTextControl(self._render_right), style=P["bg"],
                       wrap_lines=False)
        footer = Window(FormattedTextControl(self._render_footer), height=1,
                        style="bg:#0f172a")

        layout = Layout(HSplit([VSplit([left, divider, right]), footer]))
        self._app = Application(layout=layout, key_bindings=self._kb(),
                                full_screen=True, refresh_interval=0.5)
        try:
            await self._app.run_async()
        finally:
            self._app = None
            self._loop = None
            if on_exit:
                try:
                    on_exit()
                except Exception:
                    pass

    def invalidate(self):
        app, loop = self._app, getattr(self, "_loop", None)
        if not app:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop and running is not loop:
            loop.call_soon_threadsafe(app.invalidate)
        else:
            app.invalidate()

    # ── Daten-Zugriff ───────────────────────────────────────────────────────
    def _tvs(self) -> dict:
        return self._task_views or {}

    def _flat_order(self) -> list[str]:
        """Anzeigereihenfolge: Top-Level neueste zuerst, Swarm-Subs direkt
        unter ihrem Parent (Reihenfolge der sub_task_ids)."""
        tvs = self._tvs()
        sub_ids = set()
        for tv in tvs.values():
            sub_ids.update(getattr(tv, "sub_task_ids", {}).values())
        order: list[str] = []
        for tid in reversed(list(tvs.keys())):
            if tid in sub_ids:
                continue
            order.append(tid)
            tv = tvs[tid]
            for stid in getattr(tv, "sub_task_ids", {}).values():
                if stid in tvs:
                    order.append(stid)
        return order

    def _cur_tv(self):
        return self._tvs().get(self._selected)

    @staticmethod
    def _iter_health(iv) -> list[bool]:
        return [bool(ok) for (_n, ok, _t, _i) in getattr(iv, "tools", [])]

    # ── Keybindings ─────────────────────────────────────────────────────────
    def _kb(self) -> KeyBindings:
        kb = KeyBindings()

        def close(event):
            event.app.exit()

        for k in ("q", "escape", "f2"):
            kb.add(k)(close)

        # vertical
        for k in ("up", "w"):
            kb.add(k)(lambda e: self._nav_v(-1))
        for k in ("down", "s"):
            kb.add(k)(lambda e: self._nav_v(+1))
        kb.add("pageup")(lambda e: self._nav_v(-10))
        kb.add("pagedown")(lambda e: self._nav_v(+10))
        kb.add("g")(lambda e: self._jump(top=True))
        kb.add("G")(lambda e: self._jump(top=False))

        # horizontal / depth
        for k in ("right", "d", "enter"):
            kb.add(k)(lambda e: self._go_deeper())
        for k in ("left", "a", "backspace"):
            kb.add(k)(lambda e: self._go_up(e))

        return kb

    def _nav_v(self, delta: int):
        if self._depth == 0:
            order = self._flat_order()
            if not order:
                return
            try:
                i = order.index(self._selected)
            except ValueError:
                i = 0
            self._selected = order[max(0, min(len(order) - 1, i + delta))]
            self._iter_cursor = 0
            self._agent_scroll = 0
            self._step_scroll = 0
        elif self._depth == 1:
            tv = self._cur_tv()
            n = len(getattr(tv, "iterations", []) or []) if tv else 0
            if abs(delta) >= 10:
                self._agent_scroll = max(0, self._agent_scroll + delta)
            elif n:
                self._iter_cursor = max(0, min(n - 1, self._iter_cursor + delta))
                # Ansicht mitführen: jede Iteration ≈ 3+tools Zeilen, grob nachziehen
                self._agent_scroll = max(0, self._iter_cursor * 4 - 6)
        else:
            self._step_scroll = max(0, self._step_scroll + delta)
        self.invalidate()

    def _jump(self, top: bool):
        if self._depth == 1:
            self._agent_scroll = 0 if top else 10 ** 6
        elif self._depth == 2:
            self._step_scroll = 0 if top else 10 ** 6
        self.invalidate()

    def _go_deeper(self):
        if self._depth == 0:
            if self._cur_tv():
                self._depth = 1
                self._iter_cursor = 0
                self._agent_scroll = 0
        elif self._depth == 1:
            tv = self._cur_tv()
            its = getattr(tv, "iterations", []) or [] if tv else []
            if its:
                # Cursor läuft über reversed Anzeige → mappe auf echte Iteration
                iv = list(reversed(its))[self._iter_cursor]
                self._selected_iter_n = getattr(iv, "n", self._iter_cursor)
                self._depth = 2
                self._step_scroll = 0
        self.invalidate()

    def _go_up(self, event):
        if self._depth == 2:
            self._depth = 1
            self._selected_iter_n = None
            self._step_scroll = 0
        elif self._depth == 1:
            self._depth = 0
            self._agent_scroll = 0
        else:
            event.app.exit()
            return
        self.invalidate()

    # ── Render: LEFT (Fleet) ────────────────────────────────────────────────
    def _render_left(self) -> FT:
        self._loop = asyncio.get_running_loop()
        tvs = self._tvs()
        out: FT = [(P["cyan"] + " bold", "  ◎ ZEN+ FLEET\n"),
                   (P["dim"], "  ─────────────────────────\n")]
        if not tvs:
            out.append((P["dim"], "\n  (keine Tasks)\n"))
            return out

        order = self._flat_order()
        sub_ids = set()
        for tv in tvs.values():
            sub_ids.update(getattr(tv, "sub_task_ids", {}).values())

        for tid in order:
            tv = tvs[tid]
            is_sub = tid in sub_ids
            sel = tid == self._selected
            sym, col = _STATUS.get(tv.status, ("◯", P["dim"]))

            row_bg = " " + P["sel"] if sel else ""
            ptr = "▸" if sel and self._depth == 0 else ("›" if sel else " ")
            indent = "   └ " if is_sub else "  "

            name = tv.agent_name
            if getattr(tv, "is_swarm_summary", False):
                name = "🐝 " + name
            elif is_sub:
                name = name.split("_")[0]

            out.append((P["cyan"] + row_bg, f" {ptr}"))
            out.append((P["dim"] + row_bg, indent))
            out.append((col + row_bg, f"{sym} "))
            out.append((P["text"] + row_bg, f"{_short(name, 12):<12} "))
            out.append((P["dim"] + row_bg,
                        f"{tv.iteration:>2}/{tv.max_iter:<2} "))
            # Token-Heat als Einzelzeichen
            if getattr(tv, "tokens_max", 0):
                pct = min(99, int(100 * tv.tokens_used / tv.tokens_max))
                heat = P["green"] if pct < 50 else (P["amber"] if pct < 80 else P["red"])
                out.append((heat + row_bg, f"{pct:>2}%"))
            else:
                out.append((P["dim"] + row_bg, "  ·"))
            out.append(("", "\n"))

        out.append((P["dim"], "\n  ─────────────────────────\n"))
        run = sum(1 for t in tvs.values() if t.status == "running")
        done = sum(1 for t in tvs.values() if t.status in ("completed", "done"))
        bad = sum(1 for t in tvs.values() if t.status in ("failed", "error"))
        out.append((P["dim"], f"  Σ {len(tvs)}  "))
        out.append((P["cyan"], f"⟳{run} "))
        out.append((P["green"], f"●{done} "))
        if bad:
            out.append((P["red"], f"✗{bad}"))
        out.append(("", "\n"))
        return out

    # ── Render: RIGHT Router ────────────────────────────────────────────────
    def _render_right(self) -> FT:
        tv = self._cur_tv()
        if not tv:
            return [(P["dim"], "\n   Wähle links einen Task (W/S, →)\n")]
        if self._depth == 2:
            return self._render_step(tv)
        return self._render_agent(tv)

    # Stats-Header (geteilt zwischen Ebene 1 und 2)
    def _render_stats(self, tv) -> FT:
        sym, col = _STATUS.get(tv.status, ("◯", P["dim"]))
        elapsed = (getattr(tv, "completed_at", None) or time.time()) - tv.started_at
        crumb = f" {tv.agent_name} "
        if self._depth == 2:
            crumb += f"› iter {self._selected_iter_n} "
        out: FT = [
            (col + " bold", f"\n  {sym}"),
            (P["cyan"] + " bold", crumb),
            (P["dim"], f"[{tv.status}]  {_fmt_elapsed(elapsed)}\n"),
            (P["dim"], "  ┌─ query    "),
            (P["text"], _short(getattr(tv, "query", ""), 90) + "\n"),
        ]
        if getattr(tv, "persona", "") and tv.persona != "default":
            out += [(P["dim"], "  ├─ persona  "), (P["violet"], tv.persona + "\n")]
        if getattr(tv, "narrator_msg", ""):
            out += [(P["dim"], "  ├─ narrator "),
                    (P["blue"], _short(tv.narrator_msg, 90) + "\n")]
        out.append((P["dim"], "  └─ "))
        out.append((P["cyan"], _bar(tv.iteration, tv.max_iter, 14)))
        out.append((P["dim"], f" {tv.iteration}/{tv.max_iter}  "))
        if getattr(tv, "tokens_max", 0):
            pct = min(100, int(100 * tv.tokens_used / tv.tokens_max))
            heat = P["green"] if pct < 50 else (P["amber"] if pct < 80 else P["red"])
            out.append((heat, f"tok {tv.tokens_used}/{tv.tokens_max} ({pct}%)"))
        if getattr(tv, "is_swarm_summary", False):
            phase = getattr(tv, "swarm_phase", "") or "init"
            out.append((f"fg:{_PHASE_COL.get(phase, '#6b7280')} bold",
                        f"   {phase.upper()}"))
            subs = getattr(tv, "sub_agents", {}) or {}
            if subs:
                d = sum(1 for s in subs.values() if s == 1)
                out.append((P["pink"], f"  🐝 {d}/{len(subs)}"))
        out.append(("", "\n"))
        return out

    # ── Ebene 1: Agent-Overview ─────────────────────────────────────────────
    def _render_agent(self, tv) -> FT:
        head = self._render_stats(tv)
        body: FT = []
        its = list(reversed(getattr(tv, "iterations", []) or []))
        if not its:
            body.append((P["dim"], "\n   (noch keine Iterationen)\n"))
        for disp_idx, iv in enumerate(its):
            cur = disp_idx == self._iter_cursor
            ptr = "▸ " if cur else "  "
            row = " " + P["sel"] if cur else ""
            health = _spark(self._iter_health(iv))
            body.append((P["cyan"] + row, f"\n {ptr}"))
            body.append((P["dim"] + row, f"── iter {getattr(iv, 'n', '?')} ── "))
            body.append((P["green"] + row, health))
            if getattr(iv, "thinking", False):
                body.append((P["amber"] + row, "  ◎ denkt…"))
            body.append(("", "\n"))
            # 1-Zeilen-Gedanke
            th = (getattr(iv, "thoughts", [""]) or [""])[-1].strip()
            if th:
                body.append((P["dim"] + row, "      ◎ "))
                body.append((P["text"] + row, _short(th.split("\n")[-1], 84) + "\n"))
            # Tool-Tabelle kompakt
            for (name, ok, el, info) in getattr(iv, "tools", []) or []:
                oks, okc = ("✓", P["green"]) if ok else ("✗", P["red"])
                body.append((P["blue"] + row, f"      ◇ {_short(name, 22):<22} "))
                body.append((okc + row, oks))
                body.append((P["dim"] + row,
                             f" {el:5.2f}s  {_short(info or '', 44)}\n"))
            for p_name in (getattr(iv, "pending_tools", {}) or {}):
                body.append((P["amber"] + row, f"      ⟳ {_short(p_name, 40)}\n"))

        if getattr(tv, "final_answer", ""):
            body.append((P["dim"], "\n  ─── final ───\n"))
            body += _wrap_block(tv.final_answer, 92, P["green"])

        total = _count_lines(body)
        self._agent_scroll = min(self._agent_scroll, max(0, total - 5))
        return head + _apply_scroll(body, self._agent_scroll)

    # ── Ebene 2: Step-Detail (volles I/O) ───────────────────────────────────
    def _render_step(self, tv) -> FT:
        head = self._render_stats(tv)
        iv = None
        for cand in getattr(tv, "iterations", []) or []:
            if getattr(cand, "n", None) == self._selected_iter_n:
                iv = cand
                break
        if iv is None:
            return head + [(P["red"], "\n  Iteration nicht gefunden.\n")]

        body: FT = [(P["dim"], "\n  ═══ THOUGHTS ═══\n")]
        th = (getattr(iv, "thoughts", [""]) or [""])[-1].strip()
        body += _wrap_block(th, 96, P["text"]) if th else [(P["dim"], "  (leer)\n")]

        body.append((P["dim"], "\n  ═══ TOOL I/O ═══\n"))
        tools = getattr(iv, "tools", []) or []
        raws = getattr(iv, "tools_raw", []) or []
        if not tools:
            body.append((P["dim"], "  (keine Tools in dieser Iteration)\n"))
        for idx, (name, ok, el, info) in enumerate(tools):
            oks, okc = ("✓", P["green"]) if ok else ("✗", P["red"])
            body.append((P["blue"] + " bold", f"\n  ◇ {name} "))
            body.append((okc, oks))
            body.append((P["dim"], f"  {el:.2f}s"))
            if info:
                body.append((P["dim"], f"  · {_short(info, 60)}"))
            body.append(("", "\n"))

            raw_in, raw_out = "", ""
            if idx < len(raws):
                rec = raws[idx]
                # tools_raw: (name, result, input)
                raw_out = rec[1] if len(rec) > 1 else ""
                raw_in = rec[2] if len(rec) > 2 else ""

            body.append((P["violet"], "    → input\n"))
            parsed = _try_parse_struct(raw_in)
            if parsed is not None and isinstance(parsed, (dict, list)):
                body += [(s, "    " + t) if t and not t.startswith("\n") else (s, t)
                         for s, t in _format_struct(parsed, 3)]
            else:
                body += _wrap_block(raw_in or "(leer)", 92, P["text"], "      ")

            body.append((P["green"], "    ← output\n"))
            parsed = _try_parse_struct(raw_out)
            if parsed is not None and isinstance(parsed, (dict, list)):
                body += _format_struct(parsed, 3)
            else:
                body += _wrap_block(raw_out or "(leer)", 92, P["text"], "      ")

        total = _count_lines(body)
        self._step_scroll = min(self._step_scroll, max(0, total - 5))
        return head + _apply_scroll(body, self._step_scroll)

    # ── Footer ──────────────────────────────────────────────────────────────
    def _render_footer(self) -> FT:
        common = "  q/Esc=close"
        if self._depth == 0:
            hint = " FLEET   W/S=task  D/→=open agent" + common
        elif self._depth == 1:
            hint = " AGENT   W/S=iter  D/→=step detail  A/←=fleet  g/G=top/end  PgUp/Dn=scroll" + common
        else:
            hint = " STEP    W/S=scroll  PgUp/Dn=±10  g/G=top/end  A/←=back" + common
        return [("fg:#0f172a bg:#67e8f9", f" ZEN+ "),
                ("fg:#9ca3af bg:#0f172a", hint + " " * 200)]
