"""Sub-Agent als Kachel.

Klein (Standard): Aufgabe, Kennzahlen (Schritt x/max, Laufzeit, Restzeit, Tokens/Budget) und die
letzten drei Aktionen - ein einziges Widget. Klick oeffnet die grosse Ansicht: oben die volle Aufgabe,
darunter alle Aktionen in einem eigenen Scrollbereich (einzeln aufklappbar), unten Ergebnis oder
Fehler. Klick auf den Kopf schliesst sie wieder.

SubView haelt die Daten und bietet dieselbe Schnittstelle wie RunView fuer die UI-Ereignisse
(add/flush/tool_start/tool_end/note/close), dazu info() fuer das SUB-Ereignis der Engine.
"""
from __future__ import annotations

import time

from rich.text import Text
from textual import events
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from . import protocol as P
from .errors import explain, render
from .richmd import FrozenMarkdown
from .theme import DIM, G, SYM, ZEN
from .widgets import FoldLine

COMPACT_ACTIONS = 3
MAX_ACTIONS = 300
FINAL = ("completed", "failed", "max_iterations", "timeout", "cancelled", "ended")
MARK = {"running": SYM["agent"], "completed": SYM["ok"], "failed": SYM["fail"], "max_iterations": "◷",
        "timeout": "⏱", "cancelled": "■", "ended": "■"}
COLOR = {"running": G.L3_MID, "completed": ZEN["green"], "failed": ZEN["red"], "max_iterations": ZEN["amber"],
         "timeout": ZEN["amber"], "cancelled": DIM, "ended": DIM}
STATE = {"completed": "fertig", "failed": "fehlgeschlagen", "max_iterations": "Iterationslimit",
         "timeout": "Zeitlimit", "cancelled": "abgebrochen", "ended": "beendet"}


def _dur(secs: float) -> str:
    secs = max(0, int(secs))
    return f"{secs // 60}:{secs % 60:02d}" if secs < 3600 else f"{secs // 3600}:{secs % 3600 // 60:02d} h"


def _tok(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _snippet(text: str, n: int = 70) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    line = " ".join(lines[-1].split()) if lines else ""
    return line if len(line) <= n else line[: n - 1] + "…"


class SubView:
    def __init__(self, sid: str, app):
        self.id, self.app = sid, app
        self.task, self.status = sid, "running"
        self.iter: int | None = None
        self.max_iter: int | None = None
        self.tokens = self.budget = self.timeout = 0
        self.started, self.ended = time.time(), 0.0
        self.error = self.result = self.answer = ""
        self.actions: list[dict] = []
        self.tools: dict[str, dict] = {}
        self.queue: list[list] = []
        self.last_flush = 0.0
        self.tile = SubTile(self)

    # ── Kennzahlen ───────────────────────────────────────────────────────────
    @property
    def label(self) -> str:
        first = self.task.strip().splitlines()[0] if self.task.strip() else self.id
        return first if len(first) <= 60 else first[:59] + "…"

    @property
    def seconds(self) -> float:
        return (self.ended or time.time()) - self.started

    def metrics(self) -> str:
        parts = []
        if self.iter is not None:
            parts.append(f"Schritt {self.iter}/{self.max_iter}" if self.max_iter else f"Schritt {self.iter}")
        parts.append(_dur(self.seconds))
        if self.timeout and self.status == "running":
            parts.append(f"noch {_dur(self.timeout - self.seconds)}")
        if self.budget:
            parts.append(f"{_tok(self.tokens)}/{_tok(self.budget)} Tokens")
        elif self.tokens:
            parts.append(f"{_tok(self.tokens)} Tokens")
        return " · ".join(parts)

    # ── Ereignisse (gleiche Schnittstelle wie RunView) ────────────────────────
    def add(self, kind: str, text: str) -> None:
        if self.queue and self.queue[-1][0] == kind:
            self.queue[-1][1] += text
        else:
            self.queue.append([kind, text])

    async def flush(self) -> None:
        queue, self.queue = self.queue, []
        self.last_flush = time.monotonic()
        for kind, text in queue:
            if kind == P.TEXT:
                self.answer += text
            self._text_action("text" if kind == P.TEXT else "think", text)
        if queue:
            await self.tile.changed()

    def _text_action(self, kind: str, text: str) -> None:
        last = self.actions[-1] if self.actions else None
        if last is not None and last["kind"] == kind and last.get("open"):
            last["body"] += text
        else:
            self._close_text()
            self._append({"kind": kind, "body": text, "open": True})

    def _close_text(self) -> None:
        if self.actions and self.actions[-1].get("open"):
            self.actions[-1]["open"] = False

    def _append(self, action: dict) -> None:
        self.actions.append(action)
        if len(self.actions) > MAX_ACTIONS:                # sehr lange Laeufe: aelteste fallen weg
            dropped = self.actions.pop(0)
            if dropped.get("w") is not None and dropped["w"].is_attached:
                dropped["w"].remove()

    async def close_segment(self) -> None:
        self._close_text()

    async def tool_start(self, tid: str, name: str, summary: str, detail: str) -> None:
        self._close_text()
        action = {"kind": "tool", "name": name, "summary": summary, "body": detail, "ok": None}
        self.tools[tid] = action
        self._append(action)
        await self.tile.changed()

    def tool_end(self, tid: str, ok: bool, summary: str, detail: str, secs: float) -> None:
        action = self.tools.pop(tid, None)
        if action is None:
            return
        action.update(ok=ok, result=summary, secs=secs, body=f"{action['body']}\n\nErgebnis\n{detail}")
        self.tile.redraw_action(action)
        self.tile.redraw()

    async def note(self, text: str, level: str = "info") -> None:
        self._close_text()
        self._append({"kind": "note", "body": text, "level": level})
        await self.tile.changed()

    async def info(self, d: dict) -> None:
        """SUB-Ereignis der Engine: Rahmendaten und Status."""
        self.task = d.get("task") or self.task
        for key in ("iter", "max_iter"):
            if d.get(key) is not None:
                setattr(self, key, int(d[key]))
        self.tokens = int(d.get("tokens") or self.tokens)
        self.budget = int(d.get("budget") or self.budget)
        self.timeout = int(d.get("timeout") or self.timeout)
        if d.get("started"):
            self.started = float(d["started"])
        status = d.get("status") or self.status
        if d.get("final") or status in FINAL:
            self.status = status if status != "running" else "completed"
            self.ended = self.ended or time.time()
            self.error = d.get("error") or self.error
            self.result = d.get("result") or self.result
        await self.tile.changed()

    async def close(self) -> None:
        await self.flush()
        self._close_text()
        if self.status == "running":                     # Hauptlauf ist zu Ende, ohne Rueckmeldung
            self.status, self.ended = "ended", time.time()
        await self.tile.changed()

    # ── Darstellung einer Aktion ─────────────────────────────────────────────
    @staticmethod
    def title(action: dict) -> str:
        kind = action["kind"]
        if kind == "tool":
            mark = SYM["tool"] if action.get("ok") is None else SYM["ok"] if action["ok"] else SYM["fail"]
            line = f"{mark} {action['name']}"
            if action.get("summary"):
                line += f"  · {action['summary']}"
            if action.get("result"):
                line += f"  → {action['result']}"
            if action.get("secs") is not None:
                line += f"  {action['secs']:.1f}s"
            return line
        if kind == "think":
            return f"{SYM['think']} {_snippet(action['body'])}"
        if kind == "note":
            return action["body"]
        return f"✎ {_snippet(action['body'])}"


class SubTile(Vertical):
    """Klein: ein Static. Gross: Kopf, Scrollbereich mit allen Aktionen, Ergebnis/Fehler."""

    def __init__(self, view: SubView):
        super().__init__(classes="sub")
        self.view = view
        self.expanded = False
        self._lines = 0
        self.compact = Static("", classes="sub-compact", markup=False)
        self.head: Static | None = None
        self.list: VerticalScroll | None = None
        self.foot: Vertical | None = None
        self._foot_key = None

    def compose(self):
        yield self.compact

    def on_mount(self) -> None:
        self.redraw()

    # ── klein ────────────────────────────────────────────────────────────────
    def _head_text(self, full: bool) -> Text:
        v = self.view
        text = Text()
        text.append(f"{MARK.get(v.status, '?')} ", style=COLOR.get(v.status, DIM))
        text.append(f"{SYM['sub']} ", style=G.L2_DARK)
        text.append(v.task.strip() if full else v.label, style=f"bold {G.L4_BRIGHT}" if full else "bold")
        if v.status != "running":
            text.append(f"  {STATE.get(v.status, v.status)}", style=COLOR.get(v.status, DIM))
        text.append(f"\n{v.metrics()}", style=DIM)
        return text

    def compact_text(self) -> Text:
        v = self.view
        text = self._head_text(full=False)
        for action in v.actions[-COMPACT_ACTIONS:]:
            style = ZEN["red"] if action["kind"] == "tool" and action.get("ok") is False else ""
            text.append("\n  " + SubView.title(action), style=style)
        if v.status == "failed" and v.error:
            text.append(f"\n  {SYM['fail']} {explain(v.error)['title']}", style=ZEN["red"])
        more = len(v.actions) - COMPACT_ACTIONS
        hint = f"… {more} weitere · " if more > 0 else ""
        if v.actions or v.result or v.error:
            text.append(f"\n  {hint}Klick öffnet die Details", style=DIM)
        return text

    def redraw(self) -> None:
        if not self.is_mounted:
            return
        if self.expanded:
            if self.head is not None:
                self.head.update(self._head_text(full=True))
            return
        text = self.compact_text()
        lines = str(text).count("\n")
        self.compact.update(text, layout=lines != self._lines)  # gleich hoch: kein Layout-Durchlauf
        self._lines = lines

    async def changed(self) -> None:
        if self.expanded:
            await self._sync()
        self.redraw()

    # ── gross ────────────────────────────────────────────────────────────────
    async def on_click(self, event: events.Click) -> None:
        if not self.expanded:
            await self.expand()
        elif event.widget is self.head:
            await self.collapse()

    async def expand(self) -> None:
        self.expanded = True
        self.add_class("-open")
        self.head = Static("", classes="sub-head-full", markup=False)
        self.list = VerticalScroll(classes="sub-actions")
        self.foot = Vertical(classes="sub-foot")
        self._foot_key = None
        await self.compact.remove()
        await self.mount_all([self.head, self.list, self.foot])
        for action in self.view.actions:
            action.pop("w", None)
        await self._sync()
        self.redraw()
        self.list.scroll_end(animate=False)

    async def collapse(self) -> None:
        self.expanded = False
        self.remove_class("-open")
        await self.remove_children()
        for action in self.view.actions:
            action.pop("w", None)
        self.head = self.list = self.foot = None
        self.compact = Static("", classes="sub-compact", markup=False)
        self._lines = -1
        await self.mount(self.compact)
        self.redraw()

    def redraw_action(self, action: dict) -> None:
        widget = action.get("w")
        if widget is not None and widget.is_attached:
            widget.title = SubView.title(action)
            widget.body = action["body"]
            widget.set_class(action.get("ok") is False, "tool-failed")

    async def _sync(self) -> None:
        """Neue Aktionen einhaengen, bestehende aktualisieren, Ergebnis/Fehler unten zeigen."""
        if self.list is None:
            return
        at_end = self.list.scroll_y >= self.list.max_scroll_y - 1
        new = []
        for action in self.view.actions:
            if action.get("w") is None:
                classes = "tool" if action["kind"] == "tool" else "think" if action["kind"] == "think" else "sub-text"
                action["w"] = FoldLine(SubView.title(action), action["body"], classes=classes)
                new.append(action["w"])
            else:
                self.redraw_action(action)
        if new:
            await self.list.mount_all(new)
            if at_end:
                self.list.scroll_end(animate=False)
        await self._sync_foot()

    async def _sync_foot(self) -> None:
        v = self.view
        key = (v.status, v.error, v.result if v.status != "running" else "")
        if self.foot is None or key == self._foot_key:
            return
        self._foot_key = key
        await self.foot.remove_children()
        parts = []
        if v.status == "failed" and v.error:
            info = explain(v.error)
            parts.append(FoldLine(f"{SYM['fail']} {info['title']}" + (f" – {info['message'][:80]}" if info["message"] else ""),
                                  render(info), classes="problem -fail"))
        elif v.status in ("max_iterations", "timeout"):
            parts.append(Static(f"◷ {STATE[v.status]} erreicht", classes="problem -limit", markup=False))
        result = (v.result or (v.answer if v.status != "running" else "")).strip()
        if result:
            parts.append(Static("Ergebnis", classes="sub-label", markup=False))
            parts.append(FrozenMarkdown(result))
        if parts:
            await self.foot.mount_all(parts)
