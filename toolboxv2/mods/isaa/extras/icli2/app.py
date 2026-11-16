"""Textual-UI der icli 2.0.

Regel: hier laeuft nichts, was warten kann. Die UI sendet Commands an die
Engine und holt pro Frame Events ab (`Engine.poll`). Kein await auf Agenten,
kein Dateizugriff, kein Netzwerk.
"""
from __future__ import annotations

import time

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible, Input, Markdown, Static

from . import VERSION
from . import protocol as P
from .engine import Engine
from .picker import Picker
from .theme import DIM, ISAA_THEME, SYM, G, banner

FRAME = 1 / 30                 # Abholtakt fuer Engine-Events
EVENTS_PER_FRAME = 200         # Obergrenze je Frame: UI bleibt auch bei Flut bedienbar
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
END_MARK = {"done": "", "failed": f"{SYM['fail']} fehlgeschlagen", "stopped": "■ gestoppt",
            "paused": f"{SYM['pause']} pausiert – nächste Nachricht setzt fort"}

# Befehl -> (Argument-Hinweis, Beschreibung). Reihenfolge = Reihenfolge in der Palette.
COMMANDS: dict[str, tuple[str, str]] = {
    "/agent": ("[name]", "Agent wechseln"),
    "/model": ("[name] [fast|complex]", "Modell des Agenten wählen"),
    "/session": ("[name|+]", "Session wechseln oder neu anlegen"),
    "/mount": ("[ordner]", "Ordner einhängen, ohne Argument: Mounts verwalten"),
    "/new": ("", "Neue Session mit leerem Kontext"),
    "/reset": ("", "Verlauf der aktuellen Session leeren"),
    "/open": ("[n]", "Letzte n Tool-Ergebnisse aufklappen"),
    "/fold": ("", "Alle Tool-Details auf- oder zuklappen"),
    "/pack": ("[name]", "Agent mit allen Tools und Checkpoint sichern"),
    "/stop": ("", "Laufenden Lauf abbrechen"),
    "/clear": ("", "Anzeige leeren"),
    "/q": ("", "Beenden"),
}
ALIASES = {"/quit": "/q", "/exit": "/q"}
MODEL_TARGETS = [("both", "beide", "schnell und komplex"), ("fast", "schnell", "Standard-Antworten"),
                 ("complex", "komplex", "schwere Aufgaben")]


class ToolCard(Collapsible):
    """Eine Zeile pro Tool-Aufruf, aufklappbar fuer Parameter und Ergebnis."""

    def __init__(self, name: str, summary: str, detail: str):
        self._tool, self._summary, self._detail = name, summary, detail
        self._body = Static(detail, classes="tool-body", markup=False)
        super().__init__(self._body, title=self._head(SYM["tool"]), collapsed=True, classes="tool")

    def _head(self, mark: str, tail: str = "") -> str:
        line = f"{mark} {self._tool}"
        if self._summary:
            line += f"  · {self._summary}"
        return line + tail

    def finish(self, ok: bool, summary: str, detail: str, secs: float) -> None:
        tail = (f"  → {summary}" if summary else "") + f"  {secs:.1f}s"
        self.title = self._head(SYM["ok"] if ok else SYM["fail"], tail)
        self.set_class(not ok, "tool-failed")
        self._body.update(f"{self._detail}\n\nErgebnis\n{detail}")


class RunView:
    """Widgets eines Laufs. Text und Tool-Karten erscheinen in Stream-Reihenfolge."""

    def __init__(self, chat: VerticalScroll, app: "ICli2"):
        self.chat, self.app = chat, app
        self.answer = ""
        self.md: Markdown | None = None
        self.stream = None
        self.think: Static | None = None
        self.think_text = ""
        self.tools: dict[str, ToolCard] = {}
        self.started = time.monotonic()

    async def close_segment(self) -> None:
        if self.stream is not None:
            await self.stream.stop()
        self.md = self.stream = self.think = None
        self.think_text = ""

    async def text(self, delta: str) -> None:
        if self.md is None:
            await self.close_segment()
            self.md = Markdown("", classes="answer")
            await self.chat.mount(self.md)
            self.stream = Markdown.get_stream(self.md)
        self.answer += delta
        await self.stream.write(delta)

    async def thinking(self, delta: str) -> None:
        if self.think is None:
            await self.close_segment()
            self.think = Static("", classes="think-body", markup=False)
            box = Collapsible(self.think, title=f"{SYM['think']} Gedanken",
                              collapsed=not self.app.show_thinking, classes="think")
            self.app.think_boxes.append(box)
            await self.chat.mount(box)
        self.think_text += delta
        self.think.update(self.think_text)

    async def tool_start(self, tid: str, name: str, summary: str, detail: str) -> None:
        await self.close_segment()
        card = self.tools[tid] = ToolCard(name, summary, detail)
        self.app.tool_cards.append(card)
        await self.chat.mount(card)

    def tool_end(self, tid: str, ok: bool, summary: str, detail: str, secs: float) -> None:
        card = self.tools.pop(tid, None)
        if card is not None:
            card.finish(ok, summary, detail, secs)


class ICli2(App):
    TITLE = "icli"
    CSS = """
    #head { height: 1; padding: 0 1; color: $text-muted; }
    #chat { padding: 0 1; scrollbar-size-vertical: 1; }
    #banner { margin: 1 0; }
    #bottom { height: auto; }
    #row { height: 3; border: tall #3b5568; background: #0c161c; padding: 0 1; }
    #row.-run { border: tall $primary; }
    #row.-done { border: tall $success; }
    #row.-ask { border: tall $warning; }
    #row.-fail { border: tall $error; }
    #mark { width: 2; color: $accent; }
    #prompt, #prompt:focus { border: none; height: 1; padding: 0; background: transparent; }
    #foot { height: 1; padding: 0 1; color: $text-muted; }
    .user { margin: 1 0 0 0; color: $accent; text-style: bold; }
    .answer { margin: 0; padding: 0 0 0 2; background: transparent; }
    .answer MarkdownH1, .answer MarkdownH2, .answer MarkdownH3 {
        content-align: left middle; text-style: bold; color: $accent;
        background: transparent; border: none; margin: 1 0 0 0; padding: 0; }
    .answer MarkdownFence { background: $panel 40%; margin: 0 0 1 0; }
    .tool, .think { margin: 0; padding: 0 0 0 2; border: none; background: transparent; }
    .tool > CollapsibleTitle, .think > CollapsibleTitle { padding: 0; color: $text-muted; }
    .tool > CollapsibleTitle:hover, .think > CollapsibleTitle:hover,
    .tool > CollapsibleTitle:focus, .think > CollapsibleTitle:focus {
        background: transparent; color: $accent; }
    .tool-failed > CollapsibleTitle { color: $error; }
    .tool > Contents, .think > Contents { padding: 0 0 0 2; }
    .tool-body, .think-body { color: $text-muted; border-left: solid $secondary; padding: 0 1; }
    .note { padding: 0 0 0 2; color: $text-muted; }
    .note-error { color: $error; }
    .note-warn { color: $warning; }
    """
    BINDINGS = [
        Binding("escape", "stop", "Stopp", priority=True),
        Binding("ctrl+q", "quit_app", "Beenden", priority=True),
        Binding("ctrl+l", "clear", "Leeren"),
        Binding("ctrl+p", "palette", "Befehle", priority=True),
        Binding("up", "history(-1)", show=False),
        Binding("down", "history(1)", show=False),
        Binding("ctrl+o", "fold_tools", "Tools auf/zu", priority=True),
        Binding("ctrl+r", "fold_thinking", "Gedanken auf/zu", priority=True),
        *[Binding(f"alt+{n}", f"open_last({n})", show=False, priority=True) for n in range(1, 10)],
    ]

    def __init__(self, engine: Engine, agent: str = "self", session: str = "default"):
        super().__init__()
        self.engine = engine
        self.agent, self.session, self.model = agent, session, "…"
        self.complex_model, self.mounts = "", 0
        self.history: list[str] = []
        self._hist_pos = 0
        self.ready = False
        self.runs: dict[str, RunView] = {}
        self.tool_cards: list[ToolCard] = []
        self.think_boxes: list[Collapsible] = []
        self.show_thinking = False
        self.focus_run = ""
        self.status_msg = ""
        self.tokens: tuple[int, int] | None = None
        self._tick = 0
        self._esc_run = ""

    # ── Aufbau ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static(id="head")
        with VerticalScroll(id="chat"):
            yield Static(banner(VERSION), id="banner")
        with Vertical(id="bottom"):
            with Horizontal(id="row"):
                yield Static(SYM["progress"], id="mark")
                yield Input(placeholder="Nachricht oder /befehl", id="prompt")
            yield Static(id="foot")

    def on_mount(self) -> None:
        self.register_theme(ISAA_THEME)
        self.theme = "isaa"
        self.chat = self.query_one("#chat", VerticalScroll)
        self.chat.anchor()                               # bleibt unten, solange nicht gescrollt wird
        self.query_one("#prompt", Input).focus()
        self.set_interval(FRAME, self._pump)
        self.set_interval(0.1, self._chrome)
        self._chrome()

    # ── Engine -> UI ─────────────────────────────────────────────────────────
    async def _pump(self) -> None:
        for ev in self.engine.poll(EVENTS_PER_FRAME):
            await self._handle(ev)

    async def _handle(self, ev: P.Event) -> None:
        d, view = ev.data, self.runs.get(ev.run)
        if ev.kind == P.TEXT and view:
            await view.text(d["text"])
        elif ev.kind == P.THINK and view:
            await view.thinking(d["text"])
        elif ev.kind == P.TOOL_START and view:
            await view.tool_start(d["id"], d["name"], d["summary"], d["detail"])
        elif ev.kind == P.TOOL_END and view:
            view.tool_end(d["id"], d["ok"], d["summary"], d["detail"], d["secs"])
        elif ev.kind == P.STATUS:
            if d.get("msg"):
                self.status_msg = d["msg"]
            if d.get("tokens") is not None and d.get("tokens_max"):
                self.tokens = (d["tokens"], d["tokens_max"])
        elif ev.kind == P.RUN_START:
            self.runs[ev.run] = RunView(self.chat, self)
            self.focus_run = ev.run
            self._signal("run")
            if d.get("steer"):
                await self._note("↪ Nachsteuerung an laufenden Lauf gesendet")
        elif ev.kind == P.RUN_END:
            if view:
                await view.close_segment()
                secs = time.monotonic() - view.started
                mark = END_MARK.get(d["status"], d["status"])
                if d.get("error"):
                    await self._note(f"✗ {d['error']}", "error")
                elif mark:
                    await self._note(f"{mark} · {secs:.1f}s")
            self.runs.pop(ev.run, None)
            if not self.runs:
                # Signalfarbe der Eingabeleiste: fertig / Frage / Absturz
                asks = d["status"] == "paused" or (view and (d.get("answer") or view.answer).rstrip().endswith("?"))
                self._signal("fail" if d["status"] == "failed" else "ask" if asks else
                             "done" if d["status"] == "done" else "")
            if self.focus_run == ev.run:
                self.focus_run = next(reversed(self.runs), "")
            self.status_msg = ""
        elif ev.kind == P.READY:
            if self.ready and (d["agent"], d["session"], d["model"], d["complex_model"]) != \
                    (self.agent, self.session, self.model, self.complex_model):
                await self._note(f"{SYM['agent']} {d['agent']} · {d['model']} · {d['session']}")
            self.ready = True
            self.agent, self.session, self.model = d["agent"], d["session"], d["model"]
            self.complex_model, self.mounts = d["complex_model"], d["mounts"]
            if "history" in d:
                self.history, self._hist_pos = d["history"], len(d["history"])
        elif ev.kind == P.ROWS:
            self._pick(d["what"], d["title"], d["rows"])
        elif ev.kind == P.NOTICE:
            await self._note(d["text"], d.get("level", "info"))

    def _signal(self, state: str) -> None:
        row = self.query_one("#row")
        for name in ("run", "done", "ask", "fail"):
            row.set_class(name == state, f"-{name}")

    async def _note(self, text: str, level: str = "info") -> None:
        await self.chat.mount(Static(text, classes=f"note note-{level}", markup=False))

    def _chrome(self) -> None:
        """Kopf- und Fusszeile. Rein lokal, deshalb auch bei blockierter Engine fluessig."""
        self._tick += 1
        mounts = f" · {self.mounts} mount{'s' if self.mounts != 1 else ''}" if self.mounts else ""
        self.query_one("#head", Static).update(
            f"[{G.L4_BRIGHT}]{SYM['agent']} {self.agent}[/] "
            f"[{DIM}]· {self.model} · {self.session}{mounts}[/]")
        spin = SPIN[self._tick % len(SPIN)]
        if not self.ready:
            left = f"{spin} Engine startet …"
        elif self.engine.lag() > 1.5:
            left = f"{spin} Engine beschäftigt ({self.engine.lag():.0f}s) – Eingabe bleibt frei"
        elif self.runs:
            more = f" (+{len(self.runs) - 1})" if len(self.runs) > 1 else ""
            left = f"{spin} {self.status_msg or 'arbeitet'}{more}"
        else:
            left = f"[{G.L3_MID}]{SYM['done']}[/] bereit"
        if self.tokens:
            left += f" · {self.tokens[0] / 1000:.1f}k/{self.tokens[1] / 1000:.0f}k tok"
        self.query_one("#foot", Static).update(left)

    # ── UI -> Engine ─────────────────────────────────────────────────────────
    @on(Input.Submitted, "#prompt")
    async def _submit(self, message: Input.Submitted) -> None:
        text = message.value.strip()
        message.input.value = ""
        if not text:
            return
        if not self.runs:
            self._signal("")
        if not self.history or self.history[-1] != text:
            self.history.append(text)
            self.engine.send(P.DO, action="remember", text=text)
        self._hist_pos = len(self.history)
        if text.startswith("/"):
            await self._command(text)
            return
        await self.chat.mount(Static(f"{SYM['progress']} {text}", classes="user", markup=False))
        self.chat.scroll_end(animate=False)
        self.engine.send(P.CHAT, query=text, agent=self.agent, session=self.session)

    async def _command(self, text: str) -> None:
        name, _, arg = text.partition(" ")
        name, arg = ALIASES.get(name.lower(), name.lower()), arg.strip()
        do = lambda action, **kw: self.engine.send(P.DO, action=action, **kw)   # noqa: E731
        ask = lambda what: self.engine.send(P.LIST, what=what)                  # noqa: E731
        if name == "/":
            self.action_palette()
        elif name == "/q":
            self.action_quit_app()
        elif name == "/stop":
            if self.focus_run:
                self.engine.send(P.STOP, run=self.focus_run)
        elif name == "/clear":
            await self.action_clear()
        elif name == "/agent":
            do("agent", name=arg) if arg else ask("agents")
        elif name == "/model":
            if arg:
                model, _, target = arg.partition(" ")
                do("model", name=model, target=target.strip() or "both")
            else:
                ask("models")
        elif name == "/session":
            do("session", name=arg) if arg else ask("sessions")
        elif name == "/new":
            do("session", name="+")
        elif name == "/reset":
            do("session_clear")
        elif name == "/mount":
            do("mount", path=arg) if arg else ask("mounts")
        elif name == "/open":
            self.action_open_last(int(arg) if arg.isdigit() else 1)
        elif name == "/fold":
            self.action_fold_tools()
        elif name == "/pack":
            do("pack", name=arg)
        else:
            await self._note(f"Unbekannter Befehl {name} – Strg+P zeigt alle", "warn")

    def _pick(self, what: str, title: str, rows: list[dict]) -> None:
        """Picker fuer eine Liste aus der Engine oeffnen und die Wahl zurueckschicken."""
        if not rows:
            self.run_worker(self._note(title, "warn"))
            return
        send = self.engine.send

        def chosen(value: str | None) -> None:
            if value is None:
                return
            if what == "agents":
                send(P.DO, action="agent", name=value)
            elif what == "sessions":
                send(P.DO, action="session", name=value)
            elif what == "mounts":
                send(P.DO, action="unmount", at=value)
            elif what == "models":
                self.push_screen(
                    Picker(f"{value} für", MODEL_TARGETS),
                    lambda target: target and send(P.DO, action="model", name=value, target=target))

        self.push_screen(Picker(title, [(r["value"], r["label"], r["hint"]) for r in rows]), chosen)

    def action_palette(self) -> None:
        items = [(name, name, f"{hint}  {text}".strip()) for name, (hint, text) in COMMANDS.items()]

        def chosen(name: str | None) -> None:
            if name:
                self.run_worker(self._command(name))     # ohne Argument: Auswahl bzw. Standard

        self.push_screen(Picker("Befehle", items), chosen)

    def action_history(self, step: int) -> None:
        prompt = self.query_one("#prompt", Input)
        if not prompt.has_focus or not self.history:
            return
        self._hist_pos = max(0, min(len(self.history), self._hist_pos + step))
        prompt.value = self.history[self._hist_pos] if self._hist_pos < len(self.history) else ""
        prompt.cursor_position = len(prompt.value)

    def action_stop(self) -> None:
        """Esc pausiert den fokussierten Lauf (fortsetzbar), ein zweites Esc bricht hart ab.
        Kann der Lauf nicht pausieren, stoppt die Engine ihn sofort."""
        if not self.focus_run:
            return
        hard = self._esc_run == self.focus_run
        self.engine.send(P.STOP if hard else P.PAUSE, run=self.focus_run)
        self._esc_run = self.focus_run

    async def action_clear(self) -> None:
        if not self.runs:
            await self.chat.remove_children()
            self.tool_cards.clear()
            self.think_boxes.clear()

    def action_fold_tools(self) -> None:
        """Alle Tool-Details auf, beim naechsten Mal alle zu."""
        opening = any(card.collapsed for card in self.tool_cards)
        for card in self.tool_cards:
            card.collapsed = not opening

    def action_fold_thinking(self) -> None:
        self.show_thinking = not self.show_thinking
        for box in self.think_boxes:
            box.collapsed = not self.show_thinking

    def action_open_last(self, count: int) -> None:
        """Die letzten `count` Tool-Ergebnisse aufklappen, alle anderen zu."""
        for index, card in enumerate(reversed(self.tool_cards)):
            card.collapsed = index >= count
        if self.tool_cards:
            self.tool_cards[-min(count, len(self.tool_cards))].scroll_visible()

    def action_quit_app(self) -> None:
        self.exit()

    def on_key(self, event: events.Key) -> None:
        prompt = self.query_one("#prompt", Input)
        if not prompt.has_focus and event.is_printable:
            prompt.focus()
