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
AGENT_STATE = {"live": "geladen", "registered": "registriert", "disk": "gespeichert"}


class ToolCard(Collapsible):
    """Eine Zeile pro Tool-Aufruf, aufklappbar fuer Argumente und Ergebnis."""

    def __init__(self, name: str, args: str):
        self._tool, self._args = name, args
        self._body = Static(args or "(keine Argumente)", classes="tool-body", markup=False)
        super().__init__(self._body, title=f"{SYM['tool']} {name}  {self._hint()}", collapsed=True,
                         classes="tool")

    def _hint(self) -> str:
        flat = " ".join(self._args.split())
        return flat[:60] + ("…" if len(flat) > 60 else "")

    def finish(self, ok: bool, result: str, secs: float) -> None:
        self.title = f"{SYM['ok'] if ok else SYM['fail']} {self._tool}  {self._hint()}  · {secs:.1f}s"
        self.set_class(not ok, "tool-failed")
        self._body.update(f"{self._args}\n\n── Ergebnis ──\n{result}" if self._args else result)


class RunView:
    """Widgets eines Laufs. Text und Tool-Karten erscheinen in Stream-Reihenfolge."""

    def __init__(self, chat: VerticalScroll):
        self.chat = chat
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
        await self.stream.write(delta)

    async def thinking(self, delta: str) -> None:
        if self.think is None:
            await self.close_segment()
            self.think = Static("", classes="think-body", markup=False)
            await self.chat.mount(Collapsible(self.think, title=f"{SYM['think']} Gedanken", collapsed=True,
                                              classes="think"))
        self.think_text += delta
        self.think.update(self.think_text)

    async def tool_start(self, tid: str, name: str, args: str) -> None:
        await self.close_segment()
        card = self.tools[tid] = ToolCard(name, args)
        await self.chat.mount(card)

    def tool_end(self, tid: str, ok: bool, result: str, secs: float) -> None:
        card = self.tools.pop(tid, None)
        if card is not None:
            card.finish(ok, result, secs)


class ICli2(App):
    TITLE = "icli"
    CSS = """
    #head { height: 1; padding: 0 1; color: $text-muted; }
    #chat { padding: 0 1; scrollbar-size-vertical: 1; }
    #banner { margin: 1 0; }
    #bottom { height: auto; border-top: solid $secondary; }
    #row { height: 1; }
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
    ]

    def __init__(self, engine: Engine, agent: str = "self", session: str = "default"):
        super().__init__()
        self.engine = engine
        self.agent, self.session, self.model = agent, session, "…"
        self.ready = False
        self.runs: dict[str, RunView] = {}
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
            await view.tool_start(d["id"], d["name"], d["args"])
        elif ev.kind == P.TOOL_END and view:
            view.tool_end(d["id"], d["ok"], d["result"], d["secs"])
        elif ev.kind == P.STATUS:
            if d.get("msg"):
                self.status_msg = d["msg"]
            if d.get("tokens") is not None and d.get("tokens_max"):
                self.tokens = (d["tokens"], d["tokens_max"])
        elif ev.kind == P.RUN_START:
            self.runs[ev.run] = RunView(self.chat)
            self.focus_run = ev.run
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
            if self.focus_run == ev.run:
                self.focus_run = next(reversed(self.runs), "")
            self.status_msg = ""
        elif ev.kind == P.READY:
            if self.ready and d["agent"] != self.agent:
                await self._note(f"{SYM['agent']} Agent: {d['agent']} · {d['model']}")
            self.ready = True
            self.agent, self.session, self.model = d["agent"], d["session"], d["model"]
        elif ev.kind == P.AGENT_LIST:
            self._pick_agent(d["agents"])
        elif ev.kind == P.NOTICE:
            await self._note(d["text"], d.get("level", "info"))

    async def _note(self, text: str, level: str = "info") -> None:
        await self.chat.mount(Static(text, classes=f"note note-{level}", markup=False))

    def _chrome(self) -> None:
        """Kopf- und Fusszeile. Rein lokal, deshalb auch bei blockierter Engine fluessig."""
        self._tick += 1
        self.query_one("#head", Static).update(
            f"[{G.L4_BRIGHT}]{SYM['agent']} {self.agent}[/] [{DIM}]· {self.model} · {self.session}[/]")
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
        if text.startswith("/"):
            await self._command(text)
            return
        await self.chat.mount(Static(f"{SYM['progress']} {text}", classes="user", markup=False))
        self.chat.scroll_end(animate=False)
        self.engine.send(P.CHAT, query=text, agent=self.agent, session=self.session)

    async def _command(self, text: str) -> None:
        name, _, arg = text.partition(" ")
        name, arg = name.lower(), arg.strip()
        if name in ("/q", "/quit", "/exit"):
            self.action_quit_app()
        elif name == "/stop":
            self.action_stop()
        elif name == "/clear":
            await self.action_clear()
        elif name == "/agent":
            # mit Name: direkt wechseln, ohne: Auswahl (Liste kommt live aus isaa)
            self.engine.send(P.USE, agent=arg) if arg else self.engine.send(P.AGENTS)
        elif name == "/pack":
            self.engine.send(P.PACK, agent=arg or self.agent)
        else:
            await self._note(f"Unbekannter Befehl {name} – /agent /pack /stop /clear /q", "warn")

    def _pick_agent(self, agents: list[dict]) -> None:
        items = [(a["name"], a["name"],
                  AGENT_STATE.get(a["state"], a["state"]) + (" · Paket mit Tools" if a.get("pack") else "")
                  + (" · aktiv" if a["name"] == self.agent else ""))
                 for a in agents]

        def chosen(name: str | None) -> None:
            if name and name != self.agent:
                self.engine.send(P.USE, agent=name)

        self.push_screen(Picker("Agent wechseln", items), chosen)

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

    def action_quit_app(self) -> None:
        self.exit()

    def on_key(self, event: events.Key) -> None:
        prompt = self.query_one("#prompt", Input)
        if not prompt.has_focus and event.is_printable:
            prompt.focus()
