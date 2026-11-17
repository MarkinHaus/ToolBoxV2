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
from textual.message import Message
from rich.text import Text
from textual.widgets import Markdown, Static, TextArea

from . import VERSION
from . import protocol as P
from .engine import Engine
from .picker import BACK, Picker, fuzzy
from .errors import explain, render
from .richmd import FrozenMarkdown
from .subtile import SubView
from .widgets import FoldLine
from .theme import DIM, ISAA_THEME, SYM, G, banner

FRAME = 1 / 30                 # Abholtakt fuer Engine-Events
TEXT_EVERY = 0.08              # Antworttext hoechstens so oft schreiben, wenn man unten mitliest
TEXT_EVERY_AWAY = 0.6          # … wenn man hochgescrollt hat (der Text ist dann ohnehin nicht sichtbar)
SCROLL_QUIET = 0.3             # waehrend gescrollt wird, gar nicht schreiben
MAX_BLOCKS = 60                # so viele Bloecke bleiben im Bild; der Verlauf selbst bleibt in der Session
EVENTS_PER_FRAME = 200         # Obergrenze je Frame: UI bleibt auch bei Flut bedienbar
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
RUN_SYM = {"running": SYM["agent"], "done": SYM["done"], "failed": SYM["fail"],
           "stopped": "■", "paused": SYM["pause"]}
KEEP_RUNS = 30
END_MARK = {"done": "", "failed": f"{SYM['fail']} fehlgeschlagen", "stopped": "■ gestoppt",
            "paused": f"{SYM['pause']} pausiert – nächste Nachricht setzt fort"}

# Befehl -> (Argument-Hinweis, Beschreibung). Reihenfolge = Reihenfolge in der Palette.
COMMANDS: dict[str, tuple[str, str]] = {
    "/agent": ("[name]", "Agent wechseln"),
    "/model": ("[name] [fast|complex]", "Modell des Agenten wählen"),
    "/session": ("[name|+]", "Session wechseln oder neu anlegen"),
    "/mount": ("[ordner]", "Ordner einhängen, ohne Argument: Mounts verwalten"),
    "/task": ("", "Läufe ansehen, Fokus wechseln"),
    "/job": ("[name: auftrag]", "Jobs verwalten, mit Argument: Job anlegen"),
    "/dream": ("", "Nächtlichen Dream-Job für den Agenten anlegen"),
    "/network": ("[datei]", "Agent-Network laden, ohne Argument: aktuelles speichern"),
    "/feature": ("[name]", "Features des Agenten an/aus"),
    "/skill": ("[name]", "Skill für die nächsten Nachrichten an/aus"),
    "/mcp": ("[name befehl …]", "MCP-Server verbinden/trennen, mit Befehl: neu anlegen"),
    "/plugin": ("[name | install <quelle> | remove <name> | market [quelle]]",
                "Plugins an/aus, installieren, entfernen, Marketplace"),
    "/discord": ("[befehl …]", "Discord-Anbindung (wie in der alten icli)"),
    "/telegram": ("[befehl …]", "Telegram-Anbindung (wie in der alten icli)"),
    "/settings": ("", "Einstellungen: Iterationen, Kontext, Anmeldung, Zugangsdaten"),
    "/set": ("<name> <wert>", "Einstellung direkt setzen, z. B. /set max_iterations 50"),
    "/context": ("", "Kontext-Übersicht in der Seitenleiste zeigen"),
    "/vfs": ("[pfad]", "Dateien im VFS ansehen (Tab vervollständigt den Pfad)"),
    "/side": ("", "Seitenleiste ein/aus (F2)"),
    "/login": ("", "CloudM-Anmeldung prüfen oder starten"),
    "/tb": ("[mod.fn key=wert …]", "ToolBox-Funktion aufrufen"),
    "/share": ("", "LiveSync-Ordner verbinden oder trennen"),
    "/voice": ("", "Omni-Session und Geräte (F4 = Spracheingabe, #audio = Antwort vorlesen)"),
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
JOB_ACTIONS = [("fire", "jetzt ausführen", ""), ("pause", "pausieren / fortsetzen", ""),
               ("remove", "löschen", "")]
MODEL_TARGETS = [("both", "beide", "schnell und komplex"), ("fast", "schnell", "Standard-Antworten"),
                 ("complex", "komplex", "schwere Aufgaben")]


class ToolCard(FoldLine):
    """Eine Zeile pro Tool-Aufruf, aufklappbar fuer Parameter und Ergebnis."""

    def __init__(self, name: str, summary: str, detail: str):
        self._tool, self._summary, self._detail = name, summary, detail
        super().__init__(self._head(SYM["tool"]), detail, classes="tool")

    def _head(self, mark: str, tail: str = "") -> str:
        line = f"{mark} {self._tool}"
        if self._summary:
            line += f"  · {self._summary}"
        return line + tail

    def finish(self, ok: bool, summary: str, detail: str, secs: float) -> None:
        tail = (f"  → {summary}" if summary else "") + f"  {secs:.1f}s"
        self.set_class(not ok, "tool-failed")
        self.body = f"{self._detail}\n\nErgebnis\n{detail}"
        self.title = self._head(SYM["ok"] if ok else SYM["fail"], tail)


class PromptArea(TextArea):
    """Mehrzeilige Eingabe: Enter sendet, Umschalt+Enter oder Strg+J neue Zeile.
    Pfeil hoch/runter blaettert in der History, wenn der Cursor am Rand steht."""

    class Submitted(Message):
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    async def _on_key(self, event: events.Key) -> None:
        hints = self.app.completions                    # Vorschlaege zu einem angefangenen /befehl
        if event.key == "tab" and hints:
            event.prevent_default().stop()
            self.app.complete_accept()
        elif event.key in ("up", "down") and hints:
            event.prevent_default().stop()
            self.app.complete_move(-1 if event.key == "up" else 1)
        elif event.key == "escape" and hints:
            event.prevent_default().stop()
            self.app.complete_hide()
        elif event.key == "enter":
            event.prevent_default().stop()
            if self.app.complete_on_enter():
                self.app.complete_accept()               # angefangenes Wort: Vorschlag uebernehmen
            self.app.complete_hide()
            self.post_message(self.Submitted(self.text))
        elif event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default().stop()
            self.insert("\n")
        elif event.key == "up" and self.cursor_location[0] == 0:
            event.prevent_default().stop()
            self.app.action_history(-1)
        elif event.key == "down" and self.cursor_location[0] == self.document.line_count - 1:
            event.prevent_default().stop()
            self.app.action_history(1)
        else:
            await super()._on_key(event)

    def set_text(self, text: str) -> None:
        self.load_text(text)
        self.move_cursor(self.document.end)


class RunView:
    """Widgets eines Laufs in einem eigenen Block. Parallele Laeufe vermischen
    sich dadurch nicht; Sub-Agenten sind eingerueckte Bloecke im Eltern-Block."""

    def __init__(self, box: Vertical, app: "ICli2", agent: str = "", query: str = "", session: str = ""):
        self.box, self.app = box, app
        self.agent, self.query, self.session = agent, query, session
        self.status = "running"
        self.status_msg = ""
        self.iter: tuple[int, int] | None = None
        self.answer = ""
        self.md: Markdown | None = None
        self.stream = None
        self.think: FoldLine | None = None
        self.problem_line: FoldLine | None = None
        self.think_text = ""
        self.tools: dict[str, ToolCard] = {}
        self.subs: dict[str, SubView] = {}
        self.started = time.monotonic()
        self.ended = 0.0
        self.queue: list[list] = []                      # [art, text] in Ankunftsreihenfolge
        self.last_flush = 0.0

    @property
    def seconds(self) -> float:
        return (self.ended or time.monotonic()) - self.started

    def add(self, kind: str, text: str) -> None:
        """Text vormerken; geschrieben wird gebuendelt in flush()."""
        if self.queue and self.queue[-1][0] == kind:
            self.queue[-1][1] += text
        else:
            self.queue.append([kind, text])

    async def flush(self) -> None:
        queue, self.queue = self.queue, []
        self.last_flush = time.monotonic()
        for kind, text in queue:
            await (self.text(text) if kind == P.TEXT else self.thinking(text))

    async def sub(self, sid: str) -> SubView:
        """Sub-Agent dieses Laufs (Kachel, siehe subtile.py); beim ersten Auftreten einhaengen."""
        view = self.subs.get(sid)
        if view is None:
            await self.flush()                           # eigener Text gehoert ueber die Kachel
            await self.close_segment()
            view = self.subs[sid] = SubView(sid, self.app)
            await self.box.mount(view.tile)
        return view

    async def problem(self, d: dict, model: str = "") -> FoldLine:
        """Fehler- bzw. Limitzeile am Laufende: Ueberschrift + Kurzmeldung, Klick zeigt Hinweis und Details."""
        if d["status"] == "limit":
            info = explain("", kind="limit", iters=(d.get("iter") or 0, d.get("max_iter") or 0))
            body = info["hint"] + (f"\n\nZusammenfassung des Agenten:\n{d['answer'].strip()}" if d.get("answer") else "")
            line = FoldLine(f"◷ {info['title']}", body, classes="problem -limit")
        else:
            info = explain(d.get("error") or "", model=model, traceback=d.get("traceback") or "")
            short = f" – {info['message'][:90]}" if info["message"] else ""
            line = FoldLine(f"{SYM['fail']} {info['title']}{short}", render(info), classes="problem -fail")
        line.info = info
        await self.box.mount(line)
        return line

    async def close_segment(self) -> None:
        if self.stream is not None:
            await self.stream.stop()
        if self.md is not None and self.md.is_attached:
            # fertiger Abschnitt: viele Markdown-Widgets -> ein Widget (siehe richmd.py)
            frozen = FrozenMarkdown(self.md.source)
            await self.md.parent.mount(frozen, after=self.md)
            await self.md.remove()
        self.md = self.stream = self.think = None
        self.think_text = ""

    async def close(self) -> None:
        await self.flush()
        await self.close_segment()
        for view in self.subs.values():
            await view.close()

    async def text(self, delta: str) -> None:
        if self.md is None:
            await self.close_segment()
            self.md = Markdown("", classes="answer")
            await self.box.mount(self.md)
            self.stream = Markdown.get_stream(self.md)
        self.answer += delta
        await self.stream.write(delta)

    async def thinking(self, delta: str) -> None:
        if self.think is None:
            await self.close_segment()
            self.think = FoldLine(f"{SYM['think']} Gedanken", collapsed=not self.app.show_thinking,
                                  classes="think")
            self.app.think_boxes.append(self.think)
            await self.box.mount(self.think)
        self.think_text += delta
        self.think.body = self.think_text

    async def tool_start(self, tid: str, name: str, summary: str, detail: str) -> None:
        await self.close_segment()
        card = self.tools[tid] = ToolCard(name, summary, detail)
        self.app.tool_cards.append(card)
        await self.box.mount(card)

    def tool_end(self, tid: str, ok: bool, summary: str, detail: str, secs: float) -> None:
        card = self.tools.pop(tid, None)
        if card is not None:
            card.finish(ok, summary, detail, secs)

    async def note(self, text: str, level: str = "info") -> None:
        await self.box.mount(Static(text, classes=f"note note-{level}", markup=False))


class ICli2(App):
    TITLE = "icli"
    CSS = """
    #head { height: 1; padding: 0 1; color: $text-muted; }
    #main { height: 1fr; }
    #chat { padding: 0 1; scrollbar-size-vertical: 1; }
    #side { width: 38; display: none; border-left: solid $secondary; padding: 0 1;
            scrollbar-size-vertical: 1; }
    #side.-open { display: block; }
    #banner { margin: 1 0; }
    #bottom { height: auto; }
    #row { height: auto; border: tall #3b5568; background: #0c161c; padding: 0 1; }
    #row.-run { border: tall $primary; }
    #row.-done { border: tall $success; }
    #row.-ask { border: tall $warning; }
    #row.-fail { border: tall $error; }
    #mark { width: 2; color: $accent; }
    #prompt, #prompt:focus { border: none; height: auto; max-height: 8; padding: 0;
                             background: transparent; }
    #foot { height: 1; padding: 0 1; color: $text-muted; }
    #complete { height: auto; display: none; padding: 0 2; color: $text-muted; }
    #complete.-open { display: block; }
    .run { height: auto; margin: 1 0 0 0; }
    .run.-focus { border-left: thick $primary; }
    .sub { height: auto; margin: 0 0 0 2; padding: 0 1; border: round $secondary; }
    .sub:hover { border: round $primary; }
    .sub.-open { border: round $primary; }
    .sub-head-full { margin: 0 0 1 0; }
    .sub-actions { height: auto; max-height: 22; scrollbar-size-vertical: 1; }
    .sub-foot { height: auto; }
    .sub-label { margin: 1 0 0 0; color: $text-muted; text-style: bold; }
    .sub-text { color: $text; padding: 0 0 0 2; }
    .problem { padding: 0 0 0 2; background: transparent; }
    .problem.-fail, .problem.-fail:hover, .problem.-fail:focus { color: $error; }
    .problem.-limit, .problem.-limit:hover, .problem.-limit:focus { color: $warning; }
    .problem:hover, .problem:focus { text-style: bold; }
    .user { color: $accent; text-style: bold; margin: 0 0 1 0; }
    .answer { margin: 0; padding: 0 0 0 2; background: transparent; }
    .answer MarkdownH1, .answer MarkdownH2, .answer MarkdownH3 {
        content-align: left middle; text-style: bold; color: $accent;
        background: transparent; border: none; margin: 1 0 0 0; padding: 0; }
    .answer MarkdownFence { background: $panel 40%; margin: 0 0 1 0; }
    .tool, .think { margin: 0; padding: 0 0 0 2; background: transparent; color: $text-muted; }
    .tool:hover, .think:hover, .tool:focus, .think:focus { color: $accent; background: transparent; }
    .tool-failed, .tool-failed:hover { color: $error; }
    .steer { height: auto; margin: 1 0 0 0; }
    .steer-q { color: $accent; text-style: bold; }
    .note { padding: 0 0 0 2; color: $text-muted; }
    .note-error { color: $error; }
    .note-warn { color: $warning; }
    """
    BINDINGS = [
        Binding("escape", "stop", "Stopp", priority=True),
        Binding("ctrl+q", "quit_app", "Beenden", priority=True),
        Binding("ctrl+l", "clear", "Leeren"),
        Binding("ctrl+p", "palette", "Befehle", priority=True),
        Binding("f2", "side", "Seitenleiste", priority=True),
        Binding("f3", "next_run", "Nächster Lauf", priority=True),
        Binding("f4", "voice", "Spracheingabe", priority=True),
        Binding("ctrl+o", "fold_tools", "Tools auf/zu", priority=True),
        Binding("ctrl+r", "fold_thinking", "Gedanken auf/zu", priority=True),
        *[Binding(f"alt+{n}", f"open_last({n})", show=False, priority=True) for n in range(1, 10)],
    ]

    def __init__(self, engine: Engine, agent: str = "self", session: str = "default"):
        super().__init__()
        self.engine = engine
        self.agent, self.session, self.model = agent, session, "…"
        self.complex_model, self.mounts, self.user = "", 0, ""
        self.features: list[str] = []
        self.history: list[str] = []
        self._hist_pos = 0
        self.ready = False
        self.runs: dict[str, RunView] = {}          # alle Laeufe, auch beendete (begrenzt)
        self.pending: dict[str, RunView] = {}       # tag -> Block, bis die Engine den Lauf meldet
        self.side_rows: dict[str, list[dict]] = {"jobs": [], "mounts": [], "context": []}
        self._side_jump = False                      # nach dem naechsten Kontext-Update ans Ende scrollen
        self._via_palette = False                    # Picker kam aus der Palette: ← fuehrt dorthin zurueck
        self.tool_cards: list[ToolCard] = []
        self.think_boxes: list[FoldLine] = []
        self._dirty: set[RunView] = set()                # Laeufe mit noch nicht geschriebenem Text
        self._last_scroll, self._last_scroll_y = 0.0, 0.0
        self._trim_note = False
        self._head_text = self._foot_text = self._side_cache = ""
        self.show_thinking = False
        self.focus_run = ""
        # Vorschlaege zur aktuellen Eingabe: (einzusetzender Text, Anzeige, Hinweis)
        self.completions: list[tuple[str, str, str]] = []
        self._complete_pos = 0
        self.tokens: tuple[int, int] | None = None
        self._tick = 0
        self._tag = 0
        self._esc_run = ""

    @property
    def active(self) -> dict[str, RunView]:
        return {rid: v for rid, v in self.runs.items() if v.status == "running"}

    # ── Aufbau ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static(id="head")
        with Horizontal(id="main"):
            with VerticalScroll(id="chat"):
                yield Static(banner(VERSION), id="banner")
            with VerticalScroll(id="side"):
                yield Static(id="side-text")
        with Vertical(id="bottom"):
            yield Static(id="complete")
            with Horizontal(id="row"):
                yield Static(SYM["progress"], id="mark")
                yield PromptArea(placeholder="Nachricht oder /befehl", id="prompt", compact=True,
                                 show_line_numbers=False, highlight_cursor_line=False,
                                 soft_wrap=True, tab_behavior="focus")
            yield Static(id="foot")

    def on_mount(self) -> None:
        self.register_theme(ISAA_THEME)
        self.theme = "isaa"
        self.chat = self.query_one("#chat", VerticalScroll)
        self.chat.anchor()                               # bleibt unten, solange nicht gescrollt wird
        self.prompt = self.query_one("#prompt", PromptArea)
        # feste Referenzen: App.query_one sucht im obersten Screen und findet bei offenem Picker nichts
        base = self.screen
        self.w_head, self.w_foot = base.query_one("#head", Static), base.query_one("#foot", Static)
        self.w_side, self.w_side_text = base.query_one("#side"), base.query_one("#side-text", Static)
        self.w_row, self.w_complete = base.query_one("#row"), base.query_one("#complete", Static)
        self.prompt.focus()
        self.set_interval(FRAME, self._pump)
        self.set_interval(0.1, self._chrome)
        self.set_interval(2.0, self._refresh_side)
        self._chrome()

    # ── Engine -> UI ─────────────────────────────────────────────────────────
    async def _pump(self) -> None:
        events = self.engine.poll(EVENTS_PER_FRAME)
        if events:
            with self.batch_update():                    # ein Neuzeichnen fuer das ganze Buendel
                for ev in events:
                    await self._handle(ev)
        if not self._dirty:
            return
        # Text gebuendelt schreiben: unten mitlesen -> fluessig, hochgescrollt -> selten,
        # waehrend gescrollt wird -> gar nicht (jeder Schreibvorgang kostet einen Layout-Durchlauf)
        now, chat = time.monotonic(), self.chat
        if chat.scroll_y != self._last_scroll_y:
            self._last_scroll_y = chat.scroll_y
            if chat.scroll_y < chat.max_scroll_y - 1:
                self._last_scroll = now
        at_end = chat.scroll_y >= chat.max_scroll_y - 1
        if not at_end and now - self._last_scroll < SCROLL_QUIET:
            return
        every = TEXT_EVERY if at_end else TEXT_EVERY_AWAY
        due = [v for v in self._dirty if now - v.last_flush >= every]
        if due:
            with self.batch_update():
                for view in due:
                    self._dirty.discard(view)
                    await view.flush()

    async def _block(self, head: str, classes: str = "user") -> Vertical:
        box = Vertical(Static(head, classes=classes, markup=False), classes="run")
        await self.chat.mount(box)
        return box

    async def _handle(self, ev: P.Event) -> None:
        d, view = ev.data, self.runs.get(ev.run)
        if view and d.get("sub"):
            view = await view.sub(d["sub"])
        if ev.kind in (P.TEXT, P.THINK):
            if view:
                view.add(ev.kind, d["text"])
                self._dirty.add(view)
            return
        if view and view.queue:                          # Reihenfolge: erst der gesammelte Text
            await view.flush()
            self._dirty.discard(view)
        if ev.kind == P.SUB and view:
            await view.info(d)
        elif ev.kind == P.TOOL_START and view:
            await view.tool_start(d["id"], d["name"], d["summary"], d["detail"])
        elif ev.kind == P.TOOL_END and view:
            view.tool_end(d["id"], d["ok"], d["summary"], d["detail"], d["secs"])
        elif ev.kind == P.STATUS and view:
            if d.get("msg"):
                view.status_msg = d["msg"]
            if d.get("iter") is not None:
                view.iter = (d["iter"], d.get("max_iter") or 0)
            if d.get("tokens") is not None and d.get("tokens_max") and not d.get("sub"):
                self.tokens = (d["tokens"], d["tokens_max"])
        elif ev.kind == P.RUN_START:
            view = self.pending.pop(d.get("tag", ""), None)
            if view is None:                             # von der Engine gestartet (Job, Resume, …)
                box = await self._block(f"{SYM['agent']} {d.get('origin', 'lauf')} · {d['agent']} · "
                                        f"{' '.join(d['query'].split())[:70]}", "note")
                view = RunView(box, self)
            view.agent, view.query, view.session = d["agent"], d["query"], d["session"]
            view.started = time.monotonic()
            target = self.runs.get(d.get("steer") or "")
            if target is not None and target.status == "running":
                await self._steer_into(view, target, d["query"])
            self.runs[ev.run] = view
            for old in list(self.runs)[:-KEEP_RUNS]:
                if self.runs[old].status != "running":
                    del self.runs[old]
            if (d.get("tag") and target is None) or not self.focus_run:
                self._focus(ev.run)
            self._signal("run")
        elif ev.kind == P.RUN_END and view:
            self._dirty.discard(view)
            await view.close()
            view.status, view.ended = d["status"], time.monotonic()
            if d["status"] in ("failed", "limit"):
                view.problem_line = await view.problem(d, self.model)
            else:
                mark = END_MARK.get(d["status"], d["status"])
                if mark:
                    await view.note(f"{mark} · {view.seconds:.1f}s")
            if not self.active:
                # Signalfarbe der Eingabeleiste: fertig / Frage (auch Limit) / Absturz
                asks = d["status"] in ("paused", "limit") or (d.get("answer") or view.answer).rstrip().endswith("?")
                self._signal("fail" if d["status"] == "failed" else "ask" if asks else
                             "done" if d["status"] == "done" else "")
            if self.focus_run == ev.run:
                self._focus(next(reversed(self.active), ""))
            await self._trim()
            if self.w_side.has_class("-open"):
                self.engine.send(P.LIST, what="context", to="side")
        elif ev.kind == P.READY:
            changed = (d["agent"], d["session"], d["model"], d["complex_model"]) != \
                      (self.agent, self.session, self.model, self.complex_model)
            if self.ready and changed:
                await self._note(f"{SYM['agent']} {d['agent']} · {d['model']} · {d['session']}")
            self.ready = True
            self.agent, self.session, self.model = d["agent"], d["session"], d["model"]
            self.complex_model, self.mounts = d["complex_model"], d["mounts"]
            self.user = d.get("user", "")
            self.features = d.get("features", [])
            if "history" in d:
                self.history, self._hist_pos = d["history"], len(d["history"])
            for role, text in d.get("recap", []):       # letzte Nachrichten der Session
                await self._note(f"{SYM['progress'] if role == 'user' else ' '} {text}")
        elif ev.kind == P.ROWS:
            if d.get("to") == "complete":
                self._complete_paths(d.get("arg", ""), d["rows"])
            elif d.get("to") == "side":
                self.side_rows[d["what"]] = d["rows"]
                if d["what"] == "context":
                    self._set_side()
                    if self._side_jump:
                        self._side_jump = False
                        self.call_after_refresh(self.w_side.scroll_end, animate=False)
            else:
                self._pick(d["what"], d["title"], d["rows"])
        elif ev.kind == P.DRAFT:
            self.prompt.set_text((self.prompt.text + " " + d["text"]).strip())
            self.prompt.focus()
        elif ev.kind == P.NOTICE:
            if view:                                     # gehoert zu einem Lauf
                await view.note(d["text"], d.get("level", "info"))
            else:
                await self._note(d["text"], d.get("level", "info"))

    async def _steer_into(self, view: RunView, target: RunView, query: str) -> None:
        """Nachsteuerung an der aktuellen Stelle IN den laufenden Lauf setzen: Frage und
        Sidecar-Antwort stehen dort, alles Weitere des Hauptlaufs erscheint darunter."""
        await target.flush()
        await target.close_segment()                     # naechster Text des Hauptlaufs: neuer Abschnitt darunter
        old = view.box
        view.box = Vertical(Static(f"↪ {query}", classes="steer-q", markup=False), classes="steer")
        await target.box.mount(view.box)
        if old.is_attached:
            await old.remove()

    async def _trim(self) -> None:
        """Sehr alte Bloecke aus dem Bild nehmen (jeder Block kostet bei jedem Layout-Durchlauf)."""
        blocks = [w for w in self.chat.children if w.has_class("run")]
        busy = {v.box for v in self.active.values()} | {v.box for v in self.pending.values()}
        old = [w for w in blocks[:max(0, len(blocks) - MAX_BLOCKS)] if w not in busy]
        if not old:
            return
        first = blocks[0]
        drop = [w for w in self.chat.children if w is not first and w not in busy and w.id != "trim-note"
                and self.chat.children.index(w) < self.chat.children.index(old[-1])] + old
        await self.chat.remove_children(list(dict.fromkeys(drop)))
        self.tool_cards = [c for c in self.tool_cards if c.is_attached]
        self.think_boxes = [b for b in self.think_boxes if b.is_attached]
        if not self._trim_note:
            self._trim_note = True
            await self.chat.mount(Static("… ältere Ausgaben ausgeblendet – der Verlauf bleibt in der Session",
                                         classes="note", id="trim-note"), before=0)

    def _focus(self, run: str) -> None:
        self.focus_run = run
        many = len(self.active) > 1
        for rid, view in self.runs.items():
            view.box.set_class(many and rid == run, "-focus")

    def _signal(self, state: str) -> None:
        row = self.w_row
        for name in ("run", "done", "ask", "fail"):
            row.set_class(name == state, f"-{name}")

    async def _note(self, text: str, level: str = "info") -> None:
        await self.chat.mount(Static(text, classes=f"note note-{level}", markup=False))

    def _chrome(self) -> None:
        """Kopf-, Fusszeile, Seitenleiste. Rein lokal, auch bei blockierter Engine fluessig."""
        self._tick += 1
        mounts = f" · {self.mounts} mount{'s' if self.mounts != 1 else ''}" if self.mounts else ""
        self._set_line("_head_text", self.w_head,
            f"[{G.L4_BRIGHT}]{SYM['agent']} {self.agent}[/] "
            f"[{DIM}]· {self.model} · {self.session}{mounts}"
            f"{' · ' + self.user if self.user else ''}[/]"
            f"{'  [' + G.L3_MID + ']' + ' '.join(self.features) + '[/]' if self.features else ''}")
        spin = SPIN[self._tick % len(SPIN)]
        active, focus = self.active, self.runs.get(self.focus_run)
        if not self.ready:
            left = f"{spin} Engine startet …"
        elif self.engine.lag() > 1.5:
            left = f"{spin} Engine beschäftigt ({self.engine.lag():.0f}s) – Eingabe bleibt frei"
        elif active:
            view = focus if focus and focus.status == "running" else next(iter(active.values()))
            step = f" {view.iter[0]}/{view.iter[1]}" if view.iter and view.iter[1] else ""
            more = f" · {len(active)} Läufe (F3)" if len(active) > 1 else ""
            left = f"{spin} {view.agent}{step} · {view.status_msg or 'arbeitet'}{more}"
        else:
            left = f"[{G.L3_MID}]{SYM['done']}[/] bereit"
        if self.tokens:
            left += f" · {self.tokens[0] / 1000:.1f}k/{self.tokens[1] / 1000:.0f}k tok"
        self._set_line("_foot_text", self.w_foot, left)
        if self._tick % 10 == 0:                         # Laufzeit/Restzeit der Sub-Agenten
            for run in self.active.values():
                for sub in run.subs.values():
                    if sub.status == "running":
                        sub.tile.redraw()
        if self._tick % 5 == 0 and self.w_side.has_class("-open"):
            self._set_side()

    def _set_line(self, cache: str, widget: Static, text: str) -> None:
        """Einzeilige Anzeige: nur bei Aenderung und ohne Layout-Durchlauf (Hoehe bleibt 1)."""
        if getattr(self, cache) != text:
            setattr(self, cache, text)
            widget.update(text, layout=False)

    def _set_side(self) -> None:
        text = self._side_text()
        if text != self._side_cache:
            self._side_cache = text
            self.w_side_text.update(text)

    def _side_text(self) -> str:
        def title(text: str) -> str:
            return f"[{G.L4_BRIGHT}]{text}[/]"

        from rich.markup import escape
        lines = [title("Läufe")]
        for rid, view in reversed(list(self.runs.items())[-10:]):
            mark = "▸" if rid == self.focus_run else " "
            color = {"running": G.L3_MID, "failed": "red", "paused": "yellow"}.get(view.status, DIM)
            lines.append(f"{mark}[{color}]{RUN_SYM.get(view.status, '?')}[/] {escape(view.agent)} "
                         f"[{DIM}]{view.seconds:.0f}s[/]")
            lines.append(f"   [{DIM}]{escape(' '.join(view.query.split())[:30])}[/]")
            for sub in view.subs.values():
                lines.append(f"   [{DIM}]{SYM['sub']} {escape(sub.label[:30])}[/]")
        if len(lines) == 1:
            lines.append(f" [{DIM}]keine[/]")
        for what, head in (("jobs", "Jobs"), ("mounts", "Mounts")):
            lines += ["", title(head)]
            rows = [r for r in self.side_rows[what] if r["value"] not in ("*", "+")]
            lines += [f" {escape(r['label'])}\n   [{DIM}]{escape(r['hint'][:32])}[/]" for r in rows] \
                or [f" [{DIM}]keine[/]"]
        lines += ["", title("Kontext")]
        context = self.side_rows["context"]
        if context:
            for row in context:
                text = escape(row["label"])
                head = text and not text.startswith(("━", "─")) and "  " not in text.strip() and text[:1].isupper()
                lines.append(f" [{G.L3_MID}]{text}[/]" if text.startswith(("━", "─")) else
                             f" {text}" if head else f" [{DIM}]{text}[/]")
        else:
            lines.append(f" [{DIM}]wird geladen …[/]")
        return "\n".join(lines)

    def _refresh_side(self) -> None:
        if self.ready and self.w_side.has_class("-open"):
            for what in ("jobs", "mounts"):             # Kontext ist teurer: nur bei Bedarf
                self.engine.send(P.LIST, what=what, to="side")

    # ── UI -> Engine ─────────────────────────────────────────────────────────
    @on(PromptArea.Submitted)
    async def _submit(self, message: PromptArea.Submitted) -> None:
        text = message.value.strip()
        self.prompt.set_text("")
        if not text:
            return
        if not self.active:
            self._signal("")
        if not self.history or self.history[-1] != text:
            self.history.append(text)
            self.engine.send(P.DO, action="remember", text=text)
        self._hist_pos = len(self.history)
        if text.startswith("/") and "\n" not in text:
            await self._command(text)
            return
        self._tag += 1
        tag = f"t{self._tag}"
        box = await self._block(f"{SYM['progress']} {text}")
        self.pending[tag] = RunView(box, self, agent=self.agent, query=text, session=self.session)
        self.chat.scroll_end(animate=False)
        self.engine.send(P.CHAT, query=text, agent=self.agent, session=self.session, tag=tag)

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
        elif name == "/side":
            self.action_side()
        elif name == "/task":
            self._pick_run()
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
        elif name == "/job":
            do("job_add", spec=arg) if arg else ask("jobs")
        elif name == "/dream":
            do("dream")
        elif name == "/network":
            do("network_load", path=arg) if arg else do("network_save")
        elif name == "/settings":
            ask("settings")
        elif name == "/set":
            key, _, value = arg.partition(" ")
            do("set", key=key, value=value.strip())
        elif name == "/context":
            if not self.w_side.has_class("-open"):
                self.w_side.add_class("-open")
                self._refresh_side()
            self._side_jump = True
            self.engine.send(P.LIST, what="context", to="side")
        elif name == "/vfs":
            do("vfs", path=arg)
        elif name == "/feature":
            do("feature", name=arg) if arg else ask("features")
        elif name == "/skill":
            do("skill", name=arg) if arg else ask("skills")
        elif name == "/mcp":
            (do("mcp_add", spec=arg) if " " in arg else do("mcp", name=arg)) if arg else ask("mcp")
        elif name == "/plugin":
            sub, _, rest = arg.partition(" ")
            rest = rest.strip()
            if sub in ("install", "confirm", "discard", "remove") and rest:
                do(f"plugin_{sub}", **{"source" if sub == "install" else "name": rest})
            elif sub == "market":                        # ohne Quelle: stoebern, mit Quelle: merken
                do("market_add", source=rest) if rest else ask("market")
            elif sub == "markets":
                ask("markets")
            elif sub in ("install", "confirm", "discard", "remove"):
                await self._note(f"/plugin {sub} braucht eine Angabe, z. B. /plugin install owner/repo", "warn")
            else:
                do("plugin", name=arg) if arg else ask("plugins")
        elif name in ("/discord", "/telegram"):
            do(name[1:], args=arg) if arg else ask(name[1:])
        elif name == "/login":
            do("login")
        elif name == "/logout":
            do("logout")
        elif name == "/tb":
            target, _, rest = arg.partition(" ")
            do("tb", target=target, args=rest.strip()) if arg else ask("tb")
        elif name == "/share":
            ask("shares")
        elif name == "/voice":
            ask("voice")
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

        via_palette, self._via_palette = self._via_palette, False

        def chosen(value: str | None) -> None:
            if value == BACK and via_palette:            # ← : zurueck in die Befehlsliste
                return self.action_palette()
            if value in (None, BACK):
                return
            if what == "agents":
                send(P.DO, action="agent", name=value)
            elif what == "sessions":
                send(P.DO, action="session", name=value)
            elif what == "mounts":
                send(P.DO, action="unmount", at=value)
            elif what in ("features", "skills", "plugins"):
                send(P.DO, action=what[:-1], name=value)
            elif what == "mcp":
                send(P.DO, action="mcp", name=value)
            elif what in ("discord", "telegram"):
                send(P.DO, action=what, args=value)
            elif what == "market":                       # Plugin aus dem Marketplace installieren
                if value:
                    send(P.DO, action="plugin_install", source=value)
            elif what == "markets":                      # gespeicherten Marketplace entfernen
                send(P.DO, action="market_remove", source=value)
            elif what == "settings":
                if value.startswith("set:"):              # Wert eintippen lassen
                    self.prompt.set_text(f"/set {value[4:]} ")
                    self.prompt.focus()
                elif value == "context":
                    self.run_worker(self._command("/context"))
                else:
                    send(P.DO, action=value)
            elif what == "tb":
                send(P.DO, action="tb", target=value)
            elif what == "shares":
                send(P.DO, action="share", share=value)
            elif what == "voice":
                send(P.DO, action="voice", op=value)
            elif what == "jobs":
                self.push_screen(
                    Picker(f"Job {value}", JOB_ACTIONS),
                    lambda act: act not in (None, BACK) and send(P.DO, action="job", job=value, op=act))
            elif what == "models":
                self.push_screen(
                    Picker(f"{value} für", MODEL_TARGETS),
                    lambda target: target not in (None, BACK) and send(P.DO, action="model", name=value, target=target))

        self.push_screen(Picker(title, [(r["value"], r["label"], r["hint"]) for r in rows]), chosen)

    def _pick_run(self) -> None:
        items = [(rid, f"{RUN_SYM.get(v.status, '?')} {v.agent}",
                  f"{v.status} · {v.seconds:.0f}s · {' '.join(v.query.split())[:40]}")
                 for rid, v in reversed(self.runs.items())]
        if not items:
            self.run_worker(self._note("Noch keine Läufe", "warn"))
            return

        def chosen(rid: str | None) -> None:
            if rid in self.runs:
                self._focus(rid)
                self.runs[rid].box.scroll_visible(top=True)

        self.push_screen(Picker("Läufe – Enter setzt den Fokus", items), chosen)

    def action_palette(self) -> None:
        items = [(name, name, f"{hint}  {text}".strip()) for name, (hint, text) in COMMANDS.items()]

        def chosen(name: str | None) -> None:
            if name not in (None, BACK):
                self._via_palette = True                 # ← im Unter-Picker fuehrt hierher zurueck
                self.run_worker(self._command(name))     # ohne Argument: Auswahl bzw. Standard

        self.push_screen(Picker("Befehle", items), chosen)

    # ── Vervollstaendigung beim Tippen (Befehle und VFS-Pfade) ───────────────
    @on(TextArea.Changed, "#prompt")
    def _complete_update(self) -> None:
        text = self.prompt.text
        word = text.strip()
        if text.startswith("/vfs ") and "\n" not in text:
            arg = text[5:].strip()
            if " " not in arg:                           # Vorschlaege kommen aus der Engine
                self.engine.send(P.LIST, what="vfs", arg=arg, to="complete")
                return
        if not word.startswith("/") or " " in word or "\n" in text or word in COMMANDS:
            return self.complete_hide()
        starts = [c for c in COMMANDS if c.startswith(word)]
        names = (starts + [c for c in COMMANDS if c not in starts and fuzzy(word[1:], c)])[:6]
        self.completions = [(n + (" " if COMMANDS[n][0] else ""), n, " ".join(COMMANDS[n]).strip())
                            for n in names]
        self._complete_pos = 0
        self._complete_draw()

    def _complete_paths(self, arg: str, rows: list[dict]) -> None:
        """Antwort der Engine auf einen angefangenen VFS-Pfad; veraltete Antworten verwerfen."""
        text = self.prompt.text
        if not text.startswith("/vfs ") or text[5:].strip() != arg:
            return
        self.completions = [("/vfs " + r["value"], r["label"], r["hint"]) for r in rows
                            if r["value"] != arg][:8]
        self._complete_pos = 0
        self._complete_draw()

    def _complete_draw(self) -> None:
        from rich.markup import escape
        box = self.w_complete
        box.set_class(bool(self.completions), "-open")
        box.update("\n".join(
            (f"[{G.L4_BRIGHT}]{SYM['progress']} {escape(label)}[/]" if i == self._complete_pos
             else f"  {escape(label)}") + f"  [{DIM}]{escape(hint)}[/]"
            for i, (_insert, label, hint) in enumerate(self.completions))
            + (f"\n  [{DIM}]Tab übernimmt · ↑↓ wählt · Enter führt aus[/]" if self.completions else ""))

    def complete_move(self, step: int) -> None:
        if self.completions:
            self._complete_pos = (self._complete_pos + step) % len(self.completions)
            self._complete_draw()

    def complete_accept(self) -> None:
        if self.completions:
            insert = self.completions[self._complete_pos][0]
            self.completions = []
            self._complete_draw()
            self.prompt.set_text(insert)                 # loest bei Ordnern die naechste Ebene aus

    def complete_on_enter(self) -> bool:
        """Enter bei sichtbaren Vorschlaegen: angefangenes Wort erst vervollstaendigen."""
        if not self.completions:
            return False
        text = self.prompt.text.strip()
        insert = self.completions[self._complete_pos][0].strip()
        if text.startswith("/vfs"):                      # Ordner (endet auf /) wird wie getippt gezeigt
            return not text.endswith("/") and text != "/vfs" and insert.startswith(text) and insert != text
        return text not in COMMANDS and text not in ALIASES

    def complete_hide(self) -> None:
        if self.completions or self.w_complete.has_class("-open"):
            self.completions = []
            self._complete_draw()

    def action_history(self, step: int) -> None:
        if not self.history:
            return
        self._hist_pos = max(0, min(len(self.history), self._hist_pos + step))
        self.prompt.set_text(self.history[self._hist_pos] if self._hist_pos < len(self.history) else "")

    def action_side(self) -> None:
        side = self.w_side
        side.toggle_class("-open")
        if side.has_class("-open"):
            self._refresh_side()
            self.engine.send(P.LIST, what="context", to="side")
            self._set_side()

    def action_voice(self) -> None:
        """F4: Aufnahme starten bzw. beenden; der erkannte Text landet in der Eingabe."""
        self.engine.send(P.DO, action="voice_toggle")

    def action_next_run(self) -> None:
        """Fokus auf den naechsten laufenden Lauf (Esc wirkt auf den Fokus)."""
        ids = list(self.active)
        if ids:
            nxt = ids[(ids.index(self.focus_run) + 1) % len(ids)] if self.focus_run in ids else ids[0]
            self._focus(nxt)
            self.runs[nxt].box.scroll_visible(top=True)

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        # Ist ein Picker offen, gehoeren Esc & Co. ihm (sonst schloesse Esc ihn nicht).
        if len(self.screen_stack) > 1 and action in ("stop", "palette", "side", "next_run", "voice"):
            return False
        return True

    def action_stop(self) -> None:
        """Esc pausiert den fokussierten Lauf (fortsetzbar), ein zweites Esc bricht hart ab.
        Kann der Lauf nicht pausieren, stoppt die Engine ihn sofort."""
        if self.focus_run not in self.active:
            return
        hard = self._esc_run == self.focus_run
        self.engine.send(P.STOP if hard else P.PAUSE, run=self.focus_run)
        self._esc_run = self.focus_run

    async def action_clear(self) -> None:
        if not self.active and not self.pending:
            await self.chat.remove_children()
            self.runs.clear()
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
        if not self.prompt.has_focus and event.is_printable and len(self.screen_stack) == 1:
            self.prompt.focus()
            if event.character:                          # die Taste selbst gehoert schon zur Eingabe
                self.prompt.insert(event.character)
                event.prevent_default().stop()
