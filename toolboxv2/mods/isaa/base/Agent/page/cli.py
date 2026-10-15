"""Minimale CLI für die Page Engine – ausgerichtet an Pages, nicht an Chat.

Orientiert an flows/isaa/icli.py: prompt_toolkit (PromptSession, NestedCompleter,
patch_stdout, FileHistory) und das zen_renderer-Farbschema.  Lädt einen ISAA-Agenten
"einfach self" über isaa.get_agent(name) und fährt eine Page-Engine darüber.
Kompatibel mit dem ISAA-Router: FlowAgentLLM nutzt agent.a_run_llm_completion.

Aufruf:
    python -m toolboxv2.mods.isaa.base.Agent.page.cli [agent_name]
    python -m toolboxv2.mods.isaa.base.Agent.page.cli self --watch P1,P2,P5

REPL-Befehle:
    <text>              Query an die Page Engine (streamt live)
    /page              ganze Page anzeigen
    /slot P1 P2 ...    bestimmte Slots einmalig anzeigen
    /watch P1,P2       Slots permanent nach jedem Schritt anzeigen
    /unwatch           permanente Anzeige aus
    /tail [n]          letzte n Tail-Items (default 5)
    /recall <query>    offloaded Inhalte durchsuchen
    /boxes             Task-Plan (P1) mit Status
    /context           nativer FlowAgent Context-Overview (CONTEXT X-RAY)
    /versions          Commit-Historie
    /sessions          aktive Sessions
    /help · /quit

Auto-Steering: Während der Agent eine Anfrage bearbeitet, erscheint ein
"steer»"-Prompt.  Eine Eingabe ohne '/' wird als Steering-Notiz eingespeist;
eine Eingabe mit '/' läuft als normaler Befehl (auch während der Agent arbeitet).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


def _ensure_utf8_stdout():
    if sys.platform == "win32" and "pytest" not in sys.modules:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass


_ensure_utf8_stdout()
os.environ.setdefault("NARRATOR_CONSOLE_PRINT", "false")

# zen_renderer-Farbschema (wie icli).  Fallback auf schlichtes ANSI wenn nicht da.
try:
    from toolboxv2.mods.isaa.extras.zen.zen_renderer import C  # noqa: N811
except Exception:
    C = {
        "dim": "#6b7280", "cyan": "#67e8f9", "green": "#4ade80",
        "red": "#f87171", "amber": "#fbbf24", "white": "#e5e7eb",
        "bright": "#ffffff", "blue": "#60a5fa", "purple": "#a78bfa",
    }

# prompt_toolkit (wie icli).  Fallback auf input() wenn nicht installiert.
try:
    from prompt_toolkit import PromptSession, print_formatted_text, HTML
    from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.patch_stdout import patch_stdout
    _PT = True
except Exception:
    _PT = False


SYM = {"ok": "✓", "warn": "⚠", "tool": "◇", "ctx": "⟳", "arrow": "→"}

# Reusable, colored card formatter (standalone — see pretty.py).  Used to render
# tool results / recall hits as formatted cards instead of raw dict/JSON dumps.
from .pretty import Card, Theme, format_tool_result, format_value  # noqa: E402

_THEME = Theme(color=True, width=60)


# ------------------------------------------------------------------ output helpers
def _p(text: str, color: str = "white"):
    """Formatierte Ausgabe über prompt_toolkit HTML oder plain fallback."""
    if _PT:
        import html as _html
        hexc = C.get(color, C["white"])
        print_formatted_text(HTML(f'<style fg="{hexc}">{_html.escape(text)}</style>'))
    else:
        print(text)


def _short(x, n: int = 90) -> str:
    s = str(x)
    return s if len(s) <= n else s[: n - 1] + "…"


# ------------------------------------------------------------------ agent loading
async def _load_agent(name: str):
    """Lädt einen ISAA-Agenten 'self' – gleiche Kette wie icli (isaa.get_agent)."""
    from toolboxv2 import get_app
    app = get_app("page-cli")
    isaa = app.get_mod("isaa")
    if hasattr(isaa, "init_isaa"):
        try:
            await isaa.init_isaa(build=True)
        except Exception:
            pass
    return await isaa.get_agent(name)


# ------------------------------------------------------------------ rendering
def _render_slots(engine, session_id: str, slots: list[str]) -> None:
    from .core import SLOTS
    s = engine.session(session_id)
    page = s.page
    _p(f"# PAGE v{page['v']}  (session {session_id})", "bright")
    for k in slots:
        if k not in SLOTS:
            _p(f"  ?? unbekannter Slot {k}", "red")
            continue
        title, owner = SLOTS[k]
        if k == "P1":
            body = "\n".join(engine._box_line(b) for b in page["boxes"]) or "(empty)"
        else:
            body = page["slots"].get(k, "") or "(empty)"
        _p(f"## {k} {title} [{owner}]", "cyan")
        print(body)


def _render_tail(engine, session_id: str, n: int) -> None:
    s = engine.session(session_id)
    items = s.tail[-n:]
    _p(f"# TAIL  ({len(s.tail)} items, letzte {len(items)})", "bright")
    for idx, it in enumerate(items):
        label = len(s.tail) - (len(items) - 1 - idx)
        tools = ",".join(it["meta"].get("tools", [])) or it["meta"].get("status", "?")
        _p(f"  t{label}  #{it['id']}  {tools}  {it['tok']}tok  media={len(it['media'])}",
           "dim")


def _render_boxes(engine, session_id: str) -> None:
    s = engine.session(session_id)
    boxes = s.page["boxes"]
    if not boxes:
        _p("(keine Boxen im Task-Plan)", "dim")
        return
    _p("# TASK PLAN (P1)", "bright")
    for b in boxes:
        print("  " + engine._box_line(b))


# ------------------------------------------------------------------ action card
class _ActionCard:
    """Rendert Engine-Events als sortierte Status-Kacheln statt roher JSON-Zeilen.

    Pro Iteration eine Kachel:
        ┌─ action · 420 tok ────────────────────────────
        │ ◇ code_run  → ok (42)
        │ ⟳ context_fold → page v7
        │ <live gestreamter Text>
        └ ✓ final

    Text (content/content_chunk) wird live durchgereicht; Tool-Aktionen und
    Kontext-Schritte als kompakte Zeilen; final_answer als Fußzeile.
    """

    def __init__(self):
        self._streamed = False
        self._open = False
        self._tools: list[str] = []
        self._last_tool = "result"

    def _bar(self, title: str, color: str = "cyan"):
        line = f"┌─ {title} " + "─" * max(2, 48 - len(title))
        _p(line, color)
        self._open = True

    def feed(self, ev: dict):
        t = ev.get("type")
        if t == "iteration_start":
            mode = ev.get("mode", "?")
            tok = ev.get("tokens_used", 0)
            mx = ev.get("tokens_max", 0)
            self._tools = []
            self._streamed = False
            clr = {"action": "cyan", "context_fold": "amber",
                   "context_validate": "purple"}.get(mode, "cyan")
            self._bar(f"{mode} · {tok}/{mx} tok", clr)
        elif t == "reasoning":
            _p(f"│ {ev['chunk']}", "purple")
        elif t in ("content_chunk", "content"):
            if not self._streamed:
                print("│ ", end="", flush=True)
            print(ev["chunk"], end="", flush=True)
            self._streamed = True
        elif t == "tool_start":
            if self._streamed:
                print()
                self._streamed = False
            self._tools.append(ev["name"])
            self._last_tool = ev["name"]
            _p(f"│ {SYM['tool']} {ev['name']}({_short(ev.get('args', {}), 56)})", "blue")
        elif t == "tool_result":
            # Full result rendered as a colored card (reusable pretty formatter)
            # instead of a single truncated raw line.
            name = ev.get("name") or getattr(self, "_last_tool", "result")
            ok = not str(ev.get("result", "")).lstrip().lower().startswith("error")
            if self._streamed:
                print()
                self._streamed = False
            print(format_tool_result(name, ev.get("result", ""),
                                     theme=_THEME, ok=ok, max_chars=1600))
        elif t == "context_step":
            _p(f"│ {SYM['ctx']} {ev.get('mode')} {SYM['arrow']} "
               f"page v{ev.get('page_version')}", "amber")
        elif t == "final_answer":
            if self._streamed:
                print()
                self._streamed = False
            mark = SYM["ok"] if ev.get("success") else SYM["warn"]
            clr = "green" if ev.get("success") else "amber"
            # Bei gestreamtem Text keine Dopplung – nur Markierung + ggf. Kurztext
            if self._open:
                _p(f"└ {mark} final", clr)
                self._open = False
            answer = ev.get("answer", "")
            if answer:
                _p(answer, clr)

    def note(self, text: str):
        _p(f"│ {SYM['ctx']} {text}", "amber")

    def finish(self):
        if self._open:
            _p("└" + "─" * 48, "dim")
            self._open = False


# ------------------------------------------------------------------ REPL
class PageCLI:
    def __init__(self, engine, session_id: str = "default",
                 watch: list[str] | None = None, agent=None):
        self.engine = engine
        self.session_id = session_id
        self.watch = watch or []
        self.agent = agent
        self.session = None  # PromptSession

    def _completer(self):
        from .core import SLOTS
        slot_dict = {s: None for s in SLOTS}
        nested = {
            "/page": None, "/slot": slot_dict, "/watch": slot_dict, "/unwatch": None,
            "/tail": None, "/recall": None, "/boxes": None, "/context": None,
            "/versions": None, "/sessions": None, "/help": None, "/quit": None,
        }
        return FuzzyCompleter(NestedCompleter.from_nested_dict(nested))

    def _show_watch(self):
        if self.watch:
            _render_slots(self.engine, self.session_id, self.watch)

    def _render_context_breakdown(self):
        """Native CONTEXT X-RAY from Page + Tail (no V3 context_overview needed).

        Shows the SAME categories the V3 X-Ray uses (System Prompt, Tools,
        Working History, Last Input, Media) so the user sees everything the
        agent really receives — mapped from Page+Tail."""
        b = self.engine.context_breakdown(self.session_id)
        x = b.get("xray", {})
        total = x.get("Total", b["total"])
        cap = x.get("cap", b["cap"])
        pct = int(100 * total / cap) if cap else 0
        card = Card(
            f"context x-ray · page v{b['page_version']} · {total}/{cap} tok ({pct}%)",
            theme=_THEME, accent="accent",
        )
        card.line(f"System Prompt Total   {x.get('System Prompt Total', 0):>7} tok", "key")
        card.line(f"  base system prompt  {x.get('Base System Prompt', 0):>7} tok", "muted")
        card.line(f"  page plan (P1)      {x.get('Page Plan (P1)', 0):>7} tok", "muted")
        card.line(f"  page helpers (P3)   {x.get('Page Helpers (P3)', 0):>7} tok", "muted")
        card.line(
            f"Active Tools          {x.get('Active Tools', 0):>7} tok  "
            f"({x.get('tool_count', 0)} defs)", "key",
        )
        card.line(
            f"History (Working)     {x.get('History (Working)', 0):>7} tok  "
            f"({x.get('w_msg_count', 0)} tail items)", "key",
        )
        card.line(f"Last Input            {x.get('Last Input', 0):>7} tok", "muted")
        if x.get("Media", 0):
            card.line(
                f"Multimedia (Vision)   {x.get('Media', 0):>7} tok  "
                f"({x.get('media_count', 0)} images)", "muted",
            )
        card.line(f"TOTAL                 {total:>7} / {cap} tok ({pct}%)", "accent")
        print(card.render())

    async def _run_query(self, query: str):
        """Streamt die Antwort live als sortierte Status-Kacheln (kein JSON).
        Während der Agent läuft, erscheint ein 'steer»'-Prompt: eine Eingabe ohne
        '/' wird als Steering eingespeist, eine mit '/' als normaler Befehl ausgeführt."""
        done = asyncio.Event()
        card = _ActionCard()

        async def consume():
            try:
                async for ev in self.engine.stream(query, session=self.session_id):
                    card.feed(ev)
            finally:
                done.set()

        async def steer_listener():
            """Lauscht während der Agent läuft.  /-Befehle laufen normal,
            alles andere wird als Steering eingespeist."""
            while not done.is_set():
                try:
                    if _PT:
                        line = await self.session.prompt_async(
                            HTML(f'<style fg="{C["amber"]}">steer» </style>')
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        line = await loop.run_in_executor(None, input)
                except (EOFError, KeyboardInterrupt):
                    return
                line = (line or "").strip()
                if done.is_set() or not line:
                    continue
                if line.startswith("/"):
                    # Befehl auch während des Laufs normal ausführen
                    await self._command(line)
                else:
                    self.engine.steer(self.session_id, line)
                    card.note(f"steering: {_short(line, 60)}")

        consumer = asyncio.create_task(consume())
        if _PT:
            listener = asyncio.create_task(steer_listener())
            await done.wait()
            listener.cancel()
            import contextlib as _ctx
            with _ctx.suppress(asyncio.CancelledError):
                await listener
        await consumer
        card.finish()
        self._show_watch()

    async def repl(self):
        _p("Page Engine CLI", "bright")
        _p(f"session: {self.session_id}  ·  /help für Befehle, /quit beenden", "dim")
        if self.watch:
            _p(f"watching: {', '.join(self.watch)}", "dim")
            self._show_watch()

        if _PT:
            root = self.engine.session(self.session_id).dir
            self.session = PromptSession(
                history=FileHistory(str(root / ".cli_history")),
                completer=self._completer(),
                auto_suggest=AutoSuggestFromHistory(),
            )

        while True:
            try:
                if _PT:
                    with patch_stdout():
                        line = await self.session.prompt_async(
                            HTML(f'<style fg="{C["green"]}">page&gt; </style>')
                        )
                else:
                    loop = asyncio.get_event_loop()
                    line = await loop.run_in_executor(None, lambda: input("page> "))
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = (line or "").strip()
            if not line:
                continue
            if not line.startswith("/"):
                await self._run_query(line)
                continue
            if not await self._command(line):
                break

    async def _command(self, line: str) -> bool:
        parts = line.split()
        cmd, args = parts[0], parts[1:]

        if cmd in ("/quit", "/q", "/exit"):
            return False
        elif cmd == "/help":
            print(__doc__)
        elif cmd == "/page":
            from .core import SLOTS
            _render_slots(self.engine, self.session_id, list(SLOTS))
        elif cmd == "/slot":
            if not args:
                _p("Nutzung: /slot P1 P2 ...", "red")
            else:
                _render_slots(self.engine, self.session_id, args)
        elif cmd == "/watch":
            self.watch = args[0].split(",") if args else []
            _p(f"watching: {', '.join(self.watch) or '(nichts)'}", "dim")
            self._show_watch()
        elif cmd == "/unwatch":
            self.watch = []
            _p("permanente Anzeige aus", "dim")
        elif cmd == "/tail":
            n = int(args[0]) if args and args[0].isdigit() else 5
            _render_tail(self.engine, self.session_id, n)
        elif cmd == "/boxes":
            _render_boxes(self.engine, self.session_id)
        elif cmd == "/context":
            # Nativer FlowAgent Context-Overview (CONTEXT X-RAY Kachel)
            if self.agent is not None and hasattr(self.agent, "context_overview"):
                try:
                    await self.agent.context_overview(
                        session_id=self.session_id, print_visual=True
                    )
                except Exception as e:
                    _p(f"context_overview fehlgeschlagen: {e}", "red")
                    self._render_context_breakdown()
            else:
                # Pure page run: native token breakdown from Page + Tail.
                self._render_context_breakdown()
        elif cmd == "/recall":
            if not args:
                _p("Nutzung: /recall <query>", "red")
            else:
                hits = self.engine.store.search(
                    " ".join(args), 8, prefix=f"{self.session_id}/"
                )
                if not hits:
                    _p("(keine Treffer)", "dim")
                else:
                    card = Card(f"recall · {len(hits)} hits", theme=_THEME,
                                accent="accent")
                    for ref, snippet in hits:
                        card.line(ref, "key")
                        card.line(f"  {_short(snippet, 120)}", "muted")
                    print(card.render())
        elif cmd == "/versions":
            vf = self.engine.session(self.session_id).dir / "versions.jsonl"
            if vf.exists():
                import json
                for ln in vf.read_text("utf-8").splitlines()[-15:]:
                    d = json.loads(ln)
                    _p(f"  v{d['v']}  {d['reason']}", "amber")
            else:
                _p("(noch keine Versionen)", "dim")
        elif cmd == "/sessions":
            for sid in self.engine.sessions:
                _p(f"  {sid}", "cyan")
        else:
            _p(f"Unbekannter Befehl: {cmd}  (/help)", "red")
        return True


# ------------------------------------------------------------------ entry
async def _main(args) -> None:
    _p(f"Lade Agent '{args.agent}' …", "dim")
    agent = await _load_agent(args.agent)
    _p(f"Agent geladen: {getattr(agent.amd, 'name', args.agent)}", "green")

    from .isaa import make_page_engine
    engine = make_page_engine(agent, root=args.root, cfg=None)

    watch = args.watch.split(",") if args.watch else []
    cli = PageCLI(engine, session_id=args.session, watch=watch, agent=agent)
    try:
        await cli.repl()
    finally:
        await engine.close()
        _p("Engine geschlossen.", "dim")


def main() -> None:
    parser = argparse.ArgumentParser(description="Page Engine CLI (page-orientiert)")
    parser.add_argument("agent", nargs="?", default="self",
                        help="ISAA-Agentenname (default: self)")
    parser.add_argument("--session", default="default", help="Session-ID")
    parser.add_argument("--watch", default="",
                        help="Slots permanent anzeigen, z.B. P1,P2,P5")
    parser.add_argument("--root", default=None,
                        help="Root-Verzeichnis (default: aus Agent)")
    args = parser.parse_args()

    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
