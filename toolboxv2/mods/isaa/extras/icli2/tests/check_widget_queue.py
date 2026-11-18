"""RED: Widget-Queue (Modal-Stack) + Inline-Note-Widget + Progress-Live.

Szenarien:
  queue    : 2 modale Widgets waehrend eines offen -> nur 1 im Stack, zweites erst nach Dismiss
  close    : widget_close entfernt auch gequeuete (kein Ghost-Modal)
  refresh  : widget_update auf gequeuetes ersetzt Spec (nicht doppelt)
  inline   : template=note mode=inline -> Karte im Chat-Verlauf, kein Modal
  progress : template=progress mode=inline -> ProgressBar-Karte + widget_update setzt echten Stand
"""
import asyncio, sys
sys.path.insert(0, r"C:\Users\Markin\Workspace\ToolBoxV2")
from textual.widgets import Markdown
from toolboxv2.mods.isaa.extras.icli2 import protocol as P
from toolboxv2.mods.isaa.extras.icli2.engine import Engine
from toolboxv2.mods.isaa.extras.icli2.app import ICli2
from toolboxv2.mods.isaa.extras.icli2.widgets import WidgetModal

ANS = "Fertig. Zeile zwei."

class Fake:
    def __init__(s): s.st = {"agent":"self","session":"default","model":"m","complex_model":"c","mounts":0,"user":"","features":[]}
    def attach(s, spawn): pass
    async def boot(s): return dict(s.st)
    async def stream(s, agent, session, query):
        yield {"type":"content","chunk":"arbeits..."}
        yield {"type":"final_answer","answer":ANS,"success":True}
        yield {"type":"done","success":True}
    async def is_running(s,a,se): return False
    async def stop(s,a,se): pass
    async def pause(s,a,se): return False
    def tool_meta(s,n,r): return ""
    async def rows(s, what, arg=None): return {"rows":[]}
    async def do(s, action, **a): return {}
    async def close(s): pass

def _open_modals(app):
    return [s for s in app.screen_stack if isinstance(s, WidgetModal)]

async def main():
    eng = Engine(Fake()); eng.start()
    app = ICli2(eng)
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause(0.4)
        app.prompt.set_text("frage"); await pilot.press("enter")
        for _ in range(30):
            await pilot.pause(0.05)
            if not app.active: break
        await pilot.pause(0.2)

        # --- 1) queue: zweites Modal wartet hinter dem ersten ---
        app._widget_event("widget_create", {"id":"w1","mode":"modal","open":True,
            "spec":{"title":"Erstes","template":"info","text":"Eins"}})
        app._widget_event("widget_create", {"id":"w2","mode":"modal","open":True,
            "spec":{"title":"Zweites","template":"info","text":"Zwei"}})
        await pilot.pause(0.2)
        modals = _open_modals(app)
        assert len(modals) == 1, f"Queue kaputt: {len(modals)} Modale offen"
        assert modals[0].widget_id == "w1", modals[0].widget_id
        print("queue ok: nur w1 sichtbar")
        # w1 beantworten -> w2 kommt
        modals[0].dismiss({"id":"w1","values":{}})
        await pilot.pause(0.3)
        modals = _open_modals(app)
        assert len(modals) == 1 and modals[0].widget_id == "w2", f"nach Dismiss: {[m.widget_id for m in modals]}"
        print("queue ok: w2 nachgerueckt")
        modals[0].dismiss({"id":"w2","values":{}})
        await pilot.pause(0.2)

        # --- 2) close entfernt gequeuete ---
        app._widget_event("widget_create", {"id":"w3","mode":"modal","open":True,
            "spec":{"title":"Drei","template":"info","text":"D"}})
        app._widget_event("widget_create", {"id":"w4","mode":"modal","open":True,
            "spec":{"title":"Vier","template":"info","text":"V"}})
        await pilot.pause(0.15)
        app._widget_event("widget_close", {"id":"w4"})
        app._widget_event("widget_close", {"id":"w3"})
        await pilot.pause(0.2)
        assert not _open_modals(app), "close hat nicht aufgeraeumt"
        print("close ok: queue + stack leer")

        # --- 3) refresh auf gequeuetes ersetzt ---
        app._widget_event("widget_create", {"id":"w5","mode":"modal","open":True,
            "spec":{"title":"Fuenf alt","template":"info","text":"alt"}})
        app._widget_event("widget_create", {"id":"w6","mode":"modal","open":True,
            "spec":{"title":"Sechs","template":"info","text":"sechs"}})
        await pilot.pause(0.1)
        app._widget_event("widget_update", {"id":"w6","spec":{"title":"Sechs NEU","template":"info","text":"neu"}})
        await pilot.pause(0.1)
        _open_modals(app)[0].dismiss({"id":"w5","values":{}})
        await pilot.pause(0.2)
        m = _open_modals(app)
        assert m and m[0].widget_id == "w6" and "NEU" in m[0].spec.get("title",""), (m and m[0].spec)
        print("refresh ok: gequeuete Spec ersetzt")
        m[0].dismiss({"id":"w6","values":{}})
        await pilot.pause(0.15)

        # --- 4) inline note: Karte im Chat, kein Modal ---
        app._widget_event("widget_create", {"id":"w7","mode":"inline",
            "spec":{"title":"Hinweis","template":"note","text":"Wichtige **Erklaerung** im Verlauf"}})
        await pilot.pause(0.25)
        assert not any(s.widget_id == "w7" for s in _open_modals(app)), "inline darf kein Modal oeffnen"
        chat = app.query_one("#chat")
        notes = list(chat.query(".inline-note"))
        assert notes, "inline-note nicht im Chat gemountet"
        srcs = [getattr(w, "source", None) or str(getattr(w, "renderable", "")) for w in notes]
        assert any("Erklaerung" in s for s in srcs), srcs
        print("inline ok: note-Karte im Verlauf")

        # --- 5) progress live: echte Werte, update setzt Stand ---
        app._widget_event("widget_create", {"id":"w8","mode":"inline",
            "spec":{"title":"Downloads","template":"progress","text":"4 von 10 Dateien","progress":40}})
        await pilot.pause(0.25)
        cards = list(app.query_one("#chat").query(".inline-progress"))
        assert cards, "progress-Karte fehlt"
        app._widget_event("widget_update", {"id":"w8","spec":{"title":"Downloads","template":"progress","text":"10 von 10 Dateien","progress":100}})
        await pilot.pause(0.25)
        body = str(cards[0].renderable if hasattr(cards[0], "renderable") else cards[0].render())
        assert "10 von 10" in body, body
        assert not any(s.widget_id == "w8" for s in _open_modals(app))
        print("progress ok: update zeigt echten Stand")

    eng.send(P.SHUTDOWN); eng.join(5)
    print("ALL OK")

asyncio.run(main())
