"""Repro/Regression: Finale Antwort exakt EINMAL als Stream (Render 1).
Faelle: Teil-Stream + final_answer-Volltext (Overlap), final_answer ohne Stream,
final_answer nach Tool (seg-Reset), max_iterations (Zusammenfassung muss streamen)."""
import asyncio, sys
sys.path.insert(0, r"C:\Users\Markin\Workspace\ToolBoxV2")
from textual.widgets import Markdown
from toolboxv2.mods.isaa.extras.icli2 import protocol as P
from toolboxv2.mods.isaa.extras.icli2.engine import Engine
from toolboxv2.mods.isaa.extras.icli2.app import ICli2

FULL = "Erste Zeile mit Inhalt. Zweite Zeile mit mehr Inhalt. Dritte Zeile."
SUMMARY = "Zusammenfassung: drei Dateien gelesen, kein Ergebnis."

class Fake:
    def __init__(s, mode): s.mode = mode
    s_st = {"agent":"self","session":"default","model":"m","complex_model":"c","mounts":0,"user":"","features":[]}
    def attach(s, spawn): pass
    async def boot(s): return dict(s.st)
    async def stream(s, agent, session, query):
        if s.mode == "overlap":            # Teil-Stream (Praefix), dann final_answer mit VOLLTEXT
            yield {"type":"content","chunk":FULL[:22]}
            await asyncio.sleep(0.01)
            yield {"type":"final_answer","answer":FULL,"success":True}
            yield {"type":"done","success":True}
        elif s.mode == "no_stream":        # Antwort nur als final_answer-Chunk
            yield {"type":"final_answer","answer":FULL,"success":True}
            yield {"type":"done","success":True}
        elif s.mode == "tool_between":     # Hinweis, Tool, dann abweichende Antwort
            yield {"type":"content","chunk":"Kurzer Hinweis. "}
            yield {"type":"tool_start","name":"vfs_shell","args":{"command":"ls"},"id":"t1"}
            yield {"type":"tool_result","name":"vfs_shell","id":"t1","result":{"success":True}}
            yield {"type":"final_answer","answer":FULL,"success":True}
            yield {"type":"done","success":True}
        elif s.mode == "limit":            # max_iterations: Antwort darf nicht nur im FoldLine landen
            yield {"type":"status","status_msg":"Schritt 5","iter":5,"max_iter":5}
            yield {"type":"max_iterations","answer":SUMMARY,"iter":5,"max_iter":5}
            yield {"type":"done","success":False,"final_answer":"."}
    async def is_running(s,a,se): return False
    async def stop(s,a,se): pass
    async def pause(s,a,se): return False
    def tool_meta(s,n,r): return ""
    async def rows(s, what, arg=None): return {"rows":[]}
    async def do(s, action, **a): return {}
    async def close(s): pass

async def drive(mode):
    eng = Engine(Fake(mode)); eng.start()
    app = ICli2(eng)
    async with app.run_test(size=(90, 28)) as pilot:
        await pilot.pause(0.4)
        app.prompt.set_text("frage"); await pilot.press("enter")
        for _ in range(60):
            await pilot.pause(0.05)
            if not app.active: break
        await pilot.pause(0.3)
        live = "".join(m.source for m in app.query(Markdown))
        frozen = "".join(w.source for w in app.query(".answer") if not isinstance(w, Markdown))
        view = list(app.runs.values())[-1]
        body = getattr(view.problem_line, "body", "") if view.problem_line else ""
        eng.send(P.SHUTDOWN); eng.join(5)
        return live, frozen, view, body

async def main():
    # 1) Overlap: Sichtbarer Gesamttext == FULL, "Erste Zeile" genau 1x
    live, frozen, view, _ = await drive("overlap")
    total = live + frozen
    assert live.count("Erste Zeile") == 1, f"Duplikat im Stream: {live!r}"
    assert "Dritte Zeile." in live, f"Rest fehlt: {live!r}"
    print("overlap ok")
    # 2) Kein Stream: Antwort einmal sichtbar
    live, frozen, view, _ = await drive("no_stream")
    total = live + frozen
    assert total.count("Dritte Zeile.") == 1 and "Zweite Zeile" in total, total
    print("no_stream ok")
    # 3) Tool dazwischen: Hinweis + Antwort, keine Ueberschneidung
    live, frozen, view, _ = await drive("tool_between")
    total = live + frozen
    assert "Kurzer Hinweis." in total and total.count("Erste Zeile mit Inhalt.") == 1, total
    print("tool_between ok")
    # 4) Limit: Zusammenfassung muss als Stream-Text landen, FoldLine OHNE Kopie
    live, frozen, view, body = await drive("limit")
    total = live + frozen
    assert "Zusammenfassung: drei Dateien" in total, f"Summary nicht gestreamt: {total!r}"
    assert "Zusammenfassung des Agenten" not in body, f"FoldLine-Duplikat: {body!r}"
    print("limit ok")

asyncio.run(main())
