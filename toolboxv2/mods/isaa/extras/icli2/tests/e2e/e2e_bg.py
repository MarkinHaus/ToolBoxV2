"""Echter Durchlauf: Agent startet shell(background=True) und beendet seinen Zug; endet der Befehl,
startet die icli2 einen Folgelauf mit Exit-Code und Ausgabe - ohne dass jemand abfragt."""
import asyncio, os, time, traceback
os.environ.update(OPENAI_API_BASE="http://127.0.0.1:8765/v1", OPENAI_API_KEY="x", TB_DATA_DIR="/tmp/e2e/state", XDG_CONFIG_HOME="/tmp/e2e/state/cfg", APPDATA="/tmp/e2e/state/appdata", NO_PROXY="127.0.0.1,localhost",
                  no_proxy="127.0.0.1,localhost", FASTMODEL="openai/fake", COMPLEXMODEL="openai/fake",
                  BLITZMODEL="openai/fake", SUMMARYMODEL="openai/fake")
async def main():
    import numpy as np                                   # Embeddings: hier ohne HuggingFace (Netz gesperrt)
    from toolboxv2.mods.isaa.base import local_embeddings as LE
    async def no_model(texts, dimensions=None):
        return np.zeros((len(texts), dimensions or 768), dtype="float32")
    LE.local_embed = no_model
    from toolboxv2 import get_app
    from textual.widgets import Static
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    from toolboxv2.mods.isaa.extras.icli2.engine import Engine
    app_tb = get_app("isaa-host"); eng = Engine(IsaaBackend(app_tb)); eng.start(); ui = ICli2(eng)
    async with ui.run_test(size=(110, 40)) as pilot:
        t0 = time.time()
        while not ui.ready and time.time() - t0 < 150: await pilot.pause(0.5)
        from toolboxv2.mods.isaa import module as M
        M.add_background_listener(lambda e: print("BG event", {k: e[k] for k in ("agent", "session", "scope", "exit_code")}, flush=True))
        await pilot.press(*"/new"); await pilot.press("enter"); await pilot.pause(2)
        q = "bgshell bitte"
        await pilot.press(*[("space" if c == " " else c) for c in q]); await pilot.press("enter")
        t = time.time(); first = follow = None
        dumped = False
        while time.time() - t < 300:
            if not dumped and time.time() - t > 150:
                dumped = True
                import asyncio as _a, io
                def dump():
                    buf = io.StringIO()
                    for task in _a.all_tasks(eng._loop):
                        if "_chat" in repr(task.get_coro()):
                            task.print_stack(limit=40, file=buf)
                    print("BG STACK\n" + buf.getvalue(), flush=True)
                eng._loop.call_soon_threadsafe(dump)
            await pilot.pause(0.3)
            rs = [v for v in ui.runs.values() if v.query == q]
            if rs and rs[0].status != "running" and first is None:
                first = rs[0]; print(f"BG erster Lauf {first.status} nach {time.time()-t:.0f}s", flush=True)
            fs = [v for v in ui.runs.values() if "BACKGROUND COMMAND FINISHED" in (v.query or "")]
            if fs and fs[0].status != "running":
                follow = fs[0]; break
        await pilot.pause(1)
        notes = [str(w.render()) for w in ui.chat.query(Static) if w.has_class("note")]
        print("BG hinweis:", [n for n in notes if "Hintergrund" in n], flush=True)
        print("BG laeufe:", [(v.agent, (v.query or "")[:50], v.status) for v in ui.runs.values()], flush=True)
        shown = "\n".join(w.source for w in (follow.box.query(".answer") if follow else []))
        print(f"BG folgelauf={follow and follow.status} dauer={time.time()-t:.0f}s antwort={shown!r}", flush=True)
        first_shown = "\n".join(w.source for w in (first.box.query(".answer") if first else []))
        print(f"BG erste antwort={first_shown!r}", flush=True)
    eng.send(P.SHUTDOWN); eng.join(20); await app_tb.a_exit()
try: asyncio.run(main())
except BaseException: traceback.print_exc()
