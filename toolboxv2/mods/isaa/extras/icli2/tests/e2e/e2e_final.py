import asyncio, os, time, traceback
os.environ.update(OPENAI_API_BASE="http://127.0.0.1:8765/v1", OPENAI_API_KEY="x", TB_DATA_DIR="/tmp/e2e/state", XDG_CONFIG_HOME="/tmp/e2e/state/cfg", APPDATA="/tmp/e2e/state/appdata", NO_PROXY="127.0.0.1,localhost",
                  no_proxy="127.0.0.1,localhost", FASTMODEL="openai/fake", COMPLEXMODEL="openai/fake",
                  BLITZMODEL="openai/fake", SUMMARYMODEL="openai/fake")
async def main():
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    from toolboxv2.mods.isaa.extras.icli2.engine import Engine
    app_tb = get_app("isaa-host"); eng = Engine(IsaaBackend(app_tb)); eng.start(); ui = ICli2(eng)
    async with ui.run_test(size=(110, 40)) as pilot:
        t0 = time.time()
        while not ui.ready and time.time() - t0 < 150: await pilot.pause(0.5)
        await pilot.press(*"/new"); await pilot.press("enter"); await pilot.pause(2)
        q = "zwischentext bitte"
        await pilot.press(*[("space" if c == " " else c) for c in q]); await pilot.press("enter")
        t = time.time(); r = None
        while time.time() - t < 400:
            await pilot.pause(0.3)
            rs = [v for v in ui.runs.values() if v.query == q]
            if rs and rs[0].status != "running": r = rs[0]; break
        await pilot.pause(1)
        shown = "\n".join(w.source for w in (r.box.query(".answer") if r else []))
        print(f"FINAL status={r and r.status} dauer={time.time()-t:.0f}s", flush=True)
        print("FINAL anzeige:", repr(shown), flush=True)
        print("FINAL antwort sichtbar:", "42" in shown, "| zwischentext:", "Ich schaue kurz nach" in shown, flush=True)
    eng.send(P.SHUTDOWN); eng.join(20); await app_tb.a_exit()
try: asyncio.run(main())
except BaseException: traceback.print_exc()
