"""Echter Durchlauf: Fehler (429), Iterationslimit und Sub-Agent-Kachel gegen den echten isaa-Agenten."""
import asyncio, os, time, traceback
os.environ.update(TB_LLM_GATEWAY_URL="http://127.0.0.1:8765/v1", TB_DATA_DIR="/tmp/e2e/state", XDG_CONFIG_HOME="/tmp/e2e/state/cfg", APPDATA="/tmp/e2e/state/appdata", NO_PROXY="127.0.0.1,localhost",
                  no_proxy="127.0.0.1,localhost", FASTMODEL="gateway/fake", COMPLEXMODEL="gateway/fake",
                  BLITZMODEL="gateway/fake", SUMMARYMODEL="gateway/fake", DEFAULTMODELST="gateway/fake")
async def main():
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    from toolboxv2.mods.isaa.extras.icli2.engine import Engine
    app_tb = get_app("isaa-host")
    eng = Engine(IsaaBackend(app_tb)); eng.start(); ui = ICli2(eng)
    async with ui.run_test(size=(110, 40)) as pilot:
        t0 = time.time()
        while not ui.ready and time.time() - t0 < 150: await pilot.pause(0.5)
        async def typ(text):
            await pilot.press(*[("space" if c == " " else c) for c in text]); await pilot.press("enter")
        async def lauf(text, wait=180):
            await typ(text); t = time.time()
            while time.time() - t < wait:
                await pilot.pause(0.3)
                r = [v for v in ui.runs.values() if v.query == text]
                if r and r[0].status != "running": return r[0], time.time() - t
            return None, time.time() - t
        await typ("/new"); await pilot.pause(2)
        r, secs = await lauf("fehler429 bitte")
        print(f"E429 status={r and r.status} dauer={secs:.0f}s")
        if r and r.problem_line:
            print("E429 titel:", r.problem_line.title)
            r.problem_line.collapsed = False; await pilot.pause(0.2)
            print("E429 details:", " | ".join(str(r.problem_line.render()).splitlines()[:7])[:600])
        await typ("/new"); await pilot.pause(2)
        await typ("/set max_iterations 1"); await pilot.pause(1)
        r, secs = await lauf("limit-test bitte", wait=420)
        print(f"LIMIT status={r and r.status} dauer={secs:.0f}s titel={r and r.problem_line and r.problem_line.title!r}")
        if r and r.problem_line: print("LIMIT body:", r.problem_line.body[:300].replace("\n", " | "))
        await typ("/set max_iterations 30"); await pilot.pause(1)
        await typ("/new"); await pilot.pause(2)
        r, secs = await lauf("subagent-test bitte", wait=240)
        print(f"SUB status={r and r.status} dauer={secs:.0f}s subs={list(r.subs) if r else None}")
        for sv in (r.subs.values() if r else []):
            print("SUB task:", repr(sv.task), "| status:", sv.status, "| iter:", sv.iter, sv.max_iter, "| tokens:", sv.tokens, sv.budget, "| timeout:", sv.timeout)
            print("SUB kachel:\n" + str(sv.tile.compact.render()))
            await sv.tile.expand(); await pilot.pause(0.5)
            print("SUB gross: kopf=", repr(str(sv.tile.head.render())[:160]), "| aktionen=", len(list(sv.tile.list.query("FoldLine"))),
                  "| fuss=", [type(w).__name__ for w in sv.tile.foot.children])
        ui.save_screenshot("/tmp/e2e/errors.svg")
    eng.send(P.SHUTDOWN); eng.join(20)
    await app_tb.a_exit()
try: asyncio.run(main())
except BaseException: traceback.print_exc()
