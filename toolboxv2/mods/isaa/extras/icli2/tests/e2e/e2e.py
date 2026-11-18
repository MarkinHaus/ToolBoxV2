import asyncio, os, sys, time, traceback
os.environ.update(OPENAI_API_BASE="http://127.0.0.1:8765/v1", OPENAI_API_KEY="x", TB_DATA_DIR="/tmp/e2e/state", XDG_CONFIG_HOME="/tmp/e2e/state/cfg", APPDATA="/tmp/e2e/state/appdata", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                  FASTMODEL="openai/fake", COMPLEXMODEL="openai/fake", BLITZMODEL="openai/fake", SUMMARYMODEL="openai/fake", DEFAULTMODELST="openai/fake")
async def main():
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    from toolboxv2.mods.isaa.extras.icli2.engine import Engine
    from toolboxv2.mods.isaa.extras.icli2.picker import Picker
    app_tb = get_app("isaa-host")
    eng = Engine(IsaaBackend(app_tb)); eng.start()
    ui = ICli2(eng)
    def notes(): return [str(n.render()) for n in ui.query(".note")]
    async with ui.run_test(size=(110, 34)) as pilot:
        t0 = time.time()
        while not ui.ready and time.time() - t0 < 150: await pilot.pause(0.5)
        print("READY", ui.ready, round(time.time() - t0, 1), "s |", ui.agent, ui.model, ui.session, "| notes", notes()[-3:])
        if not ui.ready: ui.save_screenshot("/tmp/e2e/fail.svg"); return
        async def say(text, wait=120):
            await pilot.press(*[("space" if c == " " else c) for c in text]); await pilot.press("enter")
            t = time.time(); await pilot.pause(0.5)
            while (ui.active or ui.pending) and time.time() - t < wait: await pilot.pause(0.3)
            return list(ui.runs.values())[-1] if ui.runs else None
        r = await say("hallo welt")
        print("RUN1", r and (r.status, round(r.seconds, 1)), "| answer:", repr(r.answer[:120]) if r else None, "| tools:", [str(c.title) for c in ui.tool_cards], "| think:", [str(b.title) for b in ui.think_boxes], "| notes", notes()[-3:])
        if ui.tool_cards: print("TOOLBODY", ui.tool_cards[0].body[:300].replace("\n", " | "))
        ui.save_screenshot("/tmp/e2e/run1.svg")
        for cmd in ("/settings", "/agent", "/model", "/session", "/feature", "/skill", "/mcp", "/plugin", "/job", "/tb", "/share", "/voice"):
            await pilot.press(*cmd); await pilot.press("enter"); await pilot.pause(1.5)
            if isinstance(ui.screen, Picker):
                from textual.widgets import OptionList
                ol = ui.screen.query_one(OptionList)
                print(f"PICK {cmd}: {ui.screen._title!r} {ol.option_count} Zeilen, z.B.", [str(ol.get_option_at_index(i).prompt)[:50] for i in range(min(3, ol.option_count))])
                await pilot.press("escape"); await pilot.pause(0.2)
            else:
                print(f"NOTE {cmd}:", notes()[-1][:160] if notes() else None)
        for cmd in ("/login", "/new", "/mount /tmp/e2e", "/vfs /project/", "/vfs /project/fake_llm.py", "/vfs /gibtsnicht", "/set max_iterations 12", "/plugin install /tmp/e2e/plug/demo-plugin", "/plugin", "/plugin remove demo-plugin", "/plugin markets", "/dream", "/network", "/pack"):
            n = len(notes()); await pilot.press(*[("space" if c == " " else c) for c in cmd]); await pilot.press("enter"); await pilot.pause(2.5)
            if isinstance(ui.screen, Picker):            # Auswahl offen: Zeilen zeigen und schliessen
                from textual.widgets import OptionList
                ol = ui.screen.query_one(OptionList)
                print(f"DO {cmd}: Picker {ui.screen._title!r}", [str(ol.get_option_at_index(i).prompt)[:60] for i in range(ol.option_count)][:4])
                await pilot.press("escape"); await pilot.pause(0.3)
                continue
            print(f"DO {cmd}:", [x[:140] for x in notes()[n:]])
        # Nachsteuerung gegen den echten isaa-Agenten: zweite Nachricht, waehrend der erste Lauf arbeitet
        await pilot.press(*"/new"); await pilot.press("enter"); await pilot.pause(2)
        await pilot.press(*[("space" if c == " " else c) for c in "lange aufgabe bitte"]); await pilot.press("enter")
        t = time.time()
        while not ui.active and time.time() - t < 20: await pilot.pause(0.1)
        await pilot.pause(0.8)
        main = next(iter(ui.active.values()), None)
        await pilot.press(*[("space" if c == " " else c) for c in "kurze zwischenfrage"]); await pilot.press("enter")
        t = time.time()
        while (ui.active or ui.pending) and time.time() - t < 120: await pilot.pause(0.3)
        steer = next((v for v in ui.runs.values() if v.query == "kurze zwischenfrage"), None)
        print("STEER main lief:", main is not None, "| im Hauptlauf:", bool(main and steer and steer.box.parent is main.box),
              "| status:", steer and steer.status, "| antwort:", repr((steer.answer if steer else "")[:70]))
        if main and steer and steer.box.parent is main.box:
            kids = list(main.box.children); i = kids.index(steer.box)
            print("STEER danach im Hauptlauf:", [type(w).__name__ + ("." + " ".join(sorted(w.classes))) for w in kids[i + 1:]][:5])
        print("FROZEN", sum(type(w).__name__ == "FrozenMarkdown" for w in ui.chat.query(".answer")), "live", sum(type(w).__name__ == "Markdown" for w in ui.chat.query(".answer")))
        r = await say("zweite frage nach dem mount")
        print("RUN2", r and (r.status, r.session), "| answer:", repr(r.answer[:80]) if r else None)
        ui.prompt.set_text(""); await pilot.press(*"/vfs"); await pilot.press("space"); await pilot.press(*"/pro"); await pilot.pause(1.2)
        print("COMPLETE /vfs /pro ->", ui.completions)
        await pilot.press("tab"); await pilot.pause(1.2); print("COMPLETE nach Tab ->", ui.prompt.text, [c[1] for c in ui.completions][:6]); ui.prompt.set_text("")
        await pilot.press(*"/context"); await pilot.press("enter"); await pilot.pause(4); print("SIDE", str(ui.w_side_text.render()).replace("\n", " | ")[-700:])
        await pilot.press("ctrl+o"); await pilot.pause(0.3); ui.save_screenshot("/tmp/e2e/end.svg")
    eng.send(P.SHUTDOWN); eng.join(20); print("ENGINE STOPPED", not eng._thread.is_alive())
    await app_tb.a_exit()
try: asyncio.run(main())
except BaseException: traceback.print_exc()
