"""Benchmark der UI-Fluessigkeit (kein Pruefskript, misst nur): 25 fertige Laeufe als Verlauf, dann zwei
parallel streamende Laeufe mit Tool-Aufrufen; waehrenddessen wird wie im Terminal gescrollt.
Gemessen: Verzoegerung der UI-Schleife (10-ms-Sonde) und Zeit bis eine Scroll-Taste verarbeitet ist.

    python bench_render.py messung        # ohne Rechenlast
    python bench_render.py messung cpu    # mit Rechenlast im Engine-Thread (GIL-Konkurrenz)
"""
import asyncio, statistics, sys, time
import stub  # noqa
from toolboxv2.mods.isaa.extras.icli2 import protocol as P
from toolboxv2.mods.isaa.extras.icli2.engine import Engine
from toolboxv2.mods.isaa.extras.icli2.app import ICli2

CPU = "cpu" in sys.argv
PARA = ("Das ist ein **längerer** Absatz mit `code` und einer Aufzählung, die der Agent schreibt. " * 3)
MD = (f"## Abschnitt\n\n{PARA}\n\n- Punkt eins\n- Punkt zwei\n\n```python\nprint('x')\nfor i in range(3):\n    pass\n```\n\n") * 6

class Fake:
    def attach(s, sp): pass
    async def boot(s): return {"agent": "self", "session": "default", "model": "m", "complex_model": "m", "mounts": 0}
    async def is_running(s, a, b): return False
    async def stop(s, a, b): pass
    async def pause(s, a, b): return False
    def tool_meta(s, n, r): return "3 Zeilen"
    async def close(s): pass
    async def rows(s, w, arg=None): return {"title": "", "rows": []}
    async def do(s, action, **k): return {}
    async def stream(s, agent, session, query):
        fast = query.startswith("hist")
        words = MD.split(" ")
        yield {"type": "reasoning", "chunk": "Ich überlege. " * 20}
        for i, w in enumerate(words):
            if i % 60 == 0:
                yield {"type": "tool_start", "name": "vfs_shell", "args": {"command": f"ls {i}"}, "id": f"t{i}"}
                yield {"type": "tool_result", "name": "vfs_shell", "id": f"t{i}", "result": {"success": True, "stdout": "a\nb"}}
            yield {"type": "content", "chunk": w + " "}
            if not fast:
                if CPU and i % 10 == 0:
                    t = time.perf_counter()
                    while time.perf_counter() - t < 0.004: pass          # 4 ms Rechenlast je 10 Tokens
                await asyncio.sleep(0.0025)                               # ~400 Tokens/s
        yield {"type": "final_answer", "answer": ""}

async def main():
    eng = Engine(Fake()); eng.start()
    app = ICli2(eng)
    lags = []
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        for i in range(25):                                   # langer Verlauf
            eng.send(P.CHAT, query=f"hist {i}", agent="self", session=f"h{i}", tag="")
        t0 = time.monotonic()
        while (app.active or eng._runs) and time.monotonic() - t0 < 120: await pilot.pause(0.2)
        await pilot.pause(0.5)
        n_widgets = len(list(app.chat.query("*")))
        last = [time.perf_counter()]
        def probe():
            now = time.perf_counter(); lags.append(now - last[0] - 0.01); last[0] = now
        timer = app.set_interval(0.01, probe)
        for q in ("live a", "live b"):                           # zwei parallele Laeufe
            eng.send(P.CHAT, query=q, agent="self", session=f"s-{q}", tag="")
        await pilot.pause(0.5)
        # Scrollen wie im Terminal: Tastenereignis -> Bindung -> scroll; gemessen wird die Zeit bis
        # die App das Ereignis verarbeitet hat (ohne den Test-Piloten, der alle Widgets abwartet)
        keys, pending = [], []
        from textual import events
        def press():
            k = ["up", "up", "up", "pageup", "pagedown", "down"][len(keys) % 6]
            pending.append(time.perf_counter())
            app.chat.post_message(events.Key(k, None))
            app.call_after_refresh(lambda: keys.append(time.perf_counter() - pending.pop(0)) if pending else None)
        app.chat.focus()
        scroller = app.set_interval(0.12, press)
        t1 = time.monotonic()
        while app.active and time.monotonic() - t1 < 25: await asyncio.sleep(0.2)
        scroller.stop()
        while app.active and time.monotonic() - t0 < 240: await asyncio.sleep(0.2)
        timer.stop()
    eng.send(P.SHUTDOWN); eng.join(5)
    lags.sort(); keys.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] * 1000
    print(f"{sys.argv[1]:<12} widgets={n_widgets:<5} lag p50={q(lags,.5):5.1f} p95={q(lags,.95):6.1f} max={lags[-1]*1000:6.0f} ms"
          f" | taste p50={q(keys,.5):5.1f} p95={q(keys,.95):6.1f} max={keys[-1]*1000:5.0f} ms | n={len(lags)}")
asyncio.run(main())
