"""Blockierte Engine: /-Befehle stehen mit Spinner in der Warteschlange (max. 5), bis die Engine
antwortet; Chat-Nachrichten (Nachsteuerung) laufen daran vorbei. Hinweise des Wachhunds nur im
Debug-Modus und jede Stelle nur einmal; ins Log geht sie immer."""
import asyncio
import time
import stub  # noqa: F401
from textual.widgets import Static
from toolboxv2.mods.isaa.extras.icli2 import app as A
from toolboxv2.mods.isaa.extras.icli2.app import ICli2
from toolboxv2.mods.isaa.extras.icli2.engine import Engine
from toolboxv2.mods.isaa.extras.icli2 import protocol as P


def blocking_call():
    time.sleep(3)                                        # der "Verursacher"


class F:
    def attach(s, sp): pass
    async def boot(s): return {"agent": "self", "session": "default", "model": "m", "complex_model": "m",
                               "mounts": 0, "debug": s.debug}
    async def is_running(s, a, b): return False
    async def stop(s, a, b): pass
    def tool_meta(s, n, r): return ""
    async def close(s): pass
    async def rows(s, what, arg=None):
        if what == "block":
            blocking_call()
        return {"title": "Agent wechseln", "rows": [{"value": "self", "label": "self", "hint": ""}]}
    async def do(s, action, **a):
        s.log.append((action, a))
        return {"text": f"{action} ok"}


def warns(app):
    return [str(w.render()) for w in app.chat.query(Static) if w.has_class("note-warn")]


async def scenario(debug: bool):
    fake = F(); fake.debug, fake.log = debug, []
    eng = Engine(fake); eng.start(); app = ICli2(eng)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        assert app.debug_mode is debug
        for _ in range(2):                               # zweimal an derselben Stelle blockieren
            eng.send(P.LIST, what="block", to="side")
            await pilot.pause(0.3)
            if _ == 0:
                for cmd in ("/set max_iterations 9", "/feature coder", "/skill a", "/mcp b", "/plugin c"):
                    await app._command(cmd)
                await app._command("/agent self")          # sechster: Warteschlange voll
                await pilot.pause(0.4)
                queue = str(app.w_queue.render())
                assert app.w_queue.has_class("-open") and "/feature coder" in queue and "wartet" in queue, queue
                assert len(app.queue) == 5 and any("5 Befehle warten" in n for n in warns(app)), warns(app)
            await pilot.pause(3.0)
        await pilot.pause(0.5)
        assert not app.queue and not app.w_queue.has_class("-open")   # nach der Antwort weg
        assert ("set", {"key": "max_iterations", "value": "9"}) in fake.log and len(fake.log) == 5, fake.log
        stalls = [n for n in warns(app) if "blockiert" in n]
        if debug:
            assert len(stalls) == 1 and "blocking_call" in stalls[0], stalls   # nur einmal
        else:
            assert not stalls and "Engine beschäftigt" not in str(app.w_foot.render()), stalls
        eng.send(P.SHUTDOWN)
    eng.join(5)


async def main():
    A.STALL_SECONDS = 2.0
    await scenario(debug=False)
    await scenario(debug=True)
    print("watchdog/queue ok")


asyncio.run(main())
