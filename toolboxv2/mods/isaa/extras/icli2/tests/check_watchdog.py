"""Blockiert die Engine (synchroner Aufruf im Engine-Loop), meldet die UI einmal WO - und ein
/-Befehl in dieser Zeit bekommt sofort eine Rueckmeldung statt im Nichts zu landen."""
import asyncio
import time
import stub  # noqa: F401
from textual.widgets import Static
from toolboxv2.mods.isaa.extras.icli2 import app as A
from toolboxv2.mods.isaa.extras.icli2.app import ICli2
from toolboxv2.mods.isaa.extras.icli2.engine import Engine
from toolboxv2.mods.isaa.extras.icli2 import protocol as P


def blocking_call():
    time.sleep(4)                                        # der "Verursacher"


class F:
    def attach(s, sp): pass
    async def boot(s): return {"agent": "self", "session": "default", "model": "m", "complex_model": "m", "mounts": 0}
    async def is_running(s, a, b): return False
    async def stop(s, a, b): pass
    def tool_meta(s, n, r): return ""
    async def close(s): pass
    async def rows(s, what, arg=None):
        if what == "block":
            blocking_call()
        return {"title": "x", "rows": []}


async def main():
    A.STALL_SECONDS = 2.0
    eng = Engine(F()); eng.start(); app = ICli2(eng)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        eng.send(P.LIST, what="block", to="side")
        await pilot.pause(3.0)
        notes = [str(w.render()) for w in app.chat.query(Static) if w.has_class("note-warn")]
        assert any("blockiert" in n and "blocking_call" in n for n in notes), notes
        await app._command("/agent self")
        notes = [str(w.render()) for w in app.chat.query(Static) if w.has_class("note-warn")]
        assert any("/agent: Engine ist seit" in n for n in notes), notes
        await pilot.pause(2.0)
        assert not app._stall                            # nach der Blockade zurueckgesetzt
        eng.send(P.SHUTDOWN)
    eng.join(5)
    print("watchdog ok")


asyncio.run(main())
