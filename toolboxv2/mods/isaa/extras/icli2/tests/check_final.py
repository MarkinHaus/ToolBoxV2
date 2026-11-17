"""Die Endantwort kommt immer an.

1. Zwischentext, Tool, Zwischentext, dann final_answer -> Antwort sichtbar (war verworfen).
2. Dasselbe mit langsamem Zeichnen (Terminal im Hintergrund) -> nichts geht verloren (war leer).
3. Antwort schon woertlich gestreamt -> genau einmal sichtbar.
4. Antwort nur im done-Chunk -> sichtbar.
"""
import asyncio
import stub  # noqa: F401
from textual.widgets import Markdown
from toolboxv2.mods.isaa.extras.icli2 import protocol as P
from toolboxv2.mods.isaa.extras.icli2.app import ICli2
from toolboxv2.mods.isaa.extras.icli2.engine import Engine

ANSWER = "Die Antwort ist **42**."


class F:
    def attach(s, sp): pass
    async def boot(s): return {"agent": "self", "session": "default", "model": "m", "complex_model": "m", "mounts": 0}
    async def is_running(s, a, b): return False
    async def stop(s, a, b): pass
    def tool_meta(s, n, r): return ""
    async def close(s): pass

    async def stream(s, a, b, q):
        if q == "gestreamt":                             # Antwort als Text, danach dieselbe als final_answer
            for w in ANSWER.split(" "):
                yield {"type": "content", "chunk": w + " "}
            yield {"type": "final_answer", "answer": ANSWER}
            yield {"type": "done", "success": True}
            return
        if q == "nurdone":
            yield {"type": "tool_start", "name": "vfs_shell", "args": {"command": "ls"}, "id": "1"}
            yield {"type": "tool_result", "name": "vfs_shell", "id": "1", "result": {"success": True}}
            yield {"type": "done", "success": True, "final_answer": ANSWER}
            return
        yield {"type": "content", "chunk": "Ich schaue kurz nach. "}
        yield {"type": "tool_start", "name": "vfs_shell", "args": {"command": "ls"}, "id": "1"}
        yield {"type": "tool_result", "name": "vfs_shell", "id": "1", "result": {"success": True}}
        for w in "Fertig, ich gebe jetzt die finale Antwort .".split():
            yield {"type": "content", "chunk": w + " "}
            await asyncio.sleep(0.01)
        yield {"type": "tool_start", "name": "final_answer", "args": {"answer": ANSWER}, "id": "2"}
        yield {"type": "final_answer", "answer": ANSWER}
        yield {"type": "done", "success": True, "final_answer": ANSWER}


async def run_case(query: str) -> str:
    eng = Engine(F()); eng.start(); app = ICli2(eng)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(0.3)
        eng.send(P.CHAT, query=query, agent="self", session="default", tag="")
        for _ in range(60):
            await pilot.pause(0.1)
            done = [v for v in app.runs.values() if v.status != "running"]
            if done:
                break
        await pilot.pause(0.5)
        shown = "\n".join(w.source for w in app.chat.query(".answer"))
        eng.send(P.SHUTDOWN)
    eng.join(5)
    return shown


async def main():
    shown = await run_case("frage")
    assert "42" in shown and "finale Antwort ." in shown and shown.count("42") == 1, shown

    original = Markdown.append                           # langsames Zeichnen nachstellen
    async def slow(self, md):
        await asyncio.sleep(0.15)
        return await original(self, md)
    Markdown.append = slow
    try:
        shown = await run_case("frage")
    finally:
        Markdown.append = original
    assert "42" in shown and "Ich schaue kurz nach." in shown and "finale Antwort ." in shown, shown

    shown = await run_case("gestreamt")
    assert shown.count("42") == 1, shown

    shown = await run_case("nurdone")
    assert "42" in shown, shown
    print("final ok")


asyncio.run(main())
