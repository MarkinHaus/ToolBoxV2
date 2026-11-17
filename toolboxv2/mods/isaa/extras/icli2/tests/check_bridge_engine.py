"""Discord/Telegram-Bruecke gegen die echte Engine: ein ueber run_agent_monitored gestarteter
Lauf muss beim Aufrufer UND als Lauf in der UI ankommen."""
import logging, time, types
import stub  # noqa: F401
from toolboxv2.mods.isaa.extras.icli2 import protocol as P
from toolboxv2.mods.isaa.extras.icli2.bridges import HostShim
from toolboxv2.mods.isaa.extras.icli2.engine import Engine


class Agent:
    async def a_stream(self, query, session_id, max_iterations, user_lightning_model=None):
        self.asked = query
        yield {"type": "tool_start", "name": "discord_send", "args": {"text": "hi"}, "id": "1"}
        yield {"type": "tool_result", "name": "discord_send", "id": "1", "result": {"success": True}}
        yield {"type": "content", "chunk": "Antwort an Discord."}
        yield {"type": "final_answer", "answer": "Antwort an Discord."}


class Backend:
    agent, session, max_iterations = "self", "default", 5

    def __init__(self):
        self.app = types.SimpleNamespace(logger=logging.getLogger("b"))
        self.a = Agent()
        self.isaa = types.SimpleNamespace(get_agent=self._get)
        self.shim = HostShim(self)
        self.got = []

    async def _get(self, name): return self.a
    def attach(self, spawn): self._spawn = spawn
    async def boot(self): return {}
    async def is_running(self, a, s): return False
    async def stop(self, a, s): pass
    def tool_meta(self, n, r): return ""
    async def close(self): pass
    async def rows(self, what): return {"title": "", "rows": []}

    async def do(self, action, **kw):                    # so ruft die Discord-Erweiterung den Host auf
        async for chunk in self.shim.run_agent_monitored(
                "self", "frage aus discord", kind="discord", session_id="dc-1",
                system_prompt_override="SYS {query}"):
            self.got.append(chunk["type"])
            if action == "early" and chunk["type"] == "content":
                break
        return {"text": "fertig"}


for action in ("full", "early"):
    b = Backend(); e = Engine(b); e.start()
    e.send(P.DO, action=action); time.sleep(0.6)
    ev = e.poll()
    start = next(x for x in ev if x.kind == P.RUN_START)
    end = next(x for x in ev if x.kind == P.RUN_END)
    kinds = [x.kind for x in ev if x.run == start.run]
    assert start.data["origin"] == "discord" and start.data["session"] == "dc-1" and start.data["query"] == "frage aus discord"
    assert b.a.asked == "SYS frage aus discord"
    assert P.TOOL_START in kinds and P.TOOL_END in kinds and P.TEXT in kinds, kinds
    assert end.data["status"] == "done", end.data
    if action == "full":
        assert b.got == ["tool_start", "tool_result", "content", "final_answer"] and end.data["answer"] == "Antwort an Discord."
    else:
        assert b.got == ["tool_start", "tool_result", "content"]
    e.send(P.SHUTDOWN); e.join(5); assert not e._thread.is_alive()
print("bridge engine ok")
