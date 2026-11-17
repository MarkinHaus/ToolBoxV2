"""Pruefung von bridges.py mit Fake-Discord-/Telegram-Modulen und Fake-Agent."""
import asyncio, sys, types, logging
import stub
BASE = set(sys.modules)
from toolboxv2.mods.isaa.extras.icli2.bridges import BridgeMixin, HostShim
new = {m for m in set(sys.modules) - BASE if m.startswith("toolboxv2")}
assert new <= {"toolboxv2.mods.isaa.extras.icli2", "toolboxv2.mods.isaa.extras.icli2.bridges",
               "toolboxv2.mods.isaa.extras.icli2.fmt"} | {m for m in new if m.startswith("toolboxv2.mods.isaa.extras.icli2")}, new

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
DC = "toolboxv2.mods.isaa.extras.discord_interface"
TG = "toolboxv2.mods.isaa.extras.telegram_interface"
for n in (DC, TG): mod(n).__path__ = []

class DcExt:
    def __init__(s, host): s.host = host; s._connected = False; s.closed = 0
    async def handle_command(s, args):
        print(f"\x1b[32mdiscord {args}\x1b[0m")
    async def _disconnect(s): s.closed += 1; s._connected = False
def patch(host):
    host.discord_ext = DcExt(host); orig = host._handle_command
    async def patched(text):
        parts = text.strip().split()
        if parts and parts[0] == "/discord": await host.discord_ext.handle_command(parts[1:]); return
        await orig(text)
    host._handle_command = patched
    print("patched")
mod(DC + ".integration_example", patch_cli_for_discord=patch)
mod(DC + ".discord_cli_extension", get_completer_dict=lambda: {"connect": None, "status": None, "whitelist": {"add": None}})
class TgExt:
    def __init__(s, host): s.icli = host; s.is_connected = True; s.closed = 0
    async def handle_command(s, args): return f"tg:{args}"
    async def cmd_disconnect(s): s.closed += 1; raise RuntimeError("boom")
mod(TG + ".telegram_cli_extension", TelegramCliExtension=TgExt)

class Agent:
    mode = "ok"
    async def a_stream(self, query, session_id="default", max_iterations=1, user_lightning_model=False, **kw):
        self.last = (query, session_id, max_iterations, user_lightning_model)
        yield {"type": "content", "chunk": "a"}
        yield {"type": "content", "chunk": "b"}
        if self.mode == "raise": raise ValueError("kaputt")
        yield {"type": "final_answer", "answer": "ab"}
class Isaa:
    def __init__(s): s.agent = Agent()
    async def get_agent(s, name):
        if name == "nope": raise KeyError(name)
        return s.agent
class B(BridgeMixin):
    def __init__(s):
        s.app = types.SimpleNamespace(logger=logging.getLogger("t")); s.isaa = Isaa(); s.agent = "self"
        s.session = "default"; s.max_iterations = 7; s.rec = []; s.fail = False; s.hold = False
    async def _spawn(s, agent, session, query, kind="job", stream=None):
        if s.fail: raise RuntimeError("spawn kaputt")
        chunks = [c async for c in stream]; s.rec.append((agent, session, query, kind, chunks)); return "x"

async def settle():
    for _ in range(5): await asyncio.sleep(0)

async def main():
    b = B()
    await b._bridges_boot(); sh = b._shim
    assert isinstance(sh, HostShim) and isinstance(sh.discord_ext, DcExt) and isinstance(sh.telegram_ext, TgExt)
    d, t = sh.discord_ext, sh.telegram_ext
    await b._bridges_boot(); assert sh is b._shim and d is sh.discord_ext and t is sh.telegram_ext
    assert sh.isaa_tools is b.isaa and sh._build_completer() == ({}, None)
    sh.active_agent_name = "x"; assert b.agent == "x" and sh.active_agent_name == "x"

    r = await b._do_discord("status"); assert r == {"text": "discord ['status']"}, r
    r = await b._do_telegram("status"); assert r == {"text": "tg:status"}, r
    title, rows = await b._rows_discord()
    assert [x["value"] for x in rows] == ["connect", "status", "whitelist"] and all(set(x) == {"value", "label", "hint"} for x in rows), rows
    assert "add" in rows[2]["hint"]
    assert "status" in [x["value"] for x in (await b._rows_telegram())[1]]

    # Chunks gehen unveraendert durch und werden gespiegelt
    got = [c async for c in sh.run_agent_monitored("self", "hi", kind="discord", session_id="s1", system_prompt_override="P:{query}")]
    assert [c["type"] for c in got] == ["content", "content", "final_answer"]
    await settle()
    assert b.isaa.agent.last == ("P:hi", "s1", 7, False), b.isaa.agent.last
    assert b.rec == [("self", "s1", "hi", "discord", got)], b.rec
    # Fehler beim Agent holen
    got = [c async for c in sh.run_agent_monitored("nope", "q")]; assert got[0]["type"] == "error"
    # fruehes Abbrechen
    b.rec.clear()
    gen = sh.run_agent_monitored("self", "q2", session_id=None)
    async for c in gen: break
    await gen.aclose(); await settle()
    assert len(b.rec) == 1 and b.rec[0][1] == "default" and len(b.rec[0][4]) == 1, b.rec
    # a_stream wirft
    b.rec.clear(); b.isaa.agent.mode = "raise"
    try:
        async for c in sh.run_agent_monitored("self", "q3"): pass
        assert 0
    except ValueError: pass
    await settle(); assert b.rec[0][4][-1] == {"type": "error", "message": "kaputt"}, b.rec
    b.isaa.agent.mode = "ok"
    # _spawn scheitert -> Stream laeuft weiter
    b.fail = True
    got = [c async for c in sh.run_agent_monitored("self", "q4")]; assert len(got) == 3
    await settle()
    b.fail = False

    # close wirft nie (Telegram wirft, Discord schliesst)
    d._connected = True
    await b._bridges_close(); assert d.closed == 1 and t.closed == 1
    await B()._bridges_close()

    # Discord-Import scheitert -> Telegram haengt trotzdem, Fehler mit Grund
    del sys.modules[DC + ".integration_example"]
    def bad(*a, **k): raise ImportError("kein discord")
    mod(DC + ".integration_example").__dict__["__getattr__"] = lambda n: bad()
    b2 = B(); await b2._bridges_boot()
    assert b2._shim.discord_ext is None and b2._shim.telegram_ext is not None
    try: await b2._do_discord("status"); assert 0
    except RuntimeError as e: assert str(e).startswith("Discord-Integration nicht verfügbar: "), e
    assert (await b2._do_telegram("x"))["text"] == "tg:x"
asyncio.run(main())
print("bridges ok")
