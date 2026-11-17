"""Pruefung von connect.py mit Fake-App, Fake-Agent und Fake-mcp_session_manager."""
import asyncio, sys, types, json, tempfile
from pathlib import Path
import stub

vorher = {m for m in sys.modules if m.startswith("toolboxv2")}
from toolboxv2.mods.isaa.extras.icli2.connect import ConnectMixin
neu = {m for m in sys.modules if m.startswith("toolboxv2")} - vorher
assert all(m.startswith("toolboxv2.mods.isaa.extras.icli2") for m in neu), neu  # keine toolboxv2-Importe beim Laden

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

tmp = Path(tempfile.mkdtemp())
state = {"fail": False}

class Mgr:
    def __init__(s): s.sessions, s.capabilities_cache = {}, {}
    async def get_session_with_timeout(s, name, cfg):
        if state["fail"]: return None
        s.sessions[name] = ("sess", cfg); return s.sessions[name]
    async def _cleanup_session(s, name):
        s.sessions.pop(name, None); s.capabilities_cache.pop(name, None)
async def reg(agent, session, name, mgr):
    mgr.capabilities_cache[name] = {"tools": {"a": {}, "b": {}}}
    for t in ("a", "b"): agent.tool_manager._registry[f"{name}_{t}"] = 1
    return 2
mod("toolboxv2.mods.isaa.extras.mcp_session_manager", MCPSessionManager=Mgr, _register_mcp_session=reg)

class TM:
    def __init__(s): s._registry = {"other_x": 1}
    def unregister(s, n): return s._registry.pop(n, None) is not None
class SM:
    def __init__(s): s.skills = {}
    def import_skill(s, d, overwrite=False): s.skills[d["id"]] = d; return True
class Eng: pass
class Agent:
    def __init__(s): s.tool_manager = TM(); s.sm = SM()
    def _get_execution_engine(s):
        if state.get("boom"): raise RuntimeError("kaputt")
        e = Eng(); e.skills_manager = s.sm; return e
class Log:
    def __init__(s): s.w = []
    def warning(s, m): s.w.append(m)
class B(ConnectMixin):
    def __init__(s):
        s.app = types.SimpleNamespace(appdata=str(tmp), data_dir=str(tmp), logger=Log())
        s.agent, s.session, s.a = "self", "default", Agent()
    async def _agent(s, name=None): return s.a

# Fake-Daten: agent.json mit einem Server, Plugin mit Skill + .mcp.json
adir = tmp / "Agents" / "self"; adir.mkdir(parents=True)
(adir / "agent.json").write_text(json.dumps({"mcp": {"enabled": True, "servers": [
    {"name": "fs", "command": "npx", "args": ["srv", "x" * 80], "transport": "stdio", "env": {}}]}}))
pl = tmp / "icli" / "plugins" / "demo"; (pl / ".claude-plugin").mkdir(parents=True); (pl / "skills" / "hi").mkdir(parents=True)
(pl / ".claude-plugin" / "plugin.json").write_text(json.dumps({"name": "demo", "version": "1"}))
(pl / "skills" / "hi" / "SKILL.md").write_text("---\nname: hi\ndescription: Gruss\n---\nSag hallo.")
(pl / ".mcp.json").write_text(json.dumps({"mcpServers": {"srv": {"command": "node", "args": ["s.js"]}}}))
(tmp / "icli" / "plugins" / "kein_plugin").mkdir()

async def main():
    b = B()
    title, rows = await b._rows_mcp()
    assert len(rows) == 1 and set(rows[0]) == {"value", "label", "hint"}
    assert rows[0]["value"] == "fs" and rows[0]["label"] == "· fs" and len(rows[0]["hint"]) <= 60 and "Tools" not in rows[0]["hint"], rows
    r = await b._do_mcp("fs"); assert r["text"] == "MCP fs verbunden (2 Tools)", r
    assert "fs_a" in b.a.tool_manager._registry
    rows = (await b._rows_mcp())[1]
    assert rows[0]["label"] == "✓ fs" and rows[0]["hint"].endswith("2 Tools"), rows
    r = await b._do_mcp("fs"); assert r["text"] == "MCP fs getrennt", r
    assert list(b.a.tool_manager._registry) == ["other_x"]
    try: await b._do_mcp("nix"); assert 0
    except ValueError: pass

    r = await b._do_mcp_add("cmd python -m srv --x"); assert "verbunden (2 Tools)" in r["text"] and "gespeichert" in r["text"], r
    r = await b._do_mcp_add("web http://localhost:8000/mcp"); assert "gespeichert" in r["text"], r
    srv = {s["name"]: s for s in json.loads((adir / "agent.json").read_text())["mcp"]["servers"]}
    assert set(srv) == {"fs", "cmd", "web"}, srv
    assert srv["cmd"] == {"name": "cmd", "command": "python", "args": ["-m", "srv", "--x"], "transport": "stdio", "env": {}}
    assert srv["web"]["url"] == "http://localhost:8000/mcp" and srv["web"]["transport"] == "http", srv["web"]
    for bad in ("", "nurname", "'offen"):
        try: await b._do_mcp_add(bad); assert 0, bad
        except ValueError as e: assert "Beispiel" in str(e)
    state["fail"] = True
    try: await b._do_mcp_add("x cmd"); assert 0
    except RuntimeError as e: assert "fehlgeschlagen" in str(e)
    state["fail"] = False

    title, rows = await b._rows_plugins()
    assert [r["value"] for r in rows] == ["demo"] and rows[0]["label"] == "· demo" and rows[0]["hint"] == "1 Skills, 1 MCP-Server", rows
    r = await b._do_plugin("demo"); assert "aktiviert" in r["text"], r
    assert "plugin_demo_hi" in b.a.sm.skills and "demo_srv_a" in b.a.tool_manager._registry
    assert (await b._rows_plugins())[1][0]["label"] == "✓ demo"
    r = await b._do_plugin("demo"); assert "deaktiviert" in r["text"], r
    assert not b.a.sm.skills and "demo_srv_a" not in b.a.tool_manager._registry
    assert json.loads((tmp / "icli" / "plugins_disabled.json").read_text()) == ["demo"]
    try: await b._do_plugin("nix"); assert 0
    except ValueError: pass

    b2 = B(); b2.a = b.a
    await b2._plugins_boot(); assert not b2.a.sm.skills  # deaktiviert -> nicht geladen
    (tmp / "icli" / "plugins_disabled.json").write_text("[]")
    await b2._plugins_boot(); assert "plugin_demo_hi" in b2.a.sm.skills
    b3 = B(); b3.a.sm.skills.clear(); state["boom"] = True
    await b3._plugins_boot(); assert b3.app.logger.w, "Warnung erwartet"
    state["boom"] = False
    # Verzeichnisfehler: appdata unlesbar -> wirft nie
    b4 = B(); b4._plugins_scan = lambda: 1 / 0
    await b4._plugins_boot(); assert b4.app.logger.w
asyncio.run(main())
print("connect ok")
