"""Pruefung von features.py mit Fake-Host, Fake-Agent und Fake-Modulen."""
import asyncio, sys, types
import stub
before = set(sys.modules)
from toolboxv2.mods.isaa.extras.icli2.features import FeatureMixin, SimpleFeatureManager
new = {m for m in set(sys.modules) - before if m.startswith("toolboxv2")}
assert new <= {"toolboxv2.mods.isaa.extras.icli2", "toolboxv2.mods.isaa.extras.icli2.features"}, new

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
log = []
class Kit:
    def __init__(s, **k): pass
    async def start_browser(s): log.append("start")
    async def stop_browser(s): log.append("stop")
mod("toolboxv2.mods.isaa.extras.web_helper"); sys.modules["toolboxv2.mods.isaa.extras.web_helper"].__path__ = []
mod("toolboxv2.mods.isaa.extras.web_helper.tooklit", get_minimal_tools=lambda **k: (Kit(), ["T"]), WebAgentToolkit=Kit)
mod("toolboxv2.mods.isaa.base.Agent"); sys.modules["toolboxv2.mods.isaa.base.Agent"].__path__ = []
mod("toolboxv2.mods.isaa.base.Agent.executors", register_code_exec_tools=lambda a: [{"name": "exec_code"}])
sys.modules["toolboxv2.mods.isaa.extras.toolkit.isis_toolkit"] = None  # Import schlaegt fehl -> Lader faellt aus

class Agent:
    def __init__(s): s.added = []; s.removed = []
    def add_tools(s, t): s.added.append(t)
    def remove_tools(s, t): s.removed.append(t)
    def remove_tool(s, n): s.removed.append(n)
    def _get_execution_engine(s): return types.SimpleNamespace(skills_manager=SM)
class Sk:
    def __init__(s, id, name, triggers, instr, desc=""):
        s.id, s.name, s.triggers, s.instruction, s.description, s.confidence = id, name, triggers, instr, desc, 0.5
SM = types.SimpleNamespace(skills={"a": Sk("a", "Alpha", ["x", "y"], "IA"), "b": Sk("b", "Beta", [], "IB", "L" * 80 + "\nzwei")})
class B(FeatureMixin):
    def __init__(s): s.app = types.SimpleNamespace(logger=types.SimpleNamespace(warning=lambda m: log.append(m))); s.a = Agent(); s.agent = "self"; s.n = 0
    async def _agent(s, name=None): log.append("agent"); return s.a
    async def _ready(s): s.n += 1; return {"n": s.n}

async def main():
    b = B()
    assert b._features_active() == []
    fm1 = b._fm; b._features_init(); assert b._fm is fm1
    assert b._fm.list_features() == ["desktop_auto", "mini_web_auto", "full_web_auto", "execute"], b._fm.list_features()
    assert any("uebersprungen" in str(l) for l in log)
    title, rows = await b._rows_features()
    assert rows[1] == {"value": "mini_web_auto", "label": "· mini_web_auto", "hint": "aus"}, rows
    # async enable/disable
    log.clear(); orig = b._fm.set_agent
    b._fm.set_agent = lambda a: (log.append("set_agent"), orig(a))
    r = await b._do_feature("mini_web_auto")
    assert log[:2] == ["agent", "set_agent"] and r == {"ready": {"n": 1}, "text": "Feature mini_web_auto an"}, (log, r)
    assert "start" in log and b.a.added == [["T"]] and b._features_active() == ["mini_web_auto"]
    assert (await b._rows_features())[1][1]["label"] == "✓ mini_web_auto"
    r = await b._do_feature("mini_web_auto")
    assert r["text"] == "Feature mini_web_auto aus" and "stop" in log and b.a.removed == [["T"]] and b._features_active() == []
    # sync enable/disable
    await b._do_feature("execute"); assert b._features_active() == ["execute"]
    await b._do_feature("execute"); assert b.a.removed[-1] == "exec_code" and b._features_active() == []
    try: await b._do_feature("nix"); assert 0
    except ValueError: pass
    # Skills
    t, rows = await b._rows_skills()
    assert rows[0] == {"value": "a", "label": "· Alpha", "hint": "x, y"}, rows
    assert rows[1]["hint"] == "L" * 60, rows[1]
    assert b._skill_wrap("hi") == "hi"
    r = await b._do_skill("a"); assert "Alpha" in r["text"] and b._skill_context == {"a": "IA"}
    assert (await b._rows_skills())[1][0]["label"] == "✓ Alpha"
    assert b._skill_wrap("hi") == "[ACTIVE SKILL: a]\nIA\n\n---\nUser request:\nhi"
    await b._do_skill("b")
    assert b._skill_wrap("hi") == "[ACTIVE SKILL: a]\nIA\n\n[ACTIVE SKILL: b]\nIB\n\n---\nUser request:\nhi"
    await b._do_skill("a"); await b._do_skill("b"); assert b._skill_context == {} and b._skill_wrap("q") == "q"
    try: await b._do_skill("zz"); assert 0
    except ValueError: pass
asyncio.run(main())
print("features ok")
