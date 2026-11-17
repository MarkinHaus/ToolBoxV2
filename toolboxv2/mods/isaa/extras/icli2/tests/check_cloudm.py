"""Pruefung von cloudm.py mit Fake-App und Fake-CloudM-Modulen."""
import asyncio, sys, types, json, tempfile
from pathlib import Path
import stub
from toolboxv2.mods.isaa.extras.icli2.cloudm import CloudMMixin

def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

tmp = Path(tempfile.mkdtemp()); (tmp / "icli").mkdir()
state = {"found": None, "boom": False, "logout": 0}
def find(app, username=None):
    if state["boom"]: raise RuntimeError("x")
    return state["found"]
async def cli_logout(app):
    state["logout"] += 1; print("laute Ausgabe"); app.session.valid = False; state["found"] = None
class Mgr:
    def __init__(s, vfs): s.a = {}
    async def connect(s, vp, ld, token=None, readonly=False): s.a[vp] = (ld, token, readonly)
    async def disconnect(s, vp): return s.a.pop(vp, None) is not None
    def status(s): return {p: {"status": "live"} for p in s.a}
for n in ("toolboxv2.mods.CloudM", "toolboxv2.mods.CloudM.LiveSync"): mod(n).__path__ = []
mod("toolboxv2.mods.CloudM.LogInSystem", _find_cli_session=find, cli_logout=cli_logout)
mod("toolboxv2.mods.CloudM.LiveSync.vfs_adapter", VFSSyncManager=Mgr)

class Res:
    def __init__(s, d, err=False): s.d, s.err, s.info = d, err, None
    def is_error(s): return s.err
    def get(s): return s.d
class App:
    def __init__(s):
        s.session = types.SimpleNamespace(valid=False, username=None); s.appdata = str(tmp); s.calls = []
        s.functions = {"Mod": {"fn": {"func": print, "params": ["app", "a", "b"], "helper": "Macht was\nzweite"},
                               "plain": {"func": print, "params": []}, "on_exit": [1]}, "Leer": None}
    async def a_run_any(s, name, get_results=False, **kw):
        s.calls.append((name, kw, get_results)); return Res({"ok": kw}) if name[1] != "bad" else Res(None, True)
class Sess:
    vfs = object()
class B(CloudMMixin):
    def __init__(s): s.app = App(); s.s = Sess(); s.n = 0
    async def _session(s): return s.s
    async def _ready(s): s.n += 1; return {"mounts": s.n}

async def main():
    b = B()
    assert await b._cloud_user() == ""
    r = await b._do_login(); assert r["level"] == "warn" and "tb login" in r["text"], r
    b.app.session.valid, b.app.session.username = True, "anna"
    assert await b._cloud_user() == "anna"
    assert (await b._do_login()) == {"text": "Angemeldet als anna"}
    class Boom:  # Session wirft beim Lesen
        @property
        def valid(s): raise RuntimeError("x")
    b.app.session = Boom(); state["boom"] = True
    assert await b._cloud_user() == ""
    state["boom"] = False; state["found"] = {"username": "bob"}
    assert await b._cloud_user() == "bob"
    b.app.session = types.SimpleNamespace(valid=True, username="bob")
    r = await b._do_logout(); assert state["logout"] == 1 and "bob" in r["text"], r
    assert (await b._do_logout())["level"] == "warn"

    title, rows = await b._rows_tb(); print(rows)
    assert [x["value"] for x in rows] == ["Mod.fn", "Mod.plain"]
    assert rows[0]["hint"] == "Macht was  (a, b)" and rows[1]["hint"] == "" and set(rows[0]) == {"value", "label", "hint"}
    r = await b._do_tb("Mod.fn", 'a=1 b="x y" c=true d=[1,2]')
    assert b.app.calls[-1][0] == ("Mod", "fn") and b.app.calls[-1][1] == {"a": 1, "b": "x y", "c": True, "d": [1, 2]}, b.app.calls
    assert "ok" in r["text"], r
    for bad in (("nodot", ""), ("Mod.fn", "kaputt")):
        try: await b._do_tb(*bad); assert 0
        except ValueError: pass
    try: await b._do_tb("Mod.bad"); assert 0
    except RuntimeError as e: assert "fehlgeschlagen" in str(e)

    assert (await b._rows_shares())[1] == []
    (tmp / "icli" / "vfs_shares.json").write_text(json.dumps([{"vfs_path": "/p", "local_dir": "/d", "token": "t", "readonly": True}]))
    rows = (await b._rows_shares())[1]; assert rows[0]["value"] == "/p" and "getrennt" in rows[0]["hint"], rows
    r = await b._do_share("/p"); assert r["ready"] == {"mounts": 1} and "verbindet" in r["text"], r
    assert b.s._vfs_sync_mgr.a["/p"] == ("/d", "t", True)
    assert "verbunden" in (await b._rows_shares())[1][0]["hint"]
    r = await b._do_share("/p"); assert r["ready"] == {"mounts": 2} and "getrennt" in r["text"] and not b.s._vfs_sync_mgr.a
    try: await b._do_share("/nix"); assert 0
    except RuntimeError: pass
asyncio.run(main())
print("cloudm ok")
