"""Plugin-Installation gegen den echten isaa-Agenten. Vorher ein lokales Plugin-Repo anlegen:
    mkdir -p /tmp/e2e/plug/demo-plugin/.claude-plugin /tmp/e2e/plug/demo-plugin/skills/greet && cd /tmp/e2e/plug/demo-plugin
    echo '{"name": "demo-plugin"}' > .claude-plugin/plugin.json
    printf -- '---\nname: greet\ndescription: Begruessung\n---\nBegruesse freundlich.\n' > skills/greet/SKILL.md
    git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init
"""
import asyncio, os, traceback
os.environ.update(OPENAI_API_BASE="http://127.0.0.1:8765/v1", OPENAI_API_KEY="x", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                  FASTMODEL="openai/fake", COMPLEXMODEL="openai/fake", BLITZMODEL="openai/fake", SUMMARYMODEL="openai/fake")
async def main():
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    app = get_app("isaa-host"); b = IsaaBackend(app)
    async def spawn(*a, **k): return ""
    b.attach(spawn); await b.boot()
    sm = (await b._agent())._get_execution_engine().skills_manager
    sid = "plugin_demo-plugin_greet"
    steps = (("install", lambda: b.do("plugin_install", source="/tmp/e2e/plug/demo-plugin")),
             ("rows", lambda: b.rows("plugins")),
             ("skill registriert", lambda: asyncio.sleep(0, sid in sm.skills)),
             ("deaktivieren", lambda: b.do("plugin", name="demo-plugin")),
             ("skill nach deaktivieren weg", lambda: asyncio.sleep(0, sid not in sm.skills)),
             ("aktivieren", lambda: b.do("plugin", name="demo-plugin")),
             ("skill wieder da", lambda: asyncio.sleep(0, sid in sm.skills)),
             ("remove", lambda: b.do("plugin_remove", name="demo-plugin")),
             ("skill nach remove weg", lambda: asyncio.sleep(0, sid not in sm.skills)),
             ("ordner weg", lambda: asyncio.sleep(0, not (b._plugins_dir() / "demo-plugin").exists())),
             ("market leer", lambda: b.rows("market")))
    for label, make in steps:                            # erst hier ausfuehren, nicht beim Aufbau
        try:
            out = await asyncio.wait_for(make(), 120)
            if isinstance(out, dict) and "rows" in out:
                out = {"title": out["title"], "rows": [(r["label"], r["hint"]) for r in out["rows"]][:3]}
            print(f"PLUG {label}: {out}")
        except Exception as e:
            print(f"PLUG {label}: FEHLER {type(e).__name__}: {e}")
    await b.close(); await app.a_exit()
try: asyncio.run(main())
except BaseException: traceback.print_exc()
