import asyncio, os, traceback
os.environ.update(OPENAI_API_BASE="http://127.0.0.1:8765/v1", OPENAI_API_KEY="x", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                  FASTMODEL="openai/fake", COMPLEXMODEL="openai/fake", BLITZMODEL="openai/fake", SUMMARYMODEL="openai/fake")
async def main():
    from toolboxv2 import get_app
    from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend
    app = get_app("isaa-host"); b = IsaaBackend(app)
    async def spawn(*a, **k): return ""
    b.attach(spawn); r = await b.boot(); print("BOOT", {k: (v if k != "history" else len(v)) for k, v in r.items()})
    for what in ("agents", "models", "sessions", "mounts", "jobs", "features", "skills", "mcp", "plugins", "tb", "shares", "voice", "discord", "telegram"):
        try:
            out = await b.rows(what); rows = out["rows"]
            bad = [x for x in rows if not all(isinstance(x.get(k), str) for k in ("value", "label", "hint"))]
            print(f"ROWS {what}: {len(rows)} | {out['title']!r} | bad={bad[:2]} | {[(x['label'], x['hint'][:40]) for x in rows[:2]]}")
        except Exception as e:
            print(f"ROWS {what}: FEHLER {type(e).__name__}: {str(e)[:200]}")
    for action, kw in (("login", {}), ("feature", {"name": "execute"}), ("feature", {"name": "execute"}), ("session", {"name": "+"}), ("mount", {"path": "/tmp/e2e"}),
                       ("job_add", {"spec": "alle 30m test: sag hallo"}), ("dream", {}), ("tb", {"target": "isaa.version"}), ("pack", {}), ("network_save", {}), ("session_clear", {}),
                       ("model", {"name": "openai/fake", "target": "both"}), ("discord", {"args": "status"}), ("telegram", {"args": "status"})):
        try:
            out = await asyncio.wait_for(b.do(action, **kw), 60)
            print(f"DO {action}: text={str(out.get('text'))[:150]!r} ready={sorted(out.get('ready', {}))[:3]}")
        except Exception as e:
            print(f"DO {action}: FEHLER {type(e).__name__}: {str(e)[:220]}")
    await b.close(); await app.a_exit()
try: asyncio.run(main())
except BaseException: traceback.print_exc()
