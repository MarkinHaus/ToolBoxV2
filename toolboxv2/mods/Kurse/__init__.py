# file: toolboxv2/mods/Kurse/__init__.py
"""Kurse — Open-Mod: Aufgaben-Seiten für Coach & Teilnehmer.

Production (mount into HTTPWorker behind nginx):
    from toolboxv2.mods.Kurse import app as fast_tb_app
    worker.run(fast_tb_app=fast_tb_app)

Standalone / dev:
    tb -c Kurse web --port 8080
"""

from toolboxv2 import get_app, Style

from .app import app                      # FastTB instance (ASGI-capable)

Name = "Kurse"
version = "1.0.0"
tb = get_app(Name).tb

try:                                      # standalone WSGI (waitress)
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    handler = FastTBHandler(app)
    wsgi_app = handler.as_wsgi_app(enable_ws=True)
except Exception:
    handler = wsgi_app = None

__all__ = ["app", "handler", "wsgi_app", "run", "web"]


def run(host: str = "0.0.0.0", port: int = 5010):
    from waitress import serve
    app.serve(host=host, port=port, blocking=True)
    # serve(wsgi_app, host=host, port=port)


@tb(name="web", mod_name=Name, version=version, helper="Run the Kurse web app")
async def web(app, host: str = "0.0.0.0", port: int = 5010):
    import asyncio
    print(Style.GREEN(f"  Kurse on http://{host}:{port}  (coach: /  ·  learner: /l/<cohort>)"))
    await asyncio.to_thread(run, host, port)
    return {"served": f"http://{host}:{port}"}
