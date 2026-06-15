# file: toolboxv2/mods/llama_lab/webapp/__init__.py
"""llama_lab web app (FastTB).

Mount inside HTTPWorker (production):
    from toolboxv2.mods.llama_lab.webapp import app as fast_tb_app
    worker.run(fast_tb_app=fast_tb_app)

Standalone (local):
    from toolboxv2.mods.llama_lab.webapp import run
    run(host="0.0.0.0", port=8000)
"""

from .app import app                     # FastTB instance (ASGI-capable)

__all__ = ["app", "handler", "wsgi_app", "run"]

try:                                     # standalone WSGI (Mode C)
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    handler = FastTBHandler(app)
    wsgi_app = handler.as_wsgi_app(enable_ws=True)
except Exception:                        # framework not present at import time
    handler = wsgi_app = None


def run(host: str = "0.0.0.0", port: int = 8000):
    from waitress import serve
    serve(wsgi_app, host=host, port=port)
