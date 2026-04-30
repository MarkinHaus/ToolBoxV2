#!/usr/bin/env python3
"""
demo_fasttb_defaults.py - FastTB Default Pages Demo

Shows the auto-generated welcome page and /docs route explorer.
Add a few routes so /docs has something to show.

Run:
    python demo_fasttb_defaults.py

Then browse:
    http://localhost:8000        → Welcome page (auto-generated)
    http://localhost:8000/docs   → Route explorer with Try buttons
"""

import asyncio
import time

from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
from toolboxv2.utils.workers.server_worker import ParsedRequest

app = FastTB(title="My API")

# No "/" route registered → FastTB shows welcome page automatically


@app.get("/hello/{name}")
async def hello(name: str):
    """Greet someone by name."""
    return {"message": f"Hello {name}!", "timestamp": time.time()}


@app.get("/health")
async def health():
    return {"status": "ok", "uptime": time.time()}


@app.post("/echo")
async def echo(request: ParsedRequest):
    """Echo back the JSON body."""
    return {"received": request.json_data}


@app.sse("/sse/ticks")
async def ticks(request: ParsedRequest):
    """Stream 5 ticks, one per second."""
    for i in range(5):
        yield {"event": "tick", "data": {"n": i, "time": time.strftime("%H:%M:%S")}}
        await asyncio.sleep(1)


@app.websocket("/ws/openPing")
class PingHandler:
    async def on_connect(self, conn_id, session):
        return {"type": "connected", "conn_id": conn_id}

    async def on_message(self, payload, conn_id, session, request=None):
        return {"type": "pong", "echo": payload, "server_time": time.time()}

    async def on_disconnect(self, conn_id, session):
        pass


# =============================================================================
# Start
# =============================================================================

handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app(enable_ws=True)

if __name__ == "__main__":
    print("\n  FastTB Defaults Demo")
    print("  ====================")
    print(f"  Routes: {len(app.list_routes())}")
    print(f"  Hot reload: {app.hot_reload}")
    print()
    for r in app.list_routes():
        print(f"    {r['method'].ljust(6)} {r['path']}")
    print()
    print("  http://localhost:8000      → Welcome")
    print("  http://localhost:8000/docs → Route Explorer")
    print()

    from waitress import serve
    serve(wsgi_app, host="127.0.0.1", port=8000)
