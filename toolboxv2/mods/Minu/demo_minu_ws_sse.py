#!/usr/bin/env python3
"""
demo_minu_ws_sse.py - MinuBridge with WebSocket + SSE Demo

Shows:
  - MinuView with live WS updates (via MinuRenderer.mount())
  - SSE streaming endpoint alongside Minu views
  - Auth-gated view
  - Event handling via WS (button clicks, form submits)

The MinuBridge HTML output uses mount() which opens a WebSocket
to /ws/Minu/ui — the standard Minu WS path. State changes from
button clicks are sent over WS and patched live in the browser.

Run standalone:
    uvicorn demo_minu_ws_sse:wsgi_app --port 8000

Run inside HTTPWorker:
    worker.run(fast_tb_app=app)
"""

import asyncio
import time

from toolboxv2 import Result, get_app
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.workers.session import SessionData
from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import (
    MinuView, State, Column, Row, Heading, Text,
    Button, Input, Card, Badge, Divider, Alert, Progress, Dynamic
)

# =============================================================================
# Setup
# =============================================================================

app = FastTB(title="Minu WS+SSE Demo")
minu = MinuBridge(app)
minu.public = True       # All views public by default
minu.ws_public = True    # WS open for all by default
# =============================================================================
# Minu View — Live counter with WS updates
# =============================================================================

@minu.view("/open_live")
class LiveCounter(MinuView):
    """Counter with live WS updates — buttons trigger events over WebSocket."""
    count = State(0)
    status = State("idle")

    def live(self):
        return Heading(str(self.count.value), level=2)

    def render(self):
        dynamic = Dynamic(self.live, [self.count])
        self.register_dynamic(dynamic)
        return Card(
            Heading("Live counter"),
            Text(f"Status: {self.status.value}"),
            Row(
                Button("-", on_click="decrement", variant="secondary"),
                dynamic,
                Button("+", on_click="increment", variant="primary"),
            )
            ,
            Row(
                Button("Reset", on_click="reset", variant="ghost"),
                Button("Auto +10", on_click="auto_increment", variant="secondary"),
            ),
            title="WebSocket-powered",
            subtitle="State changes are pushed live via WS",
        )

    async def increment(self, event):
        self.count.value += 1
        self.status.value = "incremented"

    async def decrement(self, event):
        self.count.value -= 1
        self.status.value = "decremented"

    async def reset(self, event):
        self.counte.value = 0
        self.status.value = "reset"

    async def auto_increment(self, event):
        """Increment 10 times with 200ms delay — demonstrates live WS push."""
        self.status.value = "auto-incrementing..."
        for _ in range(10):
            self.count.value += "1"
            await asyncio.sleep(0.2)
        self.status.value = "done"


# =============================================================================
# Minu View — Auth-gated dashboard
# =============================================================================

@minu.view("/dashboard", require_auth=True)
class Dashboard(MinuView):
    def render(self):
        return Card(
            Heading(f"Welcome, {self.user.name}"),
            Text(f"User ID: {self.user.uid}"),
            Text(f"Level: {self.user.level}"),
            Badge("authenticated", variant="success"),
        )


# =============================================================================
# Minu View — Progress tracker with live updates
# =============================================================================

@minu.view("/progress")
class ProgressView(MinuView):
    percent = State(0)
    running = State(False)

    def render(self):
        return Card(
            Heading("Progress tracker"),
            Progress(
                value=self.percent.value,
                label="Processing",
            ),
            Text(f"{self.percent.value}%"),
            Row(
                Button(
                    "Start" if not self.running.value else "Running...",
                    on_click="start_task",
                    disabled=self.running.value,
                ),
                Button("Reset", on_click="reset", variant="ghost"),
            ),
        )

    async def start_task(self, event):
        self.running.value = True
        self.percent.value = 0
        for i in range(1, 101):
            self.percent.value = i
            await asyncio.sleep(0.05)
        self.running.value = False

    async def reset(self, event):
        self.percent.value = 0
        self.running.value = False


# =============================================================================
# SSE Endpoint — alongside Minu views
# =============================================================================

@app.get("/sse/events")
async def sse_events(request: ParsedRequest):
    """SSE stream of server events — works alongside Minu views."""
    async def event_generator():
        for i in range(20):
            yield {
                "event": "server_event",
                "data": {
                    "tick": i,
                    "time": time.strftime("%H:%M:%S"),
                    "message": f"Event #{i}",
                }
            }
            await asyncio.sleep(1)

    return Result.sse(stream_generator=event_generator())


# =============================================================================
# Index page
# =============================================================================

@app.get("/")
async def index():
    views = minu.list_views()
    links = "".join(
        f'<li><a href="{v["path"]}">{v["view"]}</a></li>'
        for v in views
    )
    return f"""<!DOCTYPE html>
<html><head><title>Minu WS+SSE Demo</title></head>
<body style="font-family:system-ui;max-width:640px;margin:2rem auto;padding:0 1rem">
<h1>MinuBridge WS + SSE Demo</h1>
<h2>Minu views (live WS)</h2>
<ul>{links}</ul>
<h2>SSE endpoint</h2>
<ul><li><a href="/sse/events">/sse/events</a> (EventSource)</li></ul>
<h2>API</h2>
<ul><li><a href="/routes">/routes</a></li></ul>
</body></html>"""


@app.get("/routes")
def routes():
    return app.list_routes()


# =============================================================================
# Standalone entry
# =============================================================================

handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app(enable_ws=True)

if __name__ == "__main__":
    print("\nRoutes:")
    for r in app.list_routes():
        print(f"  {r['method'].ljust(6)} {r['path']}")
    print("\nRun: uvicorn demo_minu_ws_sse:wsgi_app --port 8000\n")

    from waitress import serve

    serve(wsgi_app, host="127.0.0.1", port=8000)
