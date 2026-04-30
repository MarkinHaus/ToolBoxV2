#!/usr/bin/env python3
"""
demo_fasttb_ws_sse.py - FastTB WebSocket + SSE Demo (no Minu)

Shows:
  - SSE streaming via Result.stream / Result.sse
  - WebSocket handler via @app.websocket
  - Static file mounting
  - Both standalone (uvicorn) and HTTPWorker integration

Run standalone:
    uvicorn demo_fasttb_ws_sse:wsgi_app --port 8000

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

app = FastTB(title="FastTB WS+SSE Demo 2")

# =============================================================================
# SSE Endpoint — streams time ticks
# =============================================================================

@app.sse("/sse/clock")
async def sse_clock(request: ParsedRequest):
    """SSE endpoint that streams the current time every second.

    Usage:
        const es = new EventSource('/sse/clock');
        es.addEventListener('tick', e => console.log(JSON.parse(e.data)));
    """
    for i in range(30):
        yield {
            "event": "tick",
            "data": {"time": time.strftime("%H:%M:%S"), "tick": i}
        }
        await asyncio.sleep(1)


@app.sse("/sse/count")
async def sse_count(request: ParsedRequest, max: int = 10):
    """SSE endpoint that counts to max.

    Usage:
        const es = new EventSource('/sse/count?max=5');
        es.onmessage = e => console.log(e.data);
    """
    for i in range(1, max + 1):
        yield {"event": "count", "data": {"n": i, "of": max}}
        await asyncio.sleep(0.5)


# =============================================================================
# WebSocket Handler — echo + broadcast
# =============================================================================

_ws_connections: dict = {}  # conn_id -> channel

@app.websocket("/ws/openEcho")
class EchoHandler:
    """Simple echo WebSocket — sends back whatever it receives."""

    async def on_connect(self, conn_id, session):
        _ws_connections[conn_id] = "echo"
        return {"type": "connected", "conn_id": conn_id}

    async def on_message(self, payload, conn_id, session, request=None):
        msg = payload.get("message", "")
        return {
            "type": "echo",
            "message": msg,
            "from": conn_id,
            "timestamp": time.time(),
        }

    async def on_disconnect(self, conn_id, session):
        _ws_connections.pop(conn_id, None)


@app.websocket("/ws/openChat/{room}")
class ChatHandler:
    """Chat WebSocket with room support (path param)."""

    async def on_connect(self, conn_id, session):
        return {"type": "joined", "conn_id": conn_id}

    async def on_message(self, payload, conn_id, session, request=None):
        # In production, use app.ws_broadcast() to send to all in room
        return {
            "type": "chat_message",
            "from": conn_id,
            "text": payload.get("text", ""),
            "timestamp": time.time(),
        }

    async def on_disconnect(self, conn_id, session):
        pass


# =============================================================================
# Regular endpoints
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "features": ["sse", "websocket"]}


@app.get("/")
async def index():
    return """<!DOCTYPE html>
<html><head><title>FastTB WS+SSE Demo</title></head>
<body data-style="paper" style="font-family:system-ui;max-width:700px;margin:2rem auto;padding:0 1rem">
<h1>FastTB WebSocket + SSE Demo</h1>

<h2>SSE Clock <small style="font-weight:normal;color:#64748b">(/sse/clock)</small></h2>
<pre id="clock" style="background:#f1f5f9;padding:1rem;border-radius:8px;max-height:200px;overflow:auto"></pre>
<button onclick="startClock()">Start SSE Clock</button>
<button onclick="stopClock()">Stop</button>

<h2>WebSocket Echo</h2>
<input id="msg" placeholder="Type a message" style="padding:.5rem;width:60%">
<button onclick="sendMsg()">Send</button>
<pre id="ws-log" style="background:#f1f5f9;padding:1rem;border-radius:8px;max-height:200px;overflow:auto"></pre>

""" + app.hot_reload_script() + """
<script>
let es, ws;
const clockEl = document.getElementById('clock');
const wsLog = document.getElementById('ws-log');

function startClock() {
    if (es) es.close();
    clockEl.textContent = '';
    es = new EventSource('/sse/clock');
    es.addEventListener('tick', e => {
        console.log('tick', e);
        clockEl.textContent = "HALLo " +JSON.parse(e.data).time + '\\n';
        clockEl.scrollTop = clockEl.scrollHeight;
    });
    es.addEventListener('stream_end', () => { es.close(); clockEl.textContent += '--- stream ended ---\\n'; });
}
function stopClock() { if (es) es.close(); }

// WebSocket
// WebSocket — WS worker runs on separate port (default 8100)
const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsPort = window.__TB_WS_PORT__ || '8100';
ws = new WebSocket(proto + '//' + location.hostname + ':' + wsPort + '/ws/openEcho');
ws.onmessage = e => {
    const d = JSON.parse(e.data);
    wsLog.textContent += d.type + ': ' + (d.message || JSON.stringify(d)) + '\\n';
    wsLog.scrollTop = wsLog.scrollHeight;
};
ws.onopen = () => { wsLog.textContent += '--- connected ---\\n'; };
ws.onclose = () => { wsLog.textContent += '--- disconnected ---\\n'; };

function sendMsg() {
    const input = document.getElementById('msg');
    if (ws.readyState === 1) {
        ws.send(JSON.stringify({message: input.value}));
        input.value = '';
    }
}
document.getElementById('msg').addEventListener('keydown', e => { if (e.key === 'Enter') sendMsg(); });
</script>
</body></html>"""


# =============================================================================
# Standalone entry — reuses HTTPWorker infrastructure
# =============================================================================

handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app(enable_ws=True)

if __name__ == "__main__":
    print("\nRoutes:")
    for r in app.list_routes():
        print(f"  {r['method'].ljust(6)} {r['path']}")
    print("\nRun: uvicorn demo_fasttb_ws_sse:wsgi_app --port 8000\n")

    from waitress import serve

    serve(wsgi_app, host="127.0.0.1", port=8000)
