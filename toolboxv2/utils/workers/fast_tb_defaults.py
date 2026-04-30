#!/usr/bin/env python3
"""
fast_tb_defaults.py - Auto-generated default pages for FastTB

Provides:
  - Welcome page (when no routes registered)
  - /docs route explorer (always available)
  - Minu starter page (when MinuBridge has no views)

All pages use the Paper design system (nbpaper_style).
"""

import html
import json
import time

# ============================================================================
# Paper Design System Tokens
# ============================================================================
from toolboxv2 import tb_root_dir
_PAPER_CSS = (tb_root_dir / "tbjs"/"src"/"styles"/"tbjs-paper.css").read_text(encoding="utf-8", errors="ignore")
_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'
    '&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">'
)


# ============================================================================
# Welcome Page (no routes registered)
# ============================================================================

def welcome_page(app) -> str:
    """Generate welcome page with quickstart guide."""
    title = app.title or "FastTB"
    hot = "enabled" if getattr(app, 'hot_reload', False) else "disabled"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
{_FONTS}
<style>{_PAPER_CSS}
.hero {{ text-align: center; padding: 2rem 0 1rem; }}
.hero h1 {{ font-size: 2.8rem; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }}
@media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} }}
.step {{ counter-increment: step; }}
.step::before {{
  content: counter(step);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.6rem; height: 1.6rem;
  background: var(--paper-accent);
  color: white;
  border-radius: 50%;
  font-size: 0.8rem;
  font-weight: 600;
  margin-right: 0.6rem;
  flex-shrink: 0;
}}
.step {{ display: flex; align-items: flex-start; margin-bottom: 0.6rem; }}
</style>
{app.hot_reload_script()}
</head>
<body [data-style="paper"] >
<div class="container" style="counter-reset: step;">

<div class="hero">
  <p class="subtitle" style="margin-bottom:0.5rem">Welcome to</p>
  <h1>{html.escape(title)}</h1>
  <p style="margin-top:0.5rem">
    No routes registered yet. Let's fix that.
  </p>
</div>

<h2>Quickstart</h2>

<pre><span style="color:var(--paper-text-secondary)"># your_app.py</span>
from toolboxv2 import FastTB

app = FastTB(title="{html.escape(title)}")

<span style="color:var(--primary)">@app.get</span>("/index/{{name}}")
async def index(name: str):
    return get_app().web_context() + f\"\"\"
    &lt;h1&gt;Hello {{name}}!&lt;/h1&gt;
    \"\"\" + app.hot_reload_script()

<span style="color:var(--paper-accent)">@app.get</span>("/hello/{{name}}")
async def hello(name: str):
    return {{"message": f"Hello {{name}}!"}}

<span style="color:var(--paper-accent)">@app.sse</span>("/events")
async def events():
    for i in range(10):
        yield {{"event": "tick", "data": {{"n": i}}}}
        await asyncio.sleep(1)

wsgi_app = app.get_wsgi(enable_ws=True)
</pre>

<h2>Features</h2>

<div class="grid">
  <div class="card">
    <h3>HTTP Routes</h3>
    <p style="margin:0">Decorator-based routing with automatic parameter injection from path, query, body, and session.</p>
  </div>
  <div class="card">
    <h3>WebSocket</h3>
    <p style="margin:0">Class-based WS handlers via ZMQ bridge. Channels, broadcast, auth — built in.</p>
  </div>
  <div class="card">
    <h3>SSE Streaming</h3>
    <p style="margin:0"><code>@app.sse()</code> decorator. Async generators become Server-Sent Events.</p>
  </div>
  <div class="card">
    <h3>Hot Reload</h3>
    <p style="margin:0">File watcher → WS broadcast → browser refresh. State-preserving for Minu views. Currently: <strong>{hot}</strong>.</p>
  </div>
</div>

<h2>Minu UI Framework</h2>

<p>For reactive server-rendered views with live WebSocket updates:</p>

<pre>from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import MinuView, State, Column, Text, Button

minu = MinuBridge(app)

<span style="color:var(--paper-accent)">@minu.view</span>("/counter")
class Counter(MinuView):
    count = State(0)

    def render(self):
        return Column(
            Text(f"Count: {{self.count.value}}"),
            Button("+", on_click="increment"),
        )

    async def increment(self, event):
        self.count.value += 1  <span style="color:var(--paper-text-secondary)"># pushed live via WS</span></pre>

<div class="footer">
  <p>
    <a href="/docs">/docs</a> — Route explorer &nbsp;·&nbsp;
    Hot reload: {hot} &nbsp;·&nbsp;
    <a href="https://github.com/MarkinHaus/ToolBoxV2">GitHub</a>
  </p>
</div>

</div>
{app.hot_reload_script() if hasattr(app, 'hot_reload_script') else ''}
</body></html>"""


# ============================================================================
# /docs Route Explorer
# ============================================================================

def docs_page(app) -> str:
    """Generate interactive route explorer."""
    title = app.title or "FastTB"
    routes = app.list_routes()

    route_cards = []
    for i, r in enumerate(routes):
        method = r["method"]
        path = r["path"]
        handler = r.get("handler", "")
        name = r.get("name", "")

        badge_cls = {
            "GET": "badge-get", "POST": "badge-post", "PUT": "badge-put",
            "DELETE": "badge-delete", "PATCH": "badge-patch", "WS": "badge-ws",
        }.get(method, "badge-get")

        # Detect SSE
        is_sse = "sse" in name.lower() or "sse" in path.lower()
        if is_sse:
            badge_cls = "badge-sse"
            method_label = "SSE"
        else:
            method_label = method

        # Try button for GET endpoints
        try_btn = ""
        result_box = ""
        if method == "GET" and not path.startswith("/docs"):
            try_btn = f'<button class="try-btn" onclick="tryRoute({i})">Try →</button>'
            result_box = f'<div class="result-box" id="result-{i}"></div>'
        elif method == "WS":
            try_btn = f'<button class="try-btn" onclick="tryWS({i}, \'{html.escape(path)}\')">Connect</button>'
            result_box = f'<div class="result-box" id="result-{i}"></div>'

        handler_short = handler.split(".")[-1] if "." in handler else handler

        route_cards.append(f"""
<div class="card" data-path="{html.escape(path)}" data-method="{html.escape(method)}">
  <div class="route-row">
    <span class="badge {badge_cls}">{html.escape(method_label)}</span>
    <span class="route-path">{html.escape(path)}</span>
    {try_btn}
  </div>
  <div class="route-handler">{html.escape(handler_short)}{(' · ' + html.escape(name)) if name != handler_short else ''}</div>
  {result_box}
</div>""")

    routes_json = json.dumps(routes)
    count_by_method = {}
    for r in routes:
        m = r["method"]
        count_by_method[m] = count_by_method.get(m, 0) + 1
    summary_parts = [f"{c} {m}" for m, c in sorted(count_by_method.items())]
    summary = " · ".join(summary_parts)
    filters_html = ""
    for m in sorted(count_by_method.keys()):
        filters_html += f'<button class="filter-btn" onclick="filterRoutes(\'{m}\')">{m}</button>\n  '

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — Docs</title>
{_FONTS}
<style>{_PAPER_CSS}
.filter-row {{
  display: flex; gap: 0.5rem; flex-wrap: wrap;
  margin-bottom: 1.2rem;
}}
.filter-btn {{
  font-family: var(--paper-mono);
  font-size: 0.75rem;
  padding: 0.3em 0.7em;
  border: 1px solid var(--paper-border);
  border-radius: 4px;
  background: var(--paper-bg);
  color: var(--paper-text-secondary);
  cursor: pointer;
  transition: all 0.15s;
}}
.filter-btn:hover, .filter-btn.active {{
  background: var(--paper-accent);
  color: white;
  border-color: var(--paper-accent);
}}
.ws-input {{
  font-family: var(--paper-mono);
  font-size: 0.8rem;
  padding: 0.4em 0.7em;
  border: 1px solid var(--paper-border);
  border-radius: 4px;
  background: var(--paper-surface);
  color: var(--paper-text);
  width: 100%;
  margin-top: 0.4rem;
}}
</style>
</head>
<body>
<div class="container">

<h1>{html.escape(title)}</h1>
<p class="subtitle">Route Explorer — {summary}</p>

<div class="filter-row">
  <button class="filter-btn active" onclick="filterRoutes('all')">All</button>
  {filters_html}
</div>

{"".join(route_cards)}

<div class="footer">
  <p><a href="/">← Home</a> &nbsp;·&nbsp; {len(routes)} routes &nbsp;·&nbsp; {summary}</p>
</div>

</div>

<script>
var ROUTES = {routes_json};
var wsPort = window.__TB_WS_PORT__ || '8100';

function filterRoutes(method) {{
  document.querySelectorAll('.filter-btn').forEach(function(b) {{
    b.classList.toggle('active', b.textContent === method || method === 'all' && b.textContent === 'All');
  }});
  document.querySelectorAll('.card[data-method]').forEach(function(c) {{
    c.style.display = (method === 'all' || c.dataset.method === method) ? '' : 'none';
  }});
}}

function tryRoute(i) {{
  var r = ROUTES[i];
  var box = document.getElementById('result-' + i);
  box.style.display = 'block';
  box.textContent = 'Loading...';

  if (r.path.indexOf('/sse/') !== -1 || r.name && r.name.indexOf('sse') !== -1) {{
    // SSE
    var es = new EventSource(r.path);
    box.textContent = 'Connected (SSE)...\\n';
    es.onmessage = function(e) {{ box.textContent += e.data + '\\n'; box.scrollTop = box.scrollHeight; }};
    es.addEventListener('stream_end', function() {{ es.close(); box.textContent += '--- stream ended ---\\n'; }});
    es.onerror = function() {{ box.textContent += '--- error/closed ---\\n'; es.close(); }};
    setTimeout(function() {{ es.close(); }}, 60000);
  }} else {{
    fetch(r.path, {{ headers: {{ 'Accept': 'application/json' }} }})
      .then(function(res) {{ return res.text(); }})
      .then(function(text) {{
        try {{ box.textContent = JSON.stringify(JSON.parse(text), null, 2); }}
        catch(e) {{ box.textContent = text.substring(0, 2000); }}
      }})
      .catch(function(e) {{ box.textContent = 'Error: ' + e; }});
  }}
}}

function tryWS(i, path) {{
  var box = document.getElementById('result-' + i);
  box.style.display = 'block';
  box.textContent = 'Connecting...\\n';

  var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var ws = new WebSocket(proto + '//' + location.hostname + ':' + wsPort + path);

  ws.onopen = function() {{
    box.textContent += 'Connected!\\n';
    box.innerHTML += '<input class="ws-input" placeholder="Type JSON message + Enter" onkeydown="if(event.key===\\'Enter\\')sendWS(this,' + i + ')">';
    window['_ws_' + i] = ws;
  }};
  ws.onmessage = function(e) {{
    box.textContent += '← ' + e.data + '\\n';
    box.scrollTop = box.scrollHeight;
  }};
  ws.onclose = function() {{ box.textContent += '--- disconnected ---\\n'; }};
  ws.onerror = function() {{ box.textContent += '--- error ---\\n'; }};
}}

function sendWS(input, i) {{
  var ws = window['_ws_' + i];
  if (ws && ws.readyState === 1) {{
    var box = document.getElementById('result-' + i);
    box.textContent += '→ ' + input.value + '\\n';
    ws.send(input.value);
    input.value = '';
  }}
}}
</script>
{app.hot_reload_script() if hasattr(app, 'hot_reload_script') else ''}
</body></html>"""


# ============================================================================
# Minu Starter Page (no views registered)
# ============================================================================

def minu_welcome_page(app, bridge) -> str:
    """Generate Minu starter page with quickstart guide."""
    title = app.title or "FastTB + Minu"
    hot = "enabled" if getattr(app, 'hot_reload', False) else "disabled"
    ws_public = getattr(bridge, 'ws_public', True)
    public = getattr(bridge, 'public', True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
{_FONTS}
<style>{_PAPER_CSS}
.hero {{ text-align: center; padding: 2rem 0 1rem; }}
.hero h1 {{ font-size: 2.8rem; }}
.feature-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 1.5rem; }}
@media (max-width: 600px) {{ .feature-grid {{ grid-template-columns: 1fr; }} }}
.tag {{
  display: inline-block;
  font-family: var(--paper-mono);
  font-size: 0.65rem;
  padding: 0.1em 0.4em;
  border-radius: 3px;
  background: var(--paper-border);
  color: var(--paper-text-secondary);
  margin-left: 0.3rem;
  vertical-align: middle;
}}
</style>
</head>
<body>
<div class="container">

<div class="hero">
  <p class="subtitle" style="margin-bottom:0.5rem">Welcome to</p>
  <h1>{html.escape(title)}</h1>
  <p style="margin-top:0.5rem">
    Reactive server-rendered UI with live WebSocket updates.
    <br>No views registered yet — here's how to get started.
  </p>
</div>

<h2>Your First View</h2>

<pre>from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import (
    MinuView, State, Column, Row, Heading, Text, Button, Card
)

minu = MinuBridge(app)

<span style="color:var(--paper-accent)">@minu.view</span>("/counter", icon="add_circle", label="Counter")
class CounterView(MinuView):
    count = State(0)

    def render(self):
        return Card(
            Heading("Counter"),
            Text(f"Count: {{self.count.value}}"),
            Row(
                Button("-", on_click="decrement"),
                Button("+", on_click="increment"),
            ),
        )

    async def increment(self, event):
        self.count.value += 1

    async def decrement(self, event):
        self.count.value -= 1</pre>

<p>This generates three endpoints automatically:</p>
<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem">
  <code>GET /counter</code>
  <code>POST /counter/event</code>
  <code>WS /ws/openCounter</code>
</div>

<h2>How It Works</h2>

<div class="feature-grid">
  <div class="card">
    <h3>Reactive State</h3>
    <p style="margin:0"><code>State(0)</code> creates a reactive value. Changing <code>.value</code> automatically pushes patches to the browser via WebSocket.</p>
  </div>
  <div class="card">
    <h3>Live Push</h3>
    <p style="margin:0">State changes during <code>async</code> handlers (even with <code>await sleep</code>) are pushed immediately — no polling.</p>
  </div>
  <div class="card">
    <h3>HTTP Fallback</h3>
    <p style="margin:0">If WebSocket fails, events fall back to HTTP POST. Same API, same code, automatic switching.</p>
  </div>
  <div class="card">
    <h3>Error Hints</h3>
    <p style="margin:0">Typos in handler code → toast with suggestion. Type mismatches → clear hint. Full traceback in console.</p>
  </div>
</div>

<h2>Access Control</h2>

<pre>minu.public = {public}       <span style="color:var(--paper-text-secondary)"># Views public by default</span>
minu.ws_public = {ws_public}    <span style="color:var(--paper-text-secondary)"># WS open without auth</span>

<span style="color:var(--paper-accent)">@minu.view</span>("/admin", require_auth=True)         <span style="color:var(--paper-text-secondary)"># HTTP + WS auth</span>
class Admin(MinuView): ...

<span style="color:var(--paper-accent)">@minu.view</span>("/monitor", ws_public=False)          <span style="color:var(--paper-text-secondary)"># HTTP public, WS auth</span>
class Monitor(MinuView): ...</pre>

<h2>Components</h2>

<div class="feature-grid">
  <div class="card">
    <h3>Layout</h3>
    <p style="margin:0"><code>Card</code> <code>Row</code> <code>Column</code> <code>Grid</code> <code>Spacer</code> <code>Divider</code></p>
  </div>
  <div class="card">
    <h3>Content</h3>
    <p style="margin:0"><code>Text</code> <code>Heading</code> <code>Badge</code> <code>Icon</code> <code>Image</code> <code>Markdown</code></p>
  </div>
  <div class="card">
    <h3>Input</h3>
    <p style="margin:0"><code>Button</code> <code>Input</code> <code>Textarea</code> <code>Select</code> <code>Checkbox</code> <code>Switch</code> <code>Slider</code></p>
  </div>
  <div class="card">
    <h3>Feedback</h3>
    <p style="margin:0"><code>Alert</code> <code>Toast</code> <code>Progress</code> <code>Spinner</code> <code>Modal</code></p>
  </div>
</div>

<div class="footer">
  <p>
    <a href="/docs">/docs</a> — Route explorer &nbsp;·&nbsp;
    Hot reload: {hot} &nbsp;·&nbsp;
    Access: {"public" if public else "private"}
    <span class="tag">WS {"open" if ws_public else "auth"}</span>
  </p>
</div>

</div>
{app.hot_reload_script() if hasattr(app, 'hot_reload_script') else ''}
</body></html>"""


# ============================================================================
# Registration
# ============================================================================

def register_defaults(app):
    """Register /docs and optional welcome page on a FastTB instance.

    Called automatically by FastTBHandler.as_wsgi_app() if not already present.
    """
    # Always register /docs
    if not app.has_route("/docs", "GET"):
        @app.get("/docs", name="fasttb_docs")
        async def _docs():
            return docs_page(app)

    # Welcome page only if no user routes at /
    if not app.has_route("/", "GET"):
        @app.get("/", name="fasttb_welcome")
        async def _welcome():
            return welcome_page(app)


def register_minu_defaults(app, bridge):
    """Register Minu starter welcome page if no views registered.

    Called by MinuBridge after all decorators have run (lazy, on first request).
    """
    if not app.has_route("/", "GET"):
        @app.get("/", name="minu_welcome")
        async def _minu_welcome():
            # Check if views have been registered since startup
            if bridge._view_registry:
                # Views exist now — redirect to first view
                first_path = next(iter(bridge._view_registry))
                return (302, {"Location": first_path, "Content-Type": "text/plain"}, b"")
            return minu_welcome_page(app, bridge)
