#!/usr/bin/env python3
"""
fast_tb_defaults.py - Auto-generated default pages for FastTB

Provides:
  - Welcome page (when no user route on /)
  - /docs route explorer (always available)
  - Minu starter page (when MinuBridge has no views)

All pages use the Paper design system (tbjs-paper.css).
"""

import html
import json

# ============================================================================
# Paper Design System — load CSS files
# ============================================================================
from toolboxv2 import tb_root_dir

_MAIN_CSS = (tb_root_dir / "tbjs" / "src" / "styles" / "tbjs-main.css").read_text(encoding="utf-8", errors="ignore")
_PAPER_CSS = (tb_root_dir / "tbjs" / "src" / "styles" / "tbjs-paper.css").read_text(encoding="utf-8", errors="ignore")
_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600'
    '&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">'
)

# ============================================================================
# Shared inline CSS — uses real tbjs-paper.css tokens only
# ============================================================================
_SHARED_CSS = """
/* --- Layout --- */
.ftb-wrap {
  max-width: 52rem;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 3rem;
}
@media (max-width: 600px) { .ftb-wrap { padding: 1.5rem 1rem 2rem; } }

/* --- Hero --- */
.ftb-hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.ftb-hero h1 {
  font-family: var(--font-display);
  font-size: 2.4rem;
  color: var(--text-main);
  margin: 0;
}
.ftb-hero .ftb-sub {
  font-family: var(--font-body);
  color: var(--text-muted);
  margin-top: 0.4rem;
  font-size: 0.95rem;
}

/* --- Section headings --- */
.ftb-wrap h2 {
  font-family: var(--font-display);
  font-size: 1.15rem;
  color: var(--text-main);
  border-bottom: var(--border-width) solid var(--ink);
  padding-bottom: 0.3rem;
  margin: 2rem 0 1rem;
}

/* --- Cards --- */
.ftb-card {
  background: var(--bg-surface);
  border: var(--border-width) solid var(--ink);
  padding: 1rem 1.2rem;
  box-shadow: var(--shadow-micro);
  margin-bottom: 0.6rem;
}
.ftb-card h3 {
  font-family: var(--font-display);
  font-size: 0.9rem;
  color: var(--text-main);
  margin: 0 0 0.3rem;
}
.ftb-card p {
  font-family: var(--font-body);
  font-size: 0.85rem;
  color: var(--text-label);
  margin: 0;
  line-height: 1.5;
}

/* --- Grid --- */
.ftb-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6rem;
}
@media (max-width: 600px) { .ftb-grid { grid-template-columns: 1fr; } }

/* --- Code blocks --- */
.ftb-wrap pre {
  font-family: var(--font-display);
  font-size: 0.8rem;
  line-height: 1.6;
  background: var(--bg-sunken);
  border: var(--border-width) solid var(--ink);
  padding: 1rem 1.2rem;
  overflow-x: auto;
  box-shadow: var(--shadow-micro);
  color: var(--text-main);
  margin: 0.8rem 0 1.2rem;
}
.ftb-wrap code {
  font-family: var(--font-display);
  font-size: 0.8rem;
  background: var(--bg-sunken);
  padding: 0.1em 0.35em;
  border: 1px solid var(--ink);
}

/* --- Footer --- */
.ftb-footer {
  border-top: var(--border-width) solid var(--ink);
  padding-top: 0.8rem;
  margin-top: 2.5rem;
  font-family: var(--font-display);
  font-size: 0.75rem;
  color: var(--text-muted);
}
.ftb-footer a {
  color: var(--text-main);
  text-decoration: underline;
}

/* --- Accent spans in code --- */
.hl { color: var(--ink); font-weight: 600; }
.cm { color: var(--ink-faint); }

/* --- Badges --- */
.badge {
  font-family: var(--font-display);
  font-size: 0.7rem;
  font-weight: 600;
  padding: 0.15em 0.5em;
  border: var(--border-width) solid var(--ink);
  display: inline-block;
  letter-spacing: 0.03em;
}
.badge-get  { background: var(--bg-sunken); color: var(--text-main); }
.badge-post { background: var(--ink); color: var(--bg-base); }
.badge-put  { background: var(--bg-sunken); color: var(--text-main); }
.badge-delete { background: var(--ink); color: var(--bg-base); }
.badge-patch  { background: var(--bg-sunken); color: var(--text-main); }
.badge-ws   { background: var(--ink); color: var(--bg-base); }
.badge-sse  { background: var(--bg-sunken); color: var(--text-main); }
"""


# ============================================================================
# Helpers
# ============================================================================

def _head(title: str, extra_css: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en" data-style="paper">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
{_FONTS}
<style>{_MAIN_CSS}</style>
<style>{_PAPER_CSS}</style>
<style>{_SHARED_CSS}</style>
<style>{extra_css}</style>
</head>"""


def _footer(app, extra_parts: str = "") -> str:
    hot = "enabled" if getattr(app, 'hot_reload', False) else "disabled"
    hr_script = app.hot_reload_script() if hasattr(app, 'hot_reload_script') else ''
    return f"""<div class="ftb-footer">
  <a href="/docs">docs</a> — Route explorer &middot;
  Hot reload: {hot}{extra_parts}
</div>
</div>
{hr_script}
</body></html>"""


def _has_user_routes(app) -> bool:
    """Check if any routes besides the auto-generated defaults exist."""
    routes = app.list_routes()
    default_names = {"fasttb_docs", "fasttb_welcome", "minu_welcome"}
    for r in routes:
        name = r.get("name", "")
        if name not in default_names:
            return True
    return False


# ============================================================================
# Welcome Page
# ============================================================================

def welcome_page(app) -> str:
    """Generate welcome page with quickstart guide."""
    title = app.title or "FastTB"
    hot = "enabled" if getattr(app, 'hot_reload', False) else "disabled"

    has_routes = _has_user_routes(app)

    if has_routes:
        hero_sub = f'<a href="/docs">Browse {len(app.list_routes())} registered routes →</a>'
    else:
        hero_sub = "No routes registered yet. Let\u2019s fix that."

    return f"""{_head(title)}
<body>
<div class="ftb-wrap">

<div class="ftb-hero">
  <p class="ftb-sub">Welcome to</p>
  <h1>{html.escape(title)}</h1>
  <p class="ftb-sub">{hero_sub}</p>
</div>

<h2>Quickstart</h2>

<pre><span class="cm"># your_app.py</span>
from toolboxv2 import FastTB

app = FastTB(title="{html.escape(title)}")

<span class="hl">@app.get</span>("/index/{{name}}")
async def index(name: str):
    return get_app().web_context() + f\"\"\"
    &lt;h1&gt;Hello {{name}}!&lt;/h1&gt;
    \"\"\" + app.hot_reload_script()

<span class="hl">@app.get</span>("/hello/{{name}}")
async def hello(name: str):
    return {{"message": f"Hello {{name}}!"}}

<span class="hl">@app.sse</span>("/events")
async def events():
    for i in range(10):
        yield {{"event": "tick", "data": {{"n": i}}}}
        await asyncio.sleep(1)

wsgi_app = app.get_wsgi(enable_ws=True)</pre>

<h2>Features</h2>

<div class="ftb-grid">
  <div class="ftb-card">
    <h3>HTTP Routes</h3>
    <p>Decorator-based routing with automatic parameter injection from path, query, body, and session.</p>
  </div>
  <div class="ftb-card">
    <h3>WebSocket</h3>
    <p>Class-based WS handlers via ZMQ bridge. Channels, broadcast, auth \u2014 built in.</p>
  </div>
  <div class="ftb-card">
    <h3>SSE Streaming</h3>
    <p><code>@app.sse()</code> decorator. Async generators become Server-Sent Events.</p>
  </div>
  <div class="ftb-card">
    <h3>Hot Reload</h3>
    <p>File watcher \u2192 WS broadcast \u2192 browser refresh. State-preserving for Minu views. Currently: <strong>{hot}</strong>.</p>
  </div>
</div>

<h2>Minu UI Framework</h2>

<p style="font-size:0.9rem;color:var(--text-label)">For reactive server-rendered views with live WebSocket updates:</p>

<pre>from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import MinuView, State, Column, Text, Button

minu = MinuBridge(app)

<span class="hl">@minu.view</span>("/counter")
class Counter(MinuView):
    count = State(0)

    def render(self):
        return Column(
            Text(f"Count: {{self.count.value}}"),
            Button("+", on_click="increment"),
        )

    async def increment(self, event):
        self.count.value += 1  <span class="cm"># pushed live via WS</span></pre>

{_footer(app, ' &middot; <a href="https://github.com/MarkinHaus/ToolBoxV2">GitHub</a>')}"""


# ============================================================================
# /docs Route Explorer
# ============================================================================

_DOCS_CSS = """
/* --- Route cards --- */
.route-row {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.route-path {
  font-family: var(--font-display);
  font-size: 0.85rem;
  color: var(--text-main);
  word-break: break-all;
}
.route-handler {
  font-family: var(--font-body);
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

/* --- Try button --- */
.try-btn {
  font-family: var(--font-display);
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.25em 0.7em;
  border: var(--border-width) solid var(--ink);
  background: var(--bg-surface);
  color: var(--text-main);
  cursor: pointer;
  box-shadow: 3px 3px 0 var(--ink);
  margin-left: auto;
}
.try-btn:hover {
  transform: translate(-1px, -1px);
  box-shadow: 4px 4px 0 var(--ink);
}
.try-btn:active {
  transform: translate(2px, 2px);
  box-shadow: 0 0 0 var(--ink);
  background: var(--ink);
  color: var(--bg-base);
}

/* --- Filter row --- */
.filter-row {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-bottom: 1.2rem;
}
.filter-btn {
  font-family: var(--font-display);
  font-size: 0.72rem;
  padding: 0.25em 0.65em;
  border: var(--border-width) solid var(--ink);
  background: var(--bg-surface);
  color: var(--text-main);
  cursor: pointer;
  box-shadow: 2px 2px 0 var(--ink);
}
.filter-btn:hover {
  transform: translate(-1px, -1px);
  box-shadow: 3px 3px 0 var(--ink);
}
.filter-btn:active, .filter-btn.active {
  transform: translate(2px, 2px);
  box-shadow: 0 0 0 var(--ink);
  background: var(--ink);
  color: var(--bg-base);
}

/* --- Result box --- */
.result-box {
  display: none;
  font-family: var(--font-display);
  font-size: 0.78rem;
  background: var(--bg-sunken);
  border: 1px solid var(--ink);
  padding: 0.6rem 0.8rem;
  margin-top: 0.5rem;
  max-height: 20rem;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
  color: var(--text-main);
}

/* --- WS input --- */
.ws-input {
  font-family: var(--font-display);
  font-size: 0.78rem;
  padding: 0.35em 0.6em;
  border: 1px solid var(--ink);
  background: var(--input-bg);
  color: var(--text-main);
  width: 100%;
  margin-top: 0.4rem;
}
"""


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

        is_sse = "sse" in name.lower() or "sse" in path.lower()
        if is_sse:
            badge_cls = "badge-sse"
            method_label = "SSE"
        else:
            method_label = method

        try_btn = ""
        result_box = ""
        if method == "GET" and not path.startswith("/docs"):
            try_btn = f'<button class="try-btn" onclick="tryRoute({i})">TRY \u2192</button>'
            result_box = f'<div class="result-box" id="result-{i}"></div>'
        elif method == "WS":
            try_btn = f'<button class="try-btn" onclick="tryWS({i}, \'{html.escape(path)}\')">CONNECT</button>'
            result_box = f'<div class="result-box" id="result-{i}"></div>'

        handler_short = handler.split(".")[-1] if "." in handler else handler
        route_h_info = (' \u00b7 ' + html.escape(name)) if name and name != handler_short else ''
        route_cards.append(f"""
<div class="ftb-card" data-path="{html.escape(path)}" data-method="{html.escape(method)}">
  <div class="route-row">
    <span class="badge {badge_cls}">{html.escape(method_label)}</span>
    <span class="route-path">{html.escape(path)}</span>
    {try_btn}
  </div>
  <div class="route-handler">{html.escape(handler_short)}{route_h_info}</div>
  {result_box}
</div>""")

    routes_json = json.dumps(routes)
    count_by_method = {}
    for r in routes:
        m = r["method"]
        count_by_method[m] = count_by_method.get(m, 0) + 1
    summary_parts = [f"{c} {m}" for m, c in sorted(count_by_method.items())]
    summary = " \u00b7 ".join(summary_parts)

    filters_html = ""
    for m in sorted(count_by_method.keys()):
        filters_html += f'<button class="filter-btn" onclick="filterRoutes(\'{m}\')">{m}</button>\n    '

    x_1 = _head(title + " \u2014 Docs", _DOCS_CSS)
    return f"""{x_1}
<body>
<div class="ftb-wrap">

<div class="ftb-hero" style="padding-bottom:0.5rem">
  <h1>{html.escape(title)}</h1>
  <p class="ftb-sub">Route Explorer \u2014 {summary}</p>
</div>

<div class="filter-row">
  <button class="filter-btn active" onclick="filterRoutes('all')">ALL</button>
  {filters_html}
</div>

{"".join(route_cards)}

{_footer(app, f' &middot; {len(routes)} routes')}

<script>
var ROUTES = {routes_json};
var wsPort = window.__TB_WS_PORT__ || '8100';

function filterRoutes(method) {{
  document.querySelectorAll('.filter-btn').forEach(function(b) {{
    b.classList.toggle('active', b.textContent === method || (method === 'all' && b.textContent === 'ALL'));
  }});
  document.querySelectorAll('.ftb-card[data-method]').forEach(function(c) {{
    c.style.display = (method === 'all' || c.dataset.method === method) ? '' : 'none';
  }});
}}

function tryRoute(i) {{
  var r = ROUTES[i];
  var box = document.getElementById('result-' + i);
  box.style.display = 'block';
  box.textContent = 'Loading...';

  if (r.path.indexOf('/sse/') !== -1 || (r.name && r.name.indexOf('sse') !== -1)) {{
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
    box.innerHTML += '<input class="ws-input" placeholder="Type JSON + Enter" onkeydown="if(event.key===\\'Enter\\')sendWS(this,' + i + ')">';
    window['_ws_' + i] = ws;
  }};
  ws.onmessage = function(e) {{
    box.textContent += '\\u2190 ' + e.data + '\\n';
    box.scrollTop = box.scrollHeight;
  }};
  ws.onclose = function() {{ box.textContent += '--- disconnected ---\\n'; }};
  ws.onerror = function() {{ box.textContent += '--- error ---\\n'; }};
}}

function sendWS(input, i) {{
  var ws = window['_ws_' + i];
  if (ws && ws.readyState === 1) {{
    var box = document.getElementById('result-' + i);
    box.textContent += '\u2192 ' + input.value + '\\n';
    ws.send(input.value);
    input.value = '';
  }}
}}
</script>"""


# ============================================================================
# Minu Starter Page
# ============================================================================

def minu_welcome_page(app, bridge) -> str:
    """Generate Minu starter page with quickstart guide."""
    title = app.title or "FastTB + Minu"
    hot = "enabled" if getattr(app, 'hot_reload', False) else "disabled"
    ws_public = getattr(bridge, 'ws_public', True)
    public = getattr(bridge, 'public', True)

    return f"""{_head(title)}
<body>
<div class="ftb-wrap">

<div class="ftb-hero">
  <p class="ftb-sub">Welcome to</p>
  <h1>{html.escape(title)}</h1>
  <p class="ftb-sub">
    Reactive server-rendered UI with live WebSocket updates.<br>
    No views registered yet \u2014 here\u2019s how to get started.
  </p>
</div>

<h2>Your First View</h2>

<pre>from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import (
    MinuView, State, Column, Row, Heading, Text, Button, Card
)

minu = MinuBridge(app)

<span class="hl">@minu.view</span>("/counter", icon="add_circle", label="Counter")
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

<p style="font-size:0.9rem;color:var(--text-label)">This generates three endpoints automatically:</p>
<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem">
  <code>GET /counter</code>
  <code>POST /counter/event</code>
  <code>WS /ws/openCounter</code>
</div>

<h2>How It Works</h2>

<div class="ftb-grid">
  <div class="ftb-card">
    <h3>Reactive State</h3>
    <p><code>State(0)</code> creates a reactive value. Changing <code>.value</code> automatically pushes patches to the browser via WebSocket.</p>
  </div>
  <div class="ftb-card">
    <h3>Live Push</h3>
    <p>State changes during <code>async</code> handlers (even with <code>await sleep</code>) are pushed immediately \u2014 no polling.</p>
  </div>
  <div class="ftb-card">
    <h3>HTTP Fallback</h3>
    <p>If WebSocket fails, events fall back to HTTP POST. Same API, same code, automatic switching.</p>
  </div>
  <div class="ftb-card">
    <h3>Error Hints</h3>
    <p>Typos in handler code \u2192 toast with suggestion. Type mismatches \u2192 clear hint. Full traceback in console.</p>
  </div>
</div>

<h2>Access Control</h2>

<pre>minu.public = {public}       <span class="cm"># Views public by default</span>
minu.ws_public = {ws_public}    <span class="cm"># WS open without auth</span>

<span class="hl">@minu.view</span>("/admin", require_auth=True)         <span class="cm"># HTTP + WS auth</span>
class Admin(MinuView): ...

<span class="hl">@minu.view</span>("/monitor", ws_public=False)          <span class="cm"># HTTP public, WS auth</span>
class Monitor(MinuView): ...</pre>

<h2>Components</h2>

<div class="ftb-grid">
  <div class="ftb-card">
    <h3>Layout</h3>
    <p><code>Card</code> <code>Row</code> <code>Column</code> <code>Grid</code> <code>Spacer</code> <code>Divider</code></p>
  </div>
  <div class="ftb-card">
    <h3>Content</h3>
    <p><code>Text</code> <code>Heading</code> <code>Badge</code> <code>Icon</code> <code>Image</code> <code>Markdown</code></p>
  </div>
  <div class="ftb-card">
    <h3>Input</h3>
    <p><code>Button</code> <code>Input</code> <code>Textarea</code> <code>Select</code> <code>Checkbox</code> <code>Switch</code> <code>Slider</code></p>
  </div>
  <div class="ftb-card">
    <h3>Feedback</h3>
    <p><code>Alert</code> <code>Toast</code> <code>Progress</code> <code>Spinner</code> <code>Modal</code></p>
  </div>
</div>

{_footer(app, f' &middot; Access: {"public" if public else "private"} &middot; WS {"open" if ws_public else "auth"}')}"""


# ============================================================================
# Registration
# ============================================================================

def register_defaults(app):
    """Register /docs and optional welcome page on a FastTB instance."""
    if not app.has_route("/docs", "GET"):
        @app.get("/docs", name="fasttb_docs")
        async def _docs():
            return docs_page(app)

    if not app.has_route("/", "GET"):
        @app.get("/", name="fasttb_welcome")
        async def _welcome():
            return welcome_page(app)


def register_minu_defaults(app, bridge):
    """Register Minu starter welcome page if no views registered."""
    if not app.has_route("/", "GET"):
        @app.get("/", name="minu_welcome")
        async def _minu_welcome():
            if bridge._view_registry:
                first_path = next(iter(bridge._view_registry))
                return (302, {"Location": first_path, "Content-Type": "text/plain"}, b"")
            return minu_welcome_page(app, bridge)
