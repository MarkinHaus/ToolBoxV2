#!/usr/bin/env python3
"""
ToolBoxV2 Dev Runner & Auto-UI
==============================
Startet Module isoliert mit "Smart UI" Erkennung.
- Lädt Module aus dem aktuellen Verzeichnis (init_cwd)
- Erkennt native Minu-UIs automatisch
- Fallback auf "Strict Dark" Auto-Admin Panel
"""

import argparse
import asyncio
import logging
import os
import sys
import threading
import json
import inspect
from typing import Dict, Any

import uvicorn
try:
    from a2wsgi import WSGIMiddleware
except ImportError:
    WSGIMiddleware = None
# Toolbox Imports
from toolboxv2 import init_cwd
from toolboxv2.utils.workers.config import load_config
from toolboxv2.utils.workers.event_manager import ZMQEventManager
from toolboxv2.utils.workers.server_worker import HTTPWorker, ToolBoxHandler
from toolboxv2.utils.system.getting_and_closing_app import get_app

# --- STYLE DEFINITION (STRICT DARK) ---
STRICT_DARK_CSS = """
:root {
    --bg: #08080d;
    --text: #e2e2e8;
    --text-muted: rgba(255,255,255,0.45);
    --text-faint: rgba(255,255,255,0.25);
    --accent: #6366f1;
    --accent-light: #a5b4fc;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --border: rgba(255,255,255,0.06);
    --border-active: rgba(99,102,241,0.2);
    --surface: rgba(255,255,255,0.015);
    --surface-hover: rgba(99,102,241,0.06);
}

body {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    margin: 0;
    padding: 2rem;
    min-height: 100vh;
}

h1, h2, h3 { font-weight: 300; margin: 0; }
pre, code, .mono { font-family: 'IBM Plex Mono', monospace; }

/* Layout */
.container {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 2rem;
    max-width: 1400px;
    margin: 0 auto;
}

header {
    grid-column: 1 / -1;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}

.label {
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-faint);
    display: block;
    margin-bottom: 0.5rem;
}

/* Sidebar / Function List */
.sidebar {
    display: flex;
    flex-direction: column;
    gap: 1px;
}

.func-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 0.75rem 1rem;
    text-align: left;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.func-btn:hover {
    background: var(--surface-hover);
    color: var(--text);
    border-color: var(--border-active);
}

.func-btn:active {
    transform: translateX(2px);
}

.func-btn .method-badge {
    font-size: 9px;
    padding: 2px 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.05);
}

/* Main Area */
.main-area {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* Inputs */
input, textarea, select {
    background: #000;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    transition: border 0.15s;
}

input:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
}

button.action-btn {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    color: var(--accent-light);
    padding: 6px 16px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.15s;
}

button.action-btn:hover {
    background: rgba(99,102,241,0.2);
    box-shadow: 0 0 10px rgba(99,102,241,0.1);
}

/* Result Area */
#result-area {
    min-height: 300px;
    max-height: 80vh;
    overflow-y: auto;
}

.json-key { color: var(--text-muted); }
.json-string { color: var(--success); }
.json-number { color: var(--warning); }
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TBv2 Dev: {module_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>{css}</style>
</head>
<body>
    <header>
        <div>
            <span class="label">DEVELOPMENT ENVIRONMENT</span>
            <h1>{module_name}</h1>
        </div>
        <div style="text-align: right">
            <span class="label">STATUS</span>
            <span style="color: var(--success); font-family: 'IBM Plex Mono'; font-size: 12px;">● ACTIVE</span>
        </div>
    </header>

    <div class="container">
        <div class="sidebar">
            <span class="label">AVAILABLE FUNCTIONS</span>
            {sidebar_buttons}
        </div>

        <div class="main-area">
            <div id="param-area" class="card" style="display:none;">
                <span class="label">PARAMETERS</span>
                <form id="active-form" hx-post="" hx-target="#result-area" hx-swap="innerHTML">
                    <div id="form-inputs" style="display: grid; gap: 1rem; margin-bottom: 1rem;"></div>
                    <div style="display: flex; justify-content: flex-end;">
                        <button type="submit" class="action-btn">EXECUTE</button>
                    </div>
                </form>
            </div>

            <div class="card">
                <span class="label">OUTPUT CONSOLE</span>
                <div id="result-area" class="mono" style="font-size: 12px; white-space: pre-wrap;">
                    <span style="color: var(--text-faint)">Waiting for input...</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Simple client-side logic to handle parameter forms
        function loadForm(funcName, endpoint, params) {{
            const area = document.getElementById('param-area');
            const inputsDiv = document.getElementById('form-inputs');
            const form = document.getElementById('active-form');

            area.style.display = 'block';
            form.setAttribute('hx-post', endpoint);
            htmx.process(form); // Re-bind HTMX

            inputsDiv.innerHTML = '';

            if (Object.keys(params).length === 0) {{
                inputsDiv.innerHTML = '<span style="color:var(--text-faint); font-size:11px;">No parameters required.</span>';
            }} else {{
                for (const [key, type] of Object.entries(params)) {{
                    const wrapper = document.createElement('div');
                    wrapper.innerHTML = `
                        <div style="margin-bottom:4px; font-size:11px; color:var(--text-muted);">${{key}}</div>
                        <input name="${{key}}" placeholder="${{type}}" />
                    `;
                    inputsDiv.appendChild(wrapper);
                }}
            }}
        }}
    </script>
</body>
</html>
"""


# --- LOGIC ---

class DevRunnerDispatcher:
    """
    Routet Anfragen:
    1. Checkt auf native UI des Moduls (Minu Support)
    2. Wenn nicht vorhanden -> Custom Auto-UI
    3. API Calls an HTTPWorker
    """

    def __init__(self, api_app, app_instance, module_name):
        self.api_app = api_app
        self.app = app_instance
        self.module_name = module_name

        # Analysiere Modul für Auto-UI
        self.functions_meta = self._analyze_module()

    def _analyze_module(self):
        """Extrahiert API-Funktionen und deren Parameter."""
        meta = {}
        if self.module_name not in self.app.functions:
            return meta

        funcs = self.app.functions[self.module_name]
        for func_name, data in funcs.items():
            if not isinstance(data, dict) or not data.get('api', True):
                continue

            # Parametertypen extrahieren (simpel)
            params = {}
            if 'params' in data:
                # Toolbox params structure handling could go here
                pass

            # Introspection fallback
            func_obj = data.get('func')
            if func_obj:
                sig = inspect.signature(func_obj)
                for name, param in sig.parameters.items():
                    if name in ['self', 'args', 'kwargs', 'app', 'request']: continue
                    params[name] = str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"

            meta[func_name] = {"endpoint": f"/api/{self.module_name}/{func_name}", "params": params}
        return meta

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "/")

        # 1. API Pass-through
        if path.startswith(("/api", "/auth", "/ws")):
            await self.api_app(scope, receive, send)
            return

        # 2. Root Access -> Smart UI Check
        if path == "/" or path == "/index.html":
            # CHECK: Native UI Function?
            native_ui_funcs = ["ui", "index", "main", "render", "render_main"]
            target_func = None

            mod_funcs = self.app.functions.get(self.module_name, {})

            # Priorität: explizite 'ui' funktion
            for name in native_ui_funcs:
                if name in mod_funcs:
                    target_func = name
                    break

            if target_func:
                # Modul hat eine UI -> Redirect oder Render
                # Wir simulieren einen API Call an die UI Funktion
                # In Produktion würde der HTTPWorker das rendern, hier machen wir einen internen Redirect
                scope["path"] = f"/api/{self.module_name}/{target_func}"
                # Wir müssen sicherstellen, dass Minu SSR oder HTML zurückkommt
                await self.api_app(scope, receive, send)
                return

            # FALLBACK: Custom Auto-UI
            await self._serve_auto_ui(send)
            return

        # 404
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"Not Found"})

    async def _serve_auto_ui(self, send):
        sidebar_html = ""
        # Sortieren: 'open' zuerst
        sorted_keys = sorted(self.functions_meta.keys(), key=lambda k: (not k.startswith('open'), k))

        for name in sorted_keys:
            meta = self.functions_meta[name]
            params_json = json.dumps(meta['params']).replace('"', '&quot;')
            endpoint = meta['endpoint']

            # JS Call to load form
            onclick = f"loadForm('{name}', '{endpoint}', {params_json})"

            badge = "API"
            if name.startswith("get"): badge = "GET"
            if name.startswith("set") or name.startswith("do"): badge = "POST"

            sidebar_html += f"""
            <div class="func-btn" onclick="{onclick}">
                <span>{name}</span>
                <span class="method-badge">{badge}</span>
            </div>
            """

        html = HTML_TEMPLATE.format(
            module_name=self.module_name,
            css=STRICT_DARK_CSS,
            sidebar_buttons=sidebar_html
        )

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/html; charset=utf-8")]
        })
        await send({"type": "http.response.body", "body": html.encode()})


def run_dev_server(target_module: str = None, port: int = 5000):
    """
    Startet den Server. Wenn target_module None ist, versucht er alles zu laden
    und zeigt ein Dashboard (optional, hier Fokus auf Single Mod).
    """

    if WSGIMiddleware is None:
        print("No a2wsgi istalld run pip install a2wsgi")
        return

    # 1. Config & Environment
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    config = load_config()
    config.manager.web_ui_enabled = False  # Konflikte vermeiden

    # 2. Loading Modules from CWD
    print(f"\033[94m[DEV]\033[0m Loading externals from {init_cwd}")

    # Infrastructure Thread
    def start_infra():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        broker = ZMQEventManager("dev_broker",
                                 config.zmq.pub_endpoint, config.zmq.sub_endpoint,
                                 config.zmq.req_endpoint, config.zmq.rep_endpoint,
                                 config.zmq.http_to_ws_endpoint, is_broker=True)
        loop.run_until_complete(broker.start())
        loop.run_forever()

    threading.Thread(target=start_infra, daemon=True).start()

    # 3. App Initialization
    app = get_app(name="DevRunner", from_="DevRunner")

    # LOAD ALL MODS (Async Sync Hack)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(app.load_all_mods_in_file(init_cwd))
    loop.close()

    # 4. Determine Target
    if not target_module:
        # Wenn kein Modul angegeben, nimm das erste gefundene (außer System Module)
        candidates = [m for m in app.functions.keys() if m not in ["System", "CloudM", "DB"]]
        if candidates:
            target_module = candidates[0]
            print(f"\033[93m[DEV]\033[0m No module specified. Auto-selecting: {target_module}")
        else:
            print("\033[91m[ERROR]\033[0m No modules found in current directory.")
            return

    # 5. HTTP Worker Setup
    http_worker = HTTPWorker(f"dev_http_{target_module}", config, app=app)
    http_worker._init_session_manager()
    http_worker._init_access_controller()
    http_worker._init_auth_handler()

    # Event Manager Thread
    def start_em():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(http_worker._init_event_manager())
        loop.run_forever()

    threading.Thread(target=start_em, daemon=True).start()

    # Handler Init
    http_worker._toolbox_handler = ToolBoxHandler(
        app, config, http_worker._access_controller, "/api"
    )

    # 6. Dispatcher Construction
    api_asgi = WSGIMiddleware(http_worker.wsgi_app)
    dev_app = DevRunnerDispatcher(api_asgi, app, target_module)

    print(f"\n\033[92m[READY]\033[0m Dev Server running for module: \033[1m{target_module}\033[0m")
    print(f"        UI URL: http://localhost:{port}")
    print(f"        CWD   : {init_cwd}\n")

    uvicorn.run(dev_app, host="0.0.0.0", port=port, log_level="warning")

def main():

    parser = argparse.ArgumentParser(description="ToolBoxV2 Dev Runner", prog="tb x")
    parser.add_argument("module", nargs="?", help="Target module name", default=None)
    parser.add_argument("--port", type=int, default=5000, help="Server port")

    args = parser.parse_args()

    run_dev_server(args.module, args.port)

if __name__ == "__main__":
    main()
