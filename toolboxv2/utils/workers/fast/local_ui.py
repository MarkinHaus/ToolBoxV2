#!/usr/bin/env python3
"""
toolboxv2/utils/workers/fast/local_ui.py - Local Default UI (Static + Auto-Mount)

PRIMARY JOBS:
1. Asset-Server: Serves dist/ statically (including index.html/mainPagen.html).
2. Auto-Mount-Hub: Features mount themselves via feature.yaml deklaratively.

Refactored 2026-09-01: Removed HTMX legacy code.
"""

import asyncio
import importlib
import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from toolboxv2 import Result, get_app, tb_root_dir
from toolboxv2.utils.workers.fast.endpoint import dist_dir
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.extras.blobs import BlobFile

app = FastTB(title="ToolBox Local")
app.serve_login_assets = True
app.inject_style = False
# Mount dist/ as /dist
app.mount_dist()

# =============================================================================
# Helpers
# =============================================================================

def _ensure_local(request: ParsedRequest) -> Optional[Result]:
    """Refuse anything that isn't loopback."""
    host = (request.headers.get("host") or "").split(":")[0].lower()
    if host not in ("127.0.0.1", "localhost", "[::1]", "::1"):
        return Result.default_user_error(
            info="This UI is local-only. Use the public site for remote access.",
            exec_code=403,
        )
    return None

def _auto_mount_features(fast_app: FastTB):
    """Scan all features and mount their UIs based on feature.yaml."""
    features_dir = tb_root_dir / "features"
    if not features_dir.exists():
        return

    for fyaml in features_dir.glob("*/feature.yaml"):
        try:
            with open(fyaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            mount_cfg = data.get("mount")
            if mount_cfg and isinstance(mount_cfg, dict):
                prefix = mount_cfg.get("prefix")
                module_path = mount_cfg.get("module")
                app_var = mount_cfg.get("app_var", "app")

                if prefix and module_path:
                    try:
                        # Attempt to load the module
                        mod = importlib.import_module(module_path)
                        sub_app_raw = getattr(mod, app_var)

                        # Handle factory functions
                        if callable(sub_app_raw) and not isinstance(sub_app_raw, FastTB):
                            sub_app = sub_app_raw()
                        else:
                            sub_app = sub_app_raw

                        if not isinstance(sub_app, FastTB):
                             print(f"[local_ui] Warning: {module_path}.{app_var} is not a FastTB instance")
                             continue

                        # mount_app expects (other_app, prefix, source)
                        fast_app.mount_app(sub_app, prefix, source=f"feature:{data.get('name', 'unknown')}")
                        print(f"[local_ui] Mounted {data.get('name', 'unknown')} at {prefix}")
                    except Exception as e:
                        print(f"[local_ui] Failed to mount feature {data.get('name')}: {e}")
        except Exception:
            continue

def get_installation_state() -> Dict[str, Any]:
    """Get installation and first-start state from Python blob setting."""
    try:
        with BlobFile("system/installation_state.json", "r") as f:
            state = f.read_json()
            if state: return state
    except Exception:
        pass
    return {"installed": False, "first_run": True}

# =============================================================================
# Routes
# =============================================================================

@app.get("/", auth=False)
async def root(request: ParsedRequest):
    err = _ensure_local(request)
    if err is not None: return err

    # Check if first run
    state = get_installation_state()
    if state.get("first_run", True):
        return await welcome(request)

    # SPA Fallback / Entry point
    d = dist_dir()
    if d:
        # Priority for mainPagen.html if it exists, otherwise index.html
        for name in ["mainPagen.html", "index.html"]:
            if (d / name).exists():
                from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
                return FastTBHandler._serve_static_file(str(d / name))

    return "ToolBox Local UI - dist/index.html not found."

@app.get("/welcome", auth=False)
async def welcome(request: ParsedRequest):
    """Empty welcome page, only title + tbjs."""
    state = get_installation_state()
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Welcome to ToolBox</title>
    <link rel="stylesheet" href="/dist/tbjs/main.css">
    <script src="/dist/tbjs/main.js"></script>
    <script>
        const state = {json.dumps(state)};
        localStorage.setItem('tb_installation_state', JSON.stringify(state));
    </script>
</head>
<body class="glass-v3">
    <h1>Welcome to ToolBox</h1>
    <div id="app"></div>
</body>
</html>"""

@app.get("/install", auth=False)
async def install(request: ParsedRequest):
    """Empty installation page with system data display via JS."""
    state = get_installation_state()
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ToolBox Installation</title>
    <link rel="stylesheet" href="/dist/tbjs/main.css">
    <script src="/dist/tbjs/main.js"></script>
    <script>
        const state = {json.dumps(state)};
        localStorage.setItem('tb_installation_state', JSON.stringify(state));

        window.addEventListener('load', () => {{
            const sysData = {{
                platform: navigator.platform,
                agent: navigator.userAgent,
                cores: navigator.hardwareConcurrency,
                memory: navigator.deviceMemory,
                language: navigator.language,
                screen: window.screen.width + "x" + window.screen.height,
                timestamp: new Date().toISOString()
            }};
            console.log("Detected System Data:", sysData);
            const container = document.getElementById('app');
            const infoDiv = document.createElement('pre');
            infoDiv.style.background = "rgba(0,0,0,0.05)";
            infoDiv.style.padding = "20px";
            infoDiv.style.borderRadius = "8px";
            infoDiv.textContent = "System Data (Detected):\\n" + JSON.stringify(sysData, null, 2);
            container.appendChild(infoDiv);
        }});
    </script>
</head>
<body class="glass-v3">
    <h1>ToolBox Installation</h1>
    <div id="app"></div>
</body>
</html>"""

@app.get("/{path:path}", auth=False)
async def spa_fallback(request: ParsedRequest, path: str):
    """Catch-all for SPA routing."""
    d = dist_dir()
    if d:
        target = d / path
        if target.is_file():
             from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
             return FastTBHandler._serve_static_file(str(target))

        # SPA Fallback to index.html
        if (d / "index.html").exists():
            from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
            return FastTBHandler._serve_static_file(str(d / "index.html"))

    return Result.default_user_error(info=f"Not found: {path}", exec_code=404)

# =============================================================================
# Initialization
# =============================================================================

_auto_mount_features(app)

# Wire-in helper
def get_handler():
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    return FastTBHandler(app)

if __name__ == "__main__":
    from waitress import serve
    h = get_handler()
    print("Starting local_ui on http://127.0.0.1:8467")
    serve(h.as_wsgi_app(enable_ws=False), host="127.0.0.1", port=8467)
