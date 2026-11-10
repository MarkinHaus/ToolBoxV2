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
    """First-run state — same source as the CLI onboarding (manifest app.profile)."""
    from toolboxv2.utils.clis.first_run import onboarding_state
    return onboarding_state()

# =============================================================================
# Routes
# =============================================================================

def _setup_page() -> Optional[Path]:
    """The one onboarding UI for Tauri + browser.

    web/installer/index.html (copied into dist/ by the web build) is kept
    byte-identical to simple-core/src-tauri/installer.html (Tauri bundle).
    """
    candidates = []
    d = dist_dir()
    if d:
        candidates.append(d / "web" / "installer" / "index.html")
    candidates.append(tb_root_dir / "web" / "installer" / "index.html")
    return next((c for c in candidates if c.is_file()), None)


def _serve(path: Path):
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    return FastTBHandler._serve_static_file(str(path))


@app.get("/", auth=False)
async def root(request: ParsedRequest):
    err = _ensure_local(request)
    if err is not None:
        return err

    # First run -> onboarding (same state as CLI: manifest app.profile)
    if get_installation_state().get("first_run", True):
        page = _setup_page()
        if page:
            return _serve(page)
        return ("ToolBoxV2 is not set up yet. Run 'tb' in a terminal "
                "(or 'tb -init local|server|developer').")

    d = dist_dir()
    if d:
        for name in ["mainPagen.html", "index.html"]:
            if (d / name).exists():
                return _serve(d / name)
    return "ToolBox Local UI - dist/index.html not found."


@app.get("/setup", auth=False)
async def setup(request: ParsedRequest):
    """Re-run onboarding / change settings (wizard API) at any time."""
    err = _ensure_local(request)
    if err is not None:
        return err
    page = _setup_page()
    return _serve(page) if page else Result.default_user_error(
        info="Setup page not found (build dist or use 'tb -init onboard').", exec_code=404)


# =============================================================================
# Wizard-API (MUSS vor dem spa_fallback-Catch-all registriert werden)
# =============================================================================
from toolboxv2.utils.workers.fast.wizard_api import register_wizard_routes

register_wizard_routes(app)

# /live sub-mount (Dashboard-Redirect auf Worker, muss vor Catch-all)
from toolboxv2.utils.workers.fast.live_proxy import register_live_routes

register_live_routes(app)


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
