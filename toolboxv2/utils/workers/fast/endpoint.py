#!/usr/bin/env python3
"""
toolboxv2/utils/workers/fast/endpoint.py - Local-UI endpoint resolution

ONE canonical answer to "on which host:port does the local UI live?".

Before this module the answer was spread over five places that had drifted
apart:

    tb-manifest.yaml  services.manager.live_ui_port   (schema default 8700)
    workers/config.py ManagerConfig.live_ui_port      (default 5000)
    tauri_integration DEFAULT_PORT / TB_HTTP_PORT     (default 5000)
    simple-core worker_manager.rs DEFAULT_HTTP_PORT   (const 5000)
    simple-core lib.rs                                (hardcoded 127.0.0.1:5000)

Everything that needs the local UI (tray icon, Tauri app, first-run redirect,
the browser fallback) now asks `resolve_local_ui_endpoint()` instead, and the
manifest wins.

Resolution order (first hit wins):
    1. TB_LOCAL_UI_HOST / TB_LOCAL_UI_PORT   — explicit override
    2. tb-manifest.yaml services.manager.live_ui_host / live_ui_port
    3. workers config.yaml manager.live_ui_host / live_ui_port
    4. DEFAULT_LOCAL_UI_HOST / DEFAULT_LOCAL_UI_PORT

Because the Tauri shell (Rust) starts *before* any Python runs, the resolved
endpoint is also published as a small JSON file in the platform data dir:

    <data_dir>/toolboxv2/local_ui.json   {"host": ..., "port": ..., "url": ...}

worker_manager.rs reads that file, so Rust and Python cannot drift again.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_LOCAL_UI_HOST = "127.0.0.1"
DEFAULT_LOCAL_UI_PORT = 5000

ENDPOINT_FILENAME = "local_ui.json"

_cache: Optional[Tuple[str, int]] = None


# =============================================================================
# Data dir (mirrors Rust `dirs::data_dir()/toolboxv2`)
# =============================================================================

def platform_data_dir() -> Path:
    """Same directory Rust's `dirs::data_dir().join("toolboxv2")` resolves to."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "toolboxv2"


# =============================================================================
# Resolution
# =============================================================================

def _from_env() -> Tuple[Optional[str], Optional[int]]:
    host = os.getenv("TB_LOCAL_UI_HOST") or None
    port_raw = os.getenv("TB_LOCAL_UI_PORT")
    try:
        port = int(port_raw) if port_raw else None
    except ValueError:
        port = None
    return host, port


def _from_manifest() -> Tuple[Optional[str], Optional[int]]:
    try:
        from toolboxv2 import tb_root_dir
        from toolboxv2.utils.manifest.loader import ManifestLoader
        loader = ManifestLoader(base_dir=str(tb_root_dir))
        if not loader.manifest_path.exists():
            return None, None
        manifest = loader.load()
        mgr = manifest.services.manager
        return mgr.live_ui_host, int(mgr.live_ui_port)
    except Exception:
        return None, None


def _from_workers_config() -> Tuple[Optional[str], Optional[int]]:
    try:
        from toolboxv2.utils.workers.config import load_config
        mgr = load_config().manager
        return getattr(mgr, "live_ui_host", None), int(getattr(mgr, "live_ui_port", 0)) or None
    except Exception:
        return None, None


def resolve_local_ui_endpoint(refresh: bool = False) -> Tuple[str, int]:
    """Return (host, port) of the local UI. Cached — pass refresh=True to redo."""
    global _cache
    if _cache is not None and not refresh:
        return _cache

    host, port = _from_env()
    for source in (_from_manifest, _from_workers_config):
        if host and port:
            break
        s_host, s_port = source()
        host = host or s_host
        port = port or s_port

    _cache = (host or DEFAULT_LOCAL_UI_HOST, port or DEFAULT_LOCAL_UI_PORT)
    return _cache


def local_ui_port() -> int:
    return resolve_local_ui_endpoint()[1]


def local_ui_host() -> str:
    return resolve_local_ui_endpoint()[0]


def local_ui_url(path: str = "") -> str:
    """http://host:port[/path] — the one URL tray + Tauri + browser all open."""
    host, port = resolve_local_ui_endpoint()
    base = f"http://{host}:{port}"
    if not path:
        return base
    return f"{base}/{path.lstrip('/')}"


# =============================================================================
# Dist directory (what the local UI serves as static root)
# =============================================================================

def dist_dir() -> Optional[Path]:
    """Absolute path of the built web bundle, or None if it isn't built yet.

    Manifest `paths.dist_dir` wins; otherwise <tb_root_dir>/dist.
    """
    candidates = []
    env_dist = os.getenv("TB_DIST_DIR")
    if env_dist:
        candidates.append(Path(env_dist))
    try:
        from toolboxv2 import tb_root_dir
        root = Path(str(tb_root_dir))
        try:
            from toolboxv2.utils.manifest.loader import ManifestLoader
            loader = ManifestLoader(base_dir=str(root))
            if loader.manifest_path.exists():
                raw = loader.load().paths.dist_dir
                if raw and "${" not in raw:
                    p = Path(raw)
                    candidates.append(p if p.is_absolute() else (root / raw))
        except Exception:
            pass
        candidates.append(root / "dist")
    except Exception:
        pass

    for c in candidates:
        try:
            if c.is_dir():
                return c.resolve()
        except Exception:
            continue
    return None


# =============================================================================
# Publishing — so the Rust side can discover the same port
# =============================================================================

def endpoint_file() -> Path:
    return platform_data_dir() / ENDPOINT_FILENAME


def publish_endpoint(host: Optional[str] = None, port: Optional[int] = None) -> Optional[Path]:
    """Write the live endpoint to <data_dir>/toolboxv2/local_ui.json.

    Best-effort: returns the path on success, None on any failure. Called by
    whoever actually binds the port (tauri worker, live-ui worker) so the
    published value describes reality, not just configuration.
    """
    r_host, r_port = resolve_local_ui_endpoint()
    host = host or r_host
    port = int(port or r_port)
    payload = {
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "pid": os.getpid(),
    }
    try:
        target = endpoint_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(target)
        # Make the value visible to child processes (sidecars, spawned workers)
        os.environ.setdefault("TB_LOCAL_UI_HOST", host)
        os.environ.setdefault("TB_LOCAL_UI_PORT", str(port))
        return target
    except Exception:
        return None


def read_published_endpoint() -> Optional[dict]:
    """Read back what a running worker published, or None."""
    try:
        return json.loads(endpoint_file().read_text(encoding="utf-8"))
    except Exception:
        return None
