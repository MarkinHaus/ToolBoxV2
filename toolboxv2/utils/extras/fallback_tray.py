"""Fallback system tray (pystray) for ToolBoxV2.

Used whenever the Tauri desktop app is not available: the tray is the only
always-on surface the user has, so everything in here is best-effort and must
never take the host process down.

The tray talks to the sidecar's tray API (``/tray/state``). ``TB_TRAY_URL`` is
the single source of truth for its base URL and is shared with the Rust side.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import subprocess
import sys
import urllib.request

DEFAULT_TRAY_URL = "http://127.0.0.1:5000"
_TRAY_STATE_TIMEOUT = 1.5

log = logging.getLogger("toolboxv2.fallback_tray")

# Global handle so the icon can always be stopped cleanly (Tcl/Tk otherwise
# crashes on interpreter shutdown).
_active_tray_icon = None


def attach_debug_log() -> None:
    """Attach a file handler so the tray stays diagnosable under pythonw.

    Best effort: an unwritable directory must never stop the tray from running.
    Idempotent - repeated calls do not stack handlers.
    """
    log_dir = os.getenv("TB_TRAY_LOG_DIR") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), ".info")
    path = os.path.join(log_dir, "tray_debug.log")
    if any(getattr(h, "_tb_tray_debug", False) for h in log.handlers):
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        handler._tb_tray_debug = True
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    except Exception as exc:  # noqa: BLE001
        log.debug("tray debug log unavailable at %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Tray API access
# --------------------------------------------------------------------------- #

def tray_base_url() -> str:
    """Base URL of the tray API, without trailing slash.

    ``TB_TRAY_URL`` wins (the daemon may run on its own port); otherwise the
    published local_ui.json endpoint that Rust reads as well.
    """
    url = (os.getenv("TB_TRAY_URL") or "").strip()
    if not url:
        try:
            from toolboxv2.utils.workers.fast.endpoint import local_ui_url
            url = local_ui_url()
        except Exception as exc:  # noqa: BLE001 - endpoint file may not exist yet
            log.debug("local ui endpoint unavailable: %s", exc)
    return (url or DEFAULT_TRAY_URL).rstrip("/")


def fetch_tray_state() -> dict | None:
    """Pull the aggregated worker state. Returns ``None`` if unreachable."""
    try:
        with urllib.request.urlopen(
            f"{tray_base_url()}/tray/state", timeout=_TRAY_STATE_TIMEOUT
        ) as response:
            data = json.loads(response.read())
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - tray must survive any transport error
        log.debug("tray state unavailable: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Icon
# --------------------------------------------------------------------------- #

def create_gear_icon():
    """Draw a gear with a blue T and a white V. Requires Pillow."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = 32, 32
    r_outer, r_inner = 22, 14
    num_teeth, tooth_width, tooth_depth = 8, 8, 6

    for i in range(num_teeth):
        angle = i * (2 * math.pi / num_teeth)
        x1 = cx + (r_outer - tooth_depth) * math.cos(angle)
        y1 = cy + (r_outer - tooth_depth) * math.sin(angle)
        x2 = cx + (r_outer + tooth_depth) * math.cos(angle)
        y2 = cy + (r_outer + tooth_depth) * math.sin(angle)
        draw.line([x1, y1, x2, y2], fill=(128, 128, 128, 255), width=tooth_width)

    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
                 fill=(128, 128, 128, 255))
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
                 fill=(0, 0, 0, 0))

    # blue T
    draw.line([22, 24, 42, 24], fill=(0, 102, 204, 255), width=4)
    draw.line([32, 24, 32, 44], fill=(0, 102, 204, 255), width=4)
    # white V
    draw.line([25, 34, 32, 46], fill=(255, 255, 255, 255), width=3)
    draw.line([32, 46, 39, 34], fill=(255, 255, 255, 255), width=3)

    return img


# --------------------------------------------------------------------------- #
# Menu actions
# --------------------------------------------------------------------------- #

def open_dashboard(icon=None, item=None) -> None:
    """Open the live UI in the default browser."""
    import webbrowser

    url = tray_base_url()
    try:
        from toolboxv2 import get_app

        manager = get_app().manifest.services.manager
        url = f"http://{manager.live_ui_host}:{manager.live_ui_port}"
    except Exception as exc:  # noqa: BLE001 - manifest is optional here
        log.debug("no manifest live-ui config, using tray base url: %s", exc)
    webbrowser.open(url)


def open_workers_network(icon=None, item=None) -> None:
    """Launch the workers live view without blocking the tray thread."""
    try:
        subprocess.Popen(
            [sys.executable, "-m", "toolboxv2", "workers", "live"],
            start_new_session=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("could not start workers live view: %s", exc)


def cleanup_active_tray() -> None:
    """Stop the icon on interpreter shutdown."""
    global _active_tray_icon
    icon, _active_tray_icon = _active_tray_icon, None
    if icon is not None:
        try:
            icon.stop()
        except Exception as exc:  # noqa: BLE001
            log.debug("tray stop failed: %s", exc)


atexit.register(cleanup_active_tray)


def make_stop_handler(tb_app):
    """Build the 'Quit Application' handler for the given app."""

    def on_stop_all(icon=None, item=None) -> None:
        tb_app.sprint("Stopping background application from Tray...")
        if icon is not None:
            try:
                icon.stop()
            except Exception as exc:  # noqa: BLE001
                log.debug("icon stop failed: %s", exc)
        cleanup_active_tray()
        try:
            import asyncio

            tb_app.alive = False
            loop = getattr(tb_app, "loop", None)
            if loop and not loop.is_closed():
                try:
                    loop.run_until_complete(asyncio.wait_for(tb_app.a_exit(), timeout=5.0))
                except asyncio.TimeoutError:
                    tb_app.sprint("Graceful shutdown timeout, forcing exit")
                except RuntimeError as exc:
                    tb_app.sprint(f"Shutdown loop error: {exc}")
            else:
                tb_app.exit(remove_all=False)
        except Exception as exc:  # noqa: BLE001
            tb_app.sprint(f"Tray exit error: {exc}")
        os._exit(0)

    return on_stop_all


# --------------------------------------------------------------------------- #
# Menu
# --------------------------------------------------------------------------- #

def build_menu_items(tb_app):
    """Yield the current menu items.

    pystray accepts either a sequence of items or a *single callable returning
    a generator of items*; this is that generator. It is re-evaluated whenever
    the menu is shown, so the instance list stays live.
    """
    import pystray

    yield pystray.MenuItem("Open Dashboard", open_dashboard)
    yield pystray.MenuItem("Workers Network", open_workers_network)
    yield pystray.Menu.SEPARATOR

    try:
        state = fetch_tray_state()
    except Exception as exc:  # noqa: BLE001 - a broken API must not kill the menu
        log.debug("tray state lookup failed: %s", exc)
        state = None

    workers = {
        wid: info for wid, info in (state or {}).items() if isinstance(info, dict)
    }
    if workers:
        running = sum(1 for info in workers.values() if info.get("running"))
        instance_items = [
            pystray.MenuItem(
                "{dot} {label} (pid={pid})".format(
                    dot="\u25cf" if info.get("running") else "\u25cb",
                    label=info.get("label", wid),
                    pid=info.get("pid", "?"),
                ),
                None,
                enabled=False,
            )
            for wid, info in workers.items()
        ]
        yield pystray.MenuItem(
            f"Instances ({running} running)", pystray.Menu(*instance_items)
        )
        yield pystray.Menu.SEPARATOR
    elif getattr(tb_app, "daemon_app", None):
        yield pystray.MenuItem("Runner: Active", None, enabled=False)
        yield pystray.Menu.SEPARATOR

    yield pystray.MenuItem("Quit Application", make_stop_handler(tb_app))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_fallback_tray(tb_app) -> None:
    """Run the tray icon. Blocks until the icon is stopped."""
    global _active_tray_icon

    attach_debug_log()

    try:
        import pystray  # noqa: F401
    except ImportError:
        log.error("pystray/Pillow not installed - fallback tray disabled")
        tb_app.sprint("pystray oder Pillow sind nicht installiert. Fallback-Tray startet nicht.")
        return

    try:
        icon = pystray.Icon(
            "tb_tray_runner",
            icon=create_gear_icon(),
            title="ToolBox Tray Runner",
            menu=pystray.Menu(lambda: build_menu_items(tb_app)),
        )
        _active_tray_icon = icon
        log.info("fallback tray starting (api=%s)", tray_base_url())
        icon.run()
        log.info("fallback tray stopped")
    except Exception as exc:  # noqa: BLE001
        log.error("tray error: %s", exc, exc_info=True)
        tb_app.sprint(f"Fehler beim Starten des Fallback-Trays: {exc}")
    finally:
        _active_tray_icon = None
