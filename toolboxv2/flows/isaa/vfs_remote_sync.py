"""
vfs_remote_sync — icli flow
===========================
Thin command wrapper around VFSSyncManager to keep VFS share folders
live-synced across machines (LiveSync engine). GENERIC: /global and any folder.

Actions:
    connect     register + (if serving) start a folder share
    disconnect  remove + stop a folder share
    list/status show registered + live folders
    serve       run the mini-service: connect all registered folders and block
                (this is what 24/7 workers / nodes run; vfs overlay optional)

The share registry is a small JSON file, so `serve` is declarative and
restart-safe. Live, in-session control (with a VFS overlay) is done by calling
VFSSyncManager(session.vfs).connect(...) directly from the icli.

CLI:
    tb -m vfs_remote_sync --action connect --vfs-path /global \\
        --local-dir /srv/global --token <share_token>
    tb -m vfs_remote_sync --action list
    tb -m vfs_remote_sync --action serve
"""
import asyncio
import json
import os

# Engine wrapper — works from the repo or from a standalone copy.
try:
    from toolboxv2.mods.CloudM.LiveSync.vfs_adapter import VFSSyncManager
except Exception:  # pragma: no cover - sandbox/standalone fallback
    from livesync_pkg.vfs_adapter import VFSSyncManager

try:
    from toolboxv2 import App, Result
except Exception:  # pragma: no cover
    App = object

    class Result:
        @staticmethod
        def ok(data=None, info=""):
            return {"ok": True, "data": data, "info": info}

        @staticmethod
        def error(info=""):
            return {"ok": False, "error": info}


NAME = "vfs_remote_sync"
ICON = "sync"
AUTH = False


# ── share registry (declarative JSON) ──

def _config_path(app=None) -> str:
    base = getattr(app, "data_dir", None) or os.path.expanduser("~")
    return os.path.join(base, "vfs_remote_shares.json")


def _load(app=None) -> list:
    path = _config_path(app)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return []


def _save(shares: list, app=None) -> None:
    path = _config_path(app)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(shares, fh, indent=2)
    os.replace(tmp, path)


# ── per-process manager (so connect/list/disconnect cohere within one run) ──

def _get_manager(app=None, vfs=None) -> VFSSyncManager:
    mgr = getattr(app, "_vfs_sync_mgr", None)
    if mgr is None:
        mgr = VFSSyncManager(vfs)
        if app is not None:
            try:
                app._vfs_sync_mgr = mgr
            except Exception:
                pass
    return mgr


# ── actions ──

async def _connect(app, vfs_path, local_dir, token, readonly):
    if not (vfs_path and local_dir and token):
        return Result.error("connect needs --vfs-path, --local-dir and --token")
    shares = _load(app)
    shares = [s for s in shares if s.get("vfs_path") != vfs_path]
    shares.append({
        "vfs_path": vfs_path, "local_dir": local_dir,
        "token": token, "readonly": bool(readonly),
    })
    _save(shares, app)
    # Declarative: a short-lived CLI process only registers. `serve` starts the
    # live sync; in-session live connect is done via VFSSyncManager directly.
    return Result.ok(info=f"registered {vfs_path} ← {local_dir} (run `serve` to sync)")


async def _disconnect(app, vfs_path):
    if not vfs_path:
        return Result.error("disconnect needs --vfs-path")
    shares = [s for s in _load(app) if s.get("vfs_path") != vfs_path]
    _save(shares, app)
    mgr = _get_manager(app)
    await mgr.disconnect(vfs_path)
    return Result.ok(info=f"disconnected {vfs_path}")


def _status(app):
    shares = _load(app)
    mgr = getattr(app, "_vfs_sync_mgr", None)
    live = mgr.list() if mgr else {}
    rows = []
    for s in shares:
        p = s["vfs_path"]
        rows.append({
            "vfs_path": p,
            "local_dir": s["local_dir"],
            "on_disk": os.path.isdir(s["local_dir"]),
            "live": p in live,
            "readonly": s.get("readonly", False),
        })
    print(f"[vfs_remote_sync] {len(rows)} share(s):")
    for r in rows:
        flag = "●" if r["live"] else ("○" if r["on_disk"] else "✗")
        print(f"  {flag} {r['vfs_path']:<20} ← {r['local_dir']}"
              + ("  [ro]" if r["readonly"] else ""))
    return Result.ok(data=rows)


async def _serve(app):
    """Mini-service: connect every registered folder and block forever."""
    shares = _load(app)
    if not shares:
        print("[vfs_remote_sync] no shares registered — nothing to serve")
        return Result.ok(info="no shares")
    mgr = _get_manager(app)
    for s in shares:
        try:
            await mgr.connect(
                s["vfs_path"], s["local_dir"],
                token=s["token"], readonly=s.get("readonly", False),
            )
            print(f"[vfs_remote_sync] serving {s['vfs_path']} ← {s['local_dir']}")
        except Exception as e:
            print(f"[vfs_remote_sync] connect failed {s['vfs_path']}: {e}")
    print(f"[vfs_remote_sync] mini-service up — {len(mgr.list())} folder(s) live. Ctrl-C to stop.")
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await mgr.stop_all()
        print("[vfs_remote_sync] stopped")
    return Result.ok(info="served")


# ── flow entry point ──

async def run(app=None, args=None, action="status", vfs_path="",
              local_dir="", token="", readonly=False, **kwargs):
    action = (action or "status").lower()
    if action == "connect":
        return await _connect(app, vfs_path, local_dir, token, readonly)
    if action == "disconnect":
        return await _disconnect(app, vfs_path)
    if action in ("list", "status"):
        return _status(app)
    if action == "serve":
        return await _serve(app)
    return Result.error(f"unknown action '{action}' "
                        "(connect|disconnect|list|status|serve)")
