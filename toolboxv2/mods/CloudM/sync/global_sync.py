"""WP1 S3: global ISAA folder synced as blobs + one entry point.

Folder from F22 (power_vfs NOT imported - path built directly:
<data_dir>/Agents/VFS/global). Files travel as blobs "isaa/global/<rel>"
via the S2 engine, back as files, only when mtime or size changed.
Unsafe paths ("..", absolute, backslashes) and files > 50 MB are
skipped and logged; deletions are never propagated.

sync_all(app): export (files -> blobs) -> import (blobs -> files) ->
account data (S1 engine). Returns partial results + finished_at.
Without a device account the whole run is {"skipped": True}.
"""
import json
import time
from pathlib import Path
from typing import Optional

from .blob_sync import BlobStore, sync_blobs
from .device_sync import SyncStore, sync_account_data

GLOBAL_REL = Path("Agents") / "VFS" / "global"
BLOB_PREFIX = "isaa/global/"
MAX_FILE_BYTES = 50 * 1024 * 1024


def _safe_rel(rel: str):
    """Return a normalized relative path or None if unsafe."""
    if not rel or rel.startswith(("/", "\\")) or ":" in rel:
        return None
    parts = rel.replace("\\", "/").split("/")
    if any(p in ("..", "") for p in parts):
        return None
    return Path(*parts)


def _scan(global_dir: Path):
    """Yield (relpath_posix, size, mtime) for syncable files."""
    if not global_dir.is_dir():
        return
    for p in sorted(global_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(global_dir).as_posix()
        safe = _safe_rel(rel)
        if safe is None:
            continue
        size = p.stat().st_size
        if size > MAX_FILE_BYTES:
            print(f"[global_sync] skip too large: {rel} ({size} bytes)")
            continue
        yield rel, size, p.stat().st_mtime


def _account(app) -> Optional[dict]:
    try:
        from toolboxv2.mods.CloudM.auth.device_account import read_device_owner
        return read_device_owner(app)
    except Exception:
        return None


async def sync_all(app, local_blobs: BlobStore, remote_blobs: BlobStore,
                   local_account: Optional[SyncStore] = None,
                   remote_account: Optional[SyncStore] = None,
                   state_file: Optional[Path] = None,
                   last_directions: Optional[dict] = None) -> dict:
    """One full sync run: global folder blobs + account data.
    Never raises - errors land in the result dict."""
    started = time.time()
    out = {"skipped": False}

    if _account(app) is None:
        out["skipped"] = True
        out["finished_at"] = started
        return out

    data_dir = Path(getattr(app, "data_dir", "."))
    global_dir = data_dir / GLOBAL_REL

    # --- export: files -> blobs (only changed mtime/size) ---
    class _FsView:
        pass

    export = {"pushed": [], "skipped": []}
    lentries = await local_blobs.list_blobs(BLOB_PREFIX)
    lmeta = {e["blob_id"]: (float(e.get("updated_at", 0.0) or 0.0),
                            int(e.get("size", 0) or 0)) for e in lentries}
    for rel, size, mtime in _scan(global_dir):
        blob_id = BLOB_PREFIX + rel
        cur = lmeta.get(blob_id)
        local_p = global_dir / rel
        # ponytail: stores without timestamps (ts==0) compare size only;
        # upgrade when BlobManager exposes per-write mtimes everywhere
        if cur is not None and cur[1] == size and (cur[0] == 0.0 or abs(cur[0] - mtime) < 1.0):
            continue  # unchanged
        data = local_p.read_bytes()
        await local_blobs.write_blob(blob_id, data)
        export["pushed"].append(blob_id)
    out["export"] = export

    # --- blobs local <-> remote via S2 engine (blobs only, no JSON path) ---
    out["blobs"] = await sync_blobs(
        app, local_blobs, remote_blobs,
        state_file=state_file,
        last_directions=(last_directions or {}).get("blobs"))

    # --- import: blobs -> files ---
    import_res = {"written": 0, "skipped": []}
    rentries = await remote_blobs.list_blobs(BLOB_PREFIX)
    known = {e["blob_id"]: (float(e.get("updated_at", 0.0) or 0.0),
                            int(e.get("size", 0) or 0)) for e in rentries}
    for blob_id in sorted(known):
        rel = blob_id[len(BLOB_PREFIX):]
        safe = _safe_rel(rel)
        if safe is None:
            print(f"[global_sync] skip unsafe blob path: {blob_id}")
            import_res["skipped"].append(blob_id)
            continue
        target = global_dir / safe
        ts_known, size = known[blob_id]
        if target.is_file() and target.stat().st_size == size:
            # ponytail: ts==0 (store without timestamps) -> size match is enough;
            # tighten when remote entries always carry updated_at
            if ts_known == 0.0 or abs(target.stat().st_mtime - ts_known) < 1.0:
                continue
        data = await remote_blobs.read_blob(blob_id)
        if data is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        import_res["written"] += 1
    out["import"] = import_res

    # --- account data via S1 engine (optional stores) ---
    if local_account is not None and remote_account is not None:
        out["account"] = await sync_account_data(
            app, local_account, remote_account,
            state_file=state_file,
            last_directions=(last_directions or {}).get("account"))

    out["finished_at"] = time.time()
    return out

def run_sync_once(app) -> dict:
    """Trigger entry: run sync_all with the best available adapters.

    Remote adapter stays None until a remote base is configured - in
    that case the run reports skipped(no-remote) instead of touching
    anything. Called from the POST route, the 10-min loop and
    link_account (each may pass a different app instance).
    """
    import asyncio

    async def _run():
        local = None
        remote = None
        try:
            from toolboxv2.utils.extras.blobs import BlobManager
            local = BlobManager(app)
        except Exception:
            local = None
        import os
        if not os.environ.get("TOOLBOXV2_REMOTE_BASE"):
            remote = None
        if remote is None:
            # engine + local export ready, remote leg not configured
            out = {"skipped": True, "reason": "no-remote",
                   "finished_at": time.time()}
            return out
        return await sync_all(app, local, remote,
                              state_file=Path(getattr(app, "data_dir", ".")) / "_global_sync.json")

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(asyncio.run, _run).result(timeout=120)
        return loop.run_until_complete(_run())
    except RuntimeError:
        return asyncio.run(_run())
