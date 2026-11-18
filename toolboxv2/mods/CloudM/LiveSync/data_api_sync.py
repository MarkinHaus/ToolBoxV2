"""LiveSync folder sync over the CloudM Data-API (WP1 Auftrag 08, Teil 9).

End-to-end encrypted: the server only ever sees {"v": 1, "enc": <b64>} JSON
payloads. Plaintext, file names aside, never leaves the device unencrypted.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
import base64
import logging
import os

from toolboxv2.mods.CloudM.LiveSync.minio_helper import TransferError
from toolboxv2.mods.CloudM.LiveSync.crypto import (
    encrypt_bytes,
    decrypt_bytes,
    compute_checksum,
    compute_checksum_file,
)
from toolboxv2.mods.CloudM.LiveSync.conflict import (
    save_conflict_copy,
    is_conflict_copy,
    move_to_sync_trash,
)
from toolboxv2.mods.CloudM.LiveSync.protocol import should_ignore

logger = logging.getLogger(__name__)

PREFIX_GLOBAL = "livesync/global/"
PREFIX_ACCOUNT = "livesync/account/"
MAX_BYTES = 50 * 1024 * 1024
# Deleting more than this share of the known (synced) entries in one run is
# refused - a vanished folder or an empty server listing must not wipe a device.
# Fewer than MASS_DELETE_MIN deletions are never "mass" (else the last file of a
# small folder could never be deleted).
MASS_DELETE_RATIO = 0.5
MASS_DELETE_MIN = 3
PAYLOAD_V = 1


def _payload(ciphertext: bytes) -> dict:
    """The only payload form that ever goes to the server."""
    return {"v": PAYLOAD_V, "enc": base64.b64encode(ciphertext).decode()}


class DataApiTransport:
    """Talks to CloudM.UserDataAPI over the authenticated session.

    HTTPS is mandatory: either the base URL is https:// or the host is a
    loopback address. Anything else fails before a single byte is sent.
    """

    def __init__(self, app):
        self.app = app
        base = (getattr(getattr(app, "session", None), "base", None) or "")
        host = base.split("//", 1)[-1].split("/", 1)[0].split(":")[0].lower()
        if not (base.startswith("https://") or host in ("127.0.0.1", "localhost")):
            raise TransferError("insecure base")

    async def _call(self, fn: str, **kw):
        from toolboxv2 import Result
        from toolboxv2.utils.system.types import ToolBoxError
        res = await self.app.run_http("CloudM.UserDataAPI", fn, method="POST", **kw)
        if not isinstance(res, dict):
            raise TransferError(f"{fn} failed")
        r = Result.result_from_dict(
            error=res.get("error", ToolBoxError.none),
            result=res.get("result", {}),
            info=res.get("info", {}),
            origin=res.get("origin"),
        )
        if r.is_error():
            info = res.get("info", {})
            if info.get("exec_code") == 401:
                # UserDataAPI: "Nicht authentifiziert" - Anmeldung abgelaufen
                raise TransferError("reauth_required")
            if info.get("help_text") == "Session fetch failed":
                # App.run_http: kein Netz zum Server
                raise TransferError("offline")
            raise TransferError(f"{fn} failed")
        return r.get()

    async def list(self, prefix: str) -> Dict[str, Tuple[int, float]]:
        items = await self._call("list_data", prefix=prefix, scope="private")
        out = {}
        for it in items or []:
            out[it["path"]] = (it.get("size", 0), it.get("updated_at", 0.0))
        return out

    async def put(self, path: str, ciphertext: bytes) -> None:
        await self._call("set_data", path=path, scope="private", data=_payload(ciphertext))

    async def delete(self, path: str) -> None:
        await self._call("delete_data", path=path, scope="private")

    async def get(self, path: str) -> bytes:
        d = await self._call("get_data", path=path, scope="private")
        if not isinstance(d, dict) or d.get("v") != 1 or "enc" not in d:
            raise TransferError("bad payload")
        return base64.b64decode(d["enc"])


def _safe_rel(rel: str) -> Optional[str]:
    """Normalize a relative path; None means unsafe - never write it."""
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        return None
    if ":" in rel:
        return None
    parts = rel.replace("\\", "/").split("/")
    if any(p in ("", "..") for p in parts):
        return None
    return "/".join(parts)


def _inside(root: Path, path: Path) -> bool:
    """True if path, with every symlink/junction resolved, lies below root.

    root must already be resolved. Path.rglob follows junctions on Windows and
    lists file symlinks everywhere, so the walker alone cannot guarantee this.
    """
    return path.resolve().is_relative_to(root)


def _atomic_write(target: Path, data: bytes) -> None:
    tmp = target.with_name(target.name + ".livesync-tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(str(tmp), str(target))


async def sync_folder(transport, local_dir: Path, prefix: str, index, key: str,
                      device_id: str) -> dict:
    """Sync one folder bi-directionally against prefix over the transport.

    The LocalIndex is exclusively owned by this function (own DB file):
      checksum  = compute_checksum of the PLAINTEXT at the last common state
      mtime     = remote updated_at at the last common state
      size      = remote size
      remote_key= remote path
      sync_state= "synced"
    Deletions are propagated for files whose last common state is in the index:
    deleted on one side and unchanged on the other -> deleted on the other side
    (locally via .sync-trash). Changed on one side wins over deleted on the
    other. More than MASS_DELETE_RATIO of the known entries -> nothing is
    deleted and out["error"] = "mass_delete_blocked".
    """
    out = {"pushed": [], "pulled": [], "conflicts": [], "skipped": [],
           "deleted_local": [], "deleted_remote": [], "unchanged": 0}
    local_dir = Path(local_dir)
    root = local_dir.resolve()

    remote_raw = await transport.list(prefix)
    remote: Dict[str, Tuple[str, int, float]] = {}
    for path, (size, updated_at) in remote_raw.items():
        rel = _safe_rel(path[len(prefix):])
        if rel is None:
            out["skipped"].append(path)
            logger.warning("data_api_sync: unsafe remote path %r skipped", path)
            continue
        remote[rel] = (path, size, updated_at)

    local: Dict[str, Path] = {}
    held = set()  # exist locally but are not synced: never read as "deleted"
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        if should_ignore(rel) or is_conflict_copy(rel):
            continue
        if _safe_rel(rel) is None:
            held.add(rel)
            out["skipped"].append(rel)
            logger.warning("data_api_sync: %r has a name that cannot be synced, skipped", rel)
            continue
        if not _inside(root, p):
            held.add(rel)
            out["skipped"].append(rel)
            logger.warning("data_api_sync: %r resolves outside the folder, skipped", rel)
            continue
        if p.stat().st_size > MAX_BYTES:
            held.add(rel)
            out["skipped"].append(rel)
            logger.warning("data_api_sync: %r exceeds MAX_BYTES, skipped", rel)
            continue
        local[rel] = p

    touched = []  # (rel, plaintext_checksum) pairs that need an index update

    # Deletions: only from the last common state (index), only now that list()
    # succeeded in this run, and only if the other side did not change.
    known = {f["rel_path"]: f for f in await index.get_all_files()
             if f["sync_state"] == "synced"}
    del_remote, del_local = set(), set()
    for rel, base in known.items():
        if rel in held or not _inside(root, local_dir / rel):
            continue
        lpath, rmeta = local.get(rel), remote.get(rel)
        if lpath is None and rmeta is not None and rmeta[2] == base["mtime"]:
            del_remote.add(rel)
        elif (lpath is not None and rmeta is None
              and compute_checksum_file(str(lpath)) == base["checksum"]):
            del_local.add(rel)
    planned = len(del_remote) + len(del_local)
    if planned >= MASS_DELETE_MIN and planned > len(known) * MASS_DELETE_RATIO:
        out["error"] = "mass_delete_blocked"
        out["blocked"] = sorted(del_remote | del_local)
        logger.warning("data_api_sync: %d of %d known entries would be deleted, "
                       "nothing deleted", planned, len(known))
        del_remote, del_local = set(), set()

    for rel in sorted(set(remote) | set(local) | set(known)):
        if not _inside(root, local_dir / rel):
            # a remote entry below a link that points out of the folder
            if rel not in out["skipped"]:
                out["skipped"].append(rel)
                logger.warning("data_api_sync: %r resolves outside the folder, skipped", rel)
            continue
        base = await index.get_file(rel)
        lpath = local.get(rel)
        rmeta = remote.get(rel)

        l_changed = lpath is not None and (
            base is None
            or compute_checksum_file(str(lpath)) != base["checksum"]
        )
        r_changed = rmeta is not None and (
            base is None or rmeta[2] != base["mtime"]
        )

        if rel in del_remote:
            await transport.delete(rmeta[0])
            await index.delete_file(rel)
            out["deleted_remote"].append(rel)
            continue
        if rel in del_local:
            move_to_sync_trash(str(local_dir), rel)
            await index.delete_file(rel)
            out["deleted_local"].append(rel)
            continue

        if rmeta is None and lpath is None:
            # Gone on both sides (or locally deleted and remote already gone).
            if base is not None:
                await index.delete_file(rel)
            continue

        if l_changed and not r_changed:
            # Case 1: push
            plaintext = lpath.read_bytes()
            await transport.put(prefix + rel, encrypt_bytes(plaintext, key))
            out["pushed"].append(rel)
            touched.append((rel, compute_checksum(plaintext)))
            continue

        if not l_changed and r_changed:
            # Case 2 / 7: pull
            payload = await transport.get(rmeta[0])
            try:
                plaintext = decrypt_bytes(payload, key)
            except Exception:
                out["skipped"].append(rel)
                logger.warning("data_api_sync: decrypt failed for %r", rel)
                continue
            if lpath is not None and compute_checksum(plaintext) == compute_checksum_file(str(lpath)):
                # Same plaintext on both sides - only refresh the index.
                touched.append((rel, compute_checksum(plaintext)))
                out["unchanged"] += 1
                continue
            _atomic_write(local_dir / rel, plaintext)
            out["pulled"].append(rel)
            touched.append((rel, compute_checksum(plaintext)))
            continue

        if l_changed and r_changed:
            payload = await transport.get(rmeta[0])
            try:
                plaintext = decrypt_bytes(payload, key)
            except Exception:
                out["skipped"].append(rel)
                logger.warning("data_api_sync: decrypt failed for %r", rel)
                continue
            if compute_checksum(plaintext) == compute_checksum_file(str(lpath)):
                # Case 4: same plaintext - only refresh the index.
                touched.append((rel, compute_checksum(plaintext)))
                out["unchanged"] += 1
            else:
                # Case 5: real conflict - keep local aside, remote wins.
                save_conflict_copy(str(local_dir), rel, device_id)
                _atomic_write(local_dir / rel, plaintext)
                out["conflicts"].append(rel)
                touched.append((rel, compute_checksum(plaintext)))
            continue

        # Case 3: nothing changed on either side.
        out["unchanged"] += 1

    # One extra list() after all pushes: pick up remote size/updated_at.
    if touched:
        final = await transport.list(prefix)
        for rel, checksum in touched:
            meta = final.get(prefix + rel)
            if meta is None:
                continue
            await index.upsert_file(
                rel,
                mtime=meta[1],
                size=meta[0],
                checksum=checksum,
                sync_state="synced",
                remote_key=prefix + rel,
            )
    return out


# ---------------------------------------------------------------------------
# sync_all orchestration (Teil 10)
# ---------------------------------------------------------------------------
_RUN_LOCK = None
_LAST = {}


def _run_lock():
    global _RUN_LOCK
    import threading
    if _RUN_LOCK is None:
        _RUN_LOCK = threading.Lock()
    return _RUN_LOCK


async def ensure_sync_key(app, user_id: str) -> str:
    """Fetch (or reuse) the account sync key. The key is never logged."""
    from toolboxv2 import Result
    user = await _load_user(app, user_id)
    if user is not None and getattr(user, "user_pass_sync", ""):
        return user.user_pass_sync
    res = await app.run_http("CloudM.Auth", "get_sync_key", method="POST")
    if not isinstance(res, dict):
        raise TransferError("offline")
    from toolboxv2.utils.system.types import ToolBoxError
    r = Result.result_from_dict(
        error=res.get("error", ToolBoxError.none),
        result=res.get("result", {}),
        info=res.get("info", {}),
        origin=res.get("origin"),
    )
    if r.is_error():
        if res.get("info", {}).get("exec_code") == 401:
            raise TransferError("reauth_required")
        raise TransferError("offline")
    key = (r.get() or {}).get("key", "")
    if not key:
        raise TransferError("reauth_required")
    if user is not None:
        user.user_pass_sync = key
        await _save_user(app, user)
    return key


def _fill_lazy():
    """Resolve collaborators once into module globals (patchable in tests)."""
    import socket
    import time
    from toolboxv2.mods.CloudM.auth.device_account import read_device_owner
    from toolboxv2.mods.CloudM.auth.user_store import _load_user, _save_user
    from toolboxv2.mods.CloudM.LiveSync.account_bridge import (
        export_account, import_account)
    from toolboxv2.mods.CloudM.LiveSync.conflict import sanitize_device_id
    from toolboxv2.mods.CloudM.LiveSync.index import LocalIndex
    globals().update(
        socket=socket, time=time,
        read_device_owner=read_device_owner,
        _load_user=_load_user, _save_user=_save_user,
        export_account=export_account, import_account=import_account,
        sanitize_device_id=sanitize_device_id, LocalIndex=LocalIndex,
    )


async def sync_all(app) -> dict:
    """One full account + global sync run. Never raises."""
    from pathlib import Path

    if not _run_lock().acquire(blocking=False):
        return {"skipped": True, "reason": "running"}
    try:
        owner = read_device_owner(app)
        if not owner or not owner.get("user_id"):
            result = {"skipped": True, "reason": "no-account"}
            _LAST.clear()
            _LAST.update(result)
            return result
        try:
            key = await ensure_sync_key(app, owner["user_id"])
            transport = DataApiTransport(app)
            data = Path(app.data_dir)
            device_id = sanitize_device_id(socket.gethostname())
            (data / "account_sync").mkdir(parents=True, exist_ok=True)
            (data / "Agents" / "VFS" / "global").mkdir(parents=True,
                                                       exist_ok=True)
            out = {}
            # --- global folder ---
            idx = LocalIndex(str(data / "account_sync" / "global_index.db"))
            await idx.init()
            try:
                out["global"] = await sync_folder(
                    transport, data / "Agents" / "VFS" / "global",
                    PREFIX_GLOBAL, idx, key, device_id)
            finally:
                await idx.close()
            # --- account mirror ---
            user = await _load_user(app, owner["user_id"])
            if user is not None:
                mirror = data / "account_sync" / "mirror"
                export_account(user, mirror)
                idx2 = LocalIndex(str(data / "account_sync" / "account_index.db"))
                await idx2.init()
                try:
                    out["account"] = await sync_folder(
                        transport, mirror, PREFIX_ACCOUNT, idx2, key, device_id)
                finally:
                    await idx2.close()
                if import_account(user, mirror):
                    await _save_user(app, user)
            result = dict(out)
            result["finished_at"] = time.time()
        except TransferError as e:
            result = {"error": str(e), "finished_at": time.time()}
        except Exception as e:  # never raise, never leak key material
            result = {"error": type(e).__name__, "finished_at": time.time()}
        _LAST.clear()
        _LAST.update(result)
        return result
    finally:
        _run_lock().release()


def run_sync_in_background(app) -> None:
    """Fire-and-forget sync run in a daemon thread."""
    import asyncio
    import threading

    def _bg():
        try:
            asyncio.run(sync_all(app))
        except Exception:
            pass

    threading.Thread(target=_bg, daemon=True, name="account-sync-run").start()


_fill_lazy()
