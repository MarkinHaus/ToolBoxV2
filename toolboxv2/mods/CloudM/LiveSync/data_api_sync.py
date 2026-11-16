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
)
from toolboxv2.mods.CloudM.LiveSync.protocol import should_ignore

logger = logging.getLogger(__name__)

PREFIX_GLOBAL = "livesync/global/"
PREFIX_ACCOUNT = "livesync/account/"
MAX_BYTES = 50 * 1024 * 1024
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
        res = await self.app.run_http("CloudM.UserDataAPI", fn, method="POST", **kw)
        if not isinstance(res, dict):
            raise TransferError(f"{fn} failed")
        r = Result.result_from_dict(**res)
        if r.is_error():
            msg = str(res.get("info", ""))
            if "Session fetch failed" in msg:
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
    No deletions are ever propagated.
    """
    out = {"pushed": [], "pulled": [], "conflicts": [], "skipped": [], "unchanged": 0}
    local_dir = Path(local_dir)

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
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        if should_ignore(rel) or is_conflict_copy(rel) or _safe_rel(rel) is None:
            continue
        if p.stat().st_size > MAX_BYTES:
            out["skipped"].append(rel)
            logger.warning("data_api_sync: %r exceeds MAX_BYTES, skipped", rel)
            continue
        local[rel] = p

    touched = []  # (rel, plaintext_checksum) pairs that need an index update

    for rel in sorted(set(remote) | set(local)):
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

        if rmeta is None and lpath is None:
            # Case 6: locally deleted, remote unchanged or gone -> nothing.
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
