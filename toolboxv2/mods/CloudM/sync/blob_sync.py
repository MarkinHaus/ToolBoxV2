"""WP1 S2: blob sync local SQLite <-> cloud (MinIO).

Binary-safe: blob payloads are bytes end to end, never JSON. Uses the
measured BlobManager forms (blobs.py 697-975): list_blobs -> dicts with
blob_id/size/updated_at/sync_status, read_blob -> bytes, write via
create/update (data, encrypt) -> blob_id. Direction decisions reuse the
S1 table (decide/EntryMeta) with updated_at as the clock; equal ts+size
counts as equal (blobs have no version counter locally).
"""
import json
import time
from pathlib import Path
from typing import Optional, Protocol

from .device_sync import EntryMeta, decide


class BlobStore(Protocol):
    async def list_blobs(self, prefix: str = "") -> list: ...
    async def read_blob(self, blob_id: str) -> Optional[bytes]: ...
    async def write_blob(self, blob_id: str, data: bytes, encrypted: bool = True) -> str: ...


def _ts(entries):
    return {e["blob_id"]: float(e.get("updated_at", 0.0) or 0.0) for e in entries}


async def sync_blobs(app, local_store: BlobStore, remote_store: BlobStore,
                     state_file: Optional[Path] = None,
                     last_directions: Optional[dict] = None) -> dict:
    """Sync blobs by timestamp; four WP1 cases, remote wins ties-with-
    backup handled via decide(). Binary payload integrity is preserved
    (bytes in, bytes out, no serialization)."""
    result = {"pushed": [], "pulled": [], "noop": [], "conflicts": []}
    directions = dict(last_directions or {})

    lentries = await local_store.list_blobs()
    rentries = await remote_store.list_blobs()
    lts, rts = _ts(lentries), _ts(rentries)

    for blob_id in sorted(set(lts) | set(rts)):
        lmeta = EntryMeta(version=0, updated_at=lts.get(blob_id, 0.0)) if blob_id in lts else None
        rmeta = EntryMeta(version=0, updated_at=rts.get(blob_id, 0.0)) if blob_id in rts else None
        d = decide(blob_id, lmeta, rmeta, directions.get(blob_id))

        if d.action == "push":
            data = await local_store.read_blob(blob_id)
            await remote_store.write_blob(blob_id, data)
            result["pushed"].append(blob_id)
            directions[blob_id] = "push"
        elif d.action == "pull":
            data = await remote_store.read_blob(blob_id)
            await local_store.write_blob(blob_id, data)
            result["pulled"].append(blob_id)
            directions[blob_id] = "pull"
        elif d.action == "noop":
            result["noop"].append(blob_id)
        else:  # conflict-remote-wins: bytes have no in-place merge, keep
            # the local copy as .blob backup, then pull
            data = await local_store.read_blob(blob_id)
            if data is not None and state_file is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                (state_file.parent / f"backup_{ts}.{blob_id}.blob").write_bytes(data)
            rdata = await remote_store.read_blob(blob_id)
            await local_store.write_blob(blob_id, rdata)
            result["conflicts"].append(blob_id)
            directions[blob_id] = "pull"

    if state_file is not None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(
            {"last_directions": directions, "last_run": time.time()},
            indent=2), encoding="utf-8")
    return result
