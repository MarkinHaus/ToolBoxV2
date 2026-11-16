"""WP1 S1: account data sync local <-> cloud (SimpleCore).

Decision table per data key (settings, mod_data, moods, ...):
  1 local only              -> push
  2 remote only             -> pull
  3 same version + ts       -> noop
  4 remote newer            -> pull
  5 local newer             -> push
  6 same ts, diff version   -> cycle suspected: remote wins,
                              local value backed up to a file first

Measured response forms (S1 protocol, UserDataAPI.py 486-560/744-798,
blobs.py 697+): payloads are plain dicts with optional "version" and
"updated_at" fields; both stores are reduced to list_keys/read/write
here so the engine stays independent of transport (http vs direct call).
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class EntryMeta:
    version: int = 0
    updated_at: float = 0.0


@dataclass
class SyncDecision:
    action: str          # push | pull | noop | conflict-remote-wins
    reason: str = ""


class SyncStore(Protocol):
    async def list_keys(self) -> list: ...
    async def read(self, key: str): ...
    async def write(self, key: str, payload) -> None: ...


def _meta(payload) -> Optional[EntryMeta]:
    if payload is None:
        return None
    if isinstance(payload, EntryMeta):
        return payload
    d = payload if isinstance(payload, dict) else {}
    return EntryMeta(
        version=int(d.get("version", 0) or 0),
        updated_at=float(d.get("updated_at", 0.0) or 0.0),
    )


def decide(key: str, local: Optional[EntryMeta], remote: Optional[EntryMeta],
           last_direction: Optional[str] = None) -> SyncDecision:
    """Pick the sync action for one key. Pure, no IO."""
    if local is not None and remote is None:
        return SyncDecision("push", "remote-missing")
    if remote is not None and local is None:
        return SyncDecision("pull", "local-missing")
    if local.version == remote.version and local.updated_at == remote.updated_at:
        return SyncDecision("noop", "equal")
    if local.updated_at > remote.updated_at:
        return SyncDecision("push", "local-newer")
    if remote.updated_at > local.updated_at:
        return SyncDecision("pull", "remote-newer")
    # same timestamp, different versions: values were rewritten in place on
    # both sides (or clocks equal) - treat as cycle suspicion, remote wins
    return SyncDecision("conflict-remote-wins", "cycle-suspected")


async def sync_account_data(app, local_store: SyncStore, remote_store: SyncStore,
                            state_file: Optional[Path] = None,
                            last_directions: Optional[dict] = None) -> dict:
    """Run the decision table over the union of both stores.

    Writes a local backup next to state_file for every conflict before the
    remote value overwrites the local one. Persists last directions and
    run time into state_file (JSON).
    """
    result = {"pushed": 0, "pulled": 0, "noop": 0, "conflicts": []}
    directions = dict(last_directions or {})

    lkeys = set(await local_store.list_keys())
    rkeys = set(await remote_store.list_keys())

    for key in sorted(lkeys | rkeys):
        lp = await local_store.read(key)
        rp = await remote_store.read(key)
        d = decide(key, _meta(lp), _meta(rp), directions.get(key))

        if d.action == "push":
            await remote_store.write(key, lp)
            result["pushed"] += 1
            directions[key] = "push"
        elif d.action == "pull":
            await local_store.write(key, rp)
            result["pulled"] += 1
            directions[key] = "pull"
        elif d.action == "noop":
            result["noop"] += 1
        else:  # conflict-remote-wins
            if lp is not None and state_file is not None:
                ts = time.strftime("%Y%m%d_%H%M%S")
                backup = state_file.parent / f"backup_{ts}.{key}.json"
                backup.write_text(json.dumps(lp, default=str), encoding="utf-8")
            await local_store.write(key, rp)
            result["conflicts"].append(key)
            directions[key] = "pull"

    if state_file is not None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(
            {"last_directions": directions, "last_run": time.time()},
            indent=2), encoding="utf-8")
    return result
