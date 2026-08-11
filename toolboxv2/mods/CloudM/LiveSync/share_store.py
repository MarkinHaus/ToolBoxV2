"""
LiveSync Share Store
====================
Persistent, encrypted registry of the shares this node hosts or has joined.

Why it exists
-------------
A share record holds the share token, and the token holds the AES file key.
That has to survive a restart: a server running in ``replica`` mode needs the
token to start its own SyncClient after a crash, without anyone typing it in
again. It must not survive in a form anyone can read.

So the file is AES-256-GCM encrypted with a key derived from this node's TB
device key. Same trust boundary as ``.info/device.enc`` itself: whoever can
read the device key can read the shares, and nobody else. The plaintext never
touches the process environment or the command line, so it does not show up in
``ps aux`` or ``/proc/<pid>/environ``.

Location: ``<DEVICE_KEY_DIR or tb_root_dir/.info>/livesync_shares.enc``,
mode 0600.

Record shape:
    {
      "share_id":   str,
      "vault_path": str,
      "token":      str,   # signed v4 share token
      "mode":       str,   # "relay" | "replica"
      "ws_port":    int,
      "created_at": float,
    }
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .crypto import encrypt_bytes, decrypt_bytes

logger = logging.getLogger("LiveSync")

STORE_FILENAME = "livesync_shares.enc"

VALID_MODES = ("relay", "replica")


def store_dir() -> Path:
    """
    Directory holding the encrypted share store.

    Uses the same directory as the device key, so the store lives and dies
    with the key that protects it. ``DEVICE_KEY_DIR`` overrides both.
    """
    override = os.getenv("DEVICE_KEY_DIR")
    if override:
        return Path(override)
    try:
        from toolboxv2 import tb_root_dir
        return Path(str(tb_root_dir)) / ".info"
    except Exception:
        return Path.home() / ".tb_livesync"


def store_path() -> Path:
    return store_dir() / STORE_FILENAME


def _store_key() -> str:
    """
    Base64 AES-256 key derived from the TB device key.

    Derived rather than used directly: the device key is a Fernet key of its
    own format, and crypto.encrypt_bytes expects 32 raw bytes, base64-encoded.

    Raises:
        RuntimeError: if the device key is unavailable.
    """
    try:
        from toolboxv2.utils.security.cryp import DEVICE_KEY
        raw = DEVICE_KEY()
    except Exception as exc:
        raise RuntimeError(f"share store needs the TB device key: {exc}") from exc

    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw:
        raise RuntimeError("share store needs the TB device key: empty key")

    return base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")


def load_shares() -> Dict[str, Dict[str, Any]]:
    """
    Read all share records.

    Returns an empty dict when the store does not exist yet. A store that
    exists but cannot be decrypted is an error worth seeing, not something to
    paper over, so it is logged and treated as empty rather than crashing the
    caller mid-startup.
    """
    path = store_path()
    if not path.exists():
        return {}
    try:
        payload = decrypt_bytes(path.read_bytes(), _store_key())
        data = json.loads(payload.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error(
            f"[LiveSync] Share store unreadable ({path}): {exc}. "
            "Wrong TB_R_KEY, or the file was written by another node."
        )
        return {}


def _write_shares(shares: Dict[str, Dict[str, Any]]) -> None:
    """Encrypt and write the whole store atomically, owner-readable only."""
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = encrypt_bytes(
        json.dumps(shares, sort_keys=True).encode("utf-8"), _store_key())

    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def save_share(
    share_id: str,
    vault_path: str,
    token: str,
    mode: str = "relay",
    ws_port: int = 8765,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Insert or replace one share record.

    Raises:
        ValueError: on an empty share_id or an unknown mode.
    """
    import time

    if not share_id:
        raise ValueError("share_id required")
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    shares = load_shares()
    record = {
        "share_id": share_id,
        "vault_path": str(vault_path),
        "token": token,
        "mode": mode,
        "ws_port": int(ws_port),
        "created_at": created_at or shares.get(share_id, {}).get(
            "created_at", time.time()),
    }
    shares[share_id] = record
    _write_shares(shares)
    return record


def get_share(share_id: str) -> Optional[Dict[str, Any]]:
    """Return one share record, or None."""
    return load_shares().get(share_id)


def delete_share(share_id: str) -> bool:
    """Remove one share record. True if it was there."""
    shares = load_shares()
    if share_id not in shares:
        return False
    del shares[share_id]
    _write_shares(shares)
    return True


def list_share_records() -> list:
    """All share records, oldest first."""
    return sorted(load_shares().values(), key=lambda r: r.get("created_at", 0))
