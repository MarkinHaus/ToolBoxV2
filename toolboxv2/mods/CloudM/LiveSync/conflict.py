"""
LiveSync Conflict Resolution
=============================

One strategy for every file type: the server version wins on disk, and the
local version is kept next to it as

    <name>.conflict.<device_id>.<timestamp>.<ext>

Nothing is merged and nothing is thrown away. That is deliberate — it is the
only rule that works identically for Markdown, source code, PDFs, images and
spreadsheets, and it never produces a file the owning application cannot open.
Git-style merge markers were the old plan for .md; they turn a conflicted
Markdown file into something no editor renders, and they have no counterpart
for a .docx, so both nodes would have to special-case file types forever.

Conflict copies never sync (protocol.should_ignore skips them): they are one
node's answer to a collision, and pushing them would hand every peer a file it
never made.

Safety invariants:
    - BEFORE any overwrite, create a backup (see create_backup)
    - deleted files go to .sync-trash/, never straight to /dev/null
    - every conflict is logged and broadcast — never silent
"""

from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ── Detection ──

def detect_conflict(local_checksum: str, remote_checksum: str) -> bool:
    """
    Detect whether two versions conflict.

    Returns False if either checksum is empty (= new file, no conflict).
    Returns True if both are non-empty and differ.
    """
    if not local_checksum or not remote_checksum:
        return False
    return local_checksum != remote_checksum


# ── Conflict copies (all file types) ──

_UNSAFE_DEVICE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_device_id(device_id: str) -> str:
    """
    Reduce a device id to something safe for a filename.

    Hostnames carry dots and can carry worse; a device id ends up inside a
    path, so anything but letters, digits, dash and underscore is folded to a
    dash. An empty result becomes "unknown" rather than a filename that starts
    with a stray dot.
    """
    cleaned = _UNSAFE_DEVICE_CHARS.sub("-", (device_id or "").strip()).strip("-")
    return cleaned[:40] or "unknown"


def _full_suffix(name: str) -> str:
    """
    The complete extension, including compound ones.

    Path.suffix returns ".gz" for "backup.tar.gz", which would rename the file
    to "backup.tar.conflict.<...>.gz" and break the pairing. The known
    extension table has the compound forms, so ask it first.
    """
    from .protocol import _EXT_MAP

    lowered = name.lower()
    best = ""
    for ext in _EXT_MAP:
        if lowered.endswith(ext) and len(ext) > len(best):
            best = ext
    if best:
        return name[-len(best):]
    return Path(name).suffix


def make_conflict_name(
    rel_path: str, device_id: str, timestamp: Optional[float] = None
) -> str:
    """
    Build the conflict-copy path for any file type.

    "notes.md" on device "laptop" → "notes.conflict.laptop.20260811T142530Z.md"
    "report.pdf"                  → "report.conflict.laptop.20260811T142530Z.pdf"
    "Makefile"                    → "Makefile.conflict.laptop.20260811T142530Z"

    The extension is preserved so the file still opens in the application that
    owns it, and the timestamp is UTC and sortable so several conflicts on one
    file never collide.
    """
    ts = timestamp if timestamp and timestamp > 0 else time.time()
    stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    device = sanitize_device_id(device_id)

    path = Path(rel_path)
    suffix = _full_suffix(path.name)
    if suffix:
        stem = path.name[: -len(suffix)]
        marker = f"{stem}.conflict.{device}.{stamp}{suffix}"
    else:
        marker = f"{path.name}.conflict.{device}.{stamp}"
    return str(path.with_name(marker)).replace("\\", "/")


def save_conflict_copy(
    vault_path: str,
    rel_path: str,
    device_id: str,
    timestamp: Optional[float] = None,
) -> Optional[str]:
    """
    Move the local version aside before the remote version lands.

    Moved, not copied: the caller is about to overwrite the file, and a move
    guarantees the local bytes exist exactly once rather than briefly twice on
    a full disk.

    Returns:
        Absolute path of the conflict copy, or None if the file was gone.
    """
    src = Path(vault_path) / rel_path
    if not src.exists():
        return None

    dst = Path(vault_path) / make_conflict_name(rel_path, device_id, timestamp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return str(dst)


def is_conflict_copy(rel_path: str) -> bool:
    """True if this path is a conflict copy produced by save_conflict_copy."""
    return ".conflict." in Path(rel_path).name


# ── Binary Latest-Wins ──

def resolve_binary_conflict(
    local_meta: Dict[str, Any],
    remote_meta: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Decide which version goes on disk under the original name: latest mtime.

    The loser is not discarded — the caller keeps it via save_conflict_copy.
    On a tie the remote wins, so every node reaches the same answer.

    Returns:
        (winner_meta, loser_meta)
    """
    local_mtime = local_meta.get("mtime", 0.0)
    remote_mtime = remote_meta.get("mtime", 0.0)

    if local_mtime > remote_mtime:
        return local_meta, remote_meta
    else:
        # Remote wins on tie (deterministic)
        return remote_meta, local_meta


# ── Backup ──

def create_backup(file_path: str) -> Optional[str]:
    """
    Create a .backup copy of a file before overwriting.

    Returns:
        Path to the backup file, or None if source doesn't exist.
    """
    if not os.path.exists(file_path):
        return None

    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    return backup_path


# ── Sync Trash (Scenario S6) ──

def move_to_sync_trash(vault_path: str, rel_path: str) -> str:
    """
    Move a file to .sync-trash/ instead of deleting permanently.

    Safety: remotely-deleted files are NEVER immediately removed.
    User can recover from .sync-trash/ at any time.

    Returns:
        Path to the trashed file.
    """
    vault = Path(vault_path)
    src = vault / rel_path
    trash_dir = vault / ".sync-trash"
    trash_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp to avoid name collisions
    ts = int(time.time())
    trash_name = f"{ts}_{Path(rel_path).name}"
    dst = trash_dir / trash_name

    if src.exists():
        shutil.move(str(src), str(dst))

    return str(dst)
