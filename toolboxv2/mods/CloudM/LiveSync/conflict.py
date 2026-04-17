"""
LiveSync Conflict Resolution
=============================

Strategies by file type:
    .md   → merge markers (Git-style), both versions preserved
    binary → latest-wins, loser backed up as .conflict.{checksum}.ext

Safety invariant: BEFORE any overwrite, create a .backup file.
Deleted files go to .sync-trash/ (never permanently deleted by sync).

Every conflict is logged + broadcast — never silent.
"""

from __future__ import annotations

import os
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


# ── Markdown Merge ──

def resolve_md_conflict(
    local_content: str,
    remote_content: str,
    local_client: str,
    remote_client: str,
    local_timestamp: float,
    remote_timestamp: float,
) -> str:
    """
    Merge two conflicting .md versions using Git-style conflict markers.

    Both versions are preserved — the user resolves manually.
    """
    local_time = _format_ts(local_timestamp)
    remote_time = _format_ts(remote_timestamp)

    return (
        f"<<<<<<< LOCAL ({local_client} @ {local_time})\n"
        f"{local_content}\n"
        f"=======\n"
        f"{remote_content}\n"
        f">>>>>>> REMOTE ({remote_client} @ {remote_time})\n"
    )


def _format_ts(ts: float) -> str:
    """Format a Unix timestamp as HH:MM:SS for merge markers."""
    if ts <= 0:
        return "unknown"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%H:%M:%S")


# ── Binary Latest-Wins ──

def resolve_binary_conflict(
    local_meta: Dict[str, Any],
    remote_meta: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Resolve binary file conflict: latest mtime wins.

    Returns:
        (winner_meta, loser_meta)

    On tie: remote wins (deterministic).
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


def make_conflict_backup_name(rel_path: str, checksum: str) -> str:
    """
    Generate a conflict backup filename.

    Example: "notes.md" + checksum "aabb" → "notes.conflict.aabb.md"
    """
    p = Path(rel_path)
    if p.suffix:
        return str(p.with_suffix(f".conflict.{checksum}{p.suffix}"))
    else:
        return f"{rel_path}.conflict.{checksum}"


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
