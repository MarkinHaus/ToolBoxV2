"""
LiveSync Local Index
====================
Async SQLite index tracking file metadata and sync state.
Each client AND the server maintain their own index.

Tables:
  files    — per-file metadata (checksum, mtime, sync_state)
  sync_log — audit trail of sync operations

Export/import as gzipped SQLite dump for full-state sync
(Scenario S5: new client joins → downloads index from MinIO).
"""

from __future__ import annotations

import gzip
import io
import sqlite3
import time
from typing import Any, Dict, List, Optional

import aiosqlite


_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    rel_path    TEXT PRIMARY KEY,
    mtime       REAL,
    size        INTEGER,
    checksum    TEXT,
    sync_state  TEXT DEFAULT 'synced',
    remote_key  TEXT DEFAULT '',
    updated_at  REAL
);

CREATE TABLE IF NOT EXISTS sync_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_path    TEXT,
    action      TEXT,
    checksum    TEXT,
    timestamp   REAL,
    client_id   TEXT
);
"""


class LocalIndex:
    """
    Async SQLite wrapper for the local file index.

    Usage:
        idx = LocalIndex("/path/to/index.db")
        await idx.init()
        ...
        await idx.close()
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Open DB connection and create tables if needed."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close DB connection."""
        if self._db:
            await self._db.close()
            self._db = None

    def _check_open(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("LocalIndex not initialized — call await init() first")
        return self._db

    # ── File Operations ──

    async def get_file(self, rel_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata by relative path. Returns None if not found."""
        db = self._check_open()
        async with db.execute(
            "SELECT * FROM files WHERE rel_path = ?", (rel_path,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def upsert_file(
        self,
        rel_path: str,
        mtime: float,
        size: int,
        checksum: str,
        sync_state: str = "synced",
        remote_key: str = "",
    ) -> None:
        """Insert or update file metadata."""
        db = self._check_open()
        await db.execute(
            """
            INSERT OR REPLACE INTO files
                (rel_path, mtime, size, checksum, sync_state, remote_key, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (rel_path, mtime, size, checksum, sync_state, remote_key, time.time()),
        )
        await db.commit()

    async def delete_file(self, rel_path: str) -> None:
        """Remove file from index."""
        db = self._check_open()
        await db.execute("DELETE FROM files WHERE rel_path = ?", (rel_path,))
        await db.commit()

    async def set_sync_state(self, rel_path: str, state: str) -> None:
        """Update sync state for a file."""
        db = self._check_open()
        await db.execute(
            "UPDATE files SET sync_state = ?, updated_at = ? WHERE rel_path = ?",
            (state, time.time(), rel_path),
        )
        await db.commit()

    async def get_all_checksums(self) -> Dict[str, str]:
        """Return {rel_path: checksum} for all tracked files."""
        db = self._check_open()
        async with db.execute("SELECT rel_path, checksum FROM files") as cursor:
            rows = await cursor.fetchall()
            return {row["rel_path"]: row["checksum"] for row in rows}

    async def get_all_files(self) -> List[Dict[str, Any]]:
        """Return all file records."""
        db = self._check_open()
        async with db.execute("SELECT * FROM files") as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def get_pending(self) -> List[Dict[str, Any]]:
        """Return files that are not in 'synced' state."""
        db = self._check_open()
        async with db.execute(
            "SELECT * FROM files WHERE sync_state != 'synced'"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ── Sync Log ──

    async def log_sync_event(
        self,
        rel_path: str,
        action: str,
        checksum: str = "",
        client_id: str = "",
    ) -> None:
        """Record a sync event in the audit log."""
        db = self._check_open()
        await db.execute(
            """
            INSERT INTO sync_log (rel_path, action, checksum, timestamp, client_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (rel_path, action, checksum, time.time(), client_id),
        )
        await db.commit()

    async def get_sync_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent sync log entries, newest first."""
        db = self._check_open()
        async with db.execute(
            "SELECT * FROM sync_log ORDER BY id DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ── Full State Export/Import (Scenario S5) ──

    async def export_gzipped(self) -> bytes:
        """
        Export the files table as a gzipped SQLite dump.

        Used by the server to provide full state to new clients:
        server writes this to MinIO → client downloads + imports.
        """
        db = self._check_open()

        # Dump files table rows as SQL inserts
        lines: list[str] = []
        lines.append(_SCHEMA)

        async with db.execute("SELECT * FROM files") as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                d = dict(row)
                # Escape single quotes in paths
                rp = d["rel_path"].replace("'", "''")
                rk = (d.get("remote_key") or "").replace("'", "''")
                lines.append(
                    f"INSERT OR REPLACE INTO files "
                    f"(rel_path, mtime, size, checksum, sync_state, remote_key, updated_at) "
                    f"VALUES ('{rp}', {d['mtime']}, {d['size']}, '{d['checksum']}', "
                    f"'{d['sync_state']}', '{rk}', {d.get('updated_at', 0)});"
                )

        sql_dump = "\n".join(lines).encode("utf-8")
        return gzip.compress(sql_dump, compresslevel=6)

    async def import_gzipped(self, data: bytes) -> None:
        """
        Import a gzipped SQL dump into this index.

        Used by clients after downloading the full state DB from MinIO.
        """
        db = self._check_open()
        sql_dump = gzip.decompress(data).decode("utf-8")
        await db.executescript(sql_dump)
        await db.commit()
