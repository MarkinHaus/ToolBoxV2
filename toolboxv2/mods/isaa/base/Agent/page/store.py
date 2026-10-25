"""Filesystem helpers: a plain directory store with SQLite FTS5 (bm25) search, and an OS file lock."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path


class Fts:
    """SQLite FTS5 index ranked by bm25(). Standard library, no custom scoring."""

    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS d USING fts5(key UNINDEXED, body)")

    def add(self, key: str, body: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM d WHERE key = ?", (key,))
            self.db.execute("INSERT INTO d VALUES (?, ?)", (key, body))

    def delete_prefix(self, prefix: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM d WHERE key LIKE ?", (prefix + "%",))

    def search(self, query: str, k: int = 5, prefix: str = "") -> list[tuple[str, str]]:
        terms = re.findall(r"\w+", query or "")  # FTS5 query escaping, not relevance logic
        if not terms:
            return []
        return self.db.execute(
            "SELECT key, snippet(d, 1, '', '', '...', 12) FROM d "
            "WHERE d MATCH ? AND key LIKE ? ORDER BY bm25(d) LIMIT ?",
            (" OR ".join(f'"{t}"' for t in terms), prefix + "%", k),
        ).fetchall()

    def close(self) -> None:
        self.db.close()


class DirStore:
    """Default Store: files under root, text indexed in Fts. ISAA uses its VFS instead (isaa.py)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = Fts(self.root / "index.db")

    def _path(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError(f"path outside store: {path}")
        return p

    def put(self, path: str, data: str | bytes) -> str:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
        tmp.replace(p)
        if isinstance(data, str):
            self.index.add(path, data)
        return path

    def get(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    def search(self, query: str, k: int = 5, prefix: str = "") -> list[tuple[str, str]]:
        return self.index.search(query, k, prefix)

    def delete_prefix(self, prefix: str) -> None:
        """Delete every stored entry whose path starts with ``prefix`` (on disk
        and from the Fts index). ``prefix`` is a store path like ``"<sid>/"``."""
        base = self._path(prefix.rstrip("/")) if prefix.strip("/") else self.root
        import shutil
        if base.is_dir():
            shutil.rmtree(base, ignore_errors=True)
        elif base.exists():
            try:
                base.unlink()
            except OSError:
                pass
        self.index.delete_prefix(prefix)

    def close(self) -> None:
        self.index.close()


class SessionLocked(RuntimeError):
    pass


def lock_file(path: Path):
    """Exclusive OS lock, held while the returned file stays open. The OS frees it when the
    process dies, so there are no stale locks and no pid checks (os.kill kills on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SessionLocked(f"{path.parent.name} is open in another process") from None
    return f


class JsonlEvents:
    """on_event sink: one JSON line per event, references only (never full prompts)."""

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, type: str, **data) -> None:
        self.f.write(
            json.dumps({"t": round(time.time(), 3), "type": type, **data}, default=str) + "\n"
        )
        if type == "run_end":
            self.f.flush()

    def close(self) -> None:
        self.f.close()
