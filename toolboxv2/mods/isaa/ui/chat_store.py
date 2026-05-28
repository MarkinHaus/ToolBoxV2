"""
chat_store.py — Disk-backed chat persistence.

One JSONL per chat (append-only, one frame per line).
One meta JSON per chat (header + agent + run_id pointer).

Layout:
    <root>/<chat_id>.jsonl       # WS frames, each line = {"seq": N, "type": ..., ...}
    <chat_id>.meta.json          # {chat_id, title, agent, session_id, run_id, created_at, last_update, ui}

Source of truth for replay: the JSONL.
Reconnect reads frames with seq > last_seq from JSONL.
"""
from __future__ import annotations

import json
import os
import time
import uuid
import asyncio
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator


def _now_iso() -> str:
    # microsecond precision so last_update sorts stably between rapid writes
    t = time.time()
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
    micros = int((t - int(t)) * 1_000_000)
    return f"{base}.{micros:06d}"


@dataclass
class ChatMeta:
    chat_id: str
    title: str = ""
    agent: str = ""
    session_id: str = ""
    run_id: str | None = None
    created_at: str = field(default_factory=_now_iso)
    last_update: str = field(default_factory=_now_iso)
    message_count: int = 0
    ui: dict[str, Any] = field(default_factory=dict)
    # ui:
    #   pinned_widgets: list
    #   vars_global: dict
    #   expanded_steps: list

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChatMeta":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ChatStore:
    """Per-process store. Thread-safe via per-chat locks."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        # seq cache: chat_id -> last seq written
        self._seq_cache: dict[str, int] = {}

    # ---------- locking ----------

    def _lock_for(self, chat_id: str) -> threading.Lock:
        with self._locks_lock:
            lk = self._locks.get(chat_id)
            if lk is None:
                lk = threading.Lock()
                self._locks[chat_id] = lk
            return lk

    # ---------- paths ----------

    def _jsonl(self, chat_id: str) -> Path:
        return self.root / f"{chat_id}.jsonl"

    def _meta_path(self, chat_id: str) -> Path:
        return self.root / f"{chat_id}.meta.json"

    # ---------- create / list / delete ----------

    def create(self, agent: str = "", title: str = "", chat_id: str | None = None) -> ChatMeta:
        cid = chat_id or uuid.uuid4().hex[:12]
        meta = ChatMeta(
            chat_id=cid, agent=agent, session_id=cid, title=title or "New Chat"
        )
        with self._lock_for(cid):
            self._jsonl(cid).touch(exist_ok=True)
            self._write_meta(meta)
            self._seq_cache[cid] = 0
        return meta

    def exists(self, chat_id: str) -> bool:
        return self._meta_path(chat_id).exists()

    def list(self) -> list[dict]:
        out = []
        for p in self.root.glob("*.meta.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append(d)
            except (OSError, ValueError):
                continue
        out.sort(key=lambda x: x.get("last_update", ""), reverse=True)
        return out

    def get_meta(self, chat_id: str) -> ChatMeta | None:
        p = self._meta_path(chat_id)
        if not p.exists():
            return None
        try:
            return ChatMeta.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    def update_meta(self, chat_id: str, **fields) -> ChatMeta | None:
        with self._lock_for(chat_id):
            meta = self.get_meta(chat_id)
            if meta is None:
                return None
            for k, v in fields.items():
                if k == "ui" and isinstance(v, dict):
                    # merge instead of replace
                    meta.ui.update(v)
                elif hasattr(meta, k):
                    setattr(meta, k, v)
            meta.last_update = _now_iso()
            self._write_meta(meta)
            return meta

    def _write_meta(self, meta: ChatMeta) -> None:
        tmp = self._meta_path(meta.chat_id).with_suffix(".meta.json.tmp")
        tmp.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self._meta_path(meta.chat_id))

    def delete(self, chat_id: str) -> bool:
        with self._lock_for(chat_id):
            jp = self._jsonl(chat_id)
            mp = self._meta_path(chat_id)
            ok = False
            if jp.exists():
                jp.unlink()
                ok = True
            if mp.exists():
                mp.unlink()
                ok = True
            self._seq_cache.pop(chat_id, None)
            return ok

    # ---------- seq + append ----------

    def _bootstrap_seq(self, chat_id: str) -> int:
        """Walk the JSONL tail to find the highest seq. Cached after."""
        if chat_id in self._seq_cache:
            return self._seq_cache[chat_id]
        p = self._jsonl(chat_id)
        if not p.exists():
            self._seq_cache[chat_id] = 0
            return 0
        last = 0
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        s = int(obj.get("seq", 0))
                        if s > last:
                            last = s
                    except (ValueError, TypeError):
                        continue
        except OSError:
            pass
        self._seq_cache[chat_id] = last
        return last

    def append(self, chat_id: str, frame: dict) -> int:
        """Append a frame, assigning a seq. Returns the seq used."""
        with self._lock_for(chat_id):
            last = self._bootstrap_seq(chat_id)
            seq = last + 1
            frame = dict(frame)
            frame["seq"] = seq
            frame.setdefault("ts", _now_iso())
            with self._jsonl(chat_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(frame, ensure_ascii=False, default=str))
                f.write("\n")
            self._seq_cache[chat_id] = seq
            # Increment message_count when a user/assistant turn-ending frame is written
            if frame.get("type") in ("user_msg", "done", "max_iterations", "cancelled"):
                m = self.get_meta(chat_id)
                if m:
                    m.message_count += 1
                    m.last_update = _now_iso()
                    self._write_meta(m)
            return seq

    def last_seq(self, chat_id: str) -> int:
        return self._bootstrap_seq(chat_id)

    # ---------- replay ----------

    def replay(self, chat_id: str, after_seq: int = 0) -> Iterator[dict]:
        """Yield frames where seq > after_seq, in order."""
        p = self._jsonl(chat_id)
        if not p.exists():
            return
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if int(obj.get("seq", 0)) > after_seq:
                        yield obj
        except OSError:
            return

    def read_all(self, chat_id: str) -> list[dict]:
        return list(self.replay(chat_id, after_seq=0))

    # ---------- rollback ----------

    def truncate_after(self, chat_id: str, step_id: str) -> int:
        """Drop all frames where step_id matches OR comes after the given step_id.

        A step boundary is the seq of any frame with the matching step_id.
        Returns the new last_seq (= seq of the frame just BEFORE the dropped section),
        or -1 if step_id was not found.
        """
        with self._lock_for(chat_id):
            p = self._jsonl(chat_id)
            if not p.exists():
                return -1
            kept: list[dict] = []
            cut_seq = -1
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if cut_seq == -1 and obj.get("step_id") == step_id:
                            cut_seq = int(obj.get("seq", 0))
                        if cut_seq == -1 or int(obj.get("seq", 0)) < cut_seq:
                            kept.append(obj)
            except OSError:
                return -1
            if cut_seq == -1:
                return -1
            tmp = p.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for obj in kept:
                    f.write(json.dumps(obj, ensure_ascii=False, default=str))
                    f.write("\n")
            tmp.replace(p)
            new_last = kept[-1]["seq"] if kept else 0
            self._seq_cache[chat_id] = int(new_last)
            return int(new_last)
