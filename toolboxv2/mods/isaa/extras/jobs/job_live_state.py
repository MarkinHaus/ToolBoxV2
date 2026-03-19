"""
Job Live State — Shared execution state between scheduler and viewer.
Written by JobScheduler during _fire_job, read by job_viewer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class JobLiveEntry:
    job_id: str
    name: str = ""
    agent_name: str = ""
    status: str = "idle"          # idle | running | done | failed | timeout
    iteration: int = 0
    tool_calls: list[str] = field(default_factory=list)   # rolling last-N
    last_thought: str = ""
    context_used: int = 0
    context_max: int = 200_000
    started_at: str = ""
    last_update: str = ""
    current_query: str = ""

    def context_pct(self) -> float:
        return min(100.0, self.context_used / self.context_max * 100) if self.context_max else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> JobLiveEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class JobLiveStateWriter:
    """Written by the scheduler process."""
    MAX_TOOL_HISTORY = 12

    def __init__(self, live_file: Path):
        self._file = live_file
        self._entries: dict[str, JobLiveEntry] = {}

    def start(self, job_id: str, name: str, agent_name: str, query: str) -> None:
        self._entries[job_id] = JobLiveEntry(
            job_id=job_id, name=name, agent_name=agent_name,
            status="running", current_query=query[:300],
            started_at=datetime.now(timezone.utc).isoformat(),
            last_update=datetime.now(timezone.utc).isoformat(),
        )
        self._flush()

    def update_iteration(
        self,
        job_id: str,
        iteration: int,
        tool: str | None = None,
        thought: str | None = None,
        context_used: int | None = None,
        context_max: int | None = None,
    ) -> None:
        e = self._entries.get(job_id)
        if not e:
            return
        e.iteration = iteration
        if tool:
            e.tool_calls.append(tool)
            e.tool_calls = e.tool_calls[-self.MAX_TOOL_HISTORY :]
        if thought is not None:
            e.last_thought = thought[:600]
        if context_used is not None:
            e.context_used = context_used
        if context_max is not None:
            e.context_max = context_max
        e.last_update = datetime.now(timezone.utc).isoformat()
        self._flush()

    def finish(self, job_id: str, result: str = "done") -> None:
        e = self._entries.get(job_id)
        if e:
            e.status = result           # "done" | "failed" | "timeout"
            e.last_update = datetime.now(timezone.utc).isoformat()
            self._flush()
        self._entries.pop(job_id, None)
        self._flush()

    def _flush(self) -> None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps(
                    {jid: e.to_dict() for jid, e in self._entries.items()},
                    indent=2, default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass


class JobLiveStateReader:
    """Read-only accessor used by job_viewer (separate process)."""

    def __init__(self, live_file: Path):
        self._file = live_file

    def read(self) -> dict[str, JobLiveEntry]:
        try:
            if self._file.exists():
                raw = json.loads(self._file.read_text(encoding="utf-8"))
                return {jid: JobLiveEntry.from_dict(d) for jid, d in raw.items()}
        except Exception:
            pass
        return {}
