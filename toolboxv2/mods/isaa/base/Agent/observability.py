"""
ObservabilityLayer - Step-granular profiling, audit, and persistence for ISAA agents.

Provides:
- Per-step timing (LLM TTFT, tool duration, token throughput)
- VFS delta tracking per step
- Ring-buffer persistence of last N runs (configurable, default 3)
- AuditLogger integration (via app.audit_logger)
- Custom async callback for external streaming (SSE/WS/etc.)
- Step-granular ctx snapshots enabling resume from any persisted step

Storage layout:
    {obs_dir}/
        live_{run_id}.jsonl     # append-only during execution
        run_{run_id}.json       # completed run (replaces live file)
        _index.json             # ring buffer: ordered list of run_ids

Author: Observability V1
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Optional

from toolboxv2 import get_logger

logger = get_logger()

# Type alias for the external streaming callback
OnStepCallback = Callable[[dict], Awaitable[None]]


# =============================================================================
# DATA MODEL
# =============================================================================


@dataclass(slots=True)
class ToolCallRecord:
    name: str = ""
    args_summary: str = ""          # first 200 chars
    t_start: float = 0.0
    t_end: float = 0.0
    duration_s: float = 0.0
    result_summary: str = ""        # first 200 chars
    status: str = "ok"              # ok | error
    error: str = ""


@dataclass(slots=True)
class LLMCallRecord:
    model: str = ""
    t_start: float = 0.0
    t_first_token: float = 0.0
    t_end: float = 0.0
    ttft_s: float = 0.0            # time to first token
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_per_sec: float = 0.0     # output tokens / duration


@dataclass
class StepRecord:
    """One iteration of the execution loop."""
    step_id: int = 0               # iteration number
    t_start: float = 0.0
    t_end: float = 0.0
    duration_s: float = 0.0

    llm: LLMCallRecord | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # VFS changes detected this step
    vfs_deltas: list[dict] = field(default_factory=list)
    # e.g. {"path": "/project/x.py", "action": "write", "lines_added": 12, "lines_removed": 3}

    # Compression stats if triggered
    compression: dict | None = None  # {"kept": N, "summarized": N, "dropped": N}

    # ExecutionContext snapshot for resume (only every N steps)
    ctx_snapshot: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "step_id": self.step_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration_s": self.duration_s,
            "llm": asdict(self.llm) if self.llm else None,
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "vfs_deltas": self.vfs_deltas,
            "compression": self.compression,
        }
        # ctx_snapshot only in live file, not in summary dicts
        return d


@dataclass
class RunRecord:
    """Complete record of one execute() invocation."""
    run_id: str = ""
    agent_name: str = ""
    query: str = ""
    session_id: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    duration_s: float = 0.0

    steps: list[StepRecord] = field(default_factory=list)

    # Aggregated metrics (computed on end_run)
    total_llm_time_s: float = 0.0
    total_tool_time_s: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_iterations: int = 0
    tool_call_count: int = 0
    avg_ttft_s: float = 0.0

    # Outcome
    success: bool = False
    final_answer_summary: str = ""  # first 500 chars

    # VFS summary: {path: {"actions": [...], "net_lines_delta": N}}
    files_modified: dict = field(default_factory=dict)

    persona: str = ""
    skills_matched: list[str] = field(default_factory=list)

    def aggregate(self):
        """Compute aggregated metrics from steps. Call once after all steps recorded."""
        ttfts = []
        for s in self.steps:
            if s.llm:
                self.total_llm_time_s += s.llm.duration_s
                self.total_tokens_in += s.llm.tokens_in
                self.total_tokens_out += s.llm.tokens_out
                if s.llm.ttft_s > 0:
                    ttfts.append(s.llm.ttft_s)
            for tc in s.tool_calls:
                self.total_tool_time_s += tc.duration_s
                self.tool_call_count += 1
            for vd in s.vfs_deltas:
                p = vd.get("path", "?")
                entry = self.files_modified.setdefault(p, {"actions": [], "net_lines_delta": 0})
                entry["actions"].append(vd.get("action", "?"))
                entry["net_lines_delta"] += vd.get("lines_added", 0) - vd.get("lines_removed", 0)

        self.total_iterations = len(self.steps)
        self.avg_ttft_s = round(sum(ttfts) / len(ttfts), 4) if ttfts else 0.0
        self.duration_s = round(self.t_end - self.t_start, 3) if self.t_end else 0.0

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "session_id": self.session_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration_s": self.duration_s,
            "total_llm_time_s": round(self.total_llm_time_s, 3),
            "total_tool_time_s": round(self.total_tool_time_s, 3),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_iterations": self.total_iterations,
            "tool_call_count": self.tool_call_count,
            "avg_ttft_s": self.avg_ttft_s,
            "success": self.success,
            "final_answer_summary": self.final_answer_summary,
            "files_modified": self.files_modified,
            "persona": self.persona,
            "skills_matched": self.skills_matched,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        r = cls(
            run_id=d.get("run_id", ""),
            agent_name=d.get("agent_name", ""),
            query=d.get("query", ""),
            session_id=d.get("session_id", ""),
            t_start=d.get("t_start", 0),
            t_end=d.get("t_end", 0),
            duration_s=d.get("duration_s", 0),
            total_llm_time_s=d.get("total_llm_time_s", 0),
            total_tool_time_s=d.get("total_tool_time_s", 0),
            total_tokens_in=d.get("total_tokens_in", 0),
            total_tokens_out=d.get("total_tokens_out", 0),
            total_iterations=d.get("total_iterations", 0),
            tool_call_count=d.get("tool_call_count", 0),
            avg_ttft_s=d.get("avg_ttft_s", 0),
            success=d.get("success", False),
            final_answer_summary=d.get("final_answer_summary", ""),
            files_modified=d.get("files_modified", {}),
            persona=d.get("persona", ""),
            skills_matched=d.get("skills_matched", []),
        )
        # Steps without ctx_snapshot (those are only in live files)
        for sd in d.get("steps", []):
            llm = LLMCallRecord(**sd["llm"]) if sd.get("llm") else None
            tcs = [ToolCallRecord(**tc) for tc in sd.get("tool_calls", [])]
            r.steps.append(StepRecord(
                step_id=sd.get("step_id", 0),
                t_start=sd.get("t_start", 0),
                t_end=sd.get("t_end", 0),
                duration_s=sd.get("duration_s", 0),
                llm=llm,
                tool_calls=tcs,
                vfs_deltas=sd.get("vfs_deltas", []),
                compression=sd.get("compression"),
            ))
        return r


# =============================================================================
# OBSERVABILITY LAYER
# =============================================================================


class ObservabilityLayer:
    """
    Instruments ExecutionEngine runs with step-granular profiling.

    Owned by FlowAgent. Instantiated once per agent lifetime.
    Engine calls begin_run/begin_step/record_*/end_step/end_run hooks.

    Args:
        agent_name:         Agent identifier (used for audit + file naming)
        obs_dir:            Directory for ring-buffer persistence
        max_runs:           How many completed runs to keep on disk (ring buffer)
        snapshot_interval:  Persist ExecutionContext every N steps (for resume)
        audit_logger:       app.audit_logger instance (or None to skip audit)
        on_step:            Async callback invoked after each step with step dict
    """

    def __init__(
        self,
        agent_name: str,
        obs_dir: str,
        max_runs: int = 3,
        snapshot_interval: int = 5,
        audit_logger: Any = None,
        on_step: OnStepCallback | None = None,
    ):
        self.agent_name = agent_name
        self.obs_dir = obs_dir
        self.max_runs = max_runs
        self.snapshot_interval = snapshot_interval
        self.audit_logger = audit_logger
        self.on_step = on_step

        # Ensure dir exists
        os.makedirs(obs_dir, exist_ok=True)

        # Current run state (active during execution)
        self._current_run: RunRecord | None = None
        self._current_step: StepRecord | None = None
        self._current_llm: LLMCallRecord | None = None
        self._current_tool: ToolCallRecord | None = None
        self._live_fd: Any = None  # open file handle for live JSONL

    # =========================================================================
    # RUN LIFECYCLE
    # =========================================================================

    def begin_run(self, run_id: str, query: str, session_id: str = "",
                  persona: str = "", skills: list[str] | None = None):
        self._current_run = RunRecord(
            run_id=run_id,
            agent_name=self.agent_name,
            query=query,
            session_id=session_id,
            t_start=time.time(),
            persona=persona,
            skills_matched=skills or [],
        )
        # Open live JSONL file (append-only crash-safe log)
        live_path = os.path.join(self.obs_dir, f"live_{run_id}.jsonl")
        try:
            self._live_fd = open(live_path, "a", encoding="utf-8")
        except OSError as e:
            logger.warning(f"[Obs] Cannot open live file: {e}")
            self._live_fd = None

        self._audit("RUN_START", run_id, details={
            "query": query, "session_id": session_id, "persona": persona,
        })

        # Cap stale live files to max_runs
        self._cleanup_stale_live_files(exclude_run_id=run_id)

    def _cleanup_stale_live_files(self, exclude_run_id: str = ""):
        """Remove oldest live files if count exceeds max_runs."""
        if not os.path.exists(self.obs_dir):
            return

        live_files = []
        for fname in os.listdir(self.obs_dir):
            if not fname.startswith("live_") or not fname.endswith(".jsonl"):
                continue
            rid = fname.replace("live_", "").replace(".jsonl", "")
            if rid == exclude_run_id:
                continue
            # Skip if already completed (has a run_*.json)
            if os.path.exists(os.path.join(self.obs_dir, f"run_{rid}.json")):
                try:
                    os.remove(os.path.join(self.obs_dir, fname))
                except OSError:
                    pass
                continue
            fpath = os.path.join(self.obs_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                mtime = 0
            live_files.append((mtime, fpath))

        if len(live_files) < self.max_runs:
            return

        # Sort oldest first, remove excess
        live_files.sort()
        for mtime, fpath in live_files[: len(live_files) - self.max_runs + 1]:
            try:
                os.remove(fpath)
            except OSError:
                pass

    def end_run(self, success: bool, final_answer: str = ""):
        run = self._current_run
        if run is None:
            return

        run.t_end = time.time()
        run.success = success
        run.final_answer_summary = final_answer
        run.aggregate()

        # Close live file
        if self._live_fd:
            try:
                self._live_fd.close()
            except OSError:
                pass
            self._live_fd = None

        # Persist completed run
        self._persist_run(run)
        # Remove live file (data is now in run file)
        live_path = os.path.join(self.obs_dir, f"live_{run.run_id}.jsonl")
        try:
            os.remove(live_path)
        except OSError:
            pass

        # Rotate ring buffer
        self._rotate_runs()

        self._audit("RUN_END", run.run_id, details={
            "success": success,
            "duration_s": run.duration_s,
            "iterations": run.total_iterations,
            "tokens_in": run.total_tokens_in,
            "tokens_out": run.total_tokens_out,
            "tool_calls": run.tool_call_count,
            "avg_ttft_s": run.avg_ttft_s,
        })

        self._current_run = None
        self._current_step = None

    # =========================================================================
    # STEP LIFECYCLE
    # =========================================================================

    def begin_step(self, iteration: int):
        self._current_step = StepRecord(step_id=iteration, t_start=time.time())

    def end_step(self, ctx_checkpoint: dict | None = None):
        """Finalize step, persist to live file, fire callback."""
        step = self._current_step
        if step is None:
            return
        run = self._current_run

        step.t_end = time.time()
        step.duration_s = round(step.t_end - step.t_start, 4)

        # Attach ctx snapshot at configured interval or if explicitly provided
        if ctx_checkpoint is not None:
            step.ctx_snapshot = ctx_checkpoint
        elif run and (step.step_id % self.snapshot_interval == 0):
            pass  # caller must provide; we don't fabricate one

        if run:
            run.steps.append(step)

        # Persist step to live JSONL
        self._write_live_step(step)

        # Async callback
        if self.on_step and run:
            step_dict = step.to_dict()
            step_dict["run_id"] = run.run_id
            step_dict["agent_name"] = run.agent_name
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._safe_callback(step_dict))
            except RuntimeError:
                pass  # no event loop

        self._current_step = None

    # =========================================================================
    # LLM RECORDING
    # =========================================================================

    def record_llm_start(self, model: str = ""):
        self._current_llm = LLMCallRecord(model=model, t_start=time.time())

    def record_llm_first_token(self):
        if self._current_llm and self._current_llm.t_first_token == 0:
            self._current_llm.t_first_token = time.time()
            self._current_llm.ttft_s = round(
                self._current_llm.t_first_token - self._current_llm.t_start, 4
            )

    def record_llm_end(self, tokens_in: int = 0, tokens_out: int = 0, model: str = ""):
        llm = self._current_llm
        if llm is None:
            return
        llm.t_end = time.time()
        llm.duration_s = round(llm.t_end - llm.t_start, 4)
        llm.tokens_in = tokens_in
        llm.tokens_out = tokens_out
        if model:
            llm.model = model
        if llm.duration_s > 0 and tokens_out > 0:
            llm.tokens_per_sec = round(tokens_out / llm.duration_s, 2)
        if self._current_step:
            self._current_step.llm = llm
        self._current_llm = None

    # =========================================================================
    # TOOL RECORDING
    # =========================================================================

    def record_tool_start(self, name: str, args_summary: str = ""):
        self._current_tool = ToolCallRecord(
            name=name,
            args_summary=args_summary,
            t_start=time.time(),
        )

    def record_tool_end(self, name: str, result_summary: str = "",
                        status: str = "ok", error: str = ""):
        tool = self._current_tool
        if tool is None or tool.name != name:
            # Fallback: create a record without start time
            tool = ToolCallRecord(name=name, t_start=time.time())

        tool.t_end = time.time()
        tool.duration_s = round(tool.t_end - tool.t_start, 4)
        tool.result_summary = result_summary
        tool.status = status
        tool.error = error if error else ""

        if self._current_step:
            self._current_step.tool_calls.append(tool)

        self._audit("TOOL_CALL", name, details={
            "duration_s": tool.duration_s, "status": status,
            "run_id": self._current_run.run_id if self._current_run else "",
        })

        self._current_tool = None

    # =========================================================================
    # VFS + COMPRESSION RECORDING
    # =========================================================================

    def record_vfs_delta(self, path: str, action: str,
                         lines_added: int = 0, lines_removed: int = 0):
        if self._current_step:
            self._current_step.vfs_deltas.append({
                "path": path, "action": action,
                "lines_added": lines_added, "lines_removed": lines_removed,
            })

    def record_compression(self, stats: dict):
        if self._current_step:
            self._current_step.compression = stats

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _persist_run(self, run: RunRecord):
        """Write completed RunRecord as JSON."""
        path = os.path.join(self.obs_dir, f"run_{run.run_id}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(run.to_dict(), f, ensure_ascii=False, indent=1)
        except OSError as e:
            logger.warning(f"[Obs] run persist failed: {e}")
            return

        # Update index
        index = self._load_index()
        if run.run_id not in index:
            index.append(run.run_id)
        self._save_index(index)

    def _rotate_runs(self):
        """Keep only max_runs newest completed runs."""
        index = self._load_index()
        while len(index) > self.max_runs:
            old_id = index.pop(0)
            old_path = os.path.join(self.obs_dir, f"run_{old_id}.json")
            try:
                os.remove(old_path)
            except OSError:
                pass
        self._save_index(index)

    def _load_index(self) -> list[str]:
        path = os.path.join(self.obs_dir, "_index.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("runs", [])
        except (OSError, json.JSONDecodeError):
            return []

    def _save_index(self, runs: list[str]):
        path = os.path.join(self.obs_dir, "_index.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"runs": runs, "max_runs": self.max_runs}, f)
        except OSError as e:
            logger.warning(f"[Obs] index save failed: {e}")

    # =========================================================================
    # RESUME SUPPORT
    # =========================================================================

    def get_resumable_run(self, run_id: str) -> tuple[RunRecord, dict | None] | None:
        """
        Load a run and its last ctx_snapshot for resume.

        Checks live file first (incomplete run), then completed run file.

        Returns:
            (RunRecord, ctx_snapshot_dict) or None
        """
        # 1. Try live file (run was interrupted)
        live_path = os.path.join(self.obs_dir, f"live_{run_id}.jsonl")
        if os.path.exists(live_path):
            return self._load_from_live(run_id, live_path)

        # 2. Try completed run file
        run_path = os.path.join(self.obs_dir, f"run_{run_id}.json")
        if os.path.exists(run_path):
            try:
                with open(run_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                record = RunRecord.from_dict(data)
                # No ctx_snapshot in completed runs (steps don't store them)
                return record, None
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"[Obs] load run failed: {e}")
                return None

        return None

    def _load_from_live(self, run_id: str, live_path: str
                        ) -> tuple[RunRecord, dict | None] | None:
        """Reconstruct RunRecord from live JSONL + find last ctx_snapshot."""
        record = RunRecord(run_id=run_id, agent_name=self.agent_name)
        last_snapshot: dict | None = None

        try:
            with open(live_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sd = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Reconstruct StepRecord
                    llm = LLMCallRecord(**sd["llm"]) if sd.get("llm") else None
                    tcs = [ToolCallRecord(**tc) for tc in sd.get("tool_calls", [])]
                    step = StepRecord(
                        step_id=sd.get("step_id", 0),
                        t_start=sd.get("t_start", 0),
                        t_end=sd.get("t_end", 0),
                        duration_s=sd.get("duration_s", 0),
                        llm=llm, tool_calls=tcs,
                        vfs_deltas=sd.get("vfs_deltas", []),
                        compression=sd.get("compression"),
                    )
                    record.steps.append(step)
        except OSError as e:
            logger.warning(f"[Obs] live read failed: {e}")
            return None

        # Now scan for ctx_snapshot separately (stored in extended live format)
        try:
            with open(live_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sd = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "ctx_snapshot" in sd and sd["ctx_snapshot"]:
                        last_snapshot = sd["ctx_snapshot"]
        except OSError:
            pass

        return record, last_snapshot

    # =========================================================================
    # INTERRUPTED RUN DISCOVERY (for cold resume)
    # =========================================================================

    def get_interrupted_runs(self) -> list[dict]:
        """
        Find runs that were interrupted (have live_*.jsonl but no run_*.json).

        Returns:
            List of {run_id, step_count, last_step_id, has_snapshot, live_file}
        """
        results = []
        if not os.path.exists(self.obs_dir):
            return results

        for fname in os.listdir(self.obs_dir):
            if not fname.startswith("live_") or not fname.endswith(".jsonl"):
                continue
            run_id = fname.replace("live_", "").replace(".jsonl", "")

            # Skip if already completed
            if os.path.exists(os.path.join(self.obs_dir, f"run_{run_id}.json")):
                continue

            live_path = os.path.join(self.obs_dir, fname)
            step_count = 0
            last_step_id = 0
            has_snapshot = False

            try:
                with open(live_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sd = json.loads(line)
                            step_count += 1
                            last_step_id = sd.get("step_id", last_step_id)
                            if sd.get("ctx_snapshot"):
                                has_snapshot = True
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue

            if step_count > 0:
                results.append({
                    "run_id": run_id,
                    "step_count": step_count,
                    "last_step_id": last_step_id,
                    "has_snapshot": has_snapshot,
                    "live_file": live_path,
                })

        return results

    # =========================================================================
    # QUERY API
    # =========================================================================

    def list_runs(self) -> list[dict]:
        """List all persisted runs (summary only, no steps)."""
        index = self._load_index()
        summaries = []
        for run_id in reversed(index):  # newest first
            path = os.path.join(self.obs_dir, f"run_{run_id}.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Return summary without steps
                data.pop("steps", None)
                summaries.append(data)
            except (OSError, json.JSONDecodeError):
                summaries.append({"run_id": run_id, "error": "unreadable"})
        return summaries

    def get_run(self, run_id: str) -> RunRecord | None:
        """Load a completed run by ID."""
        path = os.path.join(self.obs_dir, f"run_{run_id}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return RunRecord.from_dict(json.load(f))
        except (OSError, json.JSONDecodeError):
            return None

    # =========================================================================
    # AUDIT INTEGRATION
    # =========================================================================

    def _audit(self, action: str, resource: str, details: dict | None = None):
        """Send structured event to AuditLogger if available."""
        if self.audit_logger is None:
            return
        try:
            self.audit_logger.log_action(
                user_id=self.agent_name,
                action=action,
                resource=resource,
                status="SUCCESS",
                details=details or {},
            )
        except Exception as e:
            logger.debug(f"[Obs] audit log failed: {e}")

    # =========================================================================
    # CALLBACK
    # =========================================================================

    async def _safe_callback(self, step_dict: dict):
        """Invoke on_step callback with error swallowing."""
        try:
            await self.on_step(step_dict)
        except Exception as e:
            logger.debug(f"[Obs] on_step callback error: {e}")

    # =========================================================================
    # LIVE FILE I/O
    # =========================================================================

    def _write_live_step(self, step: StepRecord):
        """Append step to live JSONL. Includes ctx_snapshot when present."""
        if self._live_fd is None:
            return
        try:
            d = step.to_dict()
            if step.ctx_snapshot:
                d["ctx_snapshot"] = step.ctx_snapshot
            line = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
            self._live_fd.write(line + "\n")
            self._live_fd.flush()
            os.fsync(self._live_fd.fileno())
        except (OSError, ValueError) as e:
            logger.warning(f"[Obs] live write failed: {e}")
