"""
Running executions — multi-agent live footer feed.

Aggregates two sources:
  1. `bridge._running` — streams attached to UI chats (top-level user-facing runs)
  2. `JobLiveStateReader` — sub-agent jobs spawned via ISAA's job scheduler

Polled by the UI footer every ~1s.
"""
from __future__ import annotations

from pathlib import Path


def register(app, ctx):
    isaa = ctx["isaa"]
    bridge = ctx["bridge"]

    @app.get("/api/running")
    async def running():
        items = []

        # 1. UI-attached streams
        for chat_id, rs in bridge._running.items():
            if rs.task is None or rs.task.done():
                continue
            items.append({
                "kind": "chat",
                "chat_id": chat_id,
                "agent": rs.agent_name,
                "run_id": rs.run_id,
                "iteration": rs.current_iter,
                "context_pct": 0.0,
                "last_thought": "",
                "started_at": rs.started_at,
            })

        # 2. Sub-agent live state from JobScheduler
        try:
            from toolboxv2.mods.isaa.extras.jobs.job_live_state import JobLiveStateReader
            jobs_file = getattr(getattr(isaa, "job_scheduler", None), "jobs_file", None)
            if jobs_file is not None:
                live_file = Path(str(jobs_file)).with_suffix(".live.json")
                reader = JobLiveStateReader(live_file)
                for job_id, entry in reader.read().items():
                    if entry.status != "running":
                        continue
                    items.append({
                        "kind": "subagent",
                        "job_id": job_id,
                        "name": entry.name,
                        "agent": entry.agent_name,
                        "run_id": "",
                        "iteration": entry.iteration,
                        "context_pct": entry.context_pct(),
                        "last_thought": entry.last_thought,
                        "started_at": entry.started_at,
                    })
        except ImportError:
            pass
        except Exception:
            # Don't let job-state errors block the footer
            pass

        return items
