"""
Pydantic models for the job HTTP API.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class TriggerIn(BaseModel):
    trigger_type: str                    # on_time | on_interval | on_cron | on_system_boot …
    at_datetime: str | None = None
    interval_seconds: int | None = None
    cron_expression: str | None = None
    watch_job_id: str | None = None
    idle_seconds: int | None = None
    agent_idle_seconds: int | None = None
    extra: dict[str, Any] | None = None


class JobAddIn(BaseModel):
    name: str
    agent_name: str
    query: str
    trigger: TriggerIn
    session_id: str = "default"
    timeout_seconds: int = 300


class JobOut(BaseModel):
    job_id: str
    name: str
    agent_name: str
    query: str
    trigger_type: str
    status: str
    run_count: int
    fail_count: int
    last_run_at: str | None
    last_result: str | None
    created_at: str
    next_fire: str | None = None        # human-readable, computed server-side
    is_running: bool = False            # from live state
    iteration: int | None = None        # from live state
    context_pct: float | None = None    # from live state
    last_thought: str | None = None     # from live state
    tool_calls: list[str] | None = None # from live state


class LaunchCLIIn(BaseModel):
    terminal: str = "auto"              # auto | wt | xterm | gnome-terminal | tmux | screen
    extra_args: list[str] | None = None


class LaunchViewerIn(BaseModel):
    mode: str = "web"                   # web | terminal | both
    port: int = 7799
    refresh: float = 1.0


class JobActionOut(BaseModel):
    ok: bool
    message: str
    job_id: str | None = None
