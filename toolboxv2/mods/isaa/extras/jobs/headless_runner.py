"""
Headless Job Runner - Auto-wake entry point
=============================================

Called by OS scheduler (schtasks/cron/launchd) when the CLI is not running.
Loads the jobs file, checks which jobs are due, fires them headlessly, exits.

Usage:
    python -m toolboxv2.mods.isaa.extras.jobs.headless_runner --jobs-file /path/to/jobs.json

Author: ISAA Team
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
_log = logging.getLogger("isaa.headless_runner")


def _check_due_jobs(jobs_file: Path) -> list[dict]:
    """Check which jobs are due to fire now (without full scheduler)."""
    if not jobs_file.exists():
        _log.info("No jobs file found: %s", jobs_file)
        return []

    try:
        data = json.loads(jobs_file.read_text(encoding="utf-8"))
    except Exception as e:
        _log.error("Failed to read jobs file: %s", e)
        return []

    due = []
    now = datetime.now(timezone.utc)
    now_ts = time.time()

    for jd in data:
        if jd.get("status") != "active":
            continue

        trigger = jd.get("trigger", {})
        trigger_type = trigger.get("trigger_type", "")

        is_due = False

        if trigger_type == "on_time":
            at_dt = trigger.get("at_datetime")
            if at_dt:
                try:
                    target = datetime.fromisoformat(at_dt)
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                    if now >= target:
                        is_due = True
                except (ValueError, TypeError):
                    pass

        elif trigger_type == "on_interval":
            interval = trigger.get("interval_seconds")
            last_run = jd.get("last_run_at")
            if interval:
                if last_run:
                    try:
                        last = datetime.fromisoformat(last_run)
                        if last.tzinfo is None:
                            last = last.replace(tzinfo=timezone.utc)
                        elapsed = (now - last).total_seconds()
                        if elapsed >= interval:
                            is_due = True
                    except (ValueError, TypeError):
                        is_due = True
                else:
                    is_due = True  # Never run before

        elif trigger_type == "on_cron":
            expr = trigger.get("cron_expression")
            if expr:
                try:
                    from croniter import croniter
                    last_run = jd.get("last_run_at")
                    base = datetime.fromisoformat(last_run) if last_run else now.replace(hour=0, minute=0, second=0)
                    if base.tzinfo is None:
                        base = base.replace(tzinfo=timezone.utc)
                    cron = croniter(expr, base)
                    next_fire = cron.get_next(datetime)
                    if next_fire.tzinfo is None:
                        next_fire = next_fire.replace(tzinfo=timezone.utc)
                    if now >= next_fire:
                        is_due = True
                except ImportError:
                    _log.debug("croniter not installed, skipping cron job")
                except Exception:
                    pass

        elif trigger_type == "on_system_boot":
            # If we're called by OS scheduler on boot, fire these
            is_due = True

        elif trigger_type == "on_cli_start":
            # These only fire when the interactive CLI starts, skip in headless
            pass

        if is_due:
            due.append(jd)

    return due


async def _run_due_jobs(jobs_file: Path, due_jobs: list[dict]):
    """Initialize minimal ISAA and run due jobs."""
    _log.info("Running %d due jobs...", len(due_jobs))

    try:
        from toolboxv2 import get_app
        from toolboxv2.mods.isaa.extras.jobs.job_manager import JobDefinition
    except ImportError as e:
        _log.error("Failed to import toolboxv2: %s", e)
        return

    app = get_app("isaa-headless")
    isaa_tools = app.get_mod("isaa")

    for jd_data in due_jobs:
        job = JobDefinition.from_dict(jd_data.copy())
        _log.info("Firing job: %s (%s) -> agent=%s", job.job_id, job.name, job.agent_name)

        try:
            agent = await isaa_tools.get_agent(job.agent_name)
            if job.query == "__dream__":
                from toolboxv2.mods.isaa.base.Agent.dreamer import DreamConfig
                dream_cfg = DreamConfig()
                if job.trigger.extra and "dream_config" in job.trigger.extra:
                    dream_cfg = DreamConfig(**job.trigger.extra["dream_config"])
                result = await asyncio.wait_for(
                    agent.a_dream(dream_cfg),
                    timeout=job.timeout_seconds,
                )
            else:
                result = await asyncio.wait_for(
                    agent.a_run(job.query, session_id=job.session_id),
                    timeout=job.timeout_seconds,
                )
            _log.info("Job %s completed: %s", job.job_id, str(result)[:200])

            # Update job in file
            jd_data["last_run_at"] = datetime.now(timezone.utc).isoformat()
            jd_data["run_count"] = jd_data.get("run_count", 0) + 1
            jd_data["last_result"] = "completed"
            if job.trigger.trigger_type == "on_time":
                jd_data["status"] = "expired"

        except asyncio.TimeoutError:
            _log.warning("Job %s timed out after %ds", job.job_id, job.timeout_seconds)
            jd_data["last_run_at"] = datetime.now(timezone.utc).isoformat()
            jd_data["run_count"] = jd_data.get("run_count", 0) + 1
            jd_data["fail_count"] = jd_data.get("fail_count", 0) + 1
            jd_data["last_result"] = "timeout"

        except Exception as e:
            _log.error("Job %s failed: %s", job.job_id, e)
            jd_data["last_run_at"] = datetime.now(timezone.utc).isoformat()
            jd_data["run_count"] = jd_data.get("run_count", 0) + 1
            jd_data["fail_count"] = jd_data.get("fail_count", 0) + 1
            jd_data["last_result"] = "failed"

    # Write back updated jobs
    try:
        all_data = json.loads(jobs_file.read_text(encoding="utf-8"))
        # Merge updates
        job_map = {j["job_id"]: j for j in all_data}
        for jd_data in due_jobs:
            if jd_data.get("job_id") in job_map:
                job_map[jd_data["job_id"]].update(jd_data)
        updated = list(job_map.values())
        jobs_file.write_text(json.dumps(updated, indent=2, default=str), encoding="utf-8")
        _log.info("Jobs file updated")
    except Exception as e:
        _log.error("Failed to update jobs file: %s", e)


def main():
    parser = argparse.ArgumentParser(description="ISAA Headless Job Runner")
    parser.add_argument("--jobs-file", required=True, help="Path to jobs JSON file")
    args = parser.parse_args()

    jobs_file = Path(args.jobs_file)
    _log.info("Checking jobs from: %s", jobs_file)

    due_jobs = _check_due_jobs(jobs_file)
    if not due_jobs:
        _log.info("No jobs due, exiting.")
        return

    _log.info("Found %d due jobs", len(due_jobs))
    asyncio.run(_run_due_jobs(jobs_file, due_jobs))
    _log.info("Headless runner finished")


if __name__ == "__main__":
    main()
