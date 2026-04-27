"""
Runner — executes tasks against a model callable, collects binary results.
Supports sync and async model functions, single and batch execution.
Circuit breaker on consecutive failures.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable

from toolboxv2.mods.isaa.base.bench.core import Attachment, CheckResult, Report, Task, TaskContext, TaskResult
from toolboxv2.mods.isaa.base.bench.validators import create_validator


async def _call_model(model_fn: Callable, prompt: str, timeout: float) -> tuple[str, dict]:
    """Call model function with timeout. Returns (response_str, cost_info)."""
    try:
        if asyncio.iscoroutinefunction(model_fn):
            result = await asyncio.wait_for(model_fn(prompt), timeout=timeout)
        else:
            result = model_fn(prompt)
    except asyncio.TimeoutError:
        return f"Error: timeout after {timeout}s", {}
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", {}

    if isinstance(result, tuple) and len(result) == 2:
        resp, cost_info = result
    else:
        resp, cost_info = result, {}

    return (resp if isinstance(resp, str) else str(resp)), (cost_info or {})


def _build_prompt(task: Task) -> str:
    """Build the final prompt string, injecting [media:path] for attachments."""
    prompt = task.prompt
    for att in task.attachments:
        prompt += f"\n[media:{att.path}]"
    return prompt


async def run_task(
    task: Task,
    model_fn: Callable,
    timeout: float = 120.0,
) -> TaskResult:
    """Run a single task and evaluate all checks."""
    prompt = _build_prompt(task)
    t0 = time.perf_counter()
    response, cost_info = await _call_model(model_fn, prompt, timeout)
    elapsed = time.perf_counter() - t0

    ctx = TaskContext(
        task=task,
        prompt=prompt,
        response=response,
        attachments=task.attachments,
        tool_calls=cost_info.get("tool_calls", []),
        execution_time=elapsed,
        token_usage={
            "in": int(cost_info.get("tokens_in", 0)),
            "out": int(cost_info.get("tokens_out", 0)),
        },
    )

    # Run all validators
    checks: list[CheckResult] = []
    for check_def in task.checks:
        try:
            validator = create_validator(
                {"type": check_def.type, **check_def.params}
            )
            result = await validator.validate(ctx)
            checks.append(result)
        except Exception as e:
            checks.append(CheckResult(check_def.type, False, f"validator crash: {e}"))

    return TaskResult(
        task_id=task.id,
        complexity=task.complexity,
        tags=task.tags,
        checks=checks,
        prompt=prompt,
        response=response,
        latency_ms=int(elapsed * 1000),
        tokens_in=int(cost_info.get("tokens_in", 0)),
        tokens_out=int(cost_info.get("tokens_out", 0)),
        cost=float(cost_info.get("total_cost", 0)),
        tool_calls=cost_info.get("tool_calls", []),
        working_history=cost_info.get("working_history", []),
    )


async def run_tasks(
    tasks: list[Task],
    model_fn: Callable,
    model_id: str = "unknown",
    suite_id: str = "default",
    mode: str = "standard",
    timeout: float = 120.0,
    max_consecutive_errors: int = 5,
    batch_size: int = 1,
    on_progress: Callable | None = None,
) -> Report:
    """Run a list of tasks sequentially (or in batches) with circuit breaker.

    Args:
        on_progress: Optional callback(done: int, total: int, task_id: str, passed: bool).
                     Called after each completed task for live progress.
    """
    report = Report(
        model_id=model_id,
        suite_id=suite_id,
        mode=mode,
        timestamp=datetime.now(),
    )
    total_start = time.perf_counter()
    consecutive_errors = 0
    total_tasks = len(tasks)

    def _progress(result: TaskResult, done: int):
        if on_progress:
            try:
                on_progress(done, total_tasks, result.task_id,
                            result.score == 1.0, result)
            except Exception:
                pass
        else:
            # Default: print to stderr
            pct = done / total_tasks * 100 if total_tasks else 0
            icon = "✓" if result.score == 1.0 else "✗" if result.score == 0.0 else "◐"
            elapsed = time.perf_counter() - total_start
            print(
                f"  [{done}/{total_tasks}] {pct:5.1f}% {icon} {result.task_id:30s} "
                f"{result.score*100:5.1f}% {result.latency_ms:5d}ms "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )

    if batch_size <= 1:
        # Sequential
        for idx, task in enumerate(tasks):
            result = await run_task(task, model_fn, timeout)
            report.results.append(result)

            # Accumulate cost
            report.total_tokens_in += result.tokens_in
            report.total_tokens_out += result.tokens_out
            report.total_cost += result.cost

            _progress(result, idx + 1)

            # Circuit breaker
            if result.response.startswith("Error:"):
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            if consecutive_errors >= max_consecutive_errors:
                report.metadata["aborted"] = True
                report.metadata["abort_reason"] = (
                    f"{consecutive_errors} consecutive errors"
                )
                break
    else:
        # Batch execution
        done_count = 0
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            coros = [run_task(t, model_fn, timeout) for t in batch]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    task = batch[results.index(r)] if results.index(r) < len(batch) else batch[0]
                    result = TaskResult(
                        task_id=task.id,
                        complexity=task.complexity,
                        tags=task.tags,
                        response=f"Error: {type(r).__name__}: {r}",
                    )
                    report.results.append(result)
                    consecutive_errors += 1
                    done_count += 1
                    _progress(result, done_count)
                else:
                    report.results.append(r)
                    report.total_tokens_in += r.tokens_in
                    report.total_tokens_out += r.tokens_out
                    report.total_cost += r.cost
                    if r.response.startswith("Error:"):
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                    done_count += 1
                    _progress(r, done_count)

            if consecutive_errors >= max_consecutive_errors:
                report.metadata["aborted"] = True
                report.metadata["abort_reason"] = (
                    f"{consecutive_errors} consecutive errors"
                )
                break

    report.total_time_s = time.perf_counter() - total_start
    return report


def run_tasks_sync(
    tasks: list[Task],
    model_fn: Callable,
    **kwargs,
) -> Report:
    """Sync wrapper for run_tasks."""
    return asyncio.run(run_tasks(tasks, model_fn, **kwargs))
