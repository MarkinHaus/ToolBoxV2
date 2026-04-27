"""
Judge Calibrator — Schicht 0.
Tests a judge LLM against ground-truth tasks to find:
1. Per-complexity accuracy (must be ≥ threshold)
2. Optimal batch sizes before degradation
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable

from pydantic import BaseModel

from toolboxv2 import Spinner
from toolboxv2.mods.isaa.base.bench.core import JudgeProfile, Task, TaskContext, CheckResult


async def _judge_single(task: Task, isaa_mod) -> bool:
    """Ask judge to evaluate a ground-truth task. Returns True if correct."""
    if not task.ground_truth:
        return True  # skip tasks without ground truth

    # Simulate: the "model response" IS the ground truth
    # Judge must confirm it's correct → should return yes
    prompt = (
        f"You are a strict binary judge. Answer ONLY 'yes' or 'no'.\n\n"
        f"## Prompt:\n{task.prompt}\n\n"
        f"## Response:\n{task.ground_truth}\n\n"
        f"## Question:\nIs this response correct and complete?\n\n"
        f"Answer 'True' or 'False'."
    )

    class Answer(BaseModel):
        answer: bool

    try:
        print(prompt)
        result = await isaa_mod.format_class(
            format_schema=Answer,
            task=prompt,
            agent_name="BenchJudgeCal",
        )
        result = result.get()
        print(result)
        if result is None:
            return False
        if isinstance(result, Answer):
            return result.answer
        answer = ""
        if isinstance(result, dict):
            answer = str(result.get("answer", result.get("result", ""))).strip().lower()
        else:
            answer = str(result).strip().lower()
        return answer in ("yes", "true", "1", "ja")
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        return False


async def _judge_batch(tasks: list[Task], isaa_mod) -> float:
    """Run judge on a batch concurrently, return accuracy 0-1."""
    if not tasks:
        return 1.0
    results = [await _judge_single(t, isaa_mod) for t in tasks]
    # results = await asyncio.gather(*coros, return_exceptions=True)
    correct = sum(1 for r in results if r is True)
    return correct / len(results)


async def calibrate(
    ground_truth_tasks: list[Task],
    isaa_mod,
    threshold: float = 0.95,
    batch_sizes: list[int] | None = None,
) -> JudgeProfile:
    """Calibrate a judge LLM.

    1. Test single-task accuracy per complexity level
    2. Ramp up batch sizes to find degradation point
    3. Return JudgeProfile with optimal batch sizes

    Args:
        ground_truth_tasks: Tasks with ground_truth field set.
        isaa_mod: ISAA module instance (has format_class).
        threshold: Minimum accuracy to pass (default 95%).
        batch_sizes: Batch sizes to test (default [2,4,8,16,32]).
    """
    if batch_sizes is None:
        batch_sizes = [2, 4, 8, 16, 32]

    profile = JudgeProfile()

    # Try to get model name from ISAA
    try:
        agent = await isaa_mod.get_agent("BenchJudgeCal")
        profile.model = os.getenv("BLITZMODEL", agent.amd.fast_llm_model)
        agent.amd.complex_llm_model = agent.amd.fast_llm_model
        agent.amd.fast_llm_model = profile.model
    except Exception:
        profile.model = "unknown"

    # Group tasks by complexity
    by_complexity: dict[str, list[Task]] = {}
    for t in ground_truth_tasks:
        if t.ground_truth:
            by_complexity.setdefault(t.complexity, []).append(t)

    for complexity in ["tutorial", "extended", "phd"]:
        print(f"Calibrating {complexity}...")
        tasks = by_complexity.get(complexity, [])
        if not tasks:
            profile.batch_sizes[complexity] = 1
            profile.accuracy[complexity] = 1.0
            continue

        # Phase 1: single accuracy
        print(f"Total Single task(s) {len(tasks)}")
        single_acc = await _judge_batch(tasks, isaa_mod)
        print(f"Single task  {single_acc}")
        profile.accuracy[complexity] = single_acc

        if single_acc < threshold:
            profile.disqualified = True
            return profile

        # Phase 2: batch size ramp-up
        best_batch = 1
        for bs in batch_sizes:
            if bs > len(tasks):
                # Not enough tasks to test this batch size, keep last good
                best_batch = bs
                break
            # Take first bs tasks as a batch
            acc = await _judge_batch(tasks[:bs], isaa_mod)
            print(f"{complexity}: Batch {bs}: {acc}")
            if acc < threshold:
                break
            best_batch = bs

        profile.batch_sizes[complexity] = best_batch

        return profile
