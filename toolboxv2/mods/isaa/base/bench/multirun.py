"""
Multi-process benchmark runner.
Each model runs in its own subprocess — fully isolated, no shared state.
Each process writes its own report JSON file.
Aggregate reports into a single dashboard afterwards.

Usage:
    from bench.multirun import multi_benchmark, aggregate_dashboard

    # Define models
    models = {
        "gemini-flash": "openrouter/google/gemini-2.5-flash",
        "claude-haiku": "openrouter/anthropic/claude-haiku-4.5",
        "qwen3-8b": "openrouter/qwen/qwen3-8b",
    }

    # Run all in parallel processes
    report_paths = multi_benchmark(
        models=models,
        task_dir="tasks/",
        output_dir="reports/",
        mode="standard",
    )

    # Generate combined dashboard
    aggregate_dashboard(report_paths, output="comparison.html")
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _worker(
    model_id: str,
    model_name: str,
    task_dir: str,
    suite_path: str,
    output_path: str,
    mode: str,
    modalities: list[str],
    timeout: float,
    seed: int | None,
    adapter_type: str,
):
    """Worker function — runs in its own process. No shared state."""
    import asyncio
    import sys

    # Force unbuffered stdout in child process
    sys.stdout.reconfigure(line_buffering=True)

    # Progress callback that prefixes with model_id
    def on_progress(done, total, task_id, passed, result):
        pct = done / total * 100 if total else 0
        icon = "✓" if result.score == 1.0 else "✗" if result.score == 0.0 else "◐"
        print(
            f"  [{model_id:20s}] [{done}/{total}] {pct:5.1f}% {icon} {task_id:30s} "
            f"{result.score*100:5.1f}% {result.latency_ms:5d}ms",
            flush=True,
        )

    # Re-import everything fresh in this process
    from toolboxv2.mods.isaa.base.bench.adapters import RowModelAdapter, AgentAdapter, AgentStreamAdapter
    from toolboxv2.mods.isaa.base.bench.core import Report

    try:
        if adapter_type == "row":
            adapter = RowModelAdapter(
                model_name=model_name,
                task_dir=task_dir,
                suite_path=suite_path,
                model_modalities=modalities,
                timeout=timeout,
                on_progress=on_progress,
            )
            report = asyncio.run(
                adapter.benchmark(model_id=model_id, mode=mode, seed=seed)
            )

        elif adapter_type in ("agent", "stream"):
            # Agent/Stream require ISAA
            from toolboxv2 import get_app

            app = get_app()
            isaa = app.get_mod("isaa")
            agent = asyncio.run(isaa.get_agent("self"))
            agent.amd.fast_llm_model = model_name
            agent.amd.complex_llm_model = model_name

            if adapter_type == "agent":
                adapter = AgentAdapter(
                    agent=agent,
                    task_dir=task_dir,
                    suite_path=suite_path,
                    model_modalities=modalities,
                    probe_timeout=timeout,
                    on_progress=on_progress,
                )
            else:
                adapter = AgentStreamAdapter(
                    agent=agent,
                    task_dir=task_dir,
                    suite_path=suite_path,
                    model_modalities=modalities,
                    probe_timeout=timeout,
                    on_progress=on_progress,
                )
            report = asyncio.run(
                adapter.benchmark(model_id=model_id, mode=mode, seed=seed)
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        # Write report JSON
        report.save(output_path)
        score = report.total_score * 100
        n = len(report.results)
        t = report.total_time_s
        print(f"[DONE] {model_id}: {score:.1f}% ({n} tasks, {t:.1f}s) -> {output_path}")
        return output_path

    except Exception as e:
        # Write error report so aggregation doesn't miss this model
        error_report = {
            "model": model_id,
            "mode": mode,
            "total": 0,
            "total_raw": 0,
            "flag_penalty": 0,
            "dimensions": {},
            "persona": {},
            "flags": [],
            "probes": 0,
            "cost": {
                "total_cost": 0, "total_tokens": 0,
                "tokens_in": 0, "tokens_out": 0,
                "total_time_s": 0, "cost_per_probe": 0,
                "time_per_probe_s": 0, "tokens_per_probe": 0,
            },
            "results": [],
            "error": f"{type(e).__name__}: {e}",
        }
        Path(output_path).write_text(json.dumps(error_report, indent=2))
        print(f"[FAIL] {model_id}: {type(e).__name__}: {e}", file=sys.stderr)
        return output_path


def multi_benchmark(
    models: dict[str, str],
    task_dir: str | Path,
    output_dir: str | Path = "reports",
    suite_path: str | Path = "",
    mode: str = "standard",
    modalities: list[str] | None = None,
    timeout: float = 90.0,
    seed: int | None = None,
    adapter_type: str | list[str] = "row",
    max_workers: int | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Run benchmarks for multiple models in parallel processes.

    Args:
        models: {model_id: litellm_model_string}
        task_dir: Directory containing task YAML files
        output_dir: Directory for report JSON files
        suite_path: Optional suite YAML file
        mode: Benchmark mode
        modalities: Model capabilities ["text", "image", ...]
        timeout: Per-task timeout in seconds
        seed: Random seed for reproducibility
        adapter_type: "row", "agent", "stream" or a list like ["row", "agent"]
                      Each model is tested with each adapter, producing separate reports.
        max_workers: Max parallel processes (default: num jobs)
        skip_existing: Skip models that already have report files

    Returns:
        List of report file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modalities = modalities or ["text"]

    # Normalize adapter_type to list
    adapters = [adapter_type] if isinstance(adapter_type, str) else list(adapter_type)

    # Determine which models × adapters to run
    jobs: list[tuple[str, str, str]] = []
    existing_paths: list[Path] = []

    for model_id, model_name in models.items():
        for adapter in adapters:
            # Suffix model_id with adapter type when testing multiple adapters
            if len(adapters) > 1:
                job_id = f"{model_id}_{adapter}"
            else:
                job_id = model_id

            safe_name = job_id.replace("/", "_").replace(":", "_")
            out_path = output_dir / f"{safe_name}.json"

            if skip_existing and out_path.exists():
                print(f"[SKIP] {job_id}: {out_path} exists")
                existing_paths.append(out_path)
                continue

            jobs.append((job_id, model_name, str(out_path), adapter))

    if not jobs:
        print("No models to run (all existing)")
        return existing_paths

    adapter_label = ", ".join(adapters) if len(adapters) > 1 else adapters[0]
    print(f"Running {len(jobs)} jobs in parallel ({adapter_label} mode)")
    t0 = time.perf_counter()

    # Use multiprocessing Pool
    workers = max_workers or len(jobs)
    pool_args = [
        (
            job_id, model_name, str(task_dir), str(suite_path),
            out_path, mode, modalities, timeout, seed, adapter,
        )
        for job_id, model_name, out_path, adapter in jobs
    ]

    result_paths = list(existing_paths)

    with mp.Pool(processes=workers) as pool:
        results = pool.starmap(_worker, pool_args)
        for r in results:
            if r:
                result_paths.append(Path(r))

    elapsed = time.perf_counter() - t0
    print(f"All done in {elapsed:.1f}s — {len(result_paths)} reports")
    return result_paths


def aggregate_dashboard(
    report_paths: list[Path | str],
    output: str = "dashboard.html",
    title: str = "Benchmark Comparison",
):
    """Load report JSONs and generate a combined dashboard."""
    from toolboxv2.mods.isaa.base.bench.dashboard import Dashboard

    reports = []
    for p in report_paths:
        try:
            with open(p) as f:
                data = json.load(f)
                reports.append(data)
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")

    if not reports:
        print("No valid reports to aggregate")
        return

    Dashboard.save(reports, output, title)
    print(f"Dashboard: {output} ({len(reports)} models)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI shortcut
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for multi-process benchmark.

    Usage:
        python -m bench.multirun --task-dir tasks/ --output-dir reports/ \\
            model1=openrouter/google/gemini-2.5-flash \\
            model2=openrouter/anthropic/claude-haiku-4.5
    """
    import argparse

    parser = argparse.ArgumentParser(prog="bench.multirun")
    parser.add_argument("models", nargs="+", help="model_id=litellm_string pairs")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--suite", default="")
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--modalities", default="text")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapter", default="row",
                        help="Adapter(s): row, agent, stream. Comma-separated for multi: row,agent")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--dashboard", "-d", default="dashboard.html")
    parser.add_argument("--title", default="Benchmark Comparison")

    args = parser.parse_args()

    # Parse model_id=model_string pairs
    models = {}
    for m in args.models:
        if "=" not in m:
            print(f"Invalid model format: {m} (expected model_id=litellm_string)")
            sys.exit(1)
        k, v = m.split("=", 1)
        models[k] = v

    paths = multi_benchmark(
        models=models,
        task_dir=args.task_dir,
        output_dir=args.output_dir,
        suite_path=args.suite,
        mode=args.mode,
        modalities=args.modalities.split(","),
        timeout=args.timeout,
        seed=args.seed,
        adapter_type=args.adapter.split(","),
        max_workers=args.workers,
        skip_existing=not args.no_skip,
    )

    if paths:
        aggregate_dashboard(paths, args.dashboard, args.title)


if __name__ == "__main__":
    main()
