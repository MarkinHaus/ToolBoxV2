"""
Interactive CLI entry point: python -m bench

Modes:
    1. Run single benchmark
    2. Run multi-model benchmark
    3. Calibrate judge LLM
    4. Generate dashboard
    5. List tasks / validators
    6. Exit

Also supports non-interactive via subcommands:
    python -m bench run ...
    python -m bench multirun ...
    python -m bench calibrate ...
    python -m bench dashboard ...
    python -m bench list ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


# -- Helpers ------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val or default


def _ask_int(prompt: str, default: int) -> int:
    raw = _ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _ask_float(prompt: str, default: float) -> float:
    raw = _ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _ask_bool(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    raw = input(f"  {prompt} [{d}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _ask_choice(prompt: str, options: list[str], default: int = 0) -> str:
    print(f"  {prompt}")
    for i, opt in enumerate(options):
        marker = ">" if i == default else " "
        print(f"    {marker} {i + 1}. {opt}")
    raw = input(f"  Choice [{default + 1}]: ").strip()
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except ValueError:
        pass
    return options[default]


def _ask_multi_choice(prompt: str, options: list[str], defaults: list[int] | None = None) -> list[str]:
    """Ask user to select one or more options by number (comma-separated)."""
    defaults = defaults or [0]
    default_str = ",".join(str(d + 1) for d in defaults)
    print(f"  {prompt}")
    for i, opt in enumerate(options):
        marker = ">" if i in defaults else " "
        print(f"    {marker} {i + 1}. {opt}")
    raw = input(f"  Select (comma-separated) [{default_str}]: ").strip()
    if not raw:
        return [options[d] for d in defaults]
    selected = []
    for part in raw.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(options) and options[idx] not in selected:
                selected.append(options[idx])
        except ValueError:
            pass
    return selected or [options[defaults[0]]]


def _header(title: str):
    w = 60
    print()
    print(f"{'─' * w}")
    print(f"  {title}")
    print(f"{'─' * w}")


def _menu() -> str:
    _header("bench — LLM Benchmark Framework")
    options = [
        "Run single benchmark",
        "Run multi-model benchmark",
        "Calibrate judge LLM",
        "Generate dashboard",
        "List tasks & validators",
        "Exit",
    ]
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print()
    raw = input("  Select [1-6]: ").strip()
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    except ValueError:
        pass
    return ""


# -- Interactive Flows --------------------------------------------------------

async def interactive_run():
    """Single model benchmark — interactive config."""
    _header("Single Benchmark Run")
    from toolboxv2 import tb_root_dir
    task_dir = _ask("Task directory", str(tb_root_dir / "mods" / "isaa" / "base" / "bench" / "tasks"))
    model = _ask("LiteLLM model string (e.g. openrouter/google/gemini-2.5-flash)", "")
    if not model:
        print("  ✗ Model is required.")
        return

    model_id = _ask("Model ID (human-readable)", model.rsplit("/", 1)[-1])
    suite = _ask("Suite YAML (leave empty for all tasks)", "")
    mode = _ask("Mode", "standard")
    modalities = _ask("Modalities (comma-separated)", "text")
    timeout = _ask_float("Timeout per task (s)", 90.0)
    output = _ask("Output JSON path", f"report_{model_id.replace('/', '_')}.json")

    print()
    print(f"  → Running {model_id} against {task_dir} ...")
    print()

    await cmd_run_with(
        task_dir=task_dir,
        model=model,
        model_id=model_id,
        suite=suite or None,
        mode=mode,
        modalities=modalities,
        timeout=timeout,
        output=output,
    )


def interactive_multirun():
    """Multi-model benchmark — interactive config."""
    _header("Multi-Model Benchmark")
    from toolboxv2 import tb_root_dir
    task_dir = _ask("Task directory", str(tb_root_dir / "mods" / "isaa" / "base" / "bench" / "tasks"))

    models: dict[str, str] = {}
    print("  Add models (empty model ID to finish):")
    while True:
        mid = _ask(f"  Model ID #{len(models) + 1}", "")
        if not mid:
            break
        mstr = _ask(f"  LiteLLM string for '{mid}'", "")
        if not mstr:
            print("    ✗ Skipped (no model string)")
            continue
        models[mid] = mstr
        print(f"    ✓ {mid} = {mstr}")

    if not models:
        print("  ✗ No models configured.")
        return

    output_dir = _ask("Output directory", "reports/")
    suite = _ask("Suite YAML (leave empty for all)", "")

    # Multi-adapter selection
    adapters = _ask_multi_choice(
        "Adapter type(s) — select multiple for comparison",
        ["row", "agent", "stream"],
        defaults=[0],
    )

    modalities = _ask("Modalities", "text")
    timeout = _ask_float("Timeout per task (s)", 90.0)
    seed_raw = _ask("Seed (empty for none)", "")
    seed = int(seed_raw) if seed_raw else None
    skip = _ask_bool("Skip existing reports?", True)
    dashboard_path = _ask("Dashboard output", "dashboard.html")

    adapter_label = ", ".join(adapters)
    total_jobs = len(models) * len(adapters)
    print()
    print(f"  → Running {len(models)} models × {len(adapters)} adapters = {total_jobs} jobs ({adapter_label}) ...")
    print()

    cmd_multirun_with(
        models=models,
        task_dir=task_dir,
        output_dir=output_dir,
        suite_path=suite,
        mode="standard",
        modalities=modalities.split(","),
        timeout=timeout,
        seed=seed,
        adapter_type=adapters if len(adapters) > 1 else adapters[0],
        skip_existing=skip,
        dashboard_path=dashboard_path,
    )


async def interactive_calibrate():
    """Judge calibration — interactive config."""
    _header("Judge LLM Calibration")
    from toolboxv2 import tb_root_dir
    task_dir = _ask("Task directory with ground-truth tasks",
                    str(tb_root_dir / "mods" / "isaa" / "base" / "bench" / "tasks" / "calibration"))
    threshold = _ask_float("Accuracy threshold", 0.95)
    output = _ask("Output profile JSON", "judge_profile.json")

    print()
    print(f"  → Calibrating judge against {task_dir} (threshold={threshold}) ...")
    print()

    await cmd_calibrate_with(
        task_dir=task_dir,
        threshold=threshold,
        output=output,
    )


def interactive_dashboard():
    """Dashboard generation — interactive config."""
    _header("Dashboard Generation")

    files_raw = _ask("Report JSON files (space-separated or glob)", "reports/*.json")

    import glob
    files = []
    for part in files_raw.split():
        expanded = glob.glob(part)
        if expanded:
            files.extend(expanded)
        else:
            files.append(part)

    if not files:
        print("  ✗ No files found.")
        return

    print(f"  Found {len(files)} file(s): {', '.join(Path(f).name for f in files)}")
    output = _ask("Output HTML", "dashboard.html")
    title = _ask("Dashboard title", "Benchmark Comparison")

    cmd_dashboard_with(files=files, output=output, title=title)


def interactive_list():
    """List tasks and validators."""
    _header("List Tasks & Validators")
    from toolboxv2 import tb_root_dir

    task_dir = _ask("Task directory (empty to skip)",
                    str(tb_root_dir / "mods" / "isaa" / "base" / "bench" / "tasks"))
    cmd_list_with(task_dir=task_dir if task_dir else None)


# -- Execution Functions ------------------------------------------------------

async def cmd_run_with(
    task_dir: str,
    model: str,
    model_id: str,
    suite: str | None,
    mode: str,
    modalities: str,
    timeout: float,
    output: str,
):
    from toolboxv2.mods.isaa.base.bench import RowModelAdapter, load_tasks_from_dir

    tasks = load_tasks_from_dir(Path(task_dir))
    if not tasks:
        print(f"  ✗ No tasks found in {task_dir}")
        return
    print(f"  Loaded {len(tasks)} tasks")

    mod_list = [m.strip() for m in modalities.split(",")]
    adapter = RowModelAdapter(
        model_name=model,
        task_dir=task_dir,
        suite_path=suite or "",
        model_modalities=mod_list,
        timeout=timeout,
    )

    report = await adapter.benchmark(model_id=model_id, mode=mode)
    report.save(output)

    print(f"  Score: {report.total_score * 100:.1f}%")
    print(f"  Tasks: {len(report.results)}, Time: {report.total_time_s:.1f}s")
    print(f"  Saved: {output}")
    for tag, score in sorted(report.scores_by_tag().items()):
        print(f"    {tag}: {score * 100:.1f}%")


def cmd_multirun_with(
    models: dict[str, str],
    task_dir: str,
    output_dir: str,
    suite_path: str,
    mode: str,
    modalities: list[str],
    timeout: float,
    seed: int | None,
    adapter_type: str | list[str],
    skip_existing: bool,
    dashboard_path: str,
):
    from toolboxv2.mods.isaa.base.bench.multirun import multi_benchmark, aggregate_dashboard

    paths = multi_benchmark(
        models=models,
        task_dir=task_dir,
        output_dir=output_dir,
        suite_path=suite_path,
        mode=mode,
        modalities=modalities,
        timeout=timeout,
        seed=seed,
        adapter_type=adapter_type,
        skip_existing=skip_existing,
    )

    if paths:
        aggregate_dashboard(paths, dashboard_path)


async def cmd_calibrate_with(task_dir: str, threshold: float, output: str):
    from toolboxv2.mods.isaa.base.bench import load_tasks_from_dir
    from toolboxv2.mods.isaa.base.bench.calibrator import calibrate

    tasks = load_tasks_from_dir(Path(task_dir))
    gt_tasks = [t for t in tasks if t.ground_truth]
    if not gt_tasks:
        print(f"  ✗ No ground-truth tasks found in {task_dir}")
        return
    print(f"  Found {len(gt_tasks)} ground-truth tasks")

    from toolboxv2 import get_app
    app = get_app()
    isaa = app.get_mod("isaa")
    profile = await calibrate(gt_tasks, isaa, threshold=threshold)

    Path(output).write_text(json.dumps(profile.to_dict(), indent=2))
    print(f"  Judge: {profile.model}")
    print(f"  Disqualified: {profile.disqualified}")
    for c, bs in profile.batch_sizes.items():
        acc = profile.accuracy.get(c, 0)
        print(f"    {c}: batch_size={bs}, accuracy={acc:.2%}")
    print(f"  Saved: {output}")


def cmd_dashboard_with(files: list[str], output: str, title: str):
    from toolboxv2.mods.isaa.base.bench.dashboard import Dashboard

    reports = []
    for fp in files:
        try:
            with open(fp) as f:
                data = json.load(f)
                if isinstance(data, list):
                    reports.extend(data)
                elif isinstance(data, dict):
                    if "model" in data:
                        reports.append(data)
                    else:
                        for v in data.values():
                            if isinstance(v, list):
                                reports.extend(v)
        except Exception as e:
            print(f"  ⚠ Skipping {fp}: {e}")

    if not reports:
        print("  ✗ No reports found")
        return

    Dashboard.save(reports, output, title)
    print(f"  Dashboard: {output} ({len(reports)} reports)")


def cmd_list_with(task_dir: str | None):
    from toolboxv2.mods.isaa.base.bench.validators import list_validators

    print("  Validators:")
    for name in sorted(list_validators()):
        print(f"    • {name}")

    if task_dir:
        from toolboxv2.mods.isaa.base.bench import load_tasks_from_dir
        tasks = load_tasks_from_dir(Path(task_dir))
        print(f"\n  Tasks ({len(tasks)}):")
        for t in tasks:
            mm = ",".join(t.modality)
            tags = ",".join(t.tags) if t.tags else "—"
            gt = "✓" if t.ground_truth else " "
            print(f"    [{t.complexity:8s}] {t.id:30s} [{mm:10s}] tags={tags} gt={gt}")


# -- CLI Argument Parser (non-interactive) ------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Binary LLM Benchmark — interactive or subcommand mode",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run single benchmark")
    p_run.add_argument("--task-dir", required=True)
    p_run.add_argument("--model", required=True, help="LiteLLM model string")
    p_run.add_argument("--model-id", help="Human-readable model ID")
    p_run.add_argument("--suite", help="Suite YAML file")
    p_run.add_argument("--mode", default="standard")
    p_run.add_argument("--modalities", default="text")
    p_run.add_argument("--timeout", type=float, default=90.0)
    p_run.add_argument("--output", "-o")

    # multirun
    p_multi = sub.add_parser("multirun", help="Run multi-model benchmark")
    p_multi.add_argument("models", nargs="+", help="model_id=litellm_string pairs")
    p_multi.add_argument("--task-dir", required=True)
    p_multi.add_argument("--output-dir", default="reports")
    p_multi.add_argument("--suite", default="")
    p_multi.add_argument("--mode", default="standard")
    p_multi.add_argument("--modalities", default="text")
    p_multi.add_argument("--timeout", type=float, default=90.0)
    p_multi.add_argument("--seed", type=int, default=None)
    p_multi.add_argument("--adapter", default="row",
                         help="Adapter(s): row, agent, stream. Comma-separated for multi: row,agent,stream")
    p_multi.add_argument("--workers", type=int, default=None)
    p_multi.add_argument("--no-skip", action="store_true")
    p_multi.add_argument("--dashboard", "-d", default="dashboard.html")
    p_multi.add_argument("--title", default="Benchmark Comparison")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate judge LLM")
    p_cal.add_argument("--task-dir", required=True)
    p_cal.add_argument("--threshold", type=float, default=0.95)
    p_cal.add_argument("--output", "-o")

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Generate HTML dashboard")
    p_dash.add_argument("files", nargs="+", help="Report JSON files")
    p_dash.add_argument("--output", "-o")
    p_dash.add_argument("--title", "-t")

    # list
    p_list = sub.add_parser("list", help="List validators and tasks")
    p_list.add_argument("--task-dir")

    return parser


async def dispatch_subcommand(args):
    """Handle non-interactive subcommands."""
    if args.command == "run":
        await cmd_run_with(
            task_dir=args.task_dir,
            model=args.model,
            model_id=args.model_id or args.model,
            suite=args.suite,
            mode=args.mode,
            modalities=args.modalities,
            timeout=args.timeout,
            output=args.output or f"report_{(args.model_id or args.model).replace('/', '_')}.json",
        )

    elif args.command == "multirun":
        models = {}
        for m in args.models:
            if "=" not in m:
                print(f"  ✗ Invalid model format: {m} (expected model_id=litellm_string)")
                sys.exit(1)
            k, v = m.split("=", 1)
            models[k] = v

        # Parse comma-separated adapters
        adapter_list = [a.strip() for a in args.adapter.split(",")]
        adapter_type = adapter_list if len(adapter_list) > 1 else adapter_list[0]

        cmd_multirun_with(
            models=models,
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            suite_path=args.suite,
            mode=args.mode,
            modalities=args.modalities.split(","),
            timeout=args.timeout,
            seed=args.seed,
            adapter_type=adapter_type,
            skip_existing=not args.no_skip,
            dashboard_path=args.dashboard,
        )

    elif args.command == "calibrate":
        await cmd_calibrate_with(
            task_dir=args.task_dir,
            threshold=args.threshold,
            output=args.output or "judge_profile.json",
        )

    elif args.command == "dashboard":
        cmd_dashboard_with(
            files=args.files,
            output=args.output or "dashboard.html",
            title=args.title or "Benchmark Comparison",
        )

    elif args.command == "list":
        cmd_list_with(task_dir=args.task_dir)


# -- Main ---------------------------------------------------------------------

async def main():
    # If subcommand args given, dispatch directly
    if len(sys.argv) > 1 and sys.argv[1] in ("run", "multirun", "calibrate", "dashboard", "list"):
        parser = build_parser()
        args = parser.parse_args()
        await dispatch_subcommand(args)
        return

    # Interactive mode
    try:
        while True:
            choice = _menu()
            if choice == "Run single benchmark":
                await interactive_run()
            elif choice == "Run multi-model benchmark":
                interactive_multirun()
            elif choice == "Calibrate judge LLM":
                await interactive_calibrate()
            elif choice == "Generate dashboard":
                interactive_dashboard()
            elif choice == "List tasks & validators":
                interactive_list()
            elif choice == "Exit":
                print("  Bye.")
                break
            else:
                print("  ✗ Invalid selection, try again.")
    except (KeyboardInterrupt, EOFError):
        print("\n  Bye.")


if __name__ == "__main__":
    asyncio.run(main())
