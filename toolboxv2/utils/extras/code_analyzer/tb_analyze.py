"""
tb_analyze — Unified API & CLI for static + runtime code analysis.

Delegates to tb_analyze_static and tb_analyze_runtime.
Provides a combined mode: static-analyze only the files touched at runtime.

API:
    from toolboxv2.utils.extras.code_analyzer.tb_analyze import (
        static_analyze, runtime_run, runtime_report, runtime_monitor,
        analyze_runtime_touched, full_pipeline,
    )

CLI:
    tb analyze static <target> [--metrics ...] [--exclude ...] [--html out.html] [--json out.json]
    tb analyze runtime run "<command>" [--outdir ./rd] [--interval 2] [--memray]
    tb analyze runtime report <outdir> [-o report.html]
    tb analyze runtime monitor [--pid PID] [--outdir ./rd] [--interval 2]
    tb analyze touched <runtime_outdir> [--metrics ...] [--html out.html]
    tb analyze pipeline "<command>" [--outdir ./rd] [--interval 2] [--metrics ...] [--html out.html]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger("tb_analyze")

if TYPE_CHECKING:
    from toolboxv2.utils.extras.code_analyzer.tb_analyze_static import AnalysisReport
    from toolboxv2.utils.extras.code_analyzer.tb_analyze_runtime import RuntimeMonitor as _RTMon


# ---------------------------------------------------------------------------
# Lazy imports — only pull in what's needed
# ---------------------------------------------------------------------------

def _static():
    from toolboxv2.utils.extras.code_analyzer.tb_analyze_static import (
        analyze_path, analyze_file, analyze_from_config,
        generate_html_report, discover_files, AnalysisReport,
    )
    return analyze_path, analyze_file, analyze_from_config, generate_html_report, discover_files, AnalysisReport


def _runtime():
    from toolboxv2.utils.extras.code_analyzer.tb_analyze_runtime import (
        RuntimeMonitor, run_with_monitoring, generate_runtime_report, JsonlWriter,
    )
    return RuntimeMonitor, run_with_monitoring, generate_runtime_report, JsonlWriter


# ---------------------------------------------------------------------------
# API — Static
# ---------------------------------------------------------------------------

def static_analyze(
    target: str | Path,
    metrics: list[str] | None = None,
    languages: list[str] | None = None,
    exclude: list[str] | None = None,
    max_files: int = 200,
    html_output: str | Path | None = None,
    json_output: str | Path | None = None,
) -> "AnalysisReport":
    """Run static analysis. Wraps analyze_path + optional report generation."""
    analyze_path, _, _, generate_html_report, _, _ = _static()

    report = analyze_path(target, metrics=metrics, languages=languages, exclude=exclude, max_files=max_files)

    if html_output:
        html = generate_html_report(report)
        Path(html_output).parent.mkdir(parents=True, exist_ok=True)
        Path(html_output).write_text(html, encoding="utf-8")
        logger.info("Static HTML report: %s", html_output)

    if json_output:
        report.to_json(json_output)
        logger.info("Static JSON report: %s", json_output)

    return report


# ---------------------------------------------------------------------------
# API — Runtime
# ---------------------------------------------------------------------------

def runtime_run(
    command: str,
    outdir: str = "./runtime_data",
    interval: float = 2.0,
    memray: bool = False,
    **kwargs,
) -> int:
    """Run a command with runtime monitoring. Returns exit code."""
    _, run_with_monitoring, _, _ = _runtime()
    return run_with_monitoring(command, outdir=outdir, interval=interval, memray=memray, **kwargs)


def runtime_report(
    outdir: str | Path,
    output: str | Path | None = None,
) -> str:
    """Generate HTML report from runtime data."""
    _, _, generate_runtime_report, _ = _runtime()
    return generate_runtime_report(outdir, output)


def runtime_monitor(
    pid: int | None = None,
    outdir: str = "./runtime_data",
    interval: float = 2.0,
    **kwargs,
) -> "_RTMon":
    """Create and start a RuntimeMonitor. Call .stop() when done."""
    RuntimeMonitor, _, _, _ = _runtime()
    mon = RuntimeMonitor(outdir=outdir, interval=interval, **kwargs)
    mon.start(pid=pid)
    return mon


# ---------------------------------------------------------------------------
# API — Touched-files mode: static analyze only runtime-loaded files
# ---------------------------------------------------------------------------

def _extract_touched_files(runtime_outdir: str | Path) -> list[Path]:
    """Extract unique file paths from runtime modules.jsonl."""
    _, _, _, JsonlWriter = _runtime()
    outdir = Path(runtime_outdir)
    modules_path = outdir / "modules.jsonl"

    records = JsonlWriter.read(modules_path)
    if not records:
        logger.warning("No module data in %s", modules_path)
        return []

    # Use the last snapshot (most complete)
    last = records[-1]
    modules = last.get("modules", [])

    seen: set[str] = set()
    result: list[Path] = []
    for m in modules:
        fpath = m.get("file", "")
        if not fpath or fpath in seen:
            continue
        p = Path(fpath)
        if p.exists() and p.suffix == ".py":
            seen.add(fpath)
            result.append(p)

    return result


def analyze_runtime_touched(
    runtime_outdir: str | Path,
    metrics: list[str] | None = None,
    html_output: str | Path | None = None,
    json_output: str | Path | None = None,
) -> "AnalysisReport":
    """Static-analyze only the Python files that were loaded at runtime.

    Reads modules.jsonl from a runtime analysis run, then runs static
    analysis on exactly those files — no more, no less.
    """
    _, analyze_file, _, generate_html_report, _, AnalysisReport = _static()
    import time
    from datetime import datetime, timezone

    files = _extract_touched_files(runtime_outdir)
    if not files:
        raise FileNotFoundError(f"No runtime-touched files found in {runtime_outdir}/modules.jsonl")

    t0 = time.monotonic()
    report = AnalysisReport(
        target=str(runtime_outdir),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={"metrics": metrics or [], "mode": "runtime_touched", "source_files": len(files)},
    )

    for fpath in files:
        fm = analyze_file(fpath, language="python", metrics=metrics)
        report.files.append(fm)

    report.duration_s = time.monotonic() - t0

    # Compute summary using the static module's helper
    from toolboxv2.utils.extras.code_analyzer.tb_analyze_static import _compute_summary
    report.summary = _compute_summary(report)

    if html_output:
        html = generate_html_report(report)
        Path(html_output).parent.mkdir(parents=True, exist_ok=True)
        Path(html_output).write_text(html, encoding="utf-8")

    if json_output:
        report.to_json(json_output)

    return report


# ---------------------------------------------------------------------------
# API — Full pipeline: runtime run → static on touched files
# ---------------------------------------------------------------------------

def full_pipeline(
    command: str,
    outdir: str = "./runtime_data",
    interval: float = 2.0,
    memray: bool = False,
    metrics: list[str] | None = None,
    html_output: str | Path | None = None,
) -> tuple[int, "AnalysisReport"]:
    """Run command with monitoring, then static-analyze touched files.

    Returns (exit_code, static_report).
    """
    exit_code = runtime_run(command, outdir=outdir, interval=interval, memray=memray)

    # Generate runtime report
    rt_html = str(Path(outdir) / "runtime_report.html")
    runtime_report(outdir, output=rt_html)

    # Static on touched files
    static_html = html_output or str(Path(outdir) / "touched_static_report.html")
    report = analyze_runtime_touched(outdir, metrics=metrics, html_output=static_html)

    return exit_code, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="tb_analyze",
        description="Unified static + runtime code analysis",
    )
    sub = parser.add_subparsers(dest="cmd")

    # --- static ---
    p_static = sub.add_parser("static", help="Static analysis of files/directories")
    p_static.add_argument("target", help="File or directory to analyze")
    p_static.add_argument("--metrics", nargs="*", default=None, help="Metrics to run")
    p_static.add_argument("--languages", nargs="*", default=None)
    p_static.add_argument("--exclude", nargs="*", default=None)
    p_static.add_argument("--max-files", type=int, default=200)
    p_static.add_argument("--html", default=None, help="HTML report output path")
    p_static.add_argument("--json", default=None, help="JSON report output path")

    # --- runtime ---
    p_rt = sub.add_parser("runtime", help="Runtime analysis")
    rt_sub = p_rt.add_subparsers(dest="rt_cmd")

    p_run = rt_sub.add_parser("run", help="Run command with monitoring")
    p_run.add_argument("command", help="Command to run")
    p_run.add_argument("--outdir", default="./runtime_data")
    p_run.add_argument("--interval", type=float, default=2.0)
    p_run.add_argument("--memray", action="store_true")
    p_run.add_argument("--no-objects", action="store_true")
    p_run.add_argument("--no-network", action="store_true")

    p_rep = rt_sub.add_parser("report", help="Generate runtime report")
    p_rep.add_argument("outdir", help="Runtime data directory")
    p_rep.add_argument("-o", "--output", default=None)

    p_mon = rt_sub.add_parser("monitor", help="Attach monitor to running process")
    p_mon.add_argument("--pid", type=int, default=None, help="PID (default: self)")
    p_mon.add_argument("--outdir", default="./runtime_data")
    p_mon.add_argument("--interval", type=float, default=2.0)

    # --- touched ---
    p_touched = sub.add_parser("touched", help="Static-analyze only runtime-touched files")
    p_touched.add_argument("runtime_outdir", help="Runtime data directory with modules.jsonl")
    p_touched.add_argument("--metrics", nargs="*", default=None)
    p_touched.add_argument("--html", default=None)
    p_touched.add_argument("--json", default=None)

    # --- pipeline ---
    p_pipe = sub.add_parser("pipeline", help="Full: runtime run → static on touched files")
    p_pipe.add_argument("command", help="Command to run")
    p_pipe.add_argument("--outdir", default="./runtime_data")
    p_pipe.add_argument("--interval", type=float, default=2.0)
    p_pipe.add_argument("--memray", action="store_true")
    p_pipe.add_argument("--metrics", nargs="*", default=None)
    p_pipe.add_argument("--html", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "static":
        report = static_analyze(
            args.target,
            metrics=args.metrics,
            languages=args.languages,
            exclude=args.exclude,
            max_files=args.max_files,
            html_output=args.html,
            json_output=args.json,
        )
        s = report.summary
        print(f"{s.get('files_analyzed', 0)} files | "
              f"{s.get('total_sloc', 0)} SLOC | "
              f"CC={s.get('avg_complexity', 0):.1f} | "
              f"MI={s.get('avg_maintainability', 0):.0f} | "
              f"lint={s.get('total_lint_issues', 0)} | "
              f"security={s.get('total_security_issues', 0)} | "
              f"dead={s.get('total_dead_code', 0)}")

    elif args.cmd == "runtime":
        if args.rt_cmd == "run":
            exit_code = runtime_run(
                args.command,
                outdir=args.outdir,
                interval=args.interval,
                memray=args.memray,
                collect_objects=not args.no_objects,
                collect_network=not args.no_network,
            )
            sys.exit(exit_code)

        elif args.rt_cmd == "report":
            output = args.output or str(Path(args.outdir) / "runtime_report.html")
            html = runtime_report(args.outdir, output)
            print(f"Report: {output} ({len(html)} bytes)")

        elif args.rt_cmd == "monitor":
            mon = runtime_monitor(pid=args.pid, outdir=args.outdir, interval=args.interval)
            print(f"Monitoring pid={mon._pid}. Press Ctrl+C to stop.")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                mon.stop()
                print("Stopped.")

        else:
            parser.parse_args(["runtime", "--help"])

    elif args.cmd == "touched":
        report = analyze_runtime_touched(
            args.runtime_outdir,
            metrics=args.metrics,
            html_output=args.html,
            json_output=args.json,
        )
        s = report.summary
        print(f"Touched: {s.get('files_analyzed', 0)} files | "
              f"{s.get('total_sloc', 0)} SLOC | "
              f"CC={s.get('avg_complexity', 0):.1f} | "
              f"MI={s.get('avg_maintainability', 0):.0f}")

    elif args.cmd == "pipeline":
        exit_code, report = full_pipeline(
            args.command,
            outdir=args.outdir,
            interval=args.interval,
            memray=args.memray,
            metrics=args.metrics,
            html_output=args.html,
        )
        s = report.summary
        print(f"Exit: {exit_code} | "
              f"Touched: {s.get('files_analyzed', 0)} files | "
              f"CC={s.get('avg_complexity', 0):.1f} | "
              f"MI={s.get('avg_maintainability', 0):.0f}")

    else:
        parser.print_help()

if __name__ == "__main__":
    _cli()
