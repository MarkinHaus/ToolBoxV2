"""Offline-Langzeit-Bench: python -m toolboxv2.mods.isaa.base.Agent.page.bench [N] [--report]

Nutzt die TB-interne RuntimeMonitor-Infrastruktur aus
toolboxv2.utils.extras.code_analyzer.tb_analyze:
  - RuntimeMonitor: kontinuierliches RSS/CPU/tracemalloc-Sampling im Hintergrund
  - generate_runtime_report: HTML-Report aus den gesammelten JSONL-Daten

Ausgabe:
  - Konsole: ms/call, Prompt-Wachstum, Peak-RSS
  - runtime_data/bench_report.html: vollständiger interaktiver HTML-Report (mit --report)

Beispiel:
    python -m toolboxv2.mods.isaa.base.Agent.page.bench 600
    python -m toolboxv2.mods.isaa.base.Agent.page.bench 600 --report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

from . import Config, InProcessKernel, PageEngine
from .testing import ScriptedLLM, call, reply


async def _run(n: int) -> tuple[list[int], float, int]:
    """Führt den Bench-Lauf aus.  Gibt (prompt_sizes, elapsed, n_calls) zurück."""
    with tempfile.TemporaryDirectory() as root:
        actions = [
            reply(call("code_run", code=f"d{i % 5} = 'x' * 20000\nprint(len(d{i % 5}), {i})"))
            for i in range(n)
        ] + [reply(call("final_answer", answer="ok"))]
        llm = ScriptedLLM(actions)
        eng = PageEngine(
            root,
            llm,
            kernel=InProcessKernel,
            cfg=Config(max_steps=n + 50),
        )
        t0 = time.perf_counter()
        await eng.run("long task")
        await eng.close()
        elapsed = time.perf_counter() - t0
        sizes = [len(json.dumps(c)) for c in llm.calls]
        return sizes, elapsed, len(llm.calls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Page Engine offline bench")
    parser.add_argument("n", nargs="?", type=int, default=600,
                        help="Anzahl Code-Run-Iterationen (default: 600)")
    parser.add_argument("--report", action="store_true",
                        help="HTML-Report nach dem Lauf erzeugen")
    parser.add_argument("--outdir", default="./runtime_data",
                        help="Verzeichnis für RuntimeMonitor-Daten (default: ./runtime_data)")
    args = parser.parse_args()

    # TB RuntimeMonitor starten
    try:
        from toolboxv2.utils.extras.code_analyzer.tb_analyze import (
            runtime_monitor,
            runtime_report,
        )
        mon = runtime_monitor(outdir=args.outdir, interval=1.0)
        has_monitor = True
    except Exception as e:
        print(f"[bench] RuntimeMonitor nicht verfügbar ({e}), laufe ohne.")
        mon = None
        has_monitor = False

    # Bench laufen lassen
    try:
        sizes, elapsed, n_calls = asyncio.run(_run(args.n))
    finally:
        if mon is not None:
            mon.stop()

    per_call = elapsed / max(n_calls, 1) * 1000

    print(
        f"\n=== Page Engine Bench ({args.n} Iterationen) ===\n"
        f"  calls       : {n_calls}\n"
        f"  ms/call     : {per_call:.1f}\n"
        f"  prompt chars: first={sizes[0]}  max={max(sizes)}  last={sizes[-1]}\n"
        f"  elapsed     : {elapsed:.2f}s"
    )

    if has_monitor:
        # Kurze Zusammenfassung aus den gesammelten Daten
        try:
            proc_file = Path(args.outdir) / "process.jsonl"
            if proc_file.exists():
                import json as _j
                snapshots = [_j.loads(line) for line in proc_file.read_text().splitlines() if line.strip()]
                if snapshots:
                    peak = max(s.get("rss_mb", 0) for s in snapshots)
                    final = snapshots[-1].get("rss_mb", 0)
                    print(f"  peak RSS    : {peak:.1f} MB\n"
                          f"  final RSS   : {final:.1f} MB")
        except Exception:
            pass

        if args.report:
            try:
                out_html = Path(args.outdir) / "bench_report.html"
                html = runtime_report(outdir=args.outdir, output=out_html)
                print(f"\n  HTML-Report : {out_html.resolve()}")
            except Exception as e:
                print(f"[bench] Report-Generierung fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
