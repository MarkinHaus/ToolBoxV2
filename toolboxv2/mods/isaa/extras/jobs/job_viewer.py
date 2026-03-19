"""
ISAA Job Viewer — Live terminal dashboard + optional web view
=============================================================

Öffne in einem separaten Terminal:

    python -m toolboxv2.mods.isaa.extras.jobs.job_viewer
    python -m toolboxv2.mods.isaa.extras.jobs.job_viewer --jobs-file ~/.toolboxv2/jobs.json
    python -m toolboxv2.mods.isaa.extras.jobs.job_viewer --web --port 7799
    python -m toolboxv2.mods.isaa.extras.jobs.job_viewer --refresh 0.5

Angezeigt wird pro Job:
  • Status-Badge, Trigger-Typ, Letzter / Nächster Lauf
  • Iterations-Counter + Toolcall-Historie
  • Agent-Gedanken (last_thought)
  • Context-Füllstand als Balken (used / max tokens)
  • Run / Fail-Zähler, letztes Ergebnis
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── rich ─────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text
    from rich import box as rbox
    _RICH = True
except ImportError:
    _RICH = False

from .job_live_state import JobLiveEntry, JobLiveStateReader

console = Console() if _RICH else None

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_STYLE = {
    "active":    ("●", "green"),
    "running":   ("⟳", "bright_cyan"),
    "paused":    ("⏸", "yellow"),
    "disabled":  ("○", "dim"),
    "expired":   ("✓", "dim"),
    "done":      ("✓", "green"),
    "failed":    ("✗", "red"),
    "timeout":   ("⏱", "red"),
}


def _fmt_ago(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "?"


def _next_fire(job: dict) -> str:
    trigger = job.get("trigger", {})
    tt = trigger.get("trigger_type", "")
    status = job.get("status", "")
    if status in ("paused", "disabled", "expired"):
        return status
    if tt == "on_time":
        at = trigger.get("at_datetime", "")
        return _fmt_from(at) if at else "—"
    if tt == "on_interval":
        interval = trigger.get("interval_seconds")
        last = job.get("last_run_at")
        if interval and last:
            try:
                dt = datetime.fromisoformat(last)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                remaining = interval - (datetime.now(timezone.utc) - dt).total_seconds()
                if remaining <= 0:
                    return "now"
                return f"in {int(remaining)}s"
            except Exception:
                pass
        return f"every {interval}s" if interval else "?"
    if tt == "on_cron":
        return trigger.get("cron_expression", "?")
    return tt.replace("on_", "")


def _fmt_from(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = int((dt - datetime.now(timezone.utc)).total_seconds())
        if secs < 0:
            return "overdue"
        if secs < 60:
            return f"in {secs}s"
        if secs < 3600:
            return f"in {secs // 60}m"
        return f"in {secs // 3600}h"
    except Exception:
        return "?"


def _ctx_bar(used: int, max_: int, width: int = 20) -> str:
    pct = min(1.0, used / max_) if max_ else 0.0
    filled = int(pct * width)
    color = "green" if pct < 0.6 else "yellow" if pct < 0.85 else "red"
    bar = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar}[/{color}] {used // 1000}k / {max_ // 1000}k"


# ─────────────────────────────────────────────────────────────────────────────
# Terminal TUI
# ─────────────────────────────────────────────────────────────────────────────

def _build_table(jobs: list[dict], live: dict[str, JobLiveEntry]) -> Table:
    t = Table(
        box=rbox.SIMPLE_HEAVY,
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=True,
        expand=True,
    )
    t.add_column("Job / Agent", min_width=18, no_wrap=True)
    t.add_column("Trigger",     min_width=12, no_wrap=True)
    t.add_column("Status",      min_width=10)
    t.add_column("Iter / Tools", min_width=14)
    t.add_column("Context",     min_width=24)
    t.add_column("Runs ✗",     min_width=7)
    t.add_column("Last run",    min_width=9)
    t.add_column("Next fire",   min_width=10)

    for job in jobs:
        jid = job.get("job_id", "")
        name = job.get("name", jid)
        agent = job.get("agent_name", "?")
        status = job.get("status", "?")
        le: JobLiveEntry | None = live.get(jid)

        # Effective status: live overrides stored
        eff_status = le.status if le and le.status in ("running",) else status
        sym, clr = _STATUS_STYLE.get(eff_status, ("?", "white"))
        status_cell = Text(f"{sym} {eff_status}", style=clr)

        # Trigger
        trigger = job.get("trigger", {})
        tt = trigger.get("trigger_type", "?").replace("on_", "")

        # Iteration + tools
        if le and le.status == "running":
            tools_str = " › ".join(le.tool_calls[-3:]) if le.tool_calls else "—"
            iter_cell = f"iter {le.iteration}\n[dim]{tools_str}[/dim]"
        else:
            iter_cell = "—"

        # Context bar
        if le and le.status == "running" and le.context_max:
            ctx_cell = _ctx_bar(le.context_used, le.context_max)
        else:
            ctx_cell = "—"

        # Counts
        runs = job.get("run_count", 0)
        fails = job.get("fail_count", 0)
        counts = f"{runs} [red]{fails}✗[/red]" if fails else str(runs)

        t.add_row(
            f"[bold]{name}[/bold]\n[dim]{agent}[/dim]",
            tt,
            status_cell,
            iter_cell,
            ctx_cell,
            counts,
            _fmt_ago(job.get("last_run_at")),
            _next_fire(job),
        )

        # Expanded thought panel for running jobs
        if le and le.status == "running" and le.last_thought:
            thought_preview = le.last_thought[-300:].replace("\n", " ")
            t.add_row(
                Text("  💭 thought", style="dim italic"),
                "", "", "", "",
                Text(thought_preview, style="italic dim"),
                "", "",
            )

    return t


def _build_header(jobs: list[dict], live: dict[str, JobLiveEntry]) -> Text:
    total = len(jobs)
    active = sum(1 for j in jobs if j.get("status") == "active")
    running = sum(1 for e in live.values() if e.status == "running")
    failed = sum(1 for j in jobs if j.get("last_result") == "failed")
    now = datetime.now().strftime("%H:%M:%S")
    t = Text()
    t.append("ISAA Job Viewer  ", style="bold cyan")
    t.append(f"[{now}]  ", style="dim")
    t.append(f"total:{total}  ", style="white")
    t.append(f"active:{active}  ", style="green")
    t.append(f"running:{running}  ", style="bright_cyan")
    if failed:
        t.append(f"failed:{failed}", style="red")
    t.append("  [q] quit  [r] refresh", style="dim")
    return t


def run_terminal_viewer(jobs_file: Path, live_file: Path, refresh: float):
    if not _RICH:
        print("rich not installed — pip install rich")
        return

    reader = JobLiveStateReader(live_file)

    def load_jobs() -> list[dict]:
        try:
            if jobs_file.exists():
                return json.loads(jobs_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []

    with Live(console=console, refresh_per_second=int(1 / refresh), screen=False) as live:
        import sys, select as _sel
        running = True
        while running:
            jobs = load_jobs()
            lstate = reader.read()

            layout = Layout()
            layout.split_column(
                Layout(Panel(_build_header(jobs, lstate), style="bright_black"), size=3),
                Layout(_build_table(jobs, lstate)),
            )
            live.update(layout)

            # Non-blocking key check (Unix only; Windows falls back gracefully)
            try:
                if sys.stdin in _sel.select([sys.stdin], [], [], refresh)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch == "q":
                        running = False
            except Exception:
                time.sleep(refresh)


# ─────────────────────────────────────────────────────────────────────────────
# Web View
# ─────────────────────────────────────────────────────────────────────────────

_WEB_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>ISAA Job Viewer</title>
<meta http-equiv="refresh" content="2">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; padding: 24px; }
  h1 { color: #67e8f9; font-size: 15px; font-weight: 600; margin-bottom: 4px; }
  .meta { color: #6b7280; font-size: 11px; margin-bottom: 20px; }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card.running { border-color: #67e8f9; box-shadow: 0 0 8px #67e8f930; }
  .card.failed  { border-color: #f87171; }
  .card.paused  { border-color: #fbbf24; }
  .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
  .job-name { font-weight: 700; font-size: 14px; }
  .agent-name { color: #6b7280; font-size: 11px; margin-top: 2px; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 600; }
  .badge-running { background: #0e3a4d; color: #67e8f9; }
  .badge-active  { background: #0d2e17; color: #4ade80; }
  .badge-paused  { background: #2e2a0d; color: #fbbf24; }
  .badge-failed  { background: #2e0d0d; color: #f87171; }
  .badge-expired { background: #1a1a1a; color: #6b7280; }
  .badge-done    { background: #0d2e17; color: #4ade80; }
  .row { display: flex; justify-content: space-between; margin: 4px 0; color: #8b949e; font-size: 12px; }
  .row .val { color: #e6edf3; }
  .ctx-bar-bg { background: #21262d; border-radius: 4px; height: 8px; margin: 8px 0; overflow: hidden; }
  .ctx-bar { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .ctx-green  { background: #4ade80; }
  .ctx-yellow { background: #fbbf24; }
  .ctx-red    { background: #f87171; }
  .tools { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
  .tool-pill { background: #1f2937; color: #67e8f9; border-radius: 4px; padding: 1px 6px; font-size: 11px; }
  .thought { margin-top: 8px; padding: 8px 10px; background: #0d1117; border-left: 2px solid #30363d;
             color: #8b949e; font-size: 11px; line-height: 1.5; max-height: 80px; overflow: hidden;
             text-overflow: ellipsis; white-space: pre-wrap; border-radius: 0 4px 4px 0; }
  .footer { margin-top: 24px; color: #6b7280; font-size: 11px; }
</style>
</head>
<body>
<h1>⚙ ISAA Job Viewer</h1>
<div class="meta" id="meta">Lädt…</div>
<div class="grid" id="grid"></div>
<div class="footer">Auto-refresh alle 2s &nbsp;·&nbsp; <a href="/api/jobs" style="color:#67e8f9">JSON API</a></div>
<script>
async function load() {
  try {
    const r = await fetch('/api/jobs');
    const { jobs, live, ts } = await r.json();
    document.getElementById('meta').textContent =
      `${ts}  ·  total: ${jobs.length}  ·  active: ${jobs.filter(j=>j.status==='active').length}  ·  running: ${Object.values(live).filter(e=>e.status==='running').length}`;

    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    for (const job of jobs) {
      const le = live[job.job_id];
      const effStatus = (le && le.status === 'running') ? 'running' : job.status;
      const card = document.createElement('div');
      card.className = `card ${effStatus}`;

      const pct = le ? Math.min(100, le.context_used / le.context_max * 100) : 0;
      const ctxClass = pct < 60 ? 'ctx-green' : pct < 85 ? 'ctx-yellow' : 'ctx-red';
      const tools = (le && le.tool_calls.length)
        ? le.tool_calls.slice(-6).map(t => `<span class="tool-pill">${t}</span>`).join('') : '';
      const thought = (le && le.last_thought)
        ? `<div class="thought">${le.last_thought.slice(-400)}</div>` : '';
      const trigger = (job.trigger?.trigger_type || '?').replace('on_','');
      const ctxLabel = le ? `${Math.round(le.context_used/1000)}k / ${Math.round(le.context_max/1000)}k (${pct.toFixed(1)}%)` : '—';
      const iterLabel = (le && le.status==='running') ? `Iter ${le.iteration}` : '—';

      card.innerHTML = `
        <div class="card-header">
          <div><div class="job-name">${job.name}</div><div class="agent-name">${job.agent_name}</div></div>
          <span class="badge badge-${effStatus}">${effStatus}</span>
        </div>
        <div class="row"><span>Trigger</span><span class="val">${trigger}</span></div>
        <div class="row"><span>Läufe / Fehler</span><span class="val">${job.run_count} / <span style="color:#f87171">${job.fail_count}✗</span></span></div>
        <div class="row"><span>Letzter Lauf</span><span class="val">${job.last_run_at ? new Date(job.last_run_at).toLocaleString('de') : '—'}</span></div>
        <div class="row"><span>Iter</span><span class="val">${iterLabel}</span></div>
        <div class="row"><span>Context</span><span class="val">${ctxLabel}</span></div>
        <div class="ctx-bar-bg"><div class="ctx-bar ${ctxClass}" style="width:${pct}%"></div></div>
        ${tools ? `<div class="tools">${tools}</div>` : ''}
        ${thought}
      `;
      grid.appendChild(card);
    }
  } catch(e) { document.getElementById('meta').textContent = 'Verbindung fehlgeschlagen — ' + e; }
}
load();
</script>
</body>
</html>"""


def run_web_viewer(jobs_file: Path, live_file: Path, port: int):
    import http.server

    reader = JobLiveStateReader(live_file)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # silent

        def do_GET(self):
            if self.path == "/api/jobs":
                try:
                    jobs = json.loads(jobs_file.read_text(encoding="utf-8")) if jobs_file.exists() else []
                except Exception:
                    jobs = []
                lstate = reader.read()
                payload = json.dumps({
                    "jobs": jobs,
                    "live": {jid: e.to_dict() for jid, e in lstate.items()},
                    "ts": datetime.now().strftime("%H:%M:%S"),
                }, default=str).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                body = _WEB_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"  Job Viewer → http://localhost:{port}")
    server.serve_forever()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _default_jobs_file() -> Path:
    """Heuristic: find the default jobs.json."""
    candidates = [
        Path.home() / ".toolboxv2" / "jobs.json",
        Path.cwd() / "jobs.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description="ISAA Job Viewer")
    ap.add_argument("--jobs-file", type=Path, default=None, help="Path to jobs.json")
    ap.add_argument("--web", action="store_true", help="Launch web viewer instead of TUI")
    ap.add_argument("--port", type=int, default=7799, help="Web port (default 7799)")
    ap.add_argument("--refresh", type=float, default=1.0, help="TUI refresh interval in seconds")
    args = ap.parse_args()

    jobs_file: Path = args.jobs_file or _default_jobs_file()
    live_file: Path = jobs_file.with_suffix(".live.json")

    if args.web:
        # Terminal TUI + web server in parallel
        t = threading.Thread(
            target=run_web_viewer, args=(jobs_file, live_file, args.port), daemon=True
        )
        t.start()
        if _RICH:
            run_terminal_viewer(jobs_file, live_file, args.refresh)
        else:
            print(f"Web viewer at http://localhost:{args.port}  (Ctrl+C to stop)")
            t.join()
    else:
        run_terminal_viewer(jobs_file, live_file, args.refresh)


if __name__ == "__main__":
    main()
