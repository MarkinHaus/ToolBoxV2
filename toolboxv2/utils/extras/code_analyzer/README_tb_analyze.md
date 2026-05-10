# tb_analyze — Code Analysis Suite for ToolBoxV2

Unified static + runtime code analysis with TBJS Glass HTML reports.

Three modules, one CLI entry point:

| Module | Purpose |
|---|---|
| `tb_analyze_static.py` | Cyclomatic complexity, maintainability index, dead code, linting, security scanning |
| `tb_analyze_runtime.py` | Live process monitoring: memory, CPU, threads, GC, object growth, network, tracemalloc |
| `tb_analyze.py` | **Unified API & CLI** — wraps both, adds `touched` and `pipeline` modes |

---

## Installation

All three files go into `toolboxv2/utils/extras/code_analyzer/`.

Dependencies are installed lazily on first use:

| Tool | Package | Used by |
|---|---|---|
| radon | `radon` | complexity, MI, halstead |
| vulture | `vulture` | dead code detection |
| ruff | `ruff` | linting |
| bandit | `bandit` | security analysis |
| psutil | `psutil` | process metrics |
| objgraph | `objgraph` | object growth tracking |
| memray | `memray` | deep allocation profiling (optional) |

Install order: `uv pip` → `uvx pip` → `pip`. Fails loud if none work.

---

## CLI Reference

All commands via `tb analyze <subcommand>`.

### `static` — Static Analysis

```bash
# Analyze a directory, all metrics, HTML output
tb analyze static ./src --html report.html

# Single file, specific metrics
tb analyze static ./src/main.py --metrics complexity lint

# JSON export, exclude tests
tb analyze static ./src --json results.json --exclude "tests/*" "conftest.py"

# JavaScript support
tb analyze static ./frontend --languages python javascript
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--metrics` | all | `raw` `complexity` `dead_code` `lint` `security` `imports` |
| `--languages` | `python` | `python` `javascript` |
| `--exclude` | none | Glob patterns to skip |
| `--max-files` | 200 | Discovery limit |
| `--html` | none | HTML report output path |
| `--json` | none | JSON report output path |

**Output (stdout):**
```
42 files | 8340 SLOC | CC=3.2 | MI=68 | lint=17 | security=2 | dead=11
```

---

### `runtime run` — Run Command with Monitoring

```bash
# Basic monitoring
tb analyze runtime run "python my_agent.py --flag" --outdir ./rd

# Fast sampling, with memray
tb analyze runtime run "python server.py" --interval 0.5 --memray

# Disable heavy collectors
tb analyze runtime run "python bot.py" --no-objects --no-network
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--outdir` | `./runtime_data` | Output directory for JSONL data |
| `--interval` | `2.0` | Sampling interval in seconds |
| `--memray` | off | Enable deep memray allocation profiling |
| `--no-objects` | off | Disable objgraph object tracking |
| `--no-network` | off | Disable network connection tracking |

**Produces in `outdir/`:**

```
runtime_data/
├── meta.json           # run metadata (command, times, config)
├── process.jsonl       # RSS, VMS, CPU%, threads, FDs — every cycle
├── memory.jsonl        # tracemalloc top allocators + hotspots
├── memory_diff.jsonl   # allocation diffs (what grew between snapshots)
├── gc.jsonl            # GC stats: generations, uncollectable objects
├── objects.jsonl       # objgraph object type growth with referrer tracing
├── children.jsonl      # child process info
├── network.jsonl       # network connections
├── modules.jsonl       # loaded Python modules (runtime code coverage)
├── memray_capture.bin  # memray binary (if --memray)
├── memray_stats.txt    # memray stats (if --memray)
└── memray_flamegraph.html  # flamegraph (if --memray)
```

All JSONL files are append-only with `fsync` on every write — crash-safe.

---

### `runtime report` — Generate HTML from Runtime Data

```bash
tb analyze runtime report ./runtime_data -o report.html
```

Produces a TBJS Glass HTML report with:
- Memory timeline (RSS/VMS) charts
- CPU usage chart
- Thread count chart
- Top memory allocators table
- Allocation hotspots with call stack traces
- Allocation frequency (aggregated across all snapshots)
- Leak candidates (allocation growth diffs)
- Loaded modules (runtime code coverage)
- Object type growth with referrer/location tracing
- Network connections
- GC statistics with uncollectable object details
- Memray stats (if available)

---

### `runtime monitor` — Attach to Running Process

```bash
# Monitor self (useful from Python REPL)
tb analyze runtime monitor --outdir ./mon

# Attach to existing PID
tb analyze runtime monitor --pid 12345 --interval 1.0
```

Runs until Ctrl+C. Same data output as `runtime run`.

---

### `touched` — Static-Analyze Only Runtime-Loaded Files

The key feature: cross-references runtime and static analysis.

```bash
# First: run with monitoring
tb analyze runtime run "python my_app.py" --outdir ./rd

# Then: static-analyze only the files that were actually loaded
tb analyze touched ./rd --html touched_report.html
```

Reads `modules.jsonl` from the runtime data, extracts the list of Python files
that were loaded during execution, and runs static analysis on exactly those files.
Skips everything that was never imported — no noise from unused modules.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--metrics` | all | Same as `static` |
| `--html` | none | HTML report path |
| `--json` | none | JSON report path |

---

### `pipeline` — Full Automated Run

```bash
# One command: run → monitor → runtime report → static on touched files
tb analyze pipeline "python my_agent.py" \
    --outdir ./analysis \
    --interval 1.0 \
    --metrics complexity raw imports \
    --html ./analysis/static.html
```

Produces both `runtime_report.html` and `touched_static_report.html` in `outdir/`.

**Output (stdout):**
```
Exit: 0 | Touched: 23 files | CC=4.1 | MI=62
```

---

## Python API

```python
from toolboxv2.utils.extras.code_analyzer.tb_analyze import (
    static_analyze,
    runtime_run,
    runtime_report,
    runtime_monitor,
    analyze_runtime_touched,
    full_pipeline,
)
```

### `static_analyze(target, ...) → AnalysisReport`

```python
report = static_analyze(
    "./src",
    metrics=["complexity", "lint", "security"],
    html_output="report.html",
    json_output="report.json",
)

print(report.summary["avg_complexity"])
print(report.summary["total_lint_issues"])

for f in report.files:
    if f.maintainability_index < 40:
        print(f"Low MI: {f.path} → {f.maintainability_index:.0f}")
```

### `runtime_run(command, ...) → int`

```python
exit_code = runtime_run(
    "python my_agent.py",
    outdir="./rd",
    interval=1.0,
    memray=True,
)
```

### `runtime_monitor(pid, ...) → RuntimeMonitor`

```python
# Attach to self or external process
mon = runtime_monitor(pid=None, outdir="./rd", interval=0.5)

# ... do work ...

mon.stop()
```

### `runtime_report(outdir, output) → str`

```python
html = runtime_report("./rd", output="report.html")
```

### `analyze_runtime_touched(runtime_outdir, ...) → AnalysisReport`

```python
report = analyze_runtime_touched(
    "./rd",
    metrics=["complexity", "imports"],
    html_output="touched.html",
)

for f in report.files:
    print(f"{f.path}: CC={max((b['complexity'] for b in f.complexity), default=0)}")
```

### `full_pipeline(command, ...) → (int, AnalysisReport)`

```python
exit_code, report = full_pipeline(
    "python my_agent.py",
    outdir="./analysis",
    interval=1.0,
    metrics=["complexity", "raw", "imports"],
    html_output="./analysis/touched.html",
)
```

---

## Static Metrics Reference

| Metric | Tool | Languages | What it measures |
|---|---|---|---|
| `raw` | radon | py, js | LOC, SLOC, comments, blanks |
| `complexity` | radon | py, js | Cyclomatic complexity per function (CC), rank A–F |
| `complexity` | radon | py | Maintainability Index (MI, per-function weighted avg), Halstead metrics |
| `dead_code` | vulture | py | Unused functions, variables, imports, classes |
| `lint` | ruff | py | Code style, errors, warnings |
| `security` | bandit | py | Hardcoded secrets, injection, unsafe deserialization, shell risks |
| `imports` | AST | py | Dependency extraction (top-level package names) |

---

## Runtime Collectors Reference

| Collector | Frequency | Data |
|---|---|---|
| Process metrics | every cycle | RSS, VMS, CPU%, threads, FDs |
| tracemalloc top | every cycle | Top 20 allocation locations with file:line |
| Allocation hotspots | every cycle | Top 15 with full call stack (up to 5 frames) |
| tracemalloc diff | every 3rd cycle | What grew since last snapshot |
| Child processes | every 3rd cycle | PID, name, RSS for child procs |
| GC stats | every 3rd cycle | Generations, uncollectable objects with referrer tracing |
| Object growth | every 3rd cycle (offset) | objgraph type growth + large container scanning |
| Network | every 5th cycle | Connections for process + children |
| Loaded modules | 1st + every 5th cycle | All non-stdlib Python files with sizes |

---

## Memory Hunting Workflow

For finding memory-hungry code in ToolBoxV2:

```bash
# 1. Run the problematic process with monitoring
tb analyze.py pipeline "python -m toolboxv2 ..." \
    --outdir ./mem_hunt \
    --interval 1.0 \
    --html ./mem_hunt/touched.html

# 2. Open runtime_report.html → check:
#    - RSS timeline: is it growing?
#    - Leak candidates: which file:line allocates and never frees?
#    - Object growth: which types keep growing?
#    - Allocation hotspots: hover 🔍 for call stacks

# 3. Open touched.html → check:
#    - Which loaded modules have high complexity?
#    - Dead code in loaded modules = unnecessary memory
#    - Import chains = how many modules get pulled in

# 4. Cross-reference:
#    - Runtime top allocator file:line → open in static report
#    - High-frequency allocators (100% appearance) = persistent, likely leaks
#    - GC garbage_details → shows uncollectable objects with source locations
```

---

## File Layout

```
toolboxv2/utils/extras/code_analyzer/
├── tb_analyze.py           # Unified API + CLI (this module)
├── tb_analyze_static.py    # Static analysis engine + HTML report
├── tb_analyze_runtime.py   # Runtime monitor + HTML report
└── README.md               # This file
```



---

Static code analysis for Python and JavaScript.
Outputs: TBJS Glass HTML report, JSON, audit log.

## Quick Start

```python
from toolboxv2.utils.extras.code_analyzer.tb_analyze_static import analyze_path, generate_html_report

report = analyze_path("src/")
html = generate_html_report(report)
Path("report.html").write_text(html)
```

## API

### `analyze_path(target, metrics?, languages?, exclude?, max_files?) → AnalysisReport`

Main entry point. Accepts a file or directory.

```python
report = analyze_path(
    "toolboxv2/",
    metrics=["complexity", "lint", "security"],  # default: all
    languages=["python", "javascript"],           # default: ["python"]
    exclude=["tests/*", "migrations/*"],
    max_files=200,
)
```

### `analyze_file(filepath, language?, metrics?) → FileMetrics`

Single file analysis.

```python
fm = analyze_file("my_module.py", metrics=["complexity", "dead_code"])
print(fm.maintainability_index, fm.complexity)
```

### `analyze_from_config(config_path) → AnalysisReport`

YAML-driven analysis. See `analyze_example.yaml`.

```python
report = analyze_from_config("analyze.yaml")
```

### Export

```python
# HTML
html = generate_html_report(report)
Path("report.html").write_text(html)

# JSON
report.to_json("report.json")

# Dict
data = report.to_dict()
```

### YAML Config

```yaml
target: ./src
languages: [python]
max_files: 200

metrics:
  - raw          # LOC, SLOC, comments, blanks
  - complexity   # Cyclomatic complexity, MI, Halstead
  - dead_code    # Unused functions/classes/imports/variables
  - lint         # Ruff linting
  - security     # Bandit security analysis
  - imports      # Dependency extraction

exclude:
  - "tests/*"
  - "migrations/*"

report:
  format: both                    # html | json | both
  output: ./reports/analysis.html
  json_output: ./reports/analysis.json
```

### Decorator with Audit Logging

```python
from toolboxv2.utils.extras.code_analyzer.tb_analyze_static import audit_analyze

@audit_analyze(target="./src", metrics=["complexity", "lint"])
def my_pipeline():
    ...
```

Logs `ANALYSIS_START`, `ANALYSIS_COMPLETE`, `ANALYSIS_FAILED` via `app.audit_logger`.

### Runtime Profiling (Phase 2)

```python
from toolboxv2.utils.extras.code_analyzer.tb_analyze_static import runtime_profile

@runtime_profile(interval=0.5, docker_audit=True)
def run_agents():
    ...

result, report = run_agents()
report.to_json("runtime.json")
```

Captures: RSS/VMS timeline, CPU%, threads, file descriptors, network connections, child processes, tracemalloc allocation diffs, Docker container audit.

## Dependencies

Installed lazily on first use (uv → pip fallback):

| Tool     | Metric       |
|----------|-------------|
| `radon`  | complexity  |
| `vulture`| dead_code   |
| `ruff`   | lint        |
| `bandit` | security    |
| `psutil` | runtime     |

Remove with `remove_tool("ruff")`.

## Extending

**Add a new metric:**

1. Write analyzer function `_analyze_foo(code, filepath) → data`
2. Write runner `_run_foo(fm, code, filepath, lang) → None` that sets `fm.your_field`
3. Add field to `FileMetrics` dataclass
4. Register in `_METRIC_RUNNERS["foo"] = _run_foo`
5. Register in `AVAILABLE_METRICS["foo"] = {"python"}`

**Add a language:**

1. Add extension mapping in `_LANG_MAP`
2. Add language-specific analyzer functions
3. Register supported metrics in `AVAILABLE_METRICS`

**Customize report:**

The HTML report uses the TBJS Glass v3.0 design system. CSS is in `_get_css()`, JS in `_get_js()`. Metric tooltips are in the `_TOOLTIPS` dict.
