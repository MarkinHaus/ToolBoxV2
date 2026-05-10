"""
tb_analyze — Static Code Analysis for ToolBoxV2

Decorator-based, YAML-configurable static analysis tool.
Produces polished TBJS Glass HTML reports.

Usage:
    from toolboxv2.mods.tb_analyze import analyze_path, analyze_from_config

    # Quick analysis of a file or directory
    results = analyze_path("src/my_module.py", metrics=["complexity", "dead_code", "lint"])

    # Config-driven analysis
    results = analyze_from_config("analyze.yaml")

    # As decorator (logs to audit)
    @audit_analyze(metrics=["complexity", "lint"])
    def my_pipeline():
        ...
"""

from __future__ import annotations

import ast
import fnmatch
import json
import logging
import os
import shutil
import subprocess  # nosec B404 — intentional: tool runner needs subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable

logger = logging.getLogger("tb_analyze")

# ---------------------------------------------------------------------------
# Lazy Dependency Management
# ---------------------------------------------------------------------------

_TOOL_PACKAGES = {
    "radon": "radon",
    "vulture": "vulture",
    "ruff": "ruff",
    "bandit": "bandit",
}


def _ensure_tool(name: str) -> bool:
    """Lazy install a tool. Try uv/uvx first, then pip, then raise."""
    pkg = _TOOL_PACKAGES.get(name, name)

    # Check if already available
    if name in ("ruff", "bandit"):
        if shutil.which(name) is not None:
            return True
    else:
        try:
            __import__(name)
            return True
        except ImportError:
            pass

    # Try uv first
    for installer in [
        ["uv", "pip", "install", pkg],
        ["uvx", "pip", "install", pkg],
    ]:
        try:
            r = subprocess.run(installer, capture_output=True, text=True, errors="replace",  encoding="utf-8",timeout=60)
            if r.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    # Fallback: sys.executable pip
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True, errors="replace",  encoding="utf-8",timeout=120,
        )
        if r.returncode == 0:
            return True
    except subprocess.TimeoutExpired:
        pass

    raise RuntimeError(
        f"Failed to install '{pkg}'. Tried uv, uvx, pip. "
        f"Install manually: pip install {pkg}"
    )


def remove_tool(name: str) -> bool:
    """Remove an optional analysis tool."""
    pkg = _TOOL_PACKAGES.get(name, name)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            capture_output=True, text=True, errors="replace",  encoding="utf-8",timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class FileMetrics:
    path: str
    language: str = "python"
    loc: int = 0
    sloc: int = 0
    comments: int = 0
    blank: int = 0
    complexity: list[dict] = field(default_factory=list)
    maintainability_index: float = 0.0
    mi_blocks: list[dict] = field(default_factory=list)
    halstead: dict = field(default_factory=dict)
    dead_code: list[dict] = field(default_factory=list)
    lint_issues: list[dict] = field(default_factory=list)
    security_issues: list[dict] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "language": self.language,
            "loc": self.loc,
            "sloc": self.sloc,
            "comments": self.comments,
            "blank": self.blank,
            "complexity": self.complexity,
            "maintainability_index": round(self.maintainability_index, 2),
            "mi_blocks": self.mi_blocks,
            "halstead": self.halstead,
            "dead_code": self.dead_code,
            "lint_issues": self.lint_issues,
            "security_issues": self.security_issues,
            "dependencies": self.dependencies,
            "errors": self.errors,
        }


@dataclass
class AnalysisReport:
    target: str
    timestamp: str = ""
    duration_s: float = 0.0
    files: list[FileMetrics] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 3),
            "summary": self.summary,
            "config": self.config,
            "files": [f.to_dict() for f in self.files],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Export report as JSON. Optionally write to file."""
        data = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(data, encoding="utf-8")
        return data


# ---------------------------------------------------------------------------
# Intelligent File Discovery
# ---------------------------------------------------------------------------

# Directories that are never useful to analyze
_SKIP_DIRS = {
    "__pycache__", ".git", ".hg", ".svn", "node_modules", ".mypy_cache",
    ".ruff_cache", ".pytest_cache", "dist", "build", "egg-info",
    ".eggs", ".tox", ".venv", "venv", "env", ".env",
    "site-packages", "migrations", ".idea", ".vscode",
}

# File size limits
_MAX_FILE_SIZE = 500_000  # 500KB — skip generated/minified files
_MAX_FILES_DEFAULT = 200   # Don't analyze more than this without explicit override

_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
}


def _should_skip_dir(dirname: str, exclude_patterns: list[str]) -> bool:
    """Check if a directory should be pruned during walk."""
    if dirname in _SKIP_DIRS or dirname.startswith("."):
        return True
    return any(fnmatch.fnmatch(dirname, p) for p in exclude_patterns)


def _is_minified_js(fpath: Path, size: int) -> bool:
    """Detect minified JS by avg line length > 200 chars."""
    if size <= 10_000:
        return False
    try:
        with open(fpath, "r", errors="ignore") as f:
            head = f.read(2000)
        lines = head.split("\n")
        return bool(lines and (len(head) / max(len(lines), 1)) > 200)
    except Exception:
        return True


def _check_file_size(fpath: Path) -> int:
    """Return file size or -1 if inaccessible/empty/too large."""
    try:
        size = fpath.stat().st_size
    except OSError:
        return -1
    if size == 0 or size > _MAX_FILE_SIZE:
        return -1
    return size


def discover_files(
    target: str | Path,
    languages: list[str] | None = None,
    exclude: list[str] | None = None,
    max_files: int = _MAX_FILES_DEFAULT,
) -> list[tuple[Path, str]]:
    """Intelligently discover analyzable files.

    Returns list of (path, language) tuples.
    Skips: hidden dirs, vendored code, generated files, huge files.
    """
    target = Path(target)
    if not target.exists():
        raise FileNotFoundError(f"Target not found: {target}")

    allowed_langs = set(languages or ["python", "javascript"])
    allowed_exts = {ext for ext, lang in _LANG_MAP.items() if lang in allowed_langs}
    exclude_patterns = exclude or []

    if target.is_file():
        ext = target.suffix.lower()
        lang = _LANG_MAP.get(ext)
        if lang and lang in allowed_langs:
            return [(target, lang)]
        return []

    result: list[tuple[Path, str]] = []

    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if not _should_skip_dir(d, exclude_patterns)]

        for fname in files:
            if len(result) >= max_files:
                logger.warning(
                    "Hit max_files limit (%d), stopping discovery. "
                    "Pass max_files=N to analyze_path() to raise this.",
                    max_files,
                )
                return result

            ext = Path(fname).suffix.lower()
            if ext not in allowed_exts:
                continue

            fpath = Path(root) / fname
            lang = _LANG_MAP[ext]

            size = _check_file_size(fpath)
            if size < 0:
                continue

            rel = str(fpath.relative_to(target))
            if any(fnmatch.fnmatch(rel, p) for p in exclude_patterns):
                continue

            if lang == "javascript" and _is_minified_js(fpath, size):
                logger.debug("Skipping minified: %s", fpath)
                continue

            result.append((fpath, lang))

    return result


# ---------------------------------------------------------------------------
# Analyzers — Python
# ---------------------------------------------------------------------------

def _analyze_complexity(code: str, filepath: str = "") -> tuple[list[dict], float, list[dict], dict]:
    """Radon: cyclomatic complexity, maintainability index, halstead."""
    _ensure_tool("radon")
    import warnings as _warnings
    import radon.complexity as rc
    import radon.metrics as rm

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", SyntaxWarning)
        blocks = rc.cc_visit(code)
    complexity = []
    for b in blocks:
        complexity.append({
            "name": b.name,
            "type": b.__class__.__name__,  # Function or Class
            "path": filepath,
            "line": b.lineno,
            "col": b.col_offset,
            "complexity": b.complexity,
            "rank": rc.cc_rank(b.complexity),
        })

    # MI: compute per top-level function/class, then average.
    # Whole-module MI is misleading for large files (ln(LOC) dominates).
    mi, mi_blocks = _compute_per_function_mi(code, rm)

    # Halstead via radon
    halstead = {}
    try:
        from radon.metrics import h_visit
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", SyntaxWarning)
            h = h_visit(code)
        if h and len(h) > 0:
            h0 = h[0]
            halstead = {
                "volume": round(h0.volume, 2) if h0.volume else 0,
                "difficulty": round(h0.difficulty, 2) if h0.difficulty else 0,
                "effort": round(h0.effort, 2) if h0.effort else 0,
                "bugs": round(h0.bugs, 4) if h0.bugs else 0,
            }
    except Exception:
        pass

    return complexity, mi, mi_blocks, halstead


def _compute_per_function_mi(code: str, rm_module) -> tuple[float, list[dict]]:
    """Compute MI per top-level function/class and return weighted average.

    Returns (avg_mi, blocks) where blocks is a list of per-node MI dicts.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return rm_module.mi_visit(code, multi=True), []

    lines = code.splitlines()
    weighted: list[tuple[float, int]] = []
    blocks: list[dict] = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
        chunk = "\n".join(lines[start:end])

        if not chunk.strip():
            continue

        try:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", SyntaxWarning)
                mi = rm_module.mi_visit(chunk, multi=True)
            sloc = max(end - start, 1)
            weighted.append((mi, sloc))
            blocks.append({
                "name": node.name,
                "type": node.__class__.__name__,
                "line": node.lineno,
                "sloc": sloc,
                "mi": round(mi, 2),
            })
        except Exception:
            continue

    if not weighted:
        return rm_module.mi_visit(code, multi=True), []

    total_sloc = sum(sloc for _, sloc in weighted)
    if total_sloc == 0:
        return 0.0, blocks
    avg = sum(mi * sloc for mi, sloc in weighted) / total_sloc
    return avg, blocks


def _analyze_raw(code: str) -> dict[str, int]:
    """Radon: raw LOC metrics."""
    _ensure_tool("radon")
    import radon.raw as rr
    raw = rr.analyze(code)
    return {
        "loc": raw.loc,
        "sloc": raw.sloc,
        "comments": raw.comments,
        "blank": raw.blank,
    }


def _analyze_dead_code(code: str, filepath: str) -> list[dict]:
    """Vulture: unused code detection."""
    _ensure_tool("vulture")
    import vulture

    v = vulture.Vulture()
    v.scan(code, filename=filepath)
    dead = v.get_unused_code()
    return [
        {
            "name": d.name,
            "type": d.typ,
            "line": d.first_lineno,
            "confidence": d.confidence,
            "message": d.message if isinstance(d.message, str) else str(d),
        }
        for d in dead
    ]


def _analyze_lint(filepath: str) -> list[dict]:
    """Ruff: linting."""
    _ensure_tool("ruff")
    try:
        r = subprocess.run(
            ["ruff", "check", "--output-format=json", filepath],
            capture_output=True, text=True, errors="replace",  encoding="utf-8",timeout=30,
        )
        if r.stdout:
            issues = json.loads(r.stdout)
            return [
                {
                    "code": i["code"],
                    "message": i["message"],
                    "line": i["location"]["row"],
                    "col": i["location"]["column"],
                    "severity": _ruff_severity(i["code"]),
                }
                for i in issues
            ]
    except Exception as e:
        logger.warning(f"Ruff failed on {filepath}: {e}")
    return []


def _ruff_severity(code: str) -> str:
    """Map ruff error code prefix to severity."""
    if code.startswith("E"):
        return "error"
    if code.startswith("W"):
        return "warning"
    if code.startswith("F"):
        return "error"
    return "info"


def _analyze_security(filepath: str) -> list[dict]:
    """Bandit: security analysis."""
    _ensure_tool("bandit")
    try:
        r = subprocess.run(
            ["bandit", "-f", "json", "-q", filepath],
            capture_output=True, text=True, errors="replace",  encoding="utf-8",timeout=30,
        )
        if r.stdout:
            data = json.loads(r.stdout)
            return [
                {
                    "test_id": i["test_id"],
                    "test_name": i["test_name"],
                    "message": i["issue_text"],
                    "severity": i["issue_severity"],
                    "confidence": i["issue_confidence"],
                    "line": i["line_number"],
                    "cwe": i.get("issue_cwe", {}).get("id", ""),
                }
                for i in data.get("results", [])
            ]
    except Exception as e:
        logger.warning(f"Bandit failed on {filepath}: {e}")
    return []


def _analyze_imports(code: str) -> list[str]:
    """Extract import dependencies via AST (no external tool needed)."""
    deps = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    deps.add(node.module.split(".")[0])
    except SyntaxError:
        pass
    return sorted(deps)


# ---------------------------------------------------------------------------
# Analyzers — JavaScript (basic via ruff is not available, use subprocess)
# ---------------------------------------------------------------------------

def _analyze_js_loc(code: str) -> dict[str, int]:
    """Basic LOC for JS."""
    lines = code.split("\n")
    blank = sum(1 for ln in lines if not ln.strip())
    comment = sum(1 for ln in lines if ln.strip().startswith("//") or ln.strip().startswith("/*"))
    return {
        "loc": len(lines),
        "sloc": len(lines) - blank - comment,
        "comments": comment,
        "blank": blank,
    }


def _analyze_js_complexity(code: str, filepath: str) -> list[dict]:
    """Estimate JS complexity by counting branching keywords."""
    # Simple heuristic — real analysis would need espree/acorn
    keywords = ["if", "else", "for", "while", "switch", "case", "catch", "&&", "||", "?"]
    lines = code.split("\n")
    # Per-function is hard without a parser, so give file-level estimate
    count = 1  # base complexity
    for line in lines:
        stripped = line.strip()
        for kw in keywords:
            if kw in ("&&", "||", "?"):
                count += stripped.count(kw)
            elif f"{kw} " in stripped or f"{kw}(" in stripped:
                count += 1
    rank = "A" if count <= 5 else "B" if count <= 10 else "C" if count <= 20 else "D" if count <= 30 else "F"
    return [{
        "name": Path(filepath).name,
        "type": "file",
        "path": filepath,
        "line": 1,
        "complexity": count,
        "rank": rank,
    }]


# ---------------------------------------------------------------------------
# Core Analysis Engine
# ---------------------------------------------------------------------------

# Available metrics and which languages they support
AVAILABLE_METRICS = {
    "raw":        {"python", "javascript"},
    "complexity": {"python", "javascript"},
    "dead_code":  {"python"},
    "lint":       {"python"},
    "security":   {"python"},
    "imports":    {"python"},
}

ALL_METRICS = list(AVAILABLE_METRICS.keys())


def _run_raw(fm: FileMetrics, code: str, filepath: str, language: str) -> None:
    """Dispatch raw LOC analysis by language."""
    raw = _analyze_raw(code) if language == "python" else _analyze_js_loc(code)
    fm.loc = raw["loc"]
    fm.sloc = raw["sloc"]
    fm.comments = raw["comments"]
    fm.blank = raw["blank"]


def _run_complexity(fm: FileMetrics, code: str, filepath: str, language: str) -> None:
    """Dispatch complexity analysis by language."""
    if language == "python":
        cc, mi, mi_blocks, hal = _analyze_complexity(code, filepath)
        fm.complexity = cc
        fm.maintainability_index = mi
        for b in mi_blocks:
            b["path"] = filepath
        fm.mi_blocks = mi_blocks
        fm.halstead = hal
    else:
        fm.complexity = _analyze_js_complexity(code, filepath)


def _run_dead_code(fm: FileMetrics, code: str, filepath: str, _lang: str) -> None:
    fm.dead_code = _analyze_dead_code(code, filepath)


def _run_lint(fm: FileMetrics, _code: str, filepath: str, _lang: str) -> None:
    fm.lint_issues = _analyze_lint(filepath)


def _run_security(fm: FileMetrics, _code: str, filepath: str, _lang: str) -> None:
    fm.security_issues = _analyze_security(filepath)


def _run_imports(fm: FileMetrics, code: str, _filepath: str, _lang: str) -> None:
    fm.dependencies = _analyze_imports(code)


_METRIC_RUNNERS: dict[str, Callable] = {
    "raw": _run_raw,
    "complexity": _run_complexity,
    "dead_code": _run_dead_code,
    "lint": _run_lint,
    "security": _run_security,
    "imports": _run_imports,
}

def _print_progress(idx: int, total: int, t_start: float, fpath) -> None:
    """Render single-line progress bar to stderr."""
    pct = idx / total
    elapsed = time.monotonic() - t_start
    eta = (elapsed / idx) * (total - idx) if idx else 0
    bar_w = 20
    filled = int(bar_w * pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    name = Path(fpath).name
    if len(name) > 40:
        name = name[:37] + "..."
    msg = f"\r[{bar}] {idx}/{total} ({pct*100:5.1f}%) ETA {eta:5.1f}s  {name:<40}"
    sys.stderr.write(msg)
    sys.stderr.flush()

def analyze_file(
    filepath: str | Path,
    language: str = "python",
    metrics: list[str] | None = None,
) -> FileMetrics:
    """Analyze a single file with selected metrics."""
    filepath = Path(filepath)
    metrics = metrics or ALL_METRICS
    fm = FileMetrics(path=str(filepath), language=language)

    try:
        code = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        fm.errors.append(f"Read error: {e}")
        return fm

    for metric in metrics:
        if language not in AVAILABLE_METRICS.get(metric, set()):
            continue
        runner = _METRIC_RUNNERS.get(metric)
        if runner is None:
            continue
        try:
            runner(fm, code, str(filepath), language)
        except Exception as e:
            fm.errors.append(f"{metric}: {e}")
            logger.warning("Metric '%s' failed for %s: %s", metric, filepath, e)

    return fm

def analyze_path(
    target: str | Path,
    metrics: list[str] | None = None,
    languages: list[str] | None = None,
    exclude: list[str] | None = None,
    max_files: int = _MAX_FILES_DEFAULT,
) -> AnalysisReport:
    """Analyze a file or directory. Main entry point."""
    from datetime import datetime, timezone

    t0 = time.monotonic()
    target = Path(target)
    metrics = metrics or ALL_METRICS
    languages = languages or ["python"]

    report = AnalysisReport(
        target=str(target),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config={"metrics": metrics, "languages": languages, "exclude": exclude or []},
    )

    files = discover_files(target, languages=languages, exclude=exclude, max_files=max_files)
    total = len(files)
    logger.info(f"Discovered {total} files in {target}")

    is_tty = sys.stderr.isatty()
    t_start = time.monotonic()

    for idx, (fpath, lang) in enumerate(files, 1):
        if is_tty and total:
            _print_progress(idx, total, t_start, fpath)
        fm = analyze_file(fpath, language=lang, metrics=metrics)
        report.files.append(fm)

    if is_tty and total:
        sys.stderr.write("\n")
        sys.stderr.flush()

    report.duration_s = time.monotonic() - t0
    report.summary = _compute_summary(report)
    return report


def _aggregate_by_key(items: list[dict], key: str, default: str = "unknown") -> dict[str, int]:
    """Count items grouped by a key field."""
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(key, default)
        counts[val] = counts.get(val, 0) + 1
    return counts


def _safe_avg(values: list[float]) -> float:
    """Average of a list, or 0 if empty."""
    return (sum(values) / len(values)) if values else 0


def _compute_summary(report: AnalysisReport) -> dict:
    """Compute aggregate summary from file metrics."""
    total_loc = sum(f.loc for f in report.files)
    total_sloc = sum(f.sloc for f in report.files)
    total_comments = sum(f.comments for f in report.files)

    all_cc = [b for f in report.files for b in f.complexity]
    avg_cc = _safe_avg([b["complexity"] for b in all_cc])

    mi_values = [f.maintainability_index for f in report.files if f.maintainability_index > 0]
    avg_mi = _safe_avg(mi_values)

    all_mi_blocks = [b for f in report.files for b in f.mi_blocks if b.get("mi", 100) > 0]
    worst_mi_blocks = sorted(all_mi_blocks, key=lambda b: b["mi"])[:10]

    all_lint = [i for f in report.files for i in f.lint_issues]

    all_lint = [i for f in report.files for i in f.lint_issues]
    all_security = [i for f in report.files for i in f.security_issues]
    all_dead = [d for f in report.files for d in f.dead_code]

    top_complex = sorted(all_cc, key=lambda b: b["complexity"], reverse=True)[:10]
    worst_mi = sorted(
        [(f.path, f.maintainability_index) for f in report.files if f.maintainability_index > 0],
        key=lambda x: x[1],
    )[:10]

    return {
        "files_analyzed": len(report.files),
        "total_loc": total_loc,
        "total_sloc": total_sloc,
        "total_comments": total_comments,
        "comment_ratio": round(total_comments / max(total_sloc, 1) * 100, 1),
        "avg_complexity": round(avg_cc, 2),
        "avg_maintainability": round(avg_mi, 2),
        "complexity_distribution": _aggregate_by_key(all_cc, "rank", "?"),
        "top_complex_functions": top_complex,
        "worst_maintainability": worst_mi,
        "worst_maintainability_blocks": worst_mi_blocks,
        "total_lint_issues": len(all_lint),
        "lint_by_severity": _aggregate_by_key(all_lint, "severity", "info"),
        "total_security_issues": len(all_security),
        "security_by_severity": _aggregate_by_key(all_security, "severity", "LOW"),
        "total_dead_code": len(all_dead),
        "dead_code_by_type": _aggregate_by_key(all_dead, "type"),
        "languages": list(set(f.language for f in report.files)),
        "errors": [e for f in report.files for e in f.errors],
    }


# ---------------------------------------------------------------------------
# YAML Config Support
# ---------------------------------------------------------------------------

def analyze_from_config(config_path: str | Path) -> AnalysisReport:
    """Run analysis from a YAML config file.

    Example config:
        target: ./src
        languages: [python, javascript]
        metrics: [complexity, dead_code, lint, security, imports, raw]
        exclude: ["tests/*", "conftest.py"]
        max_files: 300
        report:
          format: html
          output: ./reports/analysis.html
    """
    try:
        import yaml
    except ImportError:
        _ensure_tool("pyyaml")
        import yaml

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    target = cfg.get("target", ".")
    # Resolve relative to config file location
    target_path = config_path.parent / target
    if not target_path.exists():
        target_path = Path(target)

    report = analyze_path(
        target=target_path,
        metrics=cfg.get("metrics"),
        languages=cfg.get("languages"),
        exclude=cfg.get("exclude"),
        max_files=cfg.get("max_files", _MAX_FILES_DEFAULT),
    )
    report.config.update(cfg)

    # Generate report if configured
    report_cfg = cfg.get("report", {})
    out_dir = config_path.parent

    if report_cfg.get("format") in ("html", "both"):
        output = report_cfg.get("output", "analysis_report.html")
        output_path = out_dir / output
        html = generate_html_report(report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)

    if report_cfg.get("format") in ("json", "both"):
        json_output = report_cfg.get("json_output", "analysis_report.json")
        json_path = out_dir / json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_json(json_path)
        logger.info("JSON report written to %s", json_path)

    return report


# ---------------------------------------------------------------------------
# Decorator — with Audit Logging
# ---------------------------------------------------------------------------

def audit_analyze(
    target: str | Path | None = None,
    metrics: list[str] | None = None,
    languages: list[str] | None = None,
    report_path: str | None = None,
):
    """Decorator that runs analysis and logs to audit system.

    Usage:
        @audit_analyze(target="./src", metrics=["complexity", "lint"])
        def my_pipeline():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get audit logger
            audit = None
            try:
                from toolboxv2.utils.system.getting_and_closing_app import get_app
                app = get_app()
                audit = app.audit_logger if hasattr(app, "audit_logger") else None
            except Exception:
                pass

            _target = target or kwargs.get("target", ".")
            _metrics = metrics or kwargs.get("metrics")

            if audit:
                audit.log_action(
                    user_id="tb_analyze",
                    action="ANALYSIS_START",
                    resource=str(_target),
                    details={"metrics": _metrics or ALL_METRICS},
                )

            t0 = time.monotonic()
            try:
                report = analyze_path(
                    _target, metrics=_metrics, languages=languages
                )

                if report_path:
                    html = generate_html_report(report)
                    Path(report_path).write_text(html, encoding="utf-8")

                if audit:
                    audit.log_action(
                        user_id="tb_analyze",
                        action="ANALYSIS_COMPLETE",
                        resource=str(_target),
                        status="SUCCESS",
                        details={
                            "duration_s": round(time.monotonic() - t0, 3),
                            "files": report.summary.get("files_analyzed", 0),
                            "issues": (
                                report.summary.get("total_lint_issues", 0)
                                + report.summary.get("total_security_issues", 0)
                            ),
                        },
                    )

                # Inject report into function kwargs
                kwargs["_analysis_report"] = report
                return func(*args, **kwargs)

            except Exception as e:
                if audit:
                    audit.log_action(
                        user_id="tb_analyze",
                        action="ANALYSIS_FAILED",
                        resource=str(_target),
                        status="FAILURE",
                        details={"error": str(e)},
                    )
                raise

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Runtime Analysis — Phase 2
# ---------------------------------------------------------------------------

@dataclass
class RuntimeSnapshot:
    """Single point-in-time measurement during runtime profiling."""
    timestamp: float  # monotonic seconds since start
    rss_mb: float = 0.0
    vms_mb: float = 0.0
    cpu_percent: float = 0.0
    open_fds: int = 0
    threads: int = 0
    connections: list[dict] = field(default_factory=list)
    children: list[dict] = field(default_factory=list)
    tracemalloc_top: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 3),
            "rss_mb": round(self.rss_mb, 1),
            "vms_mb": round(self.vms_mb, 1),
            "cpu_percent": round(self.cpu_percent, 1),
            "open_fds": self.open_fds,
            "threads": self.threads,
            "connections": self.connections,
            "children": self.children,
            "tracemalloc_top": self.tracemalloc_top,
        }


@dataclass
class RuntimeReport:
    """Runtime profiling results."""
    target: str
    timestamp: str = ""
    duration_s: float = 0.0
    snapshots: list[RuntimeSnapshot] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0
    tracemalloc_diff: list[dict] = field(default_factory=list)
    process_tree: list[dict] = field(default_factory=list)
    network_connections: list[dict] = field(default_factory=list)
    docker_info: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "peak_vms_mb": round(self.peak_vms_mb, 1),
            "tracemalloc_peak_mb": round(self.tracemalloc_peak_mb, 1),
            "tracemalloc_diff": self.tracemalloc_diff,
            "process_tree": self.process_tree,
            "network_connections": self.network_connections,
            "docker_info": self.docker_info,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "errors": self.errors,
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Export runtime report as JSON."""
        data = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(data, encoding="utf-8")
        return data


def _get_psutil():
    """Lazy import psutil."""
    try:
        import psutil
        return psutil
    except ImportError:
        _ensure_tool("psutil")
        import psutil
        return psutil


def _take_snapshot(t0: float, pid: int, use_tracemalloc: bool = False) -> RuntimeSnapshot:
    """Capture a single runtime snapshot for process pid."""
    psutil = _get_psutil()
    snap = RuntimeSnapshot(timestamp=time.monotonic() - t0)

    try:
        proc = psutil.Process(pid)
        mem = proc.memory_info()
        snap.rss_mb = mem.rss / (1024 * 1024)
        snap.vms_mb = mem.vms / (1024 * 1024)
        snap.cpu_percent = proc.cpu_percent(interval=0)
        snap.threads = proc.num_threads()

        # File descriptors (Unix) or handles (Windows)
        try:
            snap.open_fds = proc.num_fds()
        except (AttributeError, psutil.AccessDenied):
            try:
                snap.open_fds = proc.num_handles()
            except (AttributeError, psutil.AccessDenied):
                pass

        # Network connections
        try:
            conns = proc.net_connections(kind="all")
            snap.connections = [
                {
                    "fd": c.fd,
                    "family": str(c.family),
                    "type": str(c.type),
                    "laddr": f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "",
                    "raddr": f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "",
                    "status": c.status,
                }
                for c in conns[:20]  # cap at 20 to avoid huge snapshots
            ]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        # Child processes
        try:
            children = proc.children(recursive=True)
            snap.children = [
                {
                    "pid": ch.pid,
                    "name": ch.name(),
                    "rss_mb": round(ch.memory_info().rss / (1024 * 1024), 1),
                }
                for ch in children[:20]
            ]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        snap.tracemalloc_top = [{"error": str(e)}]

    # Tracemalloc snapshot
    if use_tracemalloc:
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                top = tracemalloc.take_snapshot().statistics("lineno")[:10]
                snap.tracemalloc_top = [
                    {
                        "file": str(stat.traceback),
                        "size_kb": round(stat.size / 1024, 1),
                        "count": stat.count,
                    }
                    for stat in top
                ]
        except Exception:
            pass

    return snap


def _collect_docker_info() -> dict:
    """Collect Docker container audit info if Docker is available."""
    info: dict = {"available": False}
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True,  encoding="utf-8", errors="replace", timeout=10,
        )
        if r.returncode != 0:
            return info
        info["available"] = True
        containers = []
        for line in r.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 4:
                containers.append({
                    "id": parts[0],
                    "name": parts[1],
                    "image": parts[2],
                    "status": parts[3],
                    "ports": parts[4] if len(parts) > 4 else "",
                })
        info["running_containers"] = containers
        info["count"] = len(containers)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def runtime_profile(
    func: Callable | None = None,
    interval: float = 0.5,
    enable_tracemalloc: bool = True,
    docker_audit: bool = False,
):
    """Decorator that profiles a function's runtime behavior.

    Captures: memory (RSS/VMS), CPU, network connections,
    child processes, tracemalloc allocations, optional Docker audit.

    Usage:
        @runtime_profile(interval=1.0, docker_audit=True)
        def run_agents():
            ...

        # Result is injected as kwarg:
        @runtime_profile()
        def run_agents(_runtime_report=None):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            import threading
            from datetime import datetime, timezone

            pid = os.getpid()
            report = RuntimeReport(
                target=fn.__qualname__,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

            # Start tracemalloc
            if enable_tracemalloc:
                try:
                    import tracemalloc
                    tracemalloc.start()
                    snap_before = tracemalloc.take_snapshot()
                except Exception as e:
                    report.errors.append(f"tracemalloc start: {e}")
                    snap_before = None
            else:
                snap_before = None

            # Sampling thread
            stop_event = threading.Event()
            t0 = time.monotonic()

            def _sampler():
                while not stop_event.is_set():
                    try:
                        snap = _take_snapshot(t0, pid, use_tracemalloc=enable_tracemalloc)
                        report.snapshots.append(snap)
                    except Exception:
                        pass
                    stop_event.wait(interval)

            sampler = threading.Thread(target=_sampler, daemon=True)
            sampler.start()

            # Run the actual function
            try:
                result = fn(*args, **kwargs)
            finally:
                stop_event.set()
                sampler.join(timeout=2)
                report.duration_s = time.monotonic() - t0

                # Final snapshot
                try:
                    report.snapshots.append(_take_snapshot(t0, pid, use_tracemalloc=enable_tracemalloc))
                except Exception:
                    pass

                # Tracemalloc diff
                if enable_tracemalloc and snap_before is not None:
                    try:
                        import tracemalloc
                        snap_after = tracemalloc.take_snapshot()
                        diff = snap_after.compare_to(snap_before, "lineno")
                        report.tracemalloc_diff = [
                            {
                                "file": str(stat.traceback),
                                "size_diff_kb": round(stat.size_diff / 1024, 1),
                                "size_kb": round(stat.size / 1024, 1),
                                "count_diff": stat.count_diff,
                            }
                            for stat in diff[:20]
                        ]
                        peak = tracemalloc.get_traced_memory()[1]
                        report.tracemalloc_peak_mb = peak / (1024 * 1024)
                        tracemalloc.stop()
                    except Exception as e:
                        report.errors.append(f"tracemalloc diff: {e}")

                # Peaks
                if report.snapshots:
                    report.peak_rss_mb = max(s.rss_mb for s in report.snapshots)
                    report.peak_vms_mb = max(s.vms_mb for s in report.snapshots)

                # Collect unique network connections across all snapshots
                seen_conns: set[str] = set()
                for snap in report.snapshots:
                    for conn in snap.connections:
                        key = f"{conn.get('laddr', '')}>{conn.get('raddr', '')}:{conn.get('status', '')}"
                        if key not in seen_conns:
                            seen_conns.add(key)
                            report.network_connections.append(conn)

                # Collect unique child processes
                seen_pids: set[int] = set()
                for snap in report.snapshots:
                    for ch in snap.children:
                        if ch["pid"] not in seen_pids:
                            seen_pids.add(ch["pid"])
                            report.process_tree.append(ch)

                # Docker audit
                if docker_audit:
                    report.docker_info = _collect_docker_info()

            # Return (result, report) tuple — caller unpacks
            return result, report

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# HTML Report Generator — TBJS Glass Style
# ---------------------------------------------------------------------------

def generate_html_report(report: AnalysisReport) -> str:
    """Generate a polished TBJS Glass HTML report."""
    s = report.summary

    # Pre-compute data for charts
    cc_dist = s.get("complexity_distribution", {})
    lint_sev = s.get("lint_by_severity", {})
    sec_sev = s.get("security_by_severity", {})
    dead_types = s.get("dead_code_by_type", {})

    top_complex = s.get("top_complex_functions", [])
    worst_mi = s.get("worst_maintainability", [])
    worst_mi_blocks = s.get("worst_maintainability_blocks", [])

    # Build file details
    file_rows = []
    for f in report.files:
        cc_max = max((b["complexity"] for b in f.complexity), default=0)
        cc_rank = max((b.get("rank", "A") for b in f.complexity), default="A") if f.complexity else "-"
        issues = len(f.lint_issues) + len(f.security_issues)
        dead = len(f.dead_code)
        file_rows.append({
            "path": f.path,
            "lang": f.language,
            "sloc": f.sloc,
            "cc_max": cc_max,
            "cc_rank": cc_rank,
            "mi": f.maintainability_index,
            "issues": issues,
            "dead": dead,
            "security": len(f.security_issues),
            "lint_items": f.lint_issues,
            "security_items": f.security_issues,
            "dead_items": f.dead_code,
            "complexity_items": f.complexity,
        })

    # Sort by issues descending
    file_rows.sort(key=lambda r: (r["issues"] + r["dead"]), reverse=True)

    # Health score: 0-100
    health = _compute_health_score(s)

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tb_analyze — {_esc(str(report.target))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
{_get_css()}
</style>
</head>
<body>

<div class="report-container">

  <!-- Header -->
  <header class="report-header">
    <div class="header-row">
      <div>
        <h6 class="label">tb_analyze</h6>
        <h1>Static Analysis Report</h1>
        <p class="meta">
          <span class="mono">{_esc(str(report.target))}</span>
          &middot; {s.get('files_analyzed', 0)} files
          &middot; {s.get('total_sloc', 0):,} SLOC
          &middot; {report.duration_s:.1f}s
        </p>
      </div>
      <div class="health-ring" data-score="{health}">
        <svg viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" class="ring-bg"/>
          <circle cx="60" cy="60" r="52" class="ring-fg" style="--score:{health}"/>
        </svg>
        <div class="health-value">
          <span class="mono">{health}</span>
          <span class="label">health</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Summary Cards -->
  <section class="cards-grid">
    {_card("Complexity", f'{s.get("avg_complexity", 0):.1f}', "avg CC", _cc_color(s.get("avg_complexity", 0)), bar_value=s.get("avg_complexity", 0), bar_max=20, bar_invert=True)}
    {_card("Maintainability", f'{s.get("avg_maintainability", 0):.0f}', "avg MI", _mi_color(s.get("avg_maintainability", 0)), bar_value=s.get("avg_maintainability", 0), bar_max=100)}
    {_card("Lint Issues", str(s.get("total_lint_issues", 0)), "total", _issue_color(s.get("total_lint_issues", 0)), bar_value=s.get("total_lint_issues", 0), bar_max=max(s.get("total_lint_issues", 1), 50), bar_invert=True)}
    {_card("Security", str(s.get("total_security_issues", 0)), "issues", _sec_color(s.get("total_security_issues", 0)), bar_value=s.get("total_security_issues", 0), bar_max=max(s.get("total_security_issues", 1), 20), bar_invert=True)}
    {_card("Dead Code", str(s.get("total_dead_code", 0)), "items", _dead_color(s.get("total_dead_code", 0)), bar_value=s.get("total_dead_code", 0), bar_max=max(s.get("total_dead_code", 1), 30), bar_invert=True)}
    {_card("Comment Ratio", f'{s.get("comment_ratio", 0):.1f}%', "of SLOC", "var(--info)", bar_value=s.get("comment_ratio", 0), bar_max=30)}
  </section>

  <!-- Distributions -->
  <section class="distributions">
    <div class="card">
      <h6 class="label">Complexity Distribution</h6>
      <div class="bar-chart">
        {_bar_chart(cc_dist, _CC_COLORS)}
      </div>
    </div>
    <div class="card">
      <h6 class="label">Lint by Severity</h6>
      <div class="bar-chart">
        {_bar_chart(lint_sev, _SEVERITY_COLORS)}
      </div>
    </div>
    <div class="card">
      <h6 class="label has-tooltip" data-tooltip="Bandit security scanner: checks for hardcoded secrets, SQL injection, unsafe deserialization, shell injection, insecure crypto, debug flags, and subprocess risks.">Security by Severity</h6>
      <div class="bar-chart">
        {_bar_chart(sec_sev, _SEC_COLORS)}
      </div>
    </div>
    <div class="card">
      <h6 class="label">Dead Code by Type</h6>
      <div class="bar-chart">
        {_bar_chart(dead_types, _DEAD_COLORS)}
      </div>
    </div>
  </section>

  <!-- Worst Maintainability -->
  {_worst_mi_section(worst_mi, worst_mi_blocks)}

  <!-- Top Complex Functions -->
  {_top_complex_section(top_complex)}

  <!-- Dead Code Findings -->
  {_dead_code_section(report.files)}

  <!-- Security Findings -->
  {_security_findings_section(report.files)}

  <!-- File Details -->
  <section class="file-details">
    <h2>File Details</h2>
    <div class="grid-table" style="grid-template-columns: 1fr 60px 70px 50px 50px 55px 55px 50px;">
      <div class="grid-header">
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['Path'])}">Path</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['Lang'])}">Lang</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['SLOC'])}">SLOC</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['CC'])}">CC</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['Rank'])}">Rank</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['MI'])}">MI</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['Lint'])}">Lint</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['Dead'])}">Dead</div>
      </div>
      {''.join(_file_row(r, str(report.target)) for r in file_rows)}
    </div>
  </section>

  <!-- Issue Details (expandable) -->
  {_issue_details_section(file_rows, str(report.target))}

  <!-- Footer -->
  <footer class="report-footer">
    <span class="label">Generated by tb_analyze</span>
    <span class="mono">{report.timestamp}</span>
  </footer>

</div>

<script>
{_get_js()}
</script>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Report Helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _penalty_from_thresholds(value: float, thresholds: list[tuple[float, int]], higher_is_worse: bool = True) -> int:
    """Calculate penalty from sorted thresholds. Returns the first matching penalty."""
    for threshold, penalty in thresholds:
        if higher_is_worse and value > threshold:
            return penalty
        if not higher_is_worse and value < threshold:
            return penalty
    return 0


def _compute_health_score(s: dict) -> int:
    """0-100 health score from summary metrics."""
    score = 100

    # (threshold, penalty) — checked in order, first match wins
    score -= _penalty_from_thresholds(
        s.get("avg_complexity", 0), [(10, 25), (5, 10)])

    score -= _penalty_from_thresholds(
        s.get("avg_maintainability", 100), [(30, 25), (50, 15), (65, 5)], higher_is_worse=False)

    sloc = max(s.get("total_sloc", 1), 1)
    lint_density = s.get("total_lint_issues", 0) / sloc * 100
    score -= _penalty_from_thresholds(lint_density, [(5, 20), (2, 10), (0.5, 5)])

    score -= _penalty_from_thresholds(
        s.get("total_security_issues", 0), [(10, 20), (3, 10), (0, 5)])

    score -= _penalty_from_thresholds(
        s.get("total_dead_code", 0), [(20, 10), (5, 5)])

    return max(0, min(100, score))


_CC_COLORS = {"A": "var(--success)", "B": "var(--info)", "C": "var(--warning)", "D": "var(--error)", "E": "var(--error)", "F": "var(--error)"}
_SEVERITY_COLORS = {"error": "var(--error)", "warning": "var(--warning)", "info": "var(--info)"}
_DEAD_COLORS = {"function": "var(--warning)", "variable": "var(--info)", "import": "var(--text-muted)", "class": "var(--error)", "attribute": "var(--text-label)", "property": "var(--text-label)"}
_SEC_COLORS = {"HIGH": "var(--error)", "MEDIUM": "var(--warning)", "LOW": "var(--info)"}


def _cc_color(cc: float) -> str:
    if cc <= 5:
        return "var(--success)"
    if cc <= 10:
        return "var(--warning)"
    return "var(--error)"


def _mi_color(mi: float) -> str:
    if mi >= 65:
        return "var(--success)"
    if mi >= 40:
        return "var(--warning)"
    return "var(--error)"


def _issue_color(n: int) -> str:
    if n == 0:
        return "var(--success)"
    if n <= 10:
        return "var(--warning)"
    return "var(--error)"


def _sec_color(n: int) -> str:
    if n == 0:
        return "var(--success)"
    if n <= 3:
        return "var(--warning)"
    return "var(--error)"


def _dead_color(n: int) -> str:
    if n <= 3:
        return "var(--success)"
    if n <= 10:
        return "var(--warning)"
    return "var(--error)"


_TOOLTIPS = {
    "Complexity": "Cyclomatic Complexity (CC) — counts independent paths through code. "
                  "1-5 = simple (A), 6-10 = moderate (B), 11-20 = complex (C), 21+ = high risk (D+). Lower is better.",
    "Maintainability": "Maintainability Index (MI) — composite score from Halstead Volume, CC, and LOC. "
                       "0-100 scale. >65 = good, 40-65 = moderate, <40 = hard to maintain. "
                       "Computed per function/class and averaged (weighted by SLOC). Higher is better.",
    "Lint Issues": "Code style and correctness issues found by ruff (Python linter). "
                   "Includes unused imports, formatting, naming, and potential bugs.",
    "Security": "Security vulnerabilities found by bandit. "
                "Checks for hardcoded passwords, SQL injection, unsafe deserialization, subprocess risks, etc.",
    "Dead Code": "Unreachable or unused code found by vulture. "
                 "Includes unused functions, classes, variables, and imports. "
                 "Public API functions may show as false positives (60% confidence).",
    "Comment Ratio": "Percentage of SLOC that are comments. "
                     "10-30% is typical for well-documented code. Too low = underdocumented, too high = noisy.",
    "CC": "Cyclomatic Complexity — independent execution paths. A(1-5) B(6-10) C(11-20) D(21+)",
    "MI": "Maintainability Index — 0-100 composite. >65 good, <40 problematic.",
    "SLOC": "Source Lines of Code — non-blank, non-comment lines.",
    "Rank": "Complexity rank: A=simple, B=moderate, C=complex, D=high risk, F=unmaintainable.",
    "Path": "Relative file path from analysis target root.",
    "Lang": "Programming language detected from file extension.",
    "Lint": "Number of lint issues (style + correctness) in this file.",
    "Dead": "Number of dead/unused code items detected in this file.",
}


def _score_bar(value: float, max_val: float, segments: int = 10, invert: bool = False) -> str:
    """Generate LED score bar HTML. invert=True means lower is better."""
    if max_val <= 0:
        filled = 0
    else:
        ratio = min(value / max_val, 1.0)
        if invert:
            ratio = 1.0 - ratio
        filled = int(ratio * segments)

    # Color: green for high fill, warning for mid, error for low
    if filled >= 7:
        cls = "is-success"
    elif filled >= 4:
        cls = "is-warning"
    else:
        cls = "is-error"

    blocks = []
    for i in range(segments):
        active = "is-active " + cls if i < filled else ""
        blocks.append(f'<span class="score-block {active}"></span>')
    return f'<div class="score-bar">{"".join(blocks)}</div>'


def _card(title: str, value: str, subtitle: str, color: str,
          bar_value: float = 0, bar_max: float = 100, bar_invert: bool = False) -> str:
    bar_html = _score_bar(bar_value, bar_max, invert=bar_invert)
    tip = _esc(_TOOLTIPS.get(title, ""))
    tip_attr = f' class="has-tooltip" data-tooltip="{tip}"' if tip else ""
    return f"""<div class="card metric-card">
      <h6 class="label"{tip_attr}>{_esc(title)}</h6>
      <div class="metric-value mono" style="color:{color}">{_esc(value)}</div>
      <div class="metric-sub">{_esc(subtitle)}</div>
      {bar_html}
    </div>"""


def _bar_chart(data: dict, colors: dict) -> str:
    if not data:
        return '<span class="text-muted">No data</span>'
    total = max(sum(data.values()), 1)
    bars = []
    for key, count in sorted(data.items()):
        pct = count / total * 100
        color = colors.get(key, "var(--primary)")
        bars.append(
            f'<div class="bar-row">'
            f'<span class="bar-label mono">{_esc(key)}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span class="bar-count mono">{count}</span>'
            f'</div>'
        )
    return "\n".join(bars)

def _collapsible(title: str, body: str, count: int | None = None,
                 open_default: bool = False) -> str:
    """Wrap a section body in a collapsible <details> block. Closed by default."""
    if not body:
        return ""
    badge = f'<span class="section-count">{count}</span>' if count is not None else ""
    open_attr = " open" if open_default else ""
    return (
        f'<details class="report-section"{open_attr}>'
        f'<summary class="report-section-summary">'
        f'<h2 class="report-section-title">{_esc(title)}</h2>{badge}'
        f'</summary>'
        f'<div class="report-section-body">{body}</div>'
        f'</details>'
    )

def _top_complex_section(items: list[dict]) -> str:
    if not items:
        return ""
    rows = ""
    for b in items:
        color = _CC_COLORS.get(b.get("rank", "A"), "var(--info)")
        path_full = b.get("path", "")
        path_short = Path(path_full).name if path_full else ""
        rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono">{_esc(b["name"])}</div>'
            f'<div class="grid-cell mono file-path" title="{_esc(path_full)}">{_esc(path_short)}</div>'
            f'<div class="grid-cell" style="color:{color}">{b["complexity"]}</div>'
            f'<div class="grid-cell" style="color:{color}">{b.get("rank", "?")}</div>'
            f'<div class="grid-cell mono">{b.get("line", "")}</div>'
            f'</div>'
        )
    body = f"""<div class="grid-table" style="grid-template-columns:1fr 1fr 60px 50px 60px">
          <div class="grid-header">
            <div class="grid-cell">Function</div>
            <div class="grid-cell">File</div>
            <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['CC'])}">CC</div>
            <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['Rank'])}">Rank</div>
            <div class="grid-cell">Line</div>
          </div>
          {rows}
        </div>"""
    return _collapsible("Top Complex Functions", body, count=len(items))


def _worst_mi_section(items: list, blocks: list[dict] | None = None) -> str:
    if not items and not blocks:
        return ""
    file_rows = ""
    for path, mi in items:
        color = _mi_color(mi)
        file_rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono file-path" title="{_esc(str(path))}">{_esc(str(Path(path).name))}</div>'
            f'<div class="grid-cell" style="color:{color}">{mi:.1f}</div>'
            f'</div>'
        )

    block_rows = ""
    for b in (blocks or []):
        color = _mi_color(b.get("mi", 0))
        path_full = b.get("path", "")
        path_short = Path(path_full).name if path_full else ""
        block_rows += (
            f'<div class="grid-row">'
            f'<div class="grid-cell mono">{_esc(b.get("name", ""))}</div>'
            f'<div class="grid-cell">{_esc(b.get("type", ""))}</div>'
            f'<div class="grid-cell mono file-path" title="{_esc(path_full)}">{_esc(path_short)}</div>'
            f'<div class="grid-cell mono">{b.get("line", "")}</div>'
            f'<div class="grid-cell" style="color:{color}">{b.get("mi", 0):.1f}</div>'
            f'</div>'
        )

    file_table = f"""
    <h3>By File</h3>
    <div class="grid-table" style="grid-template-columns:1fr 80px">
      <div class="grid-header">
        <div class="grid-cell">File</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['MI'])}">MI</div>
      </div>
      {file_rows}
    </div>""" if file_rows else ""

    block_table = f"""
    <h3>By Function / Class</h3>
    <div class="grid-table" style="grid-template-columns:1fr 80px 1fr 60px 80px">
      <div class="grid-header">
        <div class="grid-cell">Name</div>
        <div class="grid-cell">Type</div>
        <div class="grid-cell">File</div>
        <div class="grid-cell">Line</div>
        <div class="grid-cell has-tooltip" data-tooltip="{_esc(_TOOLTIPS['MI'])}">MI</div>
      </div>
      {block_rows}
    </div>""" if block_rows else ""

    body = f"{file_table}{block_table}"
    return _collapsible(
        "Worst Maintainability",
        body,
        count=(len(items) + len(blocks or [])),
    )


def _security_findings_section(files: list) -> str:
    """Dedicated security findings section with all issues listed."""
    all_sec = []
    for f in files:
        for issue in f.security_issues:
            all_sec.append({
                "file": f.path,
                "test_id": issue.get("test_id", ""),
                "test_name": issue.get("test_name", ""),
                "message": issue.get("message", ""),
                "severity": issue.get("severity", "LOW"),
                "confidence": issue.get("confidence", "LOW"),
                "line": issue.get("line", 0),
                "cwe": issue.get("cwe", ""),
            })

    if not all_sec:
        return ""

    # Sort by severity: HIGH first
    sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    all_sec.sort(key=lambda x: sev_order.get(x["severity"], 3))

    rows = []
    for issue in all_sec:
        sev = issue["severity"]
        sev_cls = "sec-high" if sev == "HIGH" else "sec-med" if sev == "MEDIUM" else "sec-low"
        sev_color = _SEC_COLORS.get(sev, "var(--info)")
        fname = Path(issue["file"]).name
        cwe_tag = f' <span class="mono" style="color:var(--text-muted)">CWE-{issue["cwe"]}</span>' if issue["cwe"] else ""
        rows.append(
            f'<div class="grid-row {sev_cls}">'
            f'<div class="grid-cell" style="color:{sev_color}">{_esc(sev)}</div>'
            f'<div class="grid-cell mono">{_esc(issue["test_id"])}</div>'
            f'<div class="grid-cell">{_esc(issue["message"])}{cwe_tag}</div>'
            f'<div class="grid-cell mono">{_esc(fname)}:{issue["line"]}</div>'
            f'</div>'
        )

    body = f"""<div class="grid-table" style="grid-template-columns: 70px 60px 1fr 140px;">
      <div class="grid-header">
        <div class="grid-cell has-tooltip" data-tooltip="HIGH = likely exploitable, MEDIUM = potential risk, LOW = informational or minor concern">Severity</div>
        <div class="grid-cell has-tooltip" data-tooltip="Bandit test ID — e.g. B404 = import subprocess, B603 = subprocess call without shell=False, B110 = try/except/pass">Test</div>
        <div class="grid-cell">Description</div>
        <div class="grid-cell">Location</div>
      </div>
      {''.join(rows)}
    </div>"""
    return _collapsible("Security Findings", body, count=len(all_sec))

def _dead_code_section(files: list) -> str:
    """Dedicated dead code findings section with file/line context."""
    all_dead = []
    for f in files:
        for d in f.dead_code:
            all_dead.append({
                "file": f.path,
                "name": d.get("name", ""),
                "type": d.get("type", ""),
                "line": d.get("line", 0),
                "confidence": d.get("confidence", 0),
                "message": d.get("message", ""),
            })

    if not all_dead:
        return ""

    # Sort by confidence desc, then file
    all_dead.sort(key=lambda x: (-x["confidence"], x["file"], x["line"]))

    rows = []
    for d in all_dead:
        path_short = Path(d["file"]).name
        rows.append(
            f'<div class="grid-row">'
            f'<div class="grid-cell mono">{_esc(d["type"])}</div>'
            f'<div class="grid-cell mono">{_esc(d["name"])}</div>'
            f'<div class="grid-cell mono file-path" title="{_esc(d["file"])}">{_esc(path_short)}</div>'
            f'<div class="grid-cell mono">{d["line"]}</div>'
            f'<div class="grid-cell mono">{d["confidence"]}%</div>'
            f'</div>'
        )

    body = f"""<div class="grid-table" style="grid-template-columns: 100px 1fr 1fr 70px 80px;">
      <div class="grid-header">
        <div class="grid-cell">Type</div>
        <div class="grid-cell">Name</div>
        <div class="grid-cell">File</div>
        <div class="grid-cell">Line</div>
        <div class="grid-cell has-tooltip" data-tooltip="Vulture confidence: higher = more likely truly dead. 60%+ is reliable; lower may be dynamic refs.">Conf</div>
      </div>
      {''.join(rows)}
    </div>"""
    return _collapsible("Dead Code Findings", body, count=len(all_dead))

def _file_row(r: dict, base: str) -> str:
    try:
        rel = str(Path(r["path"]).relative_to(base))
    except ValueError:
        rel = r["path"]
    cc_color = _CC_COLORS.get(r["cc_rank"], "var(--text-main)")
    mi_color = _mi_color(r["mi"]) if r["mi"] > 0 else "var(--text-muted)"
    return (
        f'<div class="grid-row" data-file="{_esc(r["path"])}">'
        f'<div class="grid-cell mono file-path" title="{_esc(r["path"])}">{_esc(rel)}</div>'
        f'<div class="grid-cell">{r["lang"][:2]}</div>'
        f'<div class="grid-cell mono">{r["sloc"]}</div>'
        f'<div class="grid-cell mono" style="color:{cc_color}">{r["cc_max"]}</div>'
        f'<div class="grid-cell" style="color:{cc_color}">{r["cc_rank"]}</div>'
        f'<div class="grid-cell mono" style="color:{mi_color}">{r["mi"]:.0f}</div>'
        f'<div class="grid-cell mono">{r["issues"]}</div>'
        f'<div class="grid-cell mono">{r["dead"]}</div>'
        f'</div>'
    )


def _issue_details_section(file_rows: list[dict], base: str) -> str:
    """Expandable issue details for files with problems."""
    sections = []
    for r in file_rows:
        items = r["lint_items"] + r["security_items"] + r["dead_items"]
        if not items:
            continue
        try:
            rel = str(Path(r["path"]).relative_to(base))
        except ValueError:
            rel = r["path"]

        details = []
        for i in r["lint_items"]:
            details.append(f'<div class="issue-row lint"><span class="issue-badge">LINT</span>'
                           f'<span class="mono">L{i["line"]}</span>'
                           f'<span class="issue-code mono">{_esc(i.get("code", ""))}</span>'
                           f'<span>{_esc(i["message"])}</span></div>')
        for i in r["security_items"]:
            sev = i.get("severity", "LOW")
            cls = "sec-high" if sev == "HIGH" else "sec-med" if sev == "MEDIUM" else "sec-low"
            details.append(f'<div class="issue-row security {cls}"><span class="issue-badge">SEC</span>'
                           f'<span class="mono">L{i["line"]}</span>'
                           f'<span class="issue-code mono">{_esc(i.get("test_id", ""))}</span>'
                           f'<span>{_esc(i["message"])}</span></div>')
        for i in r["dead_items"]:
            details.append(f'<div class="issue-row dead"><span class="issue-badge">DEAD</span>'
                           f'<span class="mono">L{i["line"]}</span>'
                           f'<span class="issue-code mono">{_esc(i["type"])}</span>'
                           f'<span>{_esc(i["name"])}</span></div>')

        sections.append(
            f'<details class="file-issues">'
            f'<summary class="mono">{_esc(rel)} <span class="issue-count">{len(items)}</span></summary>'
            f'<div class="issues-body">{"".join(details)}</div>'
            f'</details>'
        )

    if not sections:
        return ""

    return f'<section class="issue-details"><h2>Issue Details</h2>{"".join(sections)}</section>'


# ---------------------------------------------------------------------------
# CSS — TBJS Glass v3.0
# ---------------------------------------------------------------------------

def _get_css() -> str:
    return """
:root, [data-theme="dark"] {
  --raw-primary: 55% 0.18 230;
  --raw-success: 65% 0.2 145;
  --raw-warning: 75% 0.18 85;
  --raw-error: 55% 0.22 25;
  --raw-info: 60% 0.15 230;

  --primary: oklch(var(--raw-primary));
  --success: oklch(var(--raw-success));
  --warning: oklch(var(--raw-warning));
  --error: oklch(var(--raw-error));
  --info: oklch(var(--raw-info));

  --bg-base: #08080d;
  --bg-surface: rgba(10, 10, 18, 0.8);
  --bg-elevated: rgba(15, 15, 25, 0.9);
  --bg-sunken: rgba(0, 0, 0, 0.3);
  --glass-bg: rgba(255, 255, 255, 0.02);
  --glass-border: rgba(255, 255, 255, 0.05);
  --glass-blur: 12px;
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-active: color-mix(in oklch, var(--primary) 30%, transparent);

  --text-main: rgba(255, 255, 255, 0.85);
  --text-label: rgba(255, 255, 255, 0.4);
  --text-muted: rgba(255, 255, 255, 0.25);

  --input-bg: rgba(0, 0, 0, 0.3);
  --surface-hover: color-mix(in oklch, var(--primary) 5%, transparent);
  --surface-active: color-mix(in oklch, var(--primary) 10%, transparent);
  --surface-badge: color-mix(in oklch, var(--primary) 15%, transparent);

  --font-sans: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;

  --text-h1: clamp(18px, 2vw, 22px);
  --text-h2: clamp(16px, 1.8vw, 19px);
  --text-h3: clamp(14px, 1.5vw, 16px);
  --text-base: 13px;
  --text-sm: 11px;
  --text-xs: 9px;

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-8: 3rem;
  --space-10: 4rem;
  --space-12: 6rem;

  --radius-sm: 2px;
  --radius-md: 6px;
  --radius-lg: 12px;

  --highlight-inset: inset 0 1px 0 rgba(255, 255, 255, 0.05);
  --shadow-micro: 0 2px 4px rgba(0, 0, 0, 0.5);

  --duration-fast: 150ms;
  --duration-normal: 250ms;
  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-sans);
  font-size: var(--text-base);
  color: var(--text-main);
  background: var(--bg-base);
  line-height: 1.6;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

.mono { font-family: var(--font-mono); }

.label {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2.5px;
  color: var(--text-label);
  user-select: none;
}

h1 { font-size: var(--text-h1); font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; margin-block-end: var(--space-3); }
h2 { font-size: var(--text-h2); font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; margin-block-end: var(--space-3); }
h6 { font-family: var(--font-mono); font-size: var(--text-xs); text-transform: uppercase; letter-spacing: 2.5px; color: var(--text-label); margin-block-end: var(--space-2); }

/* Report layout */
.report-container {
  max-width: 1100px;
  margin: 0 auto;
  padding: var(--space-8) var(--space-5);
}

/* Header */
.report-header {
  margin-block-end: var(--space-6);
}
.header-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: var(--space-5);
}
.meta {
  font-size: var(--text-sm);
  color: var(--text-label);
  margin-top: var(--space-2);
}

/* Health ring */
.health-ring {
  position: relative;
  width: 120px;
  height: 120px;
  flex-shrink: 0;
}
.health-ring svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.ring-bg {
  fill: none;
  stroke: var(--border-subtle);
  stroke-width: 6;
}
.ring-fg {
  fill: none;
  stroke: var(--primary);
  stroke-width: 6;
  stroke-linecap: round;
  stroke-dasharray: 326.7;
  stroke-dashoffset: calc(326.7 - (326.7 * var(--score) / 100));
  transition: stroke-dashoffset 1s var(--ease-default);
}
.health-ring[data-score] .ring-fg {
  stroke: var(--success);
}
.health-ring[data-score="0"] .ring-fg,
.health-ring[data-score="1"] .ring-fg { stroke: var(--error); }
.health-value {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.health-value .mono {
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
}
.health-value .label {
  font-size: var(--text-xs);
  margin-top: var(--space-1);
}

/* Cards grid */
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--space-3);
  margin-block-end: var(--space-6);
}
.card {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  box-shadow: var(--highlight-inset), var(--shadow-micro);
  backdrop-filter: blur(var(--glass-blur));
}
.metric-card {
  text-align: center;
}
.metric-value {
  font-size: 28px;
  font-weight: 700;
  line-height: 1;
  margin: var(--space-3) 0 var(--space-1);
}
.metric-sub {
  font-size: var(--text-sm);
  color: var(--text-muted);
}

/* LED score bars — §4 signature component */
.score-bar {
  display: flex;
  gap: 3px;
  justify-content: center;
  margin-top: var(--space-3);
}
.score-block {
  width: 12px;
  height: 5px;
  border-radius: 1px;
  background: var(--border-subtle);
  transition: background var(--duration-fast) var(--ease-default),
              box-shadow var(--duration-fast) var(--ease-default);
}
.score-block.is-active {
  background: var(--primary);
  box-shadow: 0 0 4px var(--surface-badge);
}
.score-block.is-active.is-success {
  background: var(--success);
  box-shadow: 0 0 4px color-mix(in oklch, var(--success) 15%, transparent);
}
.score-block.is-active.is-warning {
  background: var(--warning);
  box-shadow: 0 0 4px color-mix(in oklch, var(--warning) 15%, transparent);
}
.score-block.is-active.is-error {
  background: var(--error);
  box-shadow: 0 0 4px color-mix(in oklch, var(--error) 15%, transparent);
}

/* Distributions */
.distributions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: var(--space-3);
  margin-block-end: var(--space-6);
}
.bar-chart {
  margin-top: var(--space-3);
}
.bar-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  margin-block-end: var(--space-2);
}
.bar-label {
  width: 70px;
  font-size: var(--text-sm);
  color: var(--text-label);
  text-align: right;
  flex-shrink: 0;
}
.bar-track {
  flex: 1;
  height: 8px;
  background: var(--bg-sunken);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: var(--radius-sm);
  transition: width 0.6s var(--ease-default);
  min-width: 2px;
}
.bar-count {
  width: 40px;
  font-size: var(--text-sm);
  color: var(--text-label);
  flex-shrink: 0;
}

/* Grid table */
.grid-table {
  display: grid;
  margin-block-end: var(--space-6);
}
.grid-header, .grid-row {
  display: contents;
}
.grid-header .grid-cell {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--text-label);
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--border-subtle);
}
.grid-row .grid-cell {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: var(--text-sm);
  transition: background var(--duration-fast) var(--ease-default);
}
.grid-row:hover .grid-cell {
  background: var(--surface-hover);
}
.file-path {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Section spacing */
section {
  margin-block-end: var(--space-6);
}

/* Issue details */
.file-issues {
  margin-block-end: var(--space-2);
}
.file-issues summary {
  cursor: pointer;
  padding: var(--space-2) var(--space-3);
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  list-style: none;
  display: flex;
  align-items: center;
  gap: var(--space-3);
  transition: background var(--duration-fast) var(--ease-default);
}
.file-issues summary:hover {
  background: var(--surface-hover);
}
.file-issues summary::before {
  content: '▸';
  color: var(--text-muted);
  transition: transform var(--duration-fast);
}
.file-issues[open] summary::before {
  transform: rotate(90deg);
}
.issue-count {
  margin-left: auto;
  background: var(--surface-badge);
  padding: 1px 8px;
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  color: var(--primary);
}
.issues-body {
  padding: var(--space-2) 0 var(--space-2) var(--space-5);
}
.issue-row {
  display: flex;
  align-items: baseline;
  gap: var(--space-3);
  padding: var(--space-1) 0;
  font-size: var(--text-sm);
  color: rgba(255,255,255,0.7);
}
.issue-badge {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  flex-shrink: 0;
}
.issue-row.lint .issue-badge { background: color-mix(in oklch, var(--warning) 20%, transparent); color: var(--warning); }
.issue-row.security .issue-badge { background: color-mix(in oklch, var(--error) 20%, transparent); color: var(--error); }
.issue-row.dead .issue-badge { background: color-mix(in oklch, var(--info) 20%, transparent); color: var(--info); }
.issue-row.sec-high { border-left: 2px solid var(--error); padding-left: var(--space-2); }
.issue-row.sec-med { border-left: 2px solid var(--warning); padding-left: var(--space-2); }
.issue-code {
  font-size: var(--text-xs);
  color: var(--text-label);
  flex-shrink: 0;
}

/* Footer */
.report-footer {
  display: flex;
  justify-content: space-between;
  padding-top: var(--space-5);
  border-top: 1px solid var(--border-subtle);
  margin-top: var(--space-8);
}

/* Responsive */
@media (max-width: 767px) {
  .report-container { padding: var(--space-5) var(--space-3); }
  .header-row { flex-direction: column; align-items: flex-start; }
  .health-ring { width: 80px; height: 80px; }
  .health-value .mono { font-size: 20px; }
  .cards-grid { grid-template-columns: repeat(2, 1fr); }
  .distributions { grid-template-columns: 1fr; }
  .grid-table { font-size: var(--text-xs); overflow-x: auto; }
}

.text-muted { color: var(--text-muted); }

/* Tooltips */
.has-tooltip {
  position: relative;
  cursor: help;
  border-bottom: 1px dotted var(--text-muted);
}
.has-tooltip::after {
  content: attr(data-tooltip);
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%) scale(0.96);
  width: max-content;
  max-width: min(320px, 90vw);
  padding: var(--space-3) var(--space-4);
  background: var(--bg-elevated);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-md);
  box-shadow: var(--highlight-inset), 0 4px 8px rgba(0,0,0,0.6);
  font-family: var(--font-sans);
  font-size: var(--text-sm);
  font-weight: 400;
  color: var(--text-main);
  text-transform: none;
  letter-spacing: 0;
  line-height: 1.5;
  white-space: normal;
  z-index: var(--z-modal);
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity var(--duration-fast) var(--ease-default),
              visibility var(--duration-fast) var(--ease-default),
              transform var(--duration-fast) var(--ease-default);
}
.has-tooltip:hover::after {
  opacity: 1;
  visibility: visible;
  transform: translateX(-50%) scale(1);
}
/* Prevent right-edge overflow: cells near right side anchor left */
.grid-header .grid-cell.has-tooltip::after {
  left: auto;
  right: 0;
  transform: translateX(0) scale(0.96);
}
.grid-header .grid-cell.has-tooltip:hover::after {
  transform: translateX(0) scale(1);
}
/* First cell anchors to the left instead */
.grid-header .grid-cell:first-child.has-tooltip::after {
  left: 0;
  right: auto;
}
/* Cards: tooltip below instead of above to avoid clipping at top */
.metric-card .has-tooltip::after {
  bottom: auto;
  top: calc(100% + 6px);
}

/* Collapsible sections */
.report-section {
  margin-block-end: var(--space-6);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  background: var(--glass-bg);
  overflow: hidden;
}
.report-section-summary {
  cursor: pointer;
  list-style: none;
  padding: var(--space-3) var(--space-4);
  display: flex;
  align-items: center;
  gap: var(--space-3);
  transition: background var(--duration-fast) var(--ease-default);
  user-select: none;
}
.report-section-summary::-webkit-details-marker { display: none; }
.report-section-summary::before {
  content: '▸';
  color: var(--text-muted);
  font-size: var(--text-sm);
  transition: transform var(--duration-fast);
  flex-shrink: 0;
}
.report-section[open] > .report-section-summary::before {
  transform: rotate(90deg);
}
.report-section-summary:hover {
  background: var(--surface-hover);
}
.report-section-title {
  margin: 0;
  font-size: var(--text-lg);
  font-weight: 600;
  color: var(--text-main);
}
.section-count {
  margin-left: auto;
  background: var(--surface-badge);
  padding: 2px 10px;
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  font-family: var(--font-mono);
  color: var(--primary);
}
.report-section-body {
  padding: var(--space-4);
  border-top: 1px solid var(--border-subtle);
}
"""


def _get_js() -> str:
    return """
// Dynamic health ring color based on score
document.querySelectorAll('.health-ring').forEach(el => {
  const score = parseInt(el.dataset.score || '0');
  const fg = el.querySelector('.ring-fg');
  if (score >= 80) fg.style.stroke = 'oklch(65% 0.2 145)'; // success
  else if (score >= 50) fg.style.stroke = 'oklch(75% 0.18 85)'; // warning
  else fg.style.stroke = 'oklch(55% 0.22 25)'; // error
});
"""
