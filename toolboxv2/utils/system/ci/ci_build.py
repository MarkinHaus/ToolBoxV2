#!/usr/bin/env python3
"""
ToolBoxV2 CI Pipeline — Build · Test · Report · Upload

Usage:
    tb fbuild build                    # Pack features → wheel/sdist
    tb fbuild test                     # Build + venv-isolated tests per feature
    tb fbuild upload --test            # Upload to test.pypi.org
    tb fbuild upload --prod            # Upload to pypi.org
    tb fbuild deps --analyze           # Show dependency map
    tb fbuild deps --update            # Check for newer versions
    tb fbuild deps --minimize          # Rewrite pyproject.toml base deps to mini-only
    tb fbuild all --test               # Full pipeline → test.pypi
    tb fbuild all --prod               # Full pipeline → pypi
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from toolboxv2 import tb_root_dir

ROOT = tb_root_dir.parent
TOOLBOXV2_DIR = ROOT / "toolboxv2"
FEATURES_DIR = TOOLBOXV2_DIR / "features"
PACKED_DIR = TOOLBOXV2_DIR / "features_packed"
DIST_DIR = ROOT / "dist"
REPORT_DIR = ROOT / "build_reports"

ALWAYS_SOURCE = {"core", "mini"}

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

    class _Fallback:
        def print(self, *a, **kw):
            kw.pop("style", None)
            kw.pop("highlight", None)
            print(*a, **{k: v for k, v in kw.items() if k == "end"})

        def rule(self, title="", **kw):
            print(f"\n{'─' * 20} {title} {'─' * 20}")

    console = _Fallback()


def _p(msg: str, style: str = ""):
    console.print(msg, style=style)

# ── YAML Mini-Parser (keine externe Dep) ────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    """Minimaler YAML-Parser für feature.yaml (flache Strukturen)."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = {}

    for m in re.finditer(r'^(\w+):\s*"?([^"\n]+)"?', text, re.MULTILINE):
        key, val = m.group(1), m.group(2).strip().strip('"')
        if val.lower() == "true":
            data[key] = True
        elif val.lower() == "false":
            data[key] = False
        else:
            data[key] = val

    for section in ("files", "dependencies", "requires", "commands", "imports"):
        items = []
        pattern = rf'^{section}:\s*\n((?:\s+-\s+.+\n?)+)'
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            for line in m.group(1).strip().splitlines():
                item = line.strip().lstrip("- ").strip('"').strip("'")
                if item:
                    items.append(item)
        data[section] = items

    return data


# ── Feature Discovery ────────────────────────────────────────────────────────


def discover_features() -> dict[str, dict]:
    """Finde alle Features und lade ihre Configs."""
    features = {}
    if not FEATURES_DIR.exists():
        return features
    for d in sorted(FEATURES_DIR.iterdir()):
        yaml_path = d / "feature.yaml"
        if d.is_dir() and yaml_path.exists():
            features[d.name] = _load_yaml(yaml_path)
    return features


# ═════════════════════════════════════════════════════════════════════════════
#  BUILD
# ═════════════════════════════════════════════════════════════════════════════


def pack_feature(name: str, config: dict) -> Path | None:
    """Packe ein einzelnes Feature als ZIP."""
    version = config.get("version", "0.0.0")
    files_patterns = config.get("files", [])
    dependencies = config.get("dependencies", [])

    zip_name = f"tbv2-feature-{name}-{version}.zip"
    zip_path = PACKED_DIR / zip_name

    if zip_path.exists():
        zip_path.unlink()

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        src_yaml = FEATURES_DIR / name / "feature.yaml"
        if src_yaml.exists():
            zf.write(src_yaml, "feature.yaml")

        if dependencies:
            zf.writestr("requirements.txt", "\n".join(dependencies))

        for pattern in files_patterns:
            if pattern.endswith("/*"):
                src_dir = TOOLBOXV2_DIR / pattern[:-2]
                if src_dir.exists():
                    for fp in src_dir.rglob("*"):
                        if fp.is_file() and "__pycache__" not in str(fp):
                            rel = fp.relative_to(TOOLBOXV2_DIR)
                            zf.write(fp, f"files/{rel}")
                            file_count += 1
            else:
                src_file = TOOLBOXV2_DIR / pattern
                if src_file.exists():
                    zf.write(src_file, f"files/{pattern}")
                    file_count += 1

        zf.writestr(
            "_metadata.yaml",
            f"feature: {name}\nversion: {version}\n"
            f"packed_at: {datetime.now().isoformat()}\nfiles: {file_count}\n",
        )

    return zip_path


def create_manifest_in():
    """MANIFEST.in für setuptools."""
    content = (
        "recursive-include toolboxv2/features_packed *.zip *.md\n"
        "recursive-include toolboxv2/features/core *\n"
        "recursive-include toolboxv2/features/mini *\n"
        "global-exclude __pycache__\n"
        "global-exclude *.py[cod]\n"
    )
    (ROOT / "MANIFEST.in").write_text(content, encoding="utf-8")

def cmd_build() -> list[dict]:
    """Pack features + create wheel/sdist."""
    console.rule("BUILD")

    features = discover_features()
    if not features:
        _p("  ✗ Keine Features gefunden", style="red")
        return []

    PACKED_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name, config in features.items():
        if name in ALWAYS_SOURCE:
            _p(f"  ⊘ {name:12} (always source)")
            results.append({"feature": name, "action": "source", "size": 0})
            continue

        # Fix #10: per-feature error catching
        try:
            zip_path = pack_feature(name, config)
        except Exception as e:
            _p(f"  ✗ {name:12} PACK ERROR: {e}", style="red")
            results.append({"feature": name, "action": "failed", "size": 0, "pack_error": str(e)})
            continue

        if zip_path and zip_path.exists():
            size_kb = zip_path.stat().st_size // 1024
            _p(f"  ✓ {name:12} → {zip_path.name} ({size_kb} KB)")
            results.append({"feature": name, "action": "packed", "size": size_kb})
        else:
            _p(f"  ✗ {name:12} FAILED", style="red")
            results.append({"feature": name, "action": "failed", "size": 0})

    create_manifest_in()
    _p("  ✓ MANIFEST.in")

    _p("\n  Building wheel + sdist ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(DIST_DIR)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = list(DIST_DIR.glob("*.whl"))
        tars = list(DIST_DIR.glob("*.tar.gz"))
        for w in wheels:
            _p(f"  ✓ {w.name} ({w.stat().st_size // 1024} KB)")
        for t in tars:
            _p(f"  ✓ {t.name} ({t.stat().st_size // 1024} KB)")
    except subprocess.CalledProcessError as e:
        # Fix #6: capture start of stderr, not last 500
        stderr_output = e.stderr or ""
        _p(f"  ✗ Build failed:\n{stderr_output[:3000]}", style="red")
    except FileNotFoundError:
        _p("  ✗ 'build' module nicht installiert → pip install build", style="red")

    return results


# ═════════════════════════════════════════════════════════════════════════════
#  TEST
# ═════════════════════════════════════════════════════════════════════════════


def _find_wheel() -> Path | None:
    """Finde neuestes wheel in dist/."""
    if not DIST_DIR.exists():
        return None
    wheels = sorted(DIST_DIR.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    from toolboxv2 import get_app
    version = get_app().version
    wheels = [w for w in wheels if version in str(w)]
    return wheels[-1] if wheels else None

def _run_feature_test(
    feature_name: str,
    extras: list[str],
    wheel: Path,
    timeout: int = 240,
) -> dict:
    """Teste ein Feature in isoliertem venv."""
    result = {
        "feature": feature_name,
        "extras": extras,
        "passed": False,
        "status": "UNKNOWN",
        "tests_run": 0,
        "failures": 0,
        "errors": 0,
        "duration": 0.0,
        "output": "",
        "error_traceback": "",
        "install_ok": False,
    }

    venv_dir = None
    try:
        venv_dir = Path(tempfile.mkdtemp(prefix=f"tb_test_{feature_name}_"))
        venv_path = venv_dir / "venv"

        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )

        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        extra_str = ",".join(extras) if extras else ""
        install_target = f"{wheel}[{extra_str}]" if extra_str else str(wheel)

        proc = subprocess.run(
            [str(venv_python), "-m", "pip", "install", install_target, "--quiet"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            result["output"] = f"INSTALL FAILED:\n{proc.stderr[-1000:]}"
            result["status"] = "INSTALL_FAIL"
            return result

        proc = subprocess.run(
            [
                str(venv_python), "-m", "pip", "install",
                "pytest>=9.0.2",
                "pytest-asyncio>=0.23.0",
                "pytest-xdist>=3.5.0",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            result["output"] = f"PYTEST INSTALL FAILED:\n{proc.stderr[-1000:]}"
            result["status"] = "PYTEST_INSTALL_FAIL"
            return result

        result["install_ok"] = True

        # Resolve tests path from installed wheel (NOT source tree).
        # ROOT in cwd/rootdir would shadow site-packages via sys.path.
        resolve = subprocess.run(
            [str(venv_python), "-c",
             "import toolboxv2, pathlib; "
             "print(pathlib.Path(toolboxv2.__file__).parent / 'tests')"],
            capture_output=True, text=True, timeout=10,
        )
        if resolve.returncode != 0 or not resolve.stdout.strip():
            result["output"] = f"Cannot resolve installed tests path:\n{resolve.stderr[-800:]}"
            result["status"] = "RESOLVE_FAIL"
            return result
        installed_tests = Path(resolve.stdout.strip())
        feature_test_dir = installed_tests / f"test_{feature_name}"

        if feature_test_dir.exists():
            target = str(feature_test_dir)
        elif installed_tests.exists():
            target = str(installed_tests)
        else:
            result["output"] = f"No tests in installed wheel at {installed_tests}"
            result["passed"] = True
            result["status"] = "PASS"
            return result
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                str(venv_python), "-m", "pytest",
                target,
                "-v",
                "--tb=short",
                "-n", "auto",
                "--asyncio-mode=auto",
                "--rootdir", str(venv_dir),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(venv_dir),
        )
        result["duration"] = round(time.monotonic() - t0, 2)
        result["output"] = proc.stdout + proc.stderr

        combined = proc.stdout + proc.stderr

        m_passed = re.search(r"(\d+) passed", combined)
        m_failed = re.search(r"(\d+) failed", combined)
        m_error = re.search(r"(\d+) error", combined)

        tests_run = 0
        if m_passed:
            tests_run += int(m_passed.group(1))
        if m_failed:
            result["failures"] = int(m_failed.group(1))
            tests_run += result["failures"]
        if m_error:
            result["errors"] = int(m_error.group(1))
            tests_run += result["errors"]

        result["tests_run"] = tests_run
        result["passed"] = proc.returncode == 0
        result["status"] = "PASS" if proc.returncode == 0 else "TEST_FAIL"

        # Fix #5: pytest parse fallback
        if tests_run == 0 and result["failures"] == 0 and result["errors"] == 0:
            combined_lower = combined.lower()
            if "error" in combined_lower or "failed" in combined_lower:
                result["errors"] = 1
                result["parse_warning"] = "pytest output could not be parsed"
            if proc.returncode == 5 or "no tests ran" in combined.lower():
                result["status"] = "NO_TESTS"
                result["passed"] = False
                result["errors"] = 1
                return result

    except subprocess.TimeoutExpired:
        # Fix #3: timeout sets error counters
        result["output"] = f"TIMEOUT after {timeout}s"
        result["errors"] = 1
        result["failures"] = 0
        result["status"] = "TIMEOUT"
    except Exception as e:
        # Fix #2: separate traceback field + error counters
        result["output"] = f"ERROR: {e}"
        result["error_traceback"] = traceback.format_exc()
        result["errors"] = 1
        result["failures"] = 0
        result["status"] = "ERROR"
    finally:
        if venv_dir and venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)

    return result

def cmd_test(only_feature: str | None = None, do_build=True) -> list[dict]:
    """Teste jedes Feature isoliert."""
    console.rule("TEST")

    #wheel = _find_wheel()
    #if not wheel:
    #    _p("  Kein wheel gefunden — baue zuerst ...")
    if do_build:
        cmd_build()
    wheel = _find_wheel()
    if not wheel:
        _p("  ✗ Build fehlgeschlagen", style="red")
        return []

    _p(f"  Wheel: {wheel.name}\n")

    features = discover_features()

    test_matrix = [
        {"name": "mini", "extras": []},
    ]
    for name in features:
        if name in ALWAYS_SOURCE:
            continue
        test_matrix.append({"name": name, "extras": [name]})
    if only_feature:
        test_matrix = [t for t in test_matrix if t["name"] == only_feature]
        if not test_matrix:
            _p(f"  ✗ Feature '{only_feature}' nicht in Matrix", style="red")
            return []
    def _test_runner(entry, res, i):
        # Fix #4: thread crash protection
        try:
            name = entry["name"]
            extras = entry["extras"]
            label = f"[{','.join(extras)}]" if extras else "(base)"

            _p(f"  ⏳ Testing {name:12} {label} ...")
            r = _run_feature_test(name, extras, wheel)

            if r["passed"]:
                _p(
                    f"\r  ✓ {name:12} {label:16} "
                    f"{r['tests_run']} tests  {r['duration']}s"
                )
            elif not r["install_ok"]:
                _p(f"\r  ✗ {name:12} {label:16} INSTALL FAILED", style="red")
            else:
                _p(
                    f"\r  ✗ {name:12} {label:16} "
                    f"{r['failures']}F {r['errors']}E  {r['duration']}s",
                    style="red",
                )
            if 'feature' not in r:
                r["feature"] = name
            res[i] = r
        except Exception as e:
            res[i] = {
                "feature": entry["name"],
                "passed": False,
                "status": "CRASH",
                "errors": 1,
                "failures": 0,
                "tests_run": 0,
                "output": f"THREAD CRASH: {e}",
                "error_traceback": traceback.format_exc(),
                "install_ok": False,
                "duration": 0,
            }

    import threading

    # Fix #9: independent dicts instead of shared references
    results = [{} for _ in range(len(test_matrix))]
    ts = []
    for i, _entry in enumerate(test_matrix):
        t = threading.Thread(target=_test_runner, args=(_entry, results, i))
        t.start()
        ts.append(t)

    for t in ts:
        t.join()
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════════════


def _classify_failure(r: dict) -> tuple[str, str]:
    """Klassifiziere Fehler-Typ. Returns (label, color)."""
    status = r.get("status", "")
    if status in ("INSTALL_FAIL", "PYTEST_INSTALL_FAIL"):
        return "INSTALL FAIL", "#f97316"
    if status == "TIMEOUT":
        return "TIMEOUT", "#a855f7"
    if status == "CRASH":
        return "THREAD CRASH", "#dc2626"
    if status == "ERROR":
        return "ERROR", "#ef4444"
    if r.get("error_traceback"):
        return "EXCEPTION", "#ef4444"
    if r.get("parse_warning"):
        return "PARSE WARN", "#eab308"
    return "TEST FAIL", "#ef4444"


def cmd_report(
    build_results: list[dict],
    test_results: list[dict],
) -> Path:
    """Generiere HTML-Report."""
    console.rule("REPORT")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"build_report_{timestamp}.html"

    version = "?"
    try:
        from ci_version import read_version
        version = read_version()
    except Exception:
        pass

    def _status_badge(ok: bool, r: dict | None = None) -> str:
        if ok:
            color = "#22c55e"
            label = "PASS"
        else:
            label, color = _classify_failure(r or {})
        return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{label}</span>'

    build_rows = ""
    for r in build_results:
        action = r.get("action", "?")
        size = f'{r.get("size", 0)} KB' if r.get("size") else "—"
        color = "#22c55e" if action != "failed" else "#ef4444"
        error_detail = ""
        if r.get("pack_error"):
            error_detail = f' <span style="color:#f97316;font-size:11px">({html.escape(r["pack_error"])})</span>'
        build_rows += f"""
        <tr>
            <td>{html.escape(r['feature'])}</td>
            <td style="color:{color}">{action}{error_detail}</td>
            <td>{size}</td>
        </tr>"""

    test_rows = ""
    total_pass = 0
    total_fail = 0
    for r in test_results:
        passed = r.get("passed", False)
        if passed:
            total_pass += 1
        else:
            total_fail += 1

        extras = ",".join(r.get("extras", [])) or "mini"
        # Fix #7: no truncation — CSS handles overflow
        output_escaped = html.escape(r.get("output", ""))
        # Fix #8: error type classification
        status_detail = ""
        if not passed:
            fail_label, _ = _classify_failure(r)
            status_detail = f' <span style="opacity:.6;font-size:11px">[{fail_label}]</span>'
            if r.get("error_traceback"):
                output_escaped += f"\n\n--- TRACEBACK ---\n{html.escape(r['error_traceback'])}"
            if r.get("parse_warning"):
                output_escaped += f"\n\n--- PARSE WARNING ---\n{html.escape(r['parse_warning'])}"
        test_rows += f"""
        <tr>
            <td>{html.escape(r.get('feature', '?'))}</td>
            <td>[{html.escape(extras)}]</td>
            <td>{_status_badge(passed, r)}{status_detail}</td>
            <td>{r.get('tests_run', 0)}</td>
            <td>{r.get('failures', 0)}</td>
            <td>{r.get('errors', 0)}</td>
            <td>{r.get('duration', 0)}s</td>
            <td><details><summary>log</summary><pre style="max-height:300px;overflow:auto;font-size:11px">{output_escaped}</pre></details></td>
        </tr>"""

    all_pass = total_fail == 0 and total_pass > 0
    overall_badge = _status_badge(all_pass)
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ToolBoxV2 Build Report — v{html.escape(version)}</title>
<style>
  :root {{ --bg: #0d1117; --fg: #c9d1d9; --card: #161b22; --border: #30363d; --accent: #58a6ff; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--fg); padding: 24px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 22px; }}
  .header .meta {{ font-size: 13px; opacity: .7; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .card h2 {{ font-size: 16px; margin-bottom: 12px; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  th {{ font-weight: 600; opacity: .7; font-size: 11px; text-transform: uppercase; }}
  .summary {{ display: flex; gap: 24px; margin-bottom: 20px; }}
  .stat {{ text-align: center; }}
  .stat .n {{ font-size: 28px; font-weight: 700; }}
  .stat .label {{ font-size: 11px; opacity: .6; text-transform: uppercase; }}
  details summary {{ cursor: pointer; color: var(--accent); }}
  pre {{ background: #0d1117; padding: 8px; border-radius: 4px; white-space: pre-wrap; word-break: break-all; }}
</style>
</head>
<body>
<div class="header">
  <h1>ToolBoxV2 Build Report</h1>
  <div class="meta">v{html.escape(version)} · {datetime.now().strftime('%Y-%m-%d %H:%M')} · {overall_badge}</div>
</div>
<div class="summary">
  <div class="stat"><div class="n">{len(build_results)}</div><div class="label">Features</div></div>
  <div class="stat"><div class="n" style="color:#22c55e">{total_pass}</div><div class="label">Passed</div></div>
  <div class="stat"><div class="n" style="color:#ef4444">{total_fail}</div><div class="label">Failed</div></div>
  <div class="stat"><div class="n">{sum(r.get('tests_run', 0) for r in test_results)}</div><div class="label">Tests</div></div>
</div>
<div class="card">
  <h2>Build</h2>
  <table>
    <tr><th>Feature</th><th>Action</th><th>Size</th></tr>
    {build_rows}
  </table>
</div>
<div class="card">
  <h2>Tests</h2>
  <table>
    <tr><th>Feature</th><th>Extras</th><th>Status</th><th>Tests</th><th>Fail</th><th>Err</th><th>Time</th><th>Output</th></tr>
    {test_rows}
  </table>
</div>
</body>
</html>"""

    report_path.write_text(report_html, encoding="utf-8")
    _p(f"  ✓ {report_path}")

    json_path = REPORT_DIR / f"build_report_{timestamp}.json"
    json_path.write_text(
        json.dumps(
            {"version": version, "build": build_results, "tests": test_results},
            indent=2,
        ),
        encoding="utf-8",
    )

    return report_path


# ═════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ═════════════════════════════════════════════════════════════════════════════


def cmd_upload(production: bool = False):
    """Upload wheel + sdist via twine."""
    console.rule("UPLOAD")

    if not DIST_DIR.exists():
        _p("  ✗ dist/ nicht vorhanden — erst 'build' ausführen", style="red")
        return

    files = list(DIST_DIR.glob("*.whl")) + list(DIST_DIR.glob("*.tar.gz"))
    if not files:
        _p("  ✗ Keine Artefakte in dist/", style="red")
        return

    repo_url = (
        "https://upload.pypi.org/legacy/"
        if production
        else "https://test.pypi.org/legacy/"
    )
    label = "PyPI (PRODUCTION)" if production else "TestPyPI"

    _p(f"  Target: {label}")
    for f in files:
        _p(f"  → {f.name}")

    args = [
        sys.executable, "-m", "twine", "upload",
        "--repository-url", repo_url,
    ]
    args.extend(str(f) for f in files)

    try:
        subprocess.run(args, check=True, cwd=ROOT)
        _p(f"\n  ✓ Upload zu {label} erfolgreich")
    except subprocess.CalledProcessError as e:
        _p(f"\n  ✗ Upload fehlgeschlagen: {e}", style="red")
    except FileNotFoundError:
        _p("  ✗ twine nicht installiert → pip install twine", style="red")
    _upload_features_to_registry()

def _upload_features_to_registry(only_feature: str | None = None):
    """Upload packed feature ZIPs to TB Registry."""
    _p("\n  ── TB Registry ──")
    try:
        from toolboxv2.feature_loader_registry import upload_feature_to_registry
    except ImportError as e:
        _p(f"  ✗ feature_loader_registry import failed: {e}", style="red")
        return
    for name, config in discover_features().items():
        if only_feature and name != only_feature:
            continue
        if name in ALWAYS_SOURCE:
            _p(f"  ⊘ {name} (source-only)")
            continue
        version = config.get("version", "0.0.0")
        zip_path = PACKED_DIR / f"tbv2-feature-{name}-{version}.zip"
        if not zip_path.exists():
            _p(f"  ✗ {name}@{version}: ZIP fehlt", style="red")
            continue
        try:
            ok = upload_feature_to_registry(
                feature_name=name,
                version=version,
                zip_path=zip_path,
            )
            _p(f"  {'✓' if ok else '✗'} {name}@{version}", style="" if ok else "red")
        except Exception as e:
            _p(f"  ✗ {name}@{version}: {e}", style="red")
# ═════════════════════════════════════════════════════════════════════════════
#  DEPS
# ═════════════════════════════════════════════════════════════════════════════


def _parse_pyproject_deps() -> dict[str, list[str]]:
    """Parse alle dependency-Gruppen aus pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    groups: dict[str, list[str]] = {}

    m = re.search(
        r'^\[project\].*?^dependencies\s*=\s*\[(.*?)\]',
        text,
        re.MULTILINE | re.DOTALL,
    )
    if m:
        groups["mini"] = _extract_dep_list(m.group(1))

    for m in re.finditer(
        r'^(\w+)\s*=\s*\[(.*?)\]',
        _get_section(text, "[project.optional-dependencies]"),
        re.MULTILINE | re.DOTALL,
    ):
        name = m.group(1)
        groups[name] = _extract_dep_list(m.group(2))

    return groups


def _get_section(text: str, header: str) -> str:
    """Extrahiere TOML-Section."""
    start = text.find(header)
    if start < 0:
        return ""
    rest = text[start + len(header):]
    end = re.search(r'^\[', rest, re.MULTILINE)
    return rest[:end.start()] if end else rest


def _extract_dep_list(block: str) -> list[str]:
    """Extrahiere Dependencies aus einem TOML-Array-Block."""
    deps = []
    for line in block.splitlines():
        line = line.strip().strip(",")
        if line.startswith("#"):
            continue
        line = re.sub(r'#.*$', '', line).strip().strip(",")
        dep = line.strip('"').strip("'").strip()
        if dep and not dep.startswith("toolboxv2"):
            deps.append(dep)
    return deps


def _dep_name(dep_str: str) -> str:
    """Extrahiere Package-Name aus dep string."""
    return re.split(r'[><=~!\[;]', dep_str)[0].strip().lower()


def cmd_deps_analyze():
    """Zeige Dependency-Map."""
    console.rule("DEPENDENCY ANALYSIS")
    groups = _parse_pyproject_deps()
    features = discover_features()
    feature_deps: dict[str, list[str]] = {}
    for name, config in features.items():
        deps = config.get("dependencies", [])
        if deps:
            feature_deps[name] = deps
    base_names = {_dep_name(d) for d in groups.get("mini", [])}
    extra_names: dict[str, set[str]] = {}
    for group, deps in groups.items():
        if group == "mini":
            continue
        extra_names[group] = {_dep_name(d) for d in deps}
    _p("\n  ── Base Dependencies ──\n")
    for dep in sorted(groups.get("mini", [])):
        name = _dep_name(dep)
        also_in = [g for g, names in extra_names.items() if name in names]
        suffix = f"  (auch in: {', '.join(also_in)})" if also_in else ""
        _p(f"    {dep}{suffix}")
    _p(f"\n  Base hat {len(groups.get('base', []))} deps\n")
    _p("  ── Feature Dependencies (aus feature.yaml) ──\n")
    for fname, deps in sorted(feature_deps.items()):
        _p(f"    {fname}: {len(deps)} deps")
        for d in deps:
            in_base = "⚠ IN BASE" if _dep_name(d) in base_names else ""
            _p(f"      {d}  {in_base}")
        _p("")
    _p("  ── Empfehlung: base deps = mini feature ──\n")
    mini_deps = feature_deps.get("mini", [])
    if mini_deps:
        _p("    Sollte in base:")
        for d in mini_deps:
            _p(f"      {d}")
    else:
        _p("    (mini feature.yaml hat keine dependencies definiert)")
    bloat = len(groups.get("mini", [])) - len(mini_deps)
    if bloat > 0:
        _p(f"\n    → {bloat} deps könnten aus base entfernt werden")


def cmd_deps_update():
    """Prüfe auf neuere Versionen aller deps."""
    console.rule("DEPENDENCY UPDATE CHECK")
    groups = _parse_pyproject_deps()
    all_deps = set()
    for deps in groups.values():
        for d in deps:
            all_deps.add(_dep_name(d))
    _p(f"  Prüfe {len(all_deps)} packages ...\n")
    for dep_name in sorted(all_deps):
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", dep_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                m = re.search(r"LATEST:\s*(\S+)", proc.stdout)
                latest = m.group(1) if m else "?"
                _p(f"    {dep_name:30} latest: {latest}")
        except Exception:
            _p(f"    {dep_name:30} (check failed)")


def cmd_deps_minimize():
    """Schreibe pyproject.toml base deps auf mini-only um."""
    console.rule("MINIMIZE BASE DEPS")
    features = discover_features()
    mini_config = features.get("mini", {})
    mini_deps = mini_config.get("dependencies", [])
    if not mini_deps:
        _p("  ✗ mini feature.yaml hat keine dependencies", style="red")
        return
    pyproject_path = ROOT / "pyproject.toml"
    text = pyproject_path.read_text(encoding="utf-8")
    new_deps_str = "dependencies = [\n"
    for d in mini_deps:
        new_deps_str += f'    "{d}",\n'
    new_deps_str += "]"
    pattern = r'(^\[project\].*?^)dependencies\s*=\s*\[.*?\]'
    updated = re.sub(pattern, rf'\g<1>{new_deps_str}', text, count=1, flags=re.MULTILINE | re.DOTALL)
    if updated == text:
        _p("  ✗ Konnte dependencies Block nicht finden", style="red")
        return
    backup = pyproject_path.with_suffix(".toml.bak")
    shutil.copy2(pyproject_path, backup)
    _p(f"  ✓ Backup: {backup.name}")
    pyproject_path.write_text(updated, encoding="utf-8")
    _p(f"  ✓ pyproject.toml base deps auf mini reduziert ({len(mini_deps)} deps)")
    _p(f"\n  Neue base deps:")
    for d in mini_deps:
        _p(f"    {d}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="ToolBoxV2 CI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("build", help="Pack features + create wheel/sdist")

    p_test = sub.add_parser("test", help="Build + isolated venv tests per feature")
    p_test.add_argument("--feature", help="Nur dieses Feature testen (z.B. mini, web)")

    p_upload = sub.add_parser("upload", help="Upload to PyPI + Registry")
    p_upload.add_argument("--test", action="store_true", help="Upload to test.pypi.org")
    p_upload.add_argument("--prod", action="store_true", help="Upload to pypi.org")
    p_upload.add_argument("--feature", help="Nur dieses Feature zur Registry (skip PyPI)")

    p_deps = sub.add_parser("deps", help="Dependency management")
    p_deps.add_argument("--analyze", action="store_true", help="Show dependency map")
    p_deps.add_argument("--update", action="store_true", help="Check for newer versions")
    p_deps.add_argument("--minimize", action="store_true", help="Reduce base deps to mini")

    p_all = sub.add_parser("all", help="Full pipeline: build → test → report → upload")
    p_all.add_argument("--test", action="store_true", help="Upload to test.pypi.org")
    p_all.add_argument("--prod", action="store_true", help="Upload to pypi.org")
    p_all.add_argument("--skip-tests", action="store_true", help="Skip test phase")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    console.rule(f"ToolBoxV2 CI — {args.command}")

    if args.command == "build":
        cmd_build()


    elif args.command == "test":
        # Fix #1: exit with code 1 on failures
        results = cmd_test(only_feature=args.feature)
        cmd_report([], results)
        if any(not r.get("passed", False) for r in results if r):
            sys.exit(1)

    elif args.command == "upload":
        if args.feature:
            _upload_features_to_registry(only_feature=args.feature)
        elif args.prod:
            cmd_upload(production=True)
        elif args.test:
            cmd_upload(production=False)
        else:
            _p("  Spezifiziere --test, --prod oder --feature", style="red")

    elif args.command == "deps":
        if args.analyze:
            cmd_deps_analyze()
        elif args.update:
            cmd_deps_update()
        elif args.minimize:
            cmd_deps_minimize()
        else:
            cmd_deps_analyze()

    elif args.command == "all":
        build_results = cmd_build()
        test_results = []
        if not args.skip_tests:
            test_results = cmd_test(do_build=False)
        report_path = cmd_report(build_results, test_results)
        # Fix #1: exit with code 1 on failures
        has_test_fail = any(not r.get("passed", False) for r in test_results if r)
        has_build_fail = any(r.get("action") == "failed" for r in build_results)
        if args.prod:
            cmd_upload(production=True)
        elif args.test:
            cmd_upload(production=False)
        _p(f"\n  Report: {report_path}")
        if has_test_fail or has_build_fail:
            sys.exit(1)

    console.rule("Done")


if __name__ == "__main__":
    main()
