#!/usr/bin/env python3
"""
ToolBoxV2 CI Pipeline — Build · Test · Report · Upload

Usage:
    python ci_build.py build                    # Pack features → wheel/sdist
    python ci_build.py test                     # Build + venv-isolated tests per feature
    python ci_build.py upload --test            # Upload to test.pypi.org
    python ci_build.py upload --prod            # Upload to pypi.org
    python ci_build.py deps --analyze           # Show dependency map
    python ci_build.py deps --update            # Check for newer versions
    python ci_build.py deps --minimize          # Rewrite pyproject.toml base deps to mini-only
    python ci_build.py all --test               # Full pipeline → test.pypi
    python ci_build.py all --prod               # Full pipeline → pypi
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
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Pfade ────────────────────────────────────────────────────────────────────

from toolboxv2 import tb_root_dir

ROOT = tb_root_dir.parent
TOOLBOXV2_DIR = ROOT / "toolboxv2"
FEATURES_DIR = TOOLBOXV2_DIR / "features"
PACKED_DIR = TOOLBOXV2_DIR / "features_packed"
DIST_DIR = ROOT / "dist"
REPORT_DIR = ROOT / "build_reports"

ALWAYS_SOURCE = {"core", "mini"}  # nie geZIPt — immer im Source-Tree

# ── Rich / Fallback ─────────────────────────────────────────────────────────

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

    # Fallback: regex-basiert für die einfachen feature.yaml Felder
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = {}

    # Einfache key: value
    for m in re.finditer(r'^(\w+):\s*"?([^"\n]+)"?', text, re.MULTILINE):
        key, val = m.group(1), m.group(2).strip().strip('"')
        if val.lower() == "true":
            data[key] = True
        elif val.lower() == "false":
            data[key] = False
        else:
            data[key] = val

    # Listen (- "item")
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
        # feature.yaml
        src_yaml = FEATURES_DIR / name / "feature.yaml"
        if src_yaml.exists():
            zf.write(src_yaml, "feature.yaml")

        # requirements.txt
        if dependencies:
            zf.writestr("requirements.txt", "\n".join(dependencies))

        # Source files
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

        # Metadata
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

        zip_path = pack_feature(name, config)
        if zip_path and zip_path.exists():
            size_kb = zip_path.stat().st_size // 1024
            _p(f"  ✓ {name:12} → {zip_path.name} ({size_kb} KB)")
            results.append({"feature": name, "action": "packed", "size": size_kb})
        else:
            _p(f"  ✗ {name:12} FAILED", style="red")
            results.append({"feature": name, "action": "failed", "size": 0})

    # MANIFEST.in
    create_manifest_in()
    _p("  ✓ MANIFEST.in")

    # Build wheel + sdist
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
        _p(f"  ✗ Build failed: {e.stderr[-500:]}", style="red")
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
    return wheels[-1] if wheels else None


def _run_feature_test(
    feature_name: str,
    extras: list[str],
    wheel: Path,
    timeout: int = 500,
) -> dict:
    """
    Teste ein Feature in isoliertem venv.

    Returns dict mit: feature, passed, duration, output, errors
    """
    result = {
        "feature": feature_name,
        "extras": extras,
        "passed": False,
        "tests_run": 0,
        "failures": 0,
        "errors": 0,
        "duration": 0.0,
        "output": "",
        "install_ok": False,
    }

    venv_dir = None
    try:
        venv_dir = Path(tempfile.mkdtemp(prefix=f"tb_test_{feature_name}_"))
        venv_path = venv_dir / "venv"

        # 1. venv erstellen
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            timeout=60,
        )

        # Python im venv
        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        # 2. Wheel installieren mit extras
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
            return result

        # 2.5 pytest + plugins installieren
        proc = subprocess.run(
            [
                str(venv_python), "-m", "pip", "install",
                "pytest>=9.0.2",
                "pytest-asyncio>=0.23.0",
                "pytest-xdist>=3.5.0",  # Parallel execution
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            result["output"] = f"PYTEST INSTALL FAILED:\n{proc.stderr[-1000:]}"
            return result

        result["install_ok"] = True

        # 3. Tests ausführen
        test_dir = ROOT / "toolboxv2" / "tests"
        feature_test_dir = test_dir / feature_name

        # Suche nach Tests: tests/<feature>/ oder tests/test_<feature>.py
        test_targets = []
        if feature_test_dir.exists():
            test_targets.append(str(feature_test_dir))
        test_file = test_dir / f"test_{feature_name}"
        if test_file.exists():
            test_targets.append(str(test_file))

        if not test_targets and test_dir.exists():
            # Fallback: alle Tests aber mit installiertem Feature
            test_targets.append(str(test_dir))

        if not test_targets:
            result["output"] = "No tests found"
            result["passed"] = True  # Kein Test = kein Failure
            return result
        print(test_targets[0])
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                str(venv_python), "-m", "pytest",
                test_targets[0],
                "-v",
                "--tb=short",
                "-n", "auto",  # Parallel execution
                "--asyncio-mode=auto",  # Auto async support
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
        result["duration"] = round(time.monotonic() - t0, 2)
        result["output"] = proc.stdout + proc.stderr

        # Parse pytest output
        combined = proc.stdout + proc.stderr

        # Tests run: "X passed" or "X failed, Y passed"
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

    except subprocess.TimeoutExpired:
        result["output"] = f"TIMEOUT after {timeout}s"
    except Exception as e:
        result["output"] = f"ERROR: {e}"
    finally:
        print(venv_dir)
        #if venv_dir and venv_dir.exists():
        #    shutil.rmtree(venv_dir, ignore_errors=True)

    return result


def cmd_test() -> list[dict]:
    """Teste jedes Feature isoliert."""
    console.rule("TEST")

    # Erst bauen falls nötig
    wheel = _find_wheel()
    if not wheel:
        _p("  Kein wheel gefunden — baue zuerst ...")
        cmd_build()
        wheel = _find_wheel()
    if not wheel:
        _p("  ✗ Build fehlgeschlagen", style="red")
        return []

    _p(f"  Wheel: {wheel.name}\n")

    features = discover_features()

    # Test-Matrix: base install + jedes extra einzeln
    test_matrix = [
        {"name": "mini", "extras": []},
    ]
    for name in features:
        if name in ALWAYS_SOURCE:
            continue
        test_matrix.append({"name": name, "extras": [name]})

    def _test_runner(entry, res, i):
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
        if not 'feature' in r:
            r["feature"] = name
        res[i] = r
    import threading

    results = [{}] * len(test_matrix)
    ts = []
    for i, _entry in enumerate(test_matrix):
        t = threading.Thread(target=_test_runner, args=(_entry,results, i))
        t.start()
        ts.append(t)

    for t in ts:
        t.join()
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════════════


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

    # ── HTML generieren ──────────────────────────────────────────────────

    def _status_badge(ok: bool) -> str:
        color = "#22c55e" if ok else "#ef4444"
        label = "PASS" if ok else "FAIL"
        return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{label}</span>'

    build_rows = ""
    for r in build_results:
        action = r.get("action", "?")
        size = f'{r.get("size", 0)} KB' if r.get("size") else "—"
        color = "#22c55e" if action != "failed" else "#ef4444"
        build_rows += f"""
        <tr>
            <td>{html.escape(r['feature'])}</td>
            <td style="color:{color}">{action}</td>
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
        output_escaped = html.escape(r.get("output", "")[:2000])
        test_rows += f"""
        <tr>
            <td>{html.escape(r['feature'])}</td>
            <td>[{html.escape(extras)}]</td>
            <td>{_status_badge(passed)}</td>
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

    # JSON auch ablegen
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


# ═════════════════════════════════════════════════════════════════════════════
#  DEPS
# ═════════════════════════════════════════════════════════════════════════════


def _parse_pyproject_deps() -> dict[str, list[str]]:
    """Parse alle dependency-Gruppen aus pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    groups: dict[str, list[str]] = {}

    # Base dependencies
    m = re.search(
        r'^\[project\].*?^dependencies\s*=\s*\[(.*?)\]',
        text,
        re.MULTILINE | re.DOTALL,
    )
    if m:
        groups["mini"] = _extract_dep_list(m.group(1))

    # Optional dependencies
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
    # Finde nächste [section]
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
        # Entferne Kommentare am Ende
        line = re.sub(r'#.*$', '', line).strip().strip(",")
        # Entferne Anführungszeichen
        dep = line.strip('"').strip("'").strip()
        if dep and not dep.startswith("toolboxv2"):
            deps.append(dep)
    return deps


def _dep_name(dep_str: str) -> str:
    """Extrahiere Package-Name aus dep string (z.B. 'pydantic>=2.0' → 'pydantic')."""
    return re.split(r'[><=~!\[;]', dep_str)[0].strip().lower()


def cmd_deps_analyze():
    """Zeige Dependency-Map: welches Package in welcher Gruppe."""
    console.rule("DEPENDENCY ANALYSIS")

    groups = _parse_pyproject_deps()
    features = discover_features()

    # Sammle alle feature.yaml deps
    feature_deps: dict[str, list[str]] = {}
    for name, config in features.items():
        deps = config.get("dependencies", [])
        if deps:
            feature_deps[name] = deps

    # Analyse: base deps die in Features dupliziert sind
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

    # Mini-Feature deps = was wirklich in base sein sollte
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

    # Ersetze dependencies = [...] im [project] Block
    new_deps_str = "dependencies = [\n"
    for d in mini_deps:
        new_deps_str += f'    "{d}",\n'
    new_deps_str += "]"

    # Finde und ersetze den dependencies Block
    pattern = r'(^\[project\].*?^)dependencies\s*=\s*\[.*?\]'
    updated = re.sub(
        pattern,
        rf'\g<1>{new_deps_str}',
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )

    if updated == text:
        _p("  ✗ Konnte dependencies Block nicht finden", style="red")
        return

    # Backup
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
    sub.add_parser("test", help="Build + isolated venv tests per feature")

    p_upload = sub.add_parser("upload", help="Upload to PyPI")
    p_upload.add_argument("--test", action="store_true", help="Upload to test.pypi.org")
    p_upload.add_argument("--prod", action="store_true", help="Upload to pypi.org")

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
        results = cmd_test()
        cmd_report([], results)

    elif args.command == "upload":
        if args.prod:
            cmd_upload(production=True)
        elif args.test:
            cmd_upload(production=False)
        else:
            _p("  Spezifiziere --test oder --prod", style="red")

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
            test_results = cmd_test()
        report_path = cmd_report(build_results, test_results)
        if args.prod:
            cmd_upload(production=True)
        elif args.test:
            cmd_upload(production=False)
        _p(f"\n  Report: {report_path}")

    console.rule("Done")


if __name__ == "__main__":
    main()
