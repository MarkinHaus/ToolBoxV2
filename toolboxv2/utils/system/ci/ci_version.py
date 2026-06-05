#!/usr/bin/env python3
"""
ToolBoxV2 Version Bumper

Aktualisiert die Version in allen relevanten Dateien:
  - pyproject.toml
  - feature.yaml (alle Features)
  - __init__.py (__version__)

Usage:
    tb version show              # Zeige aktuelle Version
    tb version patch             # 0.1.25 → 0.1.26
    tb version minor             # 0.1.25 → 0.2.0
    tb version major             # 0.1.25 → 1.0.0
    tb version set 0.2.0         # Explizite Version
    tb version patch --tag       # Bump + git tag
"""
import re
import subprocess
import sys
from pathlib import Path
from toolboxv2 import tb_root_dir

ROOT = tb_root_dir.parent
TOOLBOXV2 = ROOT / "toolboxv2"
PYPROJECT = ROOT / "pyproject.toml"
FEATURES_DIR = TOOLBOXV2 / "features"
INIT_PY = TOOLBOXV2 / "__init__.py"

# ── Version lesen / schreiben ────────────────────────────────────────────────


def read_version() -> str:
    """Lese Version aus pyproject.toml (source of truth)."""
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        print("ERROR: Keine version in pyproject.toml gefunden", file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def bump(current: str, part: str) -> str:
    """Berechne neue Version."""
    parts = current.split(".")
    if len(parts) != 3:
        print(f"ERROR: Version '{current}' ist nicht semver (X.Y.Z)", file=sys.stderr)
        sys.exit(1)

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if part == "patch":
        patch += 1
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        print(f"ERROR: Unbekannter bump type '{part}'", file=sys.stderr)
        sys.exit(1)

    return f"{major}.{minor}.{patch}"


# ── Dateien aktualisieren ────────────────────────────────────────────────────


def update_pyproject(old: str, new: str) -> bool:
    if not PYPROJECT.exists():
        return False
    text = PYPROJECT.read_text(encoding="utf-8")
    updated = re.sub(
        r'^(version\s*=\s*)"[^"]+"',
        rf'\g<1>"{new}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if updated == text:
        return False
    PYPROJECT.write_text(updated, encoding="utf-8")
    return True


def update_feature_yamls(old: str, new: str) -> int:
    """Update version in allen feature.yaml Dateien. Returns Anzahl."""
    count = 0
    if not FEATURES_DIR.exists():
        return 0

    for feature_dir in sorted(FEATURES_DIR.iterdir()):
        yaml_path = feature_dir / "feature.yaml"
        if not yaml_path.exists():
            continue

        text = yaml_path.read_text(encoding="utf-8")
        updated = re.sub(
            r'^(version:\s*)"?[^"\n]+"?',
            rf'\g<1>"{new}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if updated != text:
            yaml_path.write_text(updated, encoding="utf-8")
            count += 1

    return count


def update_init_py(old: str, new: str) -> bool:
    if not INIT_PY.exists():
        return False
    text = INIT_PY.read_text(encoding="utf-8")
    updated = re.sub(
        r'^(VERSION\s*=\s*)["\'][^"\']+["\']',
        rf"\g<1>'{new}'",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if updated == text:
        return False
    INIT_PY.write_text(updated, encoding="utf-8")
    return True


def git_tag(version: str):
    """Erstelle git commit + tag."""
    try:
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"bump: v{version}"],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"],
            cwd=ROOT,
            check=True,
        )
        print(f"  git commit + tag v{version}")
    except subprocess.CalledProcessError as e:
        print(f"  WARN: git failed: {e}", file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    current = read_version()

    if cmd == "show":
        print(current)
        return

    # Neue Version bestimmen
    if cmd == "set":
        if len(sys.argv) < 3:
            print("ERROR: tb version set <VERSION>", file=sys.stderr)
            sys.exit(1)
        new_version = sys.argv[2].lstrip("v")
    elif cmd in ("patch", "minor", "major"):
        new_version = bump(current, cmd)
    else:
        print(f"ERROR: Unbekannter Befehl '{cmd}'", file=sys.stderr)
        print(__doc__)
        sys.exit(1)

    do_tag = "--tag" in sys.argv

    print(f"\n  {current} → {new_version}\n")

    # Dateien updaten
    if update_pyproject(current, new_version):
        print(f"  ✓ pyproject.toml")

    n = update_feature_yamls(current, new_version)
    if n:
        print(f"  ✓ {n} feature.yaml files")

    if update_init_py(current, new_version):
        print(f"  ✓ __init__.py")

    if do_tag:
        git_tag(new_version)

    print(f"\n  Done: v{new_version}\n")


if __name__ == "__main__":
    main()
