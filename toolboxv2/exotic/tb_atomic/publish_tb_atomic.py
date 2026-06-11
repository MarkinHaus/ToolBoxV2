#!/usr/bin/env python3
"""
publish_tb_atomic.py — build and upload the tb_atomic package.

Builds wheel + sdist from the tb_atomic directory only, then uploads via twine
to TestPyPI (default) or production PyPI (--prod).

Auth: set a token in the environment before running.
    TestPyPI : export TWINE_PASSWORD=<testpypi-token>
    PyPI     : export TWINE_PASSWORD=<pypi-token>
    (username is forced to __token__)

Usage:
    python publish_tb_atomic.py                 # -> TestPyPI
    python publish_tb_atomic.py --prod          # -> PyPI
    python publish_tb_atomic.py --build-only     # build, no upload
    python publish_tb_atomic.py --check          # build + twine check, no upload
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent  # tb_atomic package dir (this script lives here)
DIST_DIR = PKG_DIR / "dist"


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PKG_DIR, env=env)


def _clean() -> None:
    for d in (DIST_DIR, PKG_DIR / "build"):
        if d.exists():
            shutil.rmtree(d)
    for egg in PKG_DIR.glob("*.egg-info"):
        shutil.rmtree(egg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build & upload tb_atomic.")
    p.add_argument("--prod", action="store_true", help="Upload to production PyPI (default: TestPyPI).")
    p.add_argument("--build-only", action="store_true", help="Build artifacts, do not upload.")
    p.add_argument("--check", action="store_true", help="Build and run twine check, do not upload.")
    p.add_argument("--no-clean", action="store_true", help="Keep existing dist/ before building.")
    args = p.parse_args(argv)

    if not (PKG_DIR / "pyproject.toml").exists():
        print(f"error: no pyproject.toml in {PKG_DIR}", file=sys.stderr)
        return 1

    if not args.no_clean:
        _clean()

    # Build wheel + sdist from this package only.
    _run([sys.executable, "-m", "build", "--outdir", str(DIST_DIR), str(PKG_DIR)])

    artifacts = [str(p) for p in DIST_DIR.glob("*") if p.suffix in (".whl", ".gz")]
    if not artifacts:
        print("error: no build artifacts produced", file=sys.stderr)
        return 1

    _run([sys.executable, "-m", "twine", "check", *artifacts])

    if args.build_only or args.check:
        print(f"\nDone (no upload). Artifacts in {DIST_DIR}")
        return 0

    repo = "pypi" if args.prod else "testpypi"
    if "TWINE_PASSWORD" not in os.environ:
        print("error: set TWINE_PASSWORD to your API token before uploading.", file=sys.stderr)
        return 1

    env = {**os.environ, "TWINE_USERNAME": "__token__"}
    target = "PRODUCTION PyPI" if args.prod else "TestPyPI"
    print(f"\nUploading to {target} ...")
    _run([sys.executable, "-m", "twine", "upload", "--repository", repo, *artifacts], env=env)
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
