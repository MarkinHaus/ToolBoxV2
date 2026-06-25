#!/usr/bin/env python3
"""
publish_tb_atomic.py — build and upload tb_atomic-PACKED distributions.

Operates on the output directory of `tb_atomic pack` (which contains an optional
`tbv-core/` shared base plus one or more `<leaf>/` packages). Each package is a
self-contained, standalone-importable wheel.

Targets (combinable):
    --testpypi   upload wheel+sdist to TestPyPI         (default if none given)
    --pypi       upload wheel+sdist to production PyPI   (alias: --prod)
    --registry   upload a zipped package to the TB Registry

Ordering: `tbv-core` is always built/uploaded FIRST (leaves depend on it).

Auth:
    PyPI / TestPyPI : export TWINE_PASSWORD=<token>     (username forced __token__)
    TB Registry     : export TB_REGISTRY_TOKEN=<jwt>
                      export REGISTRY_BASE_URL=...       (optional override)

Usage:
    python publish_tb_atomic.py <packed_dir>                 # -> TestPyPI
    python publish_tb_atomic.py <packed_dir> --pypi --registry
    python publish_tb_atomic.py <packed_dir> --check         # build + twine check
    python publish_tb_atomic.py <packed_dir> --build-only
    python publish_tb_atomic.py <packed_dir> --packages tbv-core,tbv-fast-tb
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


# ── package discovery ────────────────────────────────────────────────────────


def discover_packages(packed_dir: Path, only: set[str] | None) -> list[Path]:
    """Sub-dirs containing a pyproject.toml, tbv-core first."""
    pkgs = [d for d in packed_dir.iterdir()
            if d.is_dir() and (d / "pyproject.toml").exists()]
    if only:
        pkgs = [d for d in pkgs if d.name in only]
    pkgs.sort(key=lambda d: (d.name != "tbv-core", d.name))  # core first
    return pkgs


def read_meta(pkg_dir: Path) -> dict:
    data = tomllib.loads((pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))
    proj = data.get("project", {})
    return {
        "name": proj.get("name", pkg_dir.name),
        "version": proj.get("version", "0.0.0"),
        "description": proj.get("description", ""),
    }


# ── build / twine (PyPI) ─────────────────────────────────────────────────────


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _clean(pkg_dir: Path) -> None:
    for d in (pkg_dir / "dist", pkg_dir / "build"):
        if d.exists():
            shutil.rmtree(d)
    for egg in pkg_dir.glob("*.egg-info"):
        shutil.rmtree(egg)


def build(pkg_dir: Path) -> list[Path]:
    """Build wheel + sdist; return artifact paths."""
    _run([sys.executable, "-m", "build", "--outdir", str(pkg_dir / "dist"),
          str(pkg_dir)], cwd=pkg_dir)
    arts = [p for p in (pkg_dir / "dist").glob("*") if p.suffix in (".whl", ".gz")]
    if not arts:
        raise RuntimeError(f"no build artifacts for {pkg_dir.name}")
    return arts


def twine_check(arts: list[Path], cwd: Path) -> None:
    _run([sys.executable, "-m", "twine", "check", *map(str, arts)], cwd=cwd)


def twine_upload(arts: list[Path], cwd: Path, prod: bool) -> None:
    if "TWINE_PASSWORD" not in os.environ:
        raise SystemExit("error: set TWINE_PASSWORD to your PyPI token.")
    env = {**os.environ, "TWINE_USERNAME": "__token__"}
    cmd = [sys.executable, "-m", "twine", "upload"]
    if not prod:
        cmd += ["--repository-url", "https://test.pypi.org/legacy/"]
    _run([*cmd, *map(str, arts)], cwd=cwd, env=env)


# ── TB Registry ──────────────────────────────────────────────────────────────


def zip_package(pkg_dir: Path) -> Path:
    """Zip the package source (tbv/** + pyproject.toml) for registry upload."""
    out = pkg_dir / f"{pkg_dir.name}.zip"
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in pkg_dir.rglob("*"):
            if f.is_file() and "dist" not in f.parts and "build" not in f.parts \
                    and f != out:
                zf.write(f, f.relative_to(pkg_dir))
    return out


async def _registry_publish(pkgs: list[Path]) -> None:
    try:
        from toolboxv2.utils.extras.registry_client import RegistryClient
    except ImportError as e:
        raise SystemExit(
            "error: RegistryClient unavailable — run from a ToolBoxV2 checkout "
            f"with toolboxv2 importable. ({e})")

    token = os.environ.get("TB_REGISTRY_TOKEN")
    if not token:
        raise SystemExit("error: set TB_REGISTRY_TOKEN to your registry JWT.")
    url = os.environ.get("REGISTRY_BASE_URL", "https://registry.simplecore.app")

    async with RegistryClient(registry_url=url, auth_token=token) as client:
        user = await client.get_current_user()
        if not user or not user.is_verified:
            raise SystemExit("error: registry token is not a verified publisher.")
        for pkg_dir in pkgs:
            meta = read_meta(pkg_dir)
            name, version = meta["name"], meta["version"]
            # create package (idempotent: ignore "already exists")
            try:
                await client.create_package(
                    name=name, display_name=name, package_type="library",
                    description=meta["description"], readme="")
                print(f"  registry: created package {name}")
            except Exception as e:  # RegistryError on 409 -> already exists
                if "exist" not in str(e).lower():
                    raise
                print(f"  registry: package {name} exists, adding version")
            zpath = zip_package(pkg_dir)
            ok = await client.upload_version(
                name=name, version=version, file_path=zpath,
                changelog=f"tb_atomic packed {name} {version}")
            print(f"  registry: {'uploaded' if ok else 'FAILED'} {name}@{version}")
            if not ok:
                raise SystemExit(f"error: registry upload failed for {name}")


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build & upload tb_atomic-packed dists.")
    p.add_argument("packed_dir", help="Output dir from `tb_atomic pack`.")
    p.add_argument("--pypi", "--prod", dest="pypi", action="store_true",
                   help="Upload to production PyPI.")
    p.add_argument("--testpypi", action="store_true", help="Upload to TestPyPI.")
    p.add_argument("--registry", action="store_true", help="Upload to TB Registry.")
    p.add_argument("--build-only", action="store_true", help="Build only, no upload.")
    p.add_argument("--check", action="store_true", help="Build + twine check only.")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--packages", default=None,
                   help="Comma-separated subset of package dir names.")
    args = p.parse_args(argv)

    packed = Path(args.packed_dir).resolve()
    if not packed.is_dir():
        return _err(f"not a directory: {packed}")

    only = set(args.packages.split(",")) if args.packages else None
    pkgs = discover_packages(packed, only)
    if not pkgs:
        return _err(f"no packages (pyproject.toml) found under {packed}")

    print(f"Packages (core-first): {[d.name for d in pkgs]}")

    # default target = TestPyPI when nothing chosen and not a dry mode
    to_pypi = args.pypi or args.testpypi or not (args.registry or args.build_only or args.check)

    # ---- build + check every package (PyPI artifacts) ----
    built: dict[Path, list[Path]] = {}
    for d in pkgs:
        if not args.no_clean:
            _clean(d)
        arts = build(d)
        twine_check(arts, d)
        built[d] = arts

    if args.build_only or args.check:
        print(f"\nDone (no upload). Artifacts under each <pkg>/dist/.")
        return 0

    # ---- upload, core first ----
    if to_pypi:
        for d in pkgs:
            print(f"\n[PyPI] {d.name} -> {'PyPI' if args.pypi else 'TestPyPI'}")
            twine_upload(built[d], d, prod=args.pypi)

    if args.registry:
        print("\n[Registry] uploading (core first) ...")
        asyncio.run(_registry_publish(pkgs))

    print("\nPublish complete.")
    return 0


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
