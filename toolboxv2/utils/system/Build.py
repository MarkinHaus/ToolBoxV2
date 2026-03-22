"""
toolboxv2/mods/Build.py
=======================
Lokale Feature-Build-Pipeline als TB-Mod.

Verwendung:
    tb build <feature> [Optionen]

Pipeline-Steps:
    1. feature.yaml  → version bump (nur Feature, nicht Haupt-App)
    2. Tests         → tests/test_features/test_{name}.py (unittest)
    3. Pack          → features_sto/tbv2-feature-{name}-{version}.zip
    4. Deploy ZIP    → toolboxv2/features_packed/  (wird ins Wheel eingebettet)
    5. PyPI          → pyproject.toml patch-bump → build → twine upload
    6. TB Registry   → upload_feature_to_registry()

Flags:
    --bump-main      Haupt-Version in pyproject.toml ebenfalls bumpen (patch)
    --no-test        Tests überspringen
    --no-registry    TB-Registry-Upload überspringen
    --no-pip         PyPI-Upload überspringen
    --dry-run        Simulieren, nichts schreiben/hochladen
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Optional

# ── TB-Mod Metadaten ──────────────────────────────────────────────────────────
Name = "build"
version = "0.1.1"
# ─────────────────────────────────────────────────────────────────────────────

_PYPROJECT = Path("pyproject.toml")


# =============================================================================
# Versions-Helfer
# =============================================================================

def _bump_semver(current: str, bump: str) -> str:
    """Bump semver-String. bump ∈ {patch, minor, major, none}."""
    if bump == "none":
        return current
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(.*)", current.strip())
    if not m:
        raise ValueError(f"Kein semver: {current!r}")
    ma, mi, pa, rest = int(m[1]), int(m[2]), int(m[3]), m[4]
    if bump == "major":
        return f"{ma + 1}.0.0{rest}"
    if bump == "minor":
        return f"{ma}.{mi + 1}.0{rest}"
    return f"{ma}.{mi}.{pa + 1}{rest}"


# ── Feature-Version ───────────────────────────────────────────────────────────

def _feature_yaml(feature_name: str) -> Path:
    return Path(__file__).parent.parent / "features" / feature_name / "feature.yaml"


def _read_feature_version(feature_name: str) -> str:
    import yaml
    p = _feature_yaml(feature_name)
    if not p.exists():
        raise FileNotFoundError(f"feature.yaml nicht gefunden: {p}")
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("version", "0.0.0")


def _write_feature_version(feature_name: str, new_version: str) -> None:
    import yaml
    p = _feature_yaml(feature_name)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data["version"] = new_version
    p.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


# ── Haupt-Version (pyproject.toml) ────────────────────────────────────────────

def _read_main_version() -> str:
    if not _PYPROJECT.exists():
        return "0.0.0"
    m = re.search(r'^version\s*=\s*"([^"]+)"', _PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _write_main_version(new_version: str) -> None:
    text = _PYPROJECT.read_text(encoding="utf-8")
    updated = re.sub(
        r'^(version\s*=\s*")[^"]+(")',
        lambda mo: f'{mo.group(1)}{new_version}{mo.group(2)}',
        text, count=1, flags=re.MULTILINE,
    )
    _PYPROJECT.write_text(updated, encoding="utf-8")


# =============================================================================
# Pipeline-Steps
# =============================================================================

def _run_tests(feature_name: str) -> bool:
    test_file = Path("tests") / "test_features" / f"test_{feature_name}.py"
    if not test_file.exists():
        print(f"  ⚠  Keine Tests: {test_file} – übersprungen.")
        return True
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_file.parent), pattern=f"test_{feature_name}.py")
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def _pack(feature_name: str, dry_run: bool) -> Optional[str]:
    if dry_run:
        print(f"  [dry-run] pack_feature({feature_name!r})")
        return f"features_sto/tbv2-feature-{feature_name}-DRY.zip"
    from toolboxv2.utils.system.feature_manager import FeatureManager
    fm = FeatureManager(features_dir=str(_feature_yaml(feature_name).parent.parent))
    return fm.pack_feature(feature_name)


def _deploy_to_features_packed(zip_path: str, dry_run: bool) -> None:
    """Kopiere ZIP nach toolboxv2/features_packed/ damit es ins Wheel kommt."""
    dest_dir = Path(__file__).parent.parent / "features_packed"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(zip_path).name
    if not dry_run:
        shutil.copy2(zip_path, dest)
    print(f"  {'[dry-run] ' if dry_run else ''}✓ features_packed/{dest.name}")


def _pip_build_and_upload(main_old: str, main_new: str, dry_run: bool) -> bool:
    """
    pyproject.toml patch-bump → python -m build → twine upload
    Nutzt PYPI_TOKEN env-var als __token__.
    """
    token = os.environ.get("PYPI_TOKEN")
    if not token:
        print("  ✗ PYPI_TOKEN nicht gesetzt – PyPI-Upload übersprungen.")
        return False

    print(f"  Haupt-Version: {main_old} → {main_new}")

    if not dry_run:
        _write_main_version(main_new)
        # Altes dist/ wegräumen
        dist_dir = Path("dist")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)

    if dry_run:
        print("  [dry-run] python -m build --wheel --sdist")
        print("  [dry-run] twine upload dist/*")
        return True

    print("  → python -m build …")
    r = subprocess.run([sys.executable, "-m", "build", "--wheel", "--sdist"])
    if r.returncode != 0:
        print("  ✗ Build fehlgeschlagen.")
        return False

    print("  → twine upload …")
    r = subprocess.run(
        [sys.executable, "-m", "twine", "upload", "--non-interactive", "dist/*"],
        env={**os.environ, "TWINE_PASSWORD": token, "TWINE_USERNAME": "__token__"},
    )
    return r.returncode == 0


def _upload_registry(feature_name: str, feat_version: str, zip_path: str, dry_run: bool) -> bool:
    if dry_run:
        print(f"  [dry-run] registry upload {feature_name} v{feat_version}")
        return True
    from toolboxv2.feature_loader_registry import upload_feature_to_registry
    return upload_feature_to_registry(
        feature_name=feature_name,
        version=feat_version,
        zip_path=Path(zip_path),
    )


# =============================================================================
# Haupt-Pipeline
# =============================================================================

def run_pipeline(
    feature_name: str,
    bump: str = "patch",
    bump_main: bool = False,
    run_tests: bool = True,
    pip_upload: bool = True,
    registry_upload: bool = True,
    dry_run: bool = False,
) -> int:

    sep = "─" * 54
    print(f"\n{sep}\n  tb build · {feature_name}\n{sep}\n")

    # ── 1. Feature-Version ────────────────────────────────────────────────
    try:
        feat_old = _read_feature_version(feature_name)
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return 1

    feat_new = _bump_semver(feat_old, bump)
    label = f"{feat_old} → {feat_new}" if bump != "none" else f"{feat_old} (kein bump)"
    print(f"  [1/6] Feature-Version: {label}")
    if bump != "none" and not dry_run:
        _write_feature_version(feature_name, feat_new)

    # ── 2. Tests ──────────────────────────────────────────────────────────
    if run_tests:
        print(f"\n  [2/6] Tests …")
        if not _run_tests(feature_name):
            print("\n✗ Tests fehlgeschlagen – Pipeline abgebrochen.")
            if bump != "none" and not dry_run:
                _write_feature_version(feature_name, feat_old)
                print(f"  ↩  Feature-Version zurück auf {feat_old}")
            return 1
        print("  ✓ Tests OK")
    else:
        print("  [2/6] Tests übersprungen (--no-test)")

    # ── 3. Pack ───────────────────────────────────────────────────────────
    print(f"\n  [3/6] Packen …")
    zip_path = _pack(feature_name, dry_run)
    if not zip_path:
        print("✗ Pack fehlgeschlagen.")
        return 1
    print(f"  ✓ {zip_path}")

    # ── 4. → features_packed/ ─────────────────────────────────────────────
    print(f"\n  [4/6] → features_packed/ …")
    _deploy_to_features_packed(zip_path, dry_run)

    # ── 5. PyPI ───────────────────────────────────────────────────────────
    if pip_upload:
        print(f"\n  [5/6] PyPI-Upload …")
        main_old = _read_main_version()
        # Haupt-Version bekommt IMMER einen mini patch-bump,
        # damit PyPI ein neues Release akzeptiert.
        # --bump-main ist das explizite Flag dafür (andernfalls wird trotzdem gebumpt).
        main_new = _bump_semver(main_old, "patch")

        if not _pip_build_and_upload(main_old, main_new, dry_run):
            print("✗ PyPI-Upload fehlgeschlagen.")
            if not dry_run:
                _write_main_version(main_old)
                print(f"  ↩  Haupt-Version zurück auf {main_old}")
            return 1
        print(f"  ✓ PyPI: toolboxv2 v{main_new}")
    else:
        print("  [5/6] PyPI übersprungen (--no-pip)")

    # ── 6. TB Registry ────────────────────────────────────────────────────
    if registry_upload:
        print(f"\n  [6/6] TB Registry …")
        if not _upload_registry(feature_name, feat_new, zip_path, dry_run):
            print("✗ Registry-Upload fehlgeschlagen.")
            return 1
        print(f"  ✓ Registry: {feature_name} v{feat_new}")
    else:
        print("  [6/6] Registry übersprungen (--no-registry)")

    print(f"\n{sep}\n  ✓ {feature_name} v{feat_new}  —  fertig.\n{sep}\n")
    return 0


# =============================================================================
# CLI
# =============================================================================

def build_parser(subparsers=None) -> argparse.ArgumentParser:
    kwargs = dict(
        description="Feature Build-Pipeline: bump → test → pack → pip → registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Beispiele:\n"
            "  tb build web                   # patch + test + pack + pip + registry\n"
            "  tb build isaa --bump minor      # minor feature-bump\n"
            "  tb build web --no-pip          # kein PyPI-Upload\n"
            "  tb build web --no-registry     # kein TB-Registry-Upload\n"
            "  tb build web --dry-run         # alles simulieren\n"
            "  tb build web --bump none       # kein version-bump\n"
        ),
    )
    p = (subparsers.add_parser("build", **kwargs)
         if subparsers else argparse.ArgumentParser(**kwargs))

    p.add_argument("feature", help="Feature-Name (z.B. web, isaa, cli)")
    p.add_argument(
        "--bump",
        choices=["patch", "minor", "major", "none"],
        default="patch",
        help="Feature-Version-Bump (default: patch)",
    )
    p.add_argument("--bump-main",   action="store_true", help="(reserviert, Haupt-bump ist immer patch)")
    p.add_argument("--no-test",     action="store_true", help="Tests überspringen")
    p.add_argument("--no-pip",      action="store_true", help="PyPI-Upload überspringen")
    p.add_argument("--no-registry", action="store_true", help="TB-Registry-Upload überspringen")
    p.add_argument("--dry-run",     action="store_true", help="Simulieren, nichts schreiben")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run_pipeline(
        feature_name=args.feature,
        bump=args.bump,
        bump_main=getattr(args, "bump_main", False),
        run_tests=not args.no_test,
        pip_upload=not args.no_pip,
        registry_upload=not args.no_registry,
        dry_run=args.dry_run,
    )


def run(app=None, args=None):
    """TB-Mod Entry Point."""
    if args is None:
        return main()
    if hasattr(args, "feature"):
        return run_pipeline(
            feature_name=args.feature,
            bump=getattr(args, "bump", "patch"),
            bump_main=getattr(args, "bump_main", False),
            run_tests=not getattr(args, "no_test", False),
            pip_upload=not getattr(args, "no_pip", False),
            registry_upload=not getattr(args, "no_registry", False),
            dry_run=getattr(args, "dry_run", False),
        )
    return main(args if isinstance(args, list) else None)

if __name__ == "__main__":
    sys.exit(main())
