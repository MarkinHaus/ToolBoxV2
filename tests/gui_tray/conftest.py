"""Test-Infrastruktur für die GUI/Tray-Schicht.

Die betroffenen Module leben tief in ``toolboxv2``, dessen ``__init__`` die
volle Runtime hochzieht (pydantic, DB, ...). Für Unit-Tests laden wir die
Dateien deshalb isoliert per importlib und stellen nur die Symbole bereit,
die das jeweilige Modul beim Import tatsächlich anfasst.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "toolboxv2"
TAURI_SRC = PKG_ROOT / "simple-core" / "src-tauri" / "src"


def load_isolated(path: Path, name: str, stubs: dict[str, types.ModuleType] | None = None):
    """Lädt eine .py-Datei als eigenständiges Modul, ohne ihr Paket zu importieren."""
    saved = {k: sys.modules.get(k) for k in (stubs or {})}
    sys.modules.update(stubs or {})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        sys.modules.pop(name, None)


def _toolboxv2_stub() -> types.ModuleType:
    mod = types.ModuleType("toolboxv2")
    mod.tb_root_dir = PKG_ROOT
    mod.get_app = lambda *a, **kw: None
    return mod


@pytest.fixture(scope="session")
def rust_lib_source() -> str:
    return (TAURI_SRC / "lib.rs").read_text(encoding="utf-8")


@pytest.fixture()
def fallback_tray():
    return load_isolated(
        PKG_ROOT / "utils" / "extras" / "fallback_tray.py",
        "_fallback_tray_under_test",
        {"toolboxv2": _toolboxv2_stub()},
    )


@pytest.fixture()
def init_onboarding():
    return load_isolated(
        PKG_ROOT / "init_onboarding.py",
        "_init_onboarding_under_test",
        {"toolboxv2": _toolboxv2_stub()},
    )


@pytest.fixture(scope="session")
def tauri_cli_source() -> str:
    return (PKG_ROOT / "utils" / "clis" / "tauri_cli.py").read_text(encoding="utf-8")


@pytest.fixture()
def tauri_cli():
    """tauri_cli nutzt relative Imports -> minimales Fake-Paket drumherum."""
    pkg = types.ModuleType("_tbclis")
    pkg.__path__ = [str(PKG_ROOT / "utils" / "clis")]
    printing = types.ModuleType("_tbclis.cli_printing")
    for fn in ("print_status", "print_box_header", "print_box_footer", "c_print"):
        setattr(printing, fn, lambda *a, **kw: None)
    return load_isolated(
        PKG_ROOT / "utils" / "clis" / "tauri_cli.py",
        "_tbclis.tauri_cli",
        {
            "toolboxv2": _toolboxv2_stub(),
            "_tbclis": pkg,
            "_tbclis.cli_printing": printing,
        },
    )
