"""tauri_cli (tb gui) und der Web-Fallback aus init_onboarding."""
from __future__ import annotations

import ast
import json
import types
from collections import Counter

import pytest


# --- tauri_cli: Struktur -------------------------------------------------------

def _toplevel_funcs(source: str) -> Counter:
    tree = ast.parse(source)
    return Counter(
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def test_keine_doppelten_funktionsdefinitionen(tauri_cli_source):
    dupes = {name: c for name, c in _toplevel_funcs(tauri_cli_source).items() if c > 1}
    assert dupes == {}, f"überschriebene Definitionen: {dupes}"


def test_worker_output_dir_heisst_nicht_nuitka(tauri_cli_source):
    """Gebaut wird mit PyInstaller - der Default-Pfad darf nicht lügen."""
    assert "nuitka-build" not in tauri_cli_source


def test_frontend_build_baut_im_paketroot_und_installiert_subpakete(tauri_cli, tmp_path):
    """`npm run build` im Paketroot baut web *und* tbjs -> dist/ für Tauri.

    Die Sub-Pakete brauchen aber ihr eigenes `npm install`, sonst schlägt der
    Webpack-Lauf fehl.
    """
    root = tmp_path
    pkg = root / "toolboxv2"
    for sub in (pkg, pkg / "web", pkg / "tbjs"):
        sub.mkdir(parents=True)
        (sub / "package.json").write_text("{}")

    calls = []

    class _Run:
        returncode = 0

    tauri_cli.subprocess.run = lambda cmd, cwd=None, **kw: (calls.append((cmd, cwd)), _Run())[1]

    assert tauri_cli.build_frontend(root) is True

    installs = {cwd for cmd, cwd in calls if cmd[:2] == ["npm", "install"]}
    builds = [(cmd, cwd) for cmd, cwd in calls if cmd[:2] == ["npm", "run"]]
    assert installs == {pkg, pkg / "web", pkg / "tbjs"}
    assert builds == [(["npm", "run", "build"], pkg)]


def test_frontend_build_ohne_package_json_ist_kein_fehler(tauri_cli, tmp_path):
    assert tauri_cli.build_frontend(tmp_path) is True


def test_run_app_faellt_auf_local_ui_zurueck(tauri_cli, monkeypatch):
    monkeypatch.setattr(tauri_cli, "get_installed_app_path", lambda: None)
    monkeypatch.setattr(tauri_cli, "_install_local_build", lambda root: False)
    monkeypatch.setattr(tauri_cli, "get_project_root", lambda: tauri_cli.Path("/tmp"))
    seen = {}
    monkeypatch.setattr(tauri_cli, "_fallback_to_local_ui", lambda port=5000: seen.setdefault("port", port))
    tauri_cli.run_app(with_worker=False, http_port=5000, download_if_missing=False)
    assert seen == {"port": 5000}


# --- init_onboarding: Web-Fallback --------------------------------------------

def test_serve_local_ui_mountet_tray_api(init_onboarding, monkeypatch):
    """Ohne mount_tray_api ist /tray/state 404 und das Tray-Menü bleibt leer."""
    mounted = []
    served = []

    waitress = types.ModuleType("waitress")
    waitress.serve = lambda app, host=None, port=None: served.append((host, port))

    local_ui = types.ModuleType("toolboxv2.utils.workers.fast.local_ui")
    local_ui.app = object()

    tray_api = types.ModuleType("toolboxv2.utils.workers.fast.tray_api")
    tray_api.mount_tray_api = lambda app: mounted.append(app)

    handler_mod = types.ModuleType("toolboxv2.utils.workers.fast_tb_handler")

    class _Handler:
        def __init__(self, app):
            self.app = app

        def as_wsgi_app(self, enable_ws=False):
            return "wsgi"

    handler_mod.FastTBHandler = _Handler

    import sys
    for name, mod in {
        "waitress": waitress,
        "toolboxv2.utils.workers.fast.local_ui": local_ui,
        "toolboxv2.utils.workers.fast.tray_api": tray_api,
        "toolboxv2.utils.workers.fast_tb_handler": handler_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    assert init_onboarding._serve_local_ui("127.0.0.1", 5000) is True
    assert mounted == [local_ui.app], "tray_api wurde nicht gemountet"
    assert served == [("127.0.0.1", 5000)]


def test_start_tray_setzt_tb_tray_url(init_onboarding, monkeypatch):
    import sys
    monkeypatch.delenv("TB_TRAY_URL", raising=False)
    monkeypatch.setitem(sys.modules, "pystray", types.ModuleType("pystray"))
    ft = types.ModuleType("toolboxv2.utils.extras.fallback_tray")
    ft.run_fallback_tray = lambda app: None
    monkeypatch.setitem(sys.modules, "toolboxv2.utils.extras.fallback_tray", ft)

    started = []
    monkeypatch.setattr(
        init_onboarding.threading if hasattr(init_onboarding, "threading") else __import__("threading"),
        "Thread",
        lambda target, args=(), daemon=None, name=None: types.SimpleNamespace(
            start=lambda: started.append(target)
        ),
        raising=False,
    )

    assert init_onboarding._start_tray(object(), "http://127.0.0.1:5000") is True
    import os
    assert os.environ["TB_TRAY_URL"] == "http://127.0.0.1:5000"
    assert started


# --- Daemon-Pfad: gemeinsame Tray-URL -----------------------------------------

def test_daemon_setzt_tb_tray_url_vor_tray_start():
    """Der Daemon lauscht auf 6587 - Tray und Tauri müssen dieselbe URL sehen."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "toolboxv2" / "__main__.py").read_text()
    assert 'os.environ["TB_TRAY_URL"] = _tray_url' in src
    assert src.index('os.environ["TB_TRAY_URL"] = _tray_url') < src.index(
        "from toolboxv2.utils.extras.fallback_tray import run_fallback_tray"
    )


def test_build_app_bricht_ohne_sidecar_ab(tauri_cli, tmp_path, monkeypatch):
    """externalBin ist Pflicht: ohne binaries/tb-worker scheitert `tauri build` erst spät."""
    tauri = tmp_path / "toolboxv2" / "simple-core" / "src-tauri"
    tauri.mkdir(parents=True)
    (tauri / "Cargo.toml").write_text("")
    (tmp_path / "toolboxv2" / "dist").mkdir(parents=True)
    monkeypatch.setattr(tauri_cli.subprocess, "run", lambda *a, **kw: pytest.fail("darf nicht bauen"))
    assert tauri_cli.build_tauri_app(tmp_path) is False


def test_build_app_laeuft_mit_sidecar_und_dist(tauri_cli, tmp_path, monkeypatch):
    tauri = tmp_path / "toolboxv2" / "simple-core" / "src-tauri"
    (tauri / "binaries").mkdir(parents=True)
    (tauri / "Cargo.toml").write_text("")
    (tauri / "binaries" / "tb-worker-x86_64-unknown-linux-gnu").write_text("")
    (tmp_path / "toolboxv2" / "dist").mkdir(parents=True)

    class _Run:
        returncode = 0

    monkeypatch.setattr(tauri_cli.subprocess, "run", lambda *a, **kw: _Run())
    monkeypatch.setattr(tauri_cli, "_install_local_build", lambda root: True)
    assert tauri_cli.build_tauri_app(tmp_path) is True
