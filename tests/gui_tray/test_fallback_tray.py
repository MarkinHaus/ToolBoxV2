"""Verhalten des pystray-Fallback-Trays — der Pfad, der ohne Tauri-Binary läuft."""
from __future__ import annotations

import json
import sys
import types

import pytest


# --- Fake-pystray, nah genug an der echten API ---------------------------------

class _FakeMenuItem:
    def __init__(self, text, action, **kw):
        self.text = text
        self.action = action
        self.kw = kw


class _FakeMenu:
    SEPARATOR = object()

    def __init__(self, *items):
        self.items = items


@pytest.fixture()
def fake_pystray(monkeypatch):
    mod = types.ModuleType("pystray")
    mod.MenuItem = _FakeMenuItem
    mod.Menu = _FakeMenu
    mod.Icon = lambda *a, **kw: types.SimpleNamespace(run=lambda: None, stop=lambda: None)
    monkeypatch.setitem(sys.modules, "pystray", mod)
    return mod


# --- Basis-URL ----------------------------------------------------------------

def test_tray_base_url_default(fallback_tray, monkeypatch):
    monkeypatch.delenv("TB_TRAY_URL", raising=False)
    assert fallback_tray.tray_base_url() == "http://127.0.0.1:8467"


def test_tray_base_url_aus_env_ohne_slash(fallback_tray, monkeypatch):
    monkeypatch.setenv("TB_TRAY_URL", "http://127.0.0.1:6587/")
    assert fallback_tray.tray_base_url() == "http://127.0.0.1:6587"


# --- Menü ---------------------------------------------------------------------

def _texts(items):
    return [i.text for i in items if isinstance(i, _FakeMenuItem)]


def test_menu_items_ist_callable_das_items_liefert(fallback_tray, fake_pystray, monkeypatch):
    """pystray.Menu erwartet Items oder ein Callable, das Items liefert - kein Menu."""
    monkeypatch.setattr(fallback_tray, "fetch_tray_state", lambda: None)
    items = list(fallback_tray.build_menu_items(object()))
    assert not any(isinstance(i, _FakeMenu) for i in items)
    assert "Open Dashboard" in _texts(items)
    assert "Quit Application" in _texts(items)


def test_menu_zeigt_laufende_instanzen(fallback_tray, fake_pystray, monkeypatch):
    state = {
        "daemon": {"label": "Daemon App", "pid": 42, "running": True},
        "worker-1": {"label": "Worker 1", "pid": 43, "running": False},
        "_meta": "kein dict-worker",
    }
    monkeypatch.setattr(fallback_tray, "fetch_tray_state", lambda: state)
    texts = _texts(list(fallback_tray.build_menu_items(object())))
    assert any(t.startswith("Instances (1 running)") for t in texts)


def test_menu_ueberlebt_kaputte_tray_api(fallback_tray, fake_pystray, monkeypatch):
    def boom():
        raise RuntimeError("api tot")

    monkeypatch.setattr(fallback_tray, "fetch_tray_state", boom)
    texts = _texts(list(fallback_tray.build_menu_items(object())))
    assert "Quit Application" in texts


def test_fetch_tray_state_parst_json(fallback_tray, monkeypatch):
    class _Resp:
        def read(self):
            return json.dumps({"daemon": {"running": True}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setenv("TB_TRAY_URL", "http://127.0.0.1:5000")
    monkeypatch.setattr(fallback_tray.urllib.request, "urlopen", lambda *a, **kw: _Resp())
    assert fallback_tray.fetch_tray_state() == {"daemon": {"running": True}}


def test_fetch_tray_state_bei_netzfehler_none(fallback_tray, monkeypatch):
    def boom(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(fallback_tray.urllib.request, "urlopen", boom)
    assert fallback_tray.fetch_tray_state() is None


# --- Nicht-blockierende Aktionen ----------------------------------------------

def test_workers_network_blockiert_den_tray_thread_nicht(fallback_tray, monkeypatch):
    calls = []
    monkeypatch.setattr(fallback_tray.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(fallback_tray.os, "system", lambda *a: pytest.fail("os.system blockiert"))
    fallback_tray.open_workers_network(None, None)
    assert calls and calls[0][:2] == [sys.executable, "-m"]


def test_icon_wird_gezeichnet_kein_totes_blob(fallback_tray):
    img = fallback_tray.create_gear_icon()
    assert img.size == (64, 64)
    assert not hasattr(fallback_tray, "image"), "unbenutztes Bytes-Literal entfernt"


def test_debug_logfile_wird_best_effort_angehaengt(fallback_tray, tmp_path, monkeypatch):
    """pythonw hat kein stdout - der Tray muss trotzdem diagnostizierbar sein."""
    monkeypatch.setenv("TB_TRAY_LOG_DIR", str(tmp_path))
    fallback_tray.attach_debug_log()
    fallback_tray.log.info("hallo")
    for handler in fallback_tray.log.handlers:
        handler.flush()
    assert (tmp_path / "tray_debug.log").read_text().strip().endswith("hallo")


def test_debug_logfile_fehler_ist_nicht_fatal(fallback_tray, monkeypatch):
    monkeypatch.setenv("TB_TRAY_LOG_DIR", "/proc/definitiv/nicht/beschreibbar")
    fallback_tray.attach_debug_log()  # darf nicht werfen


def test_tb_tray_url_schlaegt_endpoint_datei(fallback_tray, monkeypatch):
    """Der Daemon läuft ggf. auf 6587 - dann gewinnt die Env-Variable."""
    monkeypatch.setenv("TB_TRAY_URL", "http://127.0.0.1:6587")
    assert fallback_tray.tray_base_url() == "http://127.0.0.1:6587"
