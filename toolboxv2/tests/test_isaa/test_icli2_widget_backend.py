# -*- coding: utf-8 -*-
"""Welle 3: Backend-Widget-Registry, Persistenz, creator-Routing + Antwort-Format."""
import json
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend


def _backend(tmp: Path) -> IsaaBackend:
    class _App:
        data_dir = str(tmp / "data")
        appdata = str(tmp / "appdata")

        class logger:
            @staticmethod
            def warning(*a, **k):
                pass

    class _B(IsaaBackend):
        def __init__(self):
            self.app = _App()
            self.agent = "self"
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}

    return _B()


SPEC_FORM = {"title": "Aufgaben-Auswahl", "template": "form", "mode": "modal",
             "fields": [{"name": "ort", "label": "Ort", "type": "text"}]}


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_registry_add_and_persist(tmp_path):
    b = _backend(tmp_path)
    r = _run(b.widget_registry_add("a1", SPEC_FORM, creator=("self", "default")))
    assert r["ok"]
    f = b._widgets_file()
    assert f.exists()
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["a1"]["spec"]["title"] == "Aufgaben-Auswahl"
    assert data["a1"]["creator"] == ["self", "default"]


def test_persistence_reload(tmp_path):
    b = _backend(tmp_path)
    _run(b.widget_registry_add("a1", SPEC_FORM, creator=("self", "default")))
    b2 = _backend(tmp_path)                      # Neustart = neues Backend-Objekt
    reg = b2.widget_registry_load()
    assert "a1" in reg and reg["a1"]["spec"]["template"] == "form"


def test_closed_stays_closed(tmp_path):
    b = _backend(tmp_path)
    _run(b.widget_registry_add("a1", SPEC_FORM, creator=("self", "default")))
    _run(b.widget_registry_close("a1"))
    b2 = _backend(tmp_path)
    assert b2.widget_registry_load()["a1"]["closed"] is True


def test_answer_ohne_engine_ist_laut(tmp_path):
    """Ohne verbundene Engine darf die Antwort nicht still verloren gehen (Regel 11).
    Der echte Rundweg (Engine -> Erzeuger-Lauf) steht in test_icli2_widget_betrieb.py (Punkte 7, 8)."""
    b = _backend(tmp_path)
    b._spawn, b._tasks = None, set()
    _run(b.widget_registry_add("a1", SPEC_FORM, creator=("self", "default")))
    with pytest.raises(RuntimeError, match="keine Engine"):
        _run(b._do_widget_answer(id="a1", values={"ort": "Berlin"}))
    assert b.widget_registry_load()["a1"]["values"] == {"ort": "Berlin"}   # Werte sind gespeichert


def test_answer_unknown_id(tmp_path):
    b = _backend(tmp_path)
    with pytest.raises(LookupError):
        _run(b._do_widget_answer(id="nope", values={}))


def test_format_answer_pretty():
    from toolboxv2.mods.isaa.extras.icli2.widgets import format_answer
    spec = {"title": "Aufgaben-Auswahl", "template": "form", "fields": [
        {"name": "ort", "label": "Ort", "type": "text"},
        {"name": "prio", "label": "Prioritaet", "type": "select"}]}
    line = format_answer(spec, {"ort": "Berlin", "prio": "hoch"})
    assert "Aufgaben-Auswahl" in line and "Ort: Berlin" in line and "Priorit" in line
    assert "{" not in line and "a1" not in line        # kein JSON, keine ID
