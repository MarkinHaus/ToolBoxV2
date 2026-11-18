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


def test_answer_updates_registry_and_emits(tmp_path):
    b = _backend(tmp_path)
    _run(b.widget_registry_add("a1", SPEC_FORM, creator=("self", "default")))
    sent = []
    async def fake_start(prompt):                # Netz-/LLM-Grenze: was rausgeht
        sent.append(prompt)
    b._widget_start_turn = fake_start
    res = _run(b._do_widget_answer({"id": "a1", "values": {"ort": "Berlin"}}))
    assert res["ok"]
    reg = b.widget_registry_load()
    assert reg["a1"]["values"] == {"ort": "Berlin"}
    assert sent and "Berlin" in sent[0]          # Antwort erreicht creator-Turn


def test_answer_unknown_id(tmp_path):
    b = _backend(tmp_path)
    res = _run(b._do_widget_answer({"id": "nope", "values": {}}))
    assert not res["ok"]


def test_format_answer_pretty():
    from toolboxv2.mods.isaa.extras.icli2.widgets import format_answer
    spec = {"title": "Aufgaben-Auswahl", "template": "form", "fields": [
        {"name": "ort", "label": "Ort", "type": "text"},
        {"name": "prio", "label": "Prioritaet", "type": "select"}]}
    line = format_answer(spec, {"ort": "Berlin", "prio": "hoch"})
    assert "Aufgaben-Auswahl" in line and "Ort: Berlin" in line and "Priorit" in line
    assert "{" not in line and "a1" not in line        # kein JSON, keine ID
