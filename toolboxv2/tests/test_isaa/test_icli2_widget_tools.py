# -*- coding: utf-8 -*-
"""Welle 4: Widget-Tools (alle Agenten, creator-Schutz) + Bindung."""
import json
from pathlib import Path

import pytest

from toolboxv2.mods.isaa.extras.icli2.backend import IsaaBackend


def _backend(tmp: Path, agent: str = "self") -> object:
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
            self.agent = agent
            self.session = "default"
            self.max_iterations = 5
            self._settings = {}

    return _B()


SPEC = {"title": "Info", "template": "info", "text": "Hallo"}


def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_show_any_agent(tmp_path):
    b = _backend(tmp_path, agent="researcher")
    r = _run(b.widget_show(SPEC))
    assert r["ok"] and r["id"].startswith("researcher-")


def test_show_invalid_spec(tmp_path):
    b = _backend(tmp_path)
    r = _run(b.widget_show({"template": "html", "text": "x"}))
    assert not r["ok"] and r["errors"]


def test_update_only_creator(tmp_path):
    a = _backend(tmp_path, agent="alpha")
    r = _run(a.widget_show(SPEC))
    wid = r["id"]
    b = _backend(tmp_path, agent="beta")
    res = _run(b.widget_update(wid, SPEC))
    assert not res["ok"] and "Erzeuger" in res["error"]
    res2 = _run(a.widget_update(wid, {"title": "Neu", "template": "info", "text": "x"}))
    assert res2["ok"]


def test_close_only_creator(tmp_path):
    a = _backend(tmp_path, agent="alpha")
    wid = _run(a.widget_show(SPEC))["id"]
    b = _backend(tmp_path, agent="beta")
    assert not _run(b.widget_close(wid))["ok"]
    assert _run(a.widget_close(wid))["ok"]


def test_pin_roundtrip(tmp_path):
    a = _backend(tmp_path)
    wid = _run(a.widget_show(SPEC))["id"]
    r = _run(a.widget_pin(wid, True))
    assert r["mode"] == "pinned"
    reg = a.widget_registry_load()
    assert reg[wid]["mode"] == "pinned"


def test_state_panels(tmp_path):
    a = _backend(tmp_path)
    wid = _run(a.widget_show(SPEC))["id"]
    st = _run(a.widget_state())
    assert wid in st["panels"]
    assert st["panels"][wid]["title"] == "Info"


def test_id_unique(tmp_path):
    a = _backend(tmp_path)
    ids = {_run(a.widget_show(SPEC))["id"] for _ in range(5)}
    assert len(ids) == 5
