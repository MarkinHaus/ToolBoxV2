# -*- coding: utf-8 -*-
"""Welle 5: App-Event-Mapping fuer Widgets (reine Logik, kein Textural-Import)."""
import pytest

from toolboxv2.mods.isaa.extras.icli2 import widget_app as WA


def test_map_create_modal():
    act = WA.map_widget_event("widget_create", {"id": "a1", "mode": "modal"})
    assert act == ("push_modal", "a1")

def test_map_create_pinned():
    act = WA.map_widget_event("widget_create", {"id": "a2", "mode": "pinned"})
    assert act == ("refresh_board", "a2")

def test_map_update():
    assert WA.map_widget_event("widget_update", {"id": "a3"}) == ("refresh", "a3")

def test_map_close():
    assert WA.map_widget_event("widget_close", {"id": "a4"}) == ("close", "a4")

def test_map_unknown():
    assert WA.map_widget_event("text", {}) is None
    assert WA.map_widget_event("widget_create", {}) is None     # keine id -> ignorieren


def test_side_rows_widgets():
    reg = {"a1": {"spec": {"title": "Info-Board"}, "mode": "pinned", "closed": False,
                   "values": {}, "creator": ["self", "default"]},
           "a2": {"spec": {"title": "Alt"}, "mode": "pinned", "closed": True,
                   "values": {}, "creator": ["self", "default"]}}
    rows = WA.side_rows_widgets(reg)
    assert rows == [{"value": "a1", "label": "Info-Board", "hint": "pinned"}]
