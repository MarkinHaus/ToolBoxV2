# -*- coding: utf-8 -*-
"""Welle 2: Widget-Baukasten - reine Zeilen-Fabrik + Modal/Card (lazy textual)."""
from toolboxv2.mods.isaa.extras.icli2 import widgets as W


FORM = {
    "title": "Aufgaben-Auswahl",
    "template": "form",
    "mode": "modal",
    "fields": [
        {"name": "ort", "label": "Ort", "type": "text"},
        {"name": "prio", "label": "Prioritaet", "type": "select",
         "options": [{"label": "hoch", "value": "hoch"}, {"label": "niedrig", "value": "niedrig"}]},
        {"name": "ok", "label": "Bestaetigen", "type": "checkbox"},
    ],
}


def test_form_lines():
    lines = W.build_widget_lines(FORM)
    text = "\n".join(lines)
    assert "Ort" in text and "Prioritaet" in text
    assert "hoch" in text and "niedrig" in text
    assert any("[ ]" in l or "[x]" in l for l in lines)      # Checkbox-Kennung


def test_form_defaults_in_lines():
    spec = dict(FORM)
    spec["fields"] = FORM["fields"][:1] + [
        {"name": "n", "label": "Zahl", "type": "number", "default": 5}]
    lines = W.build_widget_lines(spec)
    assert any("5" in l for l in lines)                       # Default sichtbar


def test_choice_lines():
    lines = W.build_widget_lines({"title": "Wahl", "template": "choice", "mode": "modal",
                                  "options": [{"label": "a", "value": "a"},
                                              {"label": "b", "value": "b"}]})
    assert len([l for l in lines if l.strip().startswith(("a", "b"))]) == 2


def test_info_lines_wrapped():
    lines = W.build_widget_lines({"title": "Info", "template": "info", "mode": "pinned",
                                  "text": "x" * 250})
    assert lines and all(len(l) <= 100 for l in lines)        # Zeilen gekappt
