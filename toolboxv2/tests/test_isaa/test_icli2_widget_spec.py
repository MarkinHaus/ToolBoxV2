# -*- coding: utf-8 -*-
"""Widget-Spec-Validator (Schritt 1) - reine Funktion, keine IO."""
import pytest

from toolboxv2.mods.isaa.extras.icli2.widget_spec import validate_spec


CREATOR = ("self", "default")


def _form_spec(**over):
    s = {
        "title": "Aufgaben-Auswahl",
        "template": "form",
        "fields": [
            {"name": "ort", "label": "Ort", "type": "text"},
            {"name": "n", "label": "Anzahl", "type": "number", "min": 1, "max": 10},
            {"name": "notes", "label": "Notizen", "type": "textarea"},
            {"name": "prio", "label": "Prioritaet", "type": "select", "options": ["hoch", "niedrig"]},
            {"name": "ok", "label": "Bestaetigen", "type": "checkbox"},
        ],
    }
    s.update(over)
    return s


def test_form_valid():
    r = validate_spec(_form_spec(), creator=CREATOR)
    assert r["ok"] and not r["errors"]
    norm = r["spec"]
    assert norm["mode"] == "modal"                    # Default ergaenzt
    select = [f for f in norm["fields"] if f["name"] == "prio"][0]
    assert select["options"] == [{"label": "hoch", "value": "hoch"},
                                 {"label": "niedrig", "value": "niedrig"}]  # Shortcut normalisiert


def test_choice_valid_shortcut():
    r = validate_spec({"title": "Wahl", "template": "choice", "options": ["a", "b"]}, creator=CREATOR)
    assert r["ok"]
    assert r["spec"]["options"] == [{"label": "a", "value": "a"}, {"label": "b", "value": "b"}]


def test_info_valid_max_len():
    r = validate_spec({"title": "Info", "template": "info", "text": "x" * 2000}, creator=CREATOR)
    assert r["ok"]
    r2 = validate_spec({"title": "Info", "template": "info", "text": "x" * 2001}, creator=CREATOR)
    assert not r2["ok"]


def test_invalid_template():
    r = validate_spec({"title": "t", "template": "html", "text": "x"}, creator=CREATOR)
    assert not r["ok"] and any("template" in e for e in r["errors"])


def test_title_required_and_limit():
    assert not validate_spec({"template": "info", "text": "x"}, creator=CREATOR)["ok"]
    assert not validate_spec({"title": "y" * 81, "template": "info", "text": "x"}, creator=CREATOR)["ok"]


def test_max_20_fields():
    fields = [{"name": f"f{i}", "label": "x", "type": "text"} for i in range(21)]
    r = validate_spec(_form_spec(fields=fields), creator=CREATOR)
    assert not r["ok"]


def test_select_needs_2_to_12_options():
    r = validate_spec(_form_spec(fields=[{"name": "s", "label": "s", "type": "select", "options": ["a"]}]),
                      creator=CREATOR)
    assert not r["ok"]


def test_number_bounds_numeric():
    r = validate_spec(_form_spec(fields=[{"name": "n", "label": "n", "type": "number", "min": "x"}]),
                      creator=CREATOR)
    assert not r["ok"]


def test_unknown_field_type():
    r = validate_spec(_form_spec(fields=[{"name": "w", "label": "w", "type": "slider"}]), creator=CREATOR)
    assert not r["ok"]


def test_creator_required():
    r = validate_spec(_form_spec(), creator=("", "s"))
    assert not r["ok"] and any("creator" in e for e in r["errors"])
    r2 = validate_spec(_form_spec(), creator=("self", ""))
    assert not r2["ok"]


def test_pinned_mode_roundtrip():
    r = validate_spec({"title": "Board", "template": "info", "text": "hi", "mode": "pinned"}, creator=CREATOR)
    assert r["ok"] and r["spec"]["mode"] == "pinned"


def test_unknown_mode_warns_falls_back():
    r = validate_spec({"title": "t", "template": "info", "text": "x", "mode": "fullscreen"}, creator=CREATOR)
    assert r["ok"] and r["spec"]["mode"] == "modal" and r["warnings"]
