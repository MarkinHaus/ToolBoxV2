# -*- coding: utf-8 -*-
"""App-Verdrahtung fuer Widgets: reine Mapping-Logik + Integration-Hinweise.

Die eigentliche app.py-Verdrahtung (push_screen(WidgetModal), Board-Refresh,
WIDGET_ANSWER-Send) folgt dem Picker-Muster (app.py:_pick + push_screen) und
dem _handle-Event-Dispatch (app.py:_handle). Diese Datei haelt die testbare
Logik ohne textual-Import (venv-frei, wie autocomplete.py aus T3).
"""
from __future__ import annotations


def map_widget_event(kind: str, data: dict) -> tuple[str, str] | None:
    """Event-Typ -> UI-Aktion. Nur vollstaendige Payloads, sonst None (ignorieren).

    Rueckgabe: (aktion, widget_id) mit aktion in
    push_modal | refresh_board | refresh | close
    """
    wid = (data or {}).get("id")
    if not wid:
        return None
    if kind == "widget_create":
        return ("refresh_board", wid) if data.get("mode") == "pinned" else ("push_modal", wid)
    if kind == "widget_update":
        return ("refresh", wid)
    if kind == "widget_close":
        return ("close", wid)
    return None


def side_rows_widgets(registry: dict) -> list[dict]:
    """F2-Seitenleisten-Zeilen (Picker-Row-Format) fuer offene gepinnte Widgets."""
    rows = []
    for wid, e in (registry or {}).items():
        if e.get("closed") or e.get("mode") != "pinned":
            continue
        rows.append({"value": wid, "label": e.get("spec", {}).get("title", wid),
                     "hint": "pinned"})
    return rows
