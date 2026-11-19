# -*- coding: utf-8 -*-
"""REG: inline-Chat-Widgets (mode=inline + template note/progress) via Agent-Tools.

Deckt den Fix ab:
  inline_note     : widget_show(mode=inline, template=note) -> ok, mode bleibt inline
  inline_progress : template=progress -> progress 0-100 wird normalisiert
  silent_fallback : mode='chat' -> KEIN stiller Fallback mehr: warnings im Ergebnis
  guards          : note/progress nur inline; inline nur note/progress (harte Fehler)
  regression      : info/form/modal + pin/close/state unveraendert
  build_lines     : note/progress Zeilen fuer Sidebar-Hint
  map_event       : widget_create inline -> ('inline', wid), pinned -> refresh_board
"""
import asyncio, sys
sys.path.insert(0, r"C:\Users\Markin\Workspace\ToolBoxV2")
from toolboxv2.mods.isaa.extras.icli2.widget_tools import WidgetToolsMixin, TOOL_CALLER
from toolboxv2.mods.isaa.extras.icli2.widget_app import map_widget_event
from toolboxv2.mods.isaa.extras.icli2.widgets import build_widget_lines

class StubBackend(WidgetToolsMixin):
    def __init__(s):
        s.reg = {}
        s.events = []
        TOOL_CALLER.set(("self", "t"))
    def widget_registry_load(s): return s.reg
    async def widget_registry_add(s, wid, spec, *, creator):
        s.reg[wid] = {"spec": spec, "mode": spec.get("mode", "modal"),
                      "values": {}, "closed": False, "creator": list(creator)}
        return {"success": True, "id": wid}
    async def widget_registry_close(s, wid):
        if wid in s.reg:
            s.reg[wid]["closed"] = True
            return {"success": True, "id": wid}
        return {"success": False}
    def _widgets_save(s, reg): s.reg = reg
    def _emit_widget(s, kind, data): s.events.append((kind, data))

async def main():
    b = StubBackend()

    # 1) inline note: Agent kann Chat-Karte erzeugen
    r = await b.widget_show({"title": "Hinweis", "template": "note", "mode": "inline",
                             "text": "Arbeite an **Fix**"})
    assert r.get("success"), r
    assert r["mode"] == "inline", r
    assert "warnings" not in r, r
    wid = r["id"]
    assert b.reg[wid]["mode"] == "inline"
    assert b.events[-1][0] == "widget_create" and b.events[-1][1]["mode"] == "inline"
    print("inline_note ok")

    # 2) update der inline-Karte (Live-Fortschritt)
    r = await b.widget_update(wid, {"title": "Hinweis", "template": "note", "mode": "inline",
                                    "text": "Zweiter Stand"})
    assert r.get("success") and b.events[-1][0] == "widget_update", (r, b.events[-1])
    assert b.reg[wid]["mode"] == "inline"
    print("inline_update ok")

    # 3) progress mit Zahl
    r = await b.widget_show({"title": "DL", "template": "progress", "mode": "inline",
                             "text": "4/10", "progress": 40})
    assert r.get("success"), r
    spec = b.reg[r["id"]]["spec"]
    assert spec["progress"] == 40 and spec["text"] == "4/10", spec
    print("inline_progress ok")

    # 4) mode='chat' ist kein stiller Fallback mehr: Warning kommt zurueck
    r = await b.widget_show({"title": "Chat?", "template": "info", "mode": "chat", "text": "x"})
    assert r.get("success") and r["mode"] == "modal", r
    assert any("chat" in w for w in r.get("warnings", [])), r
    print("silent_fallback behoben: warnings =", r["warnings"])

    # 5) Guards: note/progress nur inline
    for tpl in ("note", "progress"):
        spec = {"title": "F", "template": tpl, "mode": "modal", "text": "x"}
        if tpl == "progress": spec["progress"] = 10
        r = await b.widget_show(spec)
        assert not r.get("success") and any("inline" in e for e in r["errors"]), (tpl, r)
    # Rollback 19.11.: inline wieder PASSIV - form/choice/info + inline ist verboten
    r = await b.widget_show({"title": "F", "template": "choice", "mode": "inline",
                             "options": [{"label": "a", "value": "1"}, {"label": "b", "value": "2"}]})
    assert not r.get("success") and any("passiv" in e.lower() for e in r["errors"]), r
    # Chart: passiv, line/bar, 2-50 Werte
    r = await b.widget_show({"title": "Chart", "template": "chart", "mode": "inline",
                             "chart_type": "line", "values": [1, 3, 2, 7, 5]})
    assert r.get("success") and r["mode"] == "inline", r
    wid_c = r["id"]
    r = await b.widget_update(wid_c, {"title": "Chart", "template": "chart", "mode": "inline",
                                     "chart_type": "bar", "values": [0, 10, 5]})
    assert r.get("success"), r
    bad = await b.widget_show({"title": "X", "template": "chart", "mode": "modal", "values": [1, 2]})
    assert not bad.get("success"), bad
    from toolboxv2.mods.isaa.extras.icli2.widgets import build_widget_lines as bwl
    ln = bwl({"template": "chart", "chart_type": "line", "values": [1, 5, 2, 8]})[0]
    assert "█" in ln and "1..8" in ln, ln
    bars = bwl({"template": "chart", "chart_type": "bar", "values": [1, 5, 2]})
    assert len(bars) == 3 and all(b.startswith("█") for b in bars), bars
    print("chart ok:", ln.strip())

    # progress-Wertebereich
    r = await b.widget_show({"title": "F", "template": "progress", "mode": "inline", "text": "x", "progress": 150})
    assert not r.get("success") and any("0-100" in e for e in r["errors"]), r
    print("guards ok")

    # 6) Regression: klassische Pfade unversehrt
    r = await b.widget_show({"title": "Info", "template": "info", "mode": "modal", "text": "hallo"})
    assert r.get("success") and r["mode"] == "modal" and "warnings" not in r, r
    r = await b.widget_show({"title": "F", "template": "form", "mode": "pinned",
                             "fields": [{"name": "n", "label": "Name", "type": "text", "required": True}]})
    assert r.get("success") and r["mode"] == "pinned", r
    wid_f = r["id"]
    r = await b.widget_pin(wid_f, True)
    assert r.get("success") and r["mode"] == "pinned", r
    r = await b.widget_pin(wid_f, False)
    assert r.get("success") and r["mode"] == "modal", r
    st = await b.widget_state()
    assert st.get("success") and any(e["mode"] == "inline" for e in st["panels"].values()), st
    r = await b.widget_close(wid)
    assert r.get("success") and b.reg[wid]["closed"], r
    print("regression (info/form/pin/state/close) ok")

    # 7) build_widget_lines: Sidebar-Hint fuer note/progress
    assert build_widget_lines({"template": "note", "text": "Zeile im Chat"}) == ["Zeile im Chat"]
    lines = build_widget_lines({"template": "progress", "progress": 40, "text": "4/10"})
    assert lines and "40%" in lines[0] and "4/10" in lines[0], lines
    print("build_lines ok:", lines[0])

    # 8) Event-Mapping (app.py-Routing-Voraussetzung)
    assert map_widget_event("widget_create", {"id": "w", "mode": "inline"}) == ("inline", "w")
    assert map_widget_event("widget_create", {"id": "w", "mode": "pinned"}) == ("refresh_board", "w")
    assert map_widget_event("widget_create", {"id": "w", "mode": "modal"}) == ("push_modal", "w")
    print("map_event ok")

    # 9) mermaid: ASCII-Baum statt Rohcode (Restart 4, Fix B2+Renderer)
    lines = build_widget_lines({"template": "mermaid", "text":
        "graph LR\n  A[User] --> B{icli2}\n  B -->|fix| C[Agent]\n  C --> D\n  D --> E((Chat))"})
    assert lines and not lines[0].startswith("```"), lines
    assert any("\u25b6" in ln for ln in lines), lines          # Pfeilspitzen
    assert any("User" in ln for ln in lines) and any("Chat" in ln for ln in lines), lines
    assert any("fix" in ln for ln in lines), lines               # Kantenlabel
    assert not any(ln.strip().startswith("graph") for ln in lines), lines
    print("mermaid_tree ok:", " | ".join(lines[:3]))
    # Fallback: unverstaendlicher Quellcode bleibt Code-Block
    fb = build_widget_lines({"template": "mermaid", "text": "sequenceDiagram\n  A->>B: hi"})
    assert fb[0] == "```mermaid", fb
    print("mermaid_fallback ok")

    print("ALL OK")

asyncio.run(main())
