# -*- coding: utf-8 -*-
"""Betriebstests der icli2-Widgets (PLAN v4, Punkte 1-10).

Eingang wie im Betrieb: echte ToolBox-App + isaa-Mod, echte Engine und IsaaBackend, echte Textual-App
(Pilot). Ersetzt ist nur das LLM (HTTP-Ersatz, Prozessgrenze) und das Laden des Embedding-Modells
(Netz) - siehe icli2_betrieb.py. Geprueft wird Zustand: widgets.json, settings.json, Tool-Registry des
Agenten, sichtbare Zeilen der App, Prompt am Ersatz-LLM.

Laufen in einem eigenen CI-Schritt:  pytest -m icli2_betrieb --timeout=120
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from icli2_betrieb import TIMEOUT, Betrieb, call_query

pytestmark = [pytest.mark.icli2_betrieb, pytest.mark.timeout(TIMEOUT)]

WIDGET_TOOLS = ("widget_show", "widget_update", "widget_close", "widget_pin", "widget_state")
SELF_TOOLS = ("self_set_setting", "self_get_setting", "self_get_agent_config",
              "self_set_agent_config", "self_list_agents")

FORM = {"title": "Reise planen", "template": "form", "mode": "modal", "fields": [
    {"name": "ort", "label": "Ort", "type": "text"},
    {"name": "bahn", "label": "Mit der Bahn", "type": "checkbox"},
    {"name": "prio", "label": "Prioritaet", "type": "select",
     "options": [{"label": "hoch", "value": "h"}, {"label": "niedrig", "value": "n"}]},
]}
CHOICE = {"title": "Farbe waehlen", "template": "choice", "mode": "modal",
          "options": [{"label": "Rot", "value": "rot"}, {"label": "Gruen", "value": "gruen"},
                      {"label": "Blau", "value": "blau"}]}


def _pinned(title: str) -> dict:
    return {"title": title, "template": "info", "mode": "pinned", "text": f"Inhalt von {title}"}


def _register(b: Betrieb, *names: str) -> None:
    """Agenten so anlegen, wie isaa es tut (Builder + register_agent), ohne LLM-Aufruf."""
    async def go():
        for name in names:
            await b.backend.isaa.register_agent(b.backend.isaa.get_agent_builder(name))
    b.call(go())


def _show_via_agent(b: Betrieb, agent: str, session: str, spec: dict, tag: str) -> str:
    """Agent erzeugt ein Widget ueber seinen Tool-Aufruf (Ersatz-LLM ruft widget_show)."""
    end = b.chat(agent, session, call_query("widget_show", **spec), tag)
    assert end["status"] == "done", end
    ids = [wid for wid, e in b.widgets_json().items() if e["spec"]["title"] == spec["title"]]
    assert ids, f"widget_show von {agent} hat nichts in widgets.json geschrieben: {end.get('answer')}"
    return ids[-1]


# ── 0. Ein echter Lauf braucht kein litellm ────────────────────────────────
def test_0_lauf_ohne_litellm():
    with Betrieb() as b:
        assert sys.modules.get("litellm", None) is None    # gesperrt: Import wuerde scheitern
        b.ready()
        end = b.chat("self", "default", "hallo ohne litellm", "t0")
        assert end["status"] == "done", end
        assert any("hallo ohne litellm" in p for p in b.llm.prompts()), "Ersatz-LLM nicht erreicht"


# ── 1. Tool-Bindung: jeder Agent der icli2 hat die Widget-Tools ─────────────
def test_1_jeder_agent_hat_widget_tools():
    with Betrieb() as b:
        b.ready()
        _register(b, "alpha", "beta")
        for i, name in enumerate(("self", "alpha", "beta")):
            b.chat(name, "default", "hallo", f"t1-{i}")
            agent = b.call(b.backend.isaa.get_agent(name))
            have = [t for t in WIDGET_TOOLS if agent.tool_manager.exists(t)]
            assert have == list(WIDGET_TOOLS), f"{name}: Widget-Tools fehlen, vorhanden: {have}"
        # self-Tools nur beim Agenten 'self'
        assert all(b.call(b.backend.isaa.get_agent("self")).tool_manager.exists(t) for t in SELF_TOOLS)
        assert not any(b.call(b.backend.isaa.get_agent("alpha")).tool_manager.exists(t) for t in SELF_TOOLS)


# ── 2. widget_show ueber den Agenten (Tool-Call aus dem Lauf) ───────────────
def test_2_widget_show_ueber_agenten_lauf():
    with Betrieb() as b:
        b.ready()
        _register(b, "alpha")
        end = b.chat("alpha", "s2", call_query("widget_show", **FORM), "t2")
        assert end["status"] == "done", end
        reg = b.widgets_json()
        assert len(reg) == 1, reg
        wid, entry = next(iter(reg.items()))
        assert entry["creator"] == ["alpha", "s2"]
        assert entry["spec"]["title"] == "Reise planen" and not entry["closed"]
        # Tool-Ergebnis kam beim Modell an: ok + id
        result = next(r["results"][-1] for r in b.llm.requests if "widget_show" in r["called"] and r["results"])
        assert '"ok": true' in result.lower().replace("'", '"').replace("True", "true") and wid in result
        # Die UI bekommt das Widget mit vollem Spec und Erzeuger
        ev = b.wait_event(lambda e: e.kind == "widget_create" and e.data.get("id") == wid, 5, "WIDGET_CREATE")
        assert ev.data["spec"]["fields"][0]["name"] == "ort"
        assert ev.data["creator"] == ["alpha", "s2"] and ev.data["mode"] == "modal"


# ── 3. self_*-Tools wirken nur fuer den Agenten 'self' ─────────────────────
def test_3_self_tools_nur_fuer_self():
    with Betrieb() as b:
        b.ready()
        _register(b, "fremd")
        q = call_query("self_set_setting", key="max_iterations", value=77)
        b.chat("fremd", "default", q, "t3a")
        assert b.settings_json().get("max_iterations") != 77, "fremder Agent hat self-Setting geaendert"
        fremd = b.call(b.backend.isaa.get_agent("fremd"))
        with pytest.raises(Exception):
            b.call(fremd.arun_function("self_set_setting", key="max_iterations", value=78))
        assert b.settings_json().get("max_iterations") not in (77, 78)
        end = b.chat("self", "default", q, "t3b")
        assert end["status"] == "done", end
        assert b.settings_json().get("max_iterations") == 77, b.settings_json()


# ── 4./5. Modal echt bedienen (Textual Pilot) ───────────────────────────────
def _host(screen_factory):
    from textual.app import App

    class Host(App):
        result = "unset"

        def on_mount(self):
            self.push_screen(screen_factory(), self._done)

        def _done(self, value):
            self.result = value

    return Host()


def test_4_form_modal_liefert_alle_werte():
    from textual.widgets import Checkbox, Input, OptionList
    from toolboxv2.mods.isaa.extras.icli2.widget_spec import validate_spec
    from toolboxv2.mods.isaa.extras.icli2.widgets import WidgetModal
    spec = validate_spec(FORM, creator=("self", "default"))["spec"]
    host = _host(lambda: WidgetModal("w4", spec, creator=("self", "default")))

    async def go():
        async with host.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            scr = host.screen
            await pilot.click(scr.query_one(Input))
            await pilot.press(*"Berlin")
            await pilot.click(scr.query_one(Checkbox))
            opts = scr.query_one(OptionList)
            opts.focus()
            await pilot.press("down", "enter")             # zweite Option: niedrig
            await pilot.click(scr.query_one(Input))
            await pilot.press("enter")
            await pilot.pause()
    asyncio.run(go())
    assert host.result == {"id": "w4", "values": {"ort": "Berlin", "bahn": True, "prio": "n"}}, host.result


def test_5_choice_modal_liefert_gewaehlte_option():
    from textual.widgets import OptionList
    from toolboxv2.mods.isaa.extras.icli2.widget_spec import validate_spec
    from toolboxv2.mods.isaa.extras.icli2.widgets import WidgetModal
    spec = validate_spec(CHOICE, creator=("self", "default"))["spec"]
    host = _host(lambda: WidgetModal("w5", spec, creator=("self", "default")))

    async def go():
        async with host.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            opts = host.screen.query_one(OptionList)
            assert [str(opts.get_option_at_index(i).prompt) for i in range(opts.option_count)] == \
                ["Rot", "Gruen", "Blau"]
            opts.focus()
            await pilot.press("down", "down", "enter")
            await pilot.pause()
    asyncio.run(go())
    assert host.result == {"id": "w5", "values": {"value": "blau"}}, host.result


# ── UI-Hilfen fuer die echte ICli2 ──────────────────────────────────────────
async def _until(pilot, pred, timeout=60.0, what="Bedingung"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        await pilot.pause(0.1)
    extra = ""
    app = getattr(pilot, "app", None)
    if app is not None and hasattr(app, "w_side_text"):
        extra = f"; Seitenleiste: {_side_text(app)[:600]!r}; board={app.side_rows.get('board')}"
    raise AssertionError(f"{what} nicht erreicht in {timeout}s{extra}")


def _side_text(ui) -> str:
    return str(ui.w_side_text.render())


def _chat_text(ui) -> str:
    from textual.widgets import Static
    return "\n".join(str(w.render()) for w in ui.chat.query(Static))


# ── 6. F2-Seitenleiste zeigt die offenen gepinnten Widgets ─────────────────
async def _ui_chat(ui, pilot, query: str, timeout: float = 60.0) -> None:
    """Nachricht ueber die Eingabe der App senden und auf das Ende dieses Laufs warten."""
    before = set(ui.runs)
    ui.prompt.text = query
    await pilot.press("enter")
    await _until(pilot, lambda: any(r not in before and v.status != "running" for r, v in ui.runs.items()),
                 timeout, f"Lauf zu {query[:40]}")


def test_6_seitenleiste_zeigt_gepinnte_widgets():
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.widgets import WidgetModal
    with Betrieb() as b:
        ui = ICli2(b.engine)

        async def go():
            async with ui.run_test(size=(140, 45)) as pilot:
                await _until(pilot, lambda: ui.ready, 60, "READY")
                for title in ("Board Eins", "Board Zwei", "Schon zu"):
                    await _ui_chat(ui, pilot, call_query("widget_show", **_pinned(title)))
                await _ui_chat(ui, pilot, call_query("widget_show", **dict(CHOICE, title="Nur Modal")))
                await _until(pilot, lambda: isinstance(ui.screen, WidgetModal), 10, "Modal 'Nur Modal'")
                await pilot.press("escape")              # nicht beantworten
                await _until(pilot, lambda: len(ui.screen_stack) == 1, 5, "Modal zu")
                gone = next(w for w, e in b.widgets_json().items() if e["spec"]["title"] == "Schon zu")
                await _ui_chat(ui, pilot, call_query("widget_close", widget_id=gone))
                assert b.widgets_json()[gone]["closed"] is True
                await pilot.press("f2")
                await _until(pilot, lambda: "Board Zwei" in _side_text(ui) and "Schon zu" not in _side_text(ui),
                             20, "Widgets in der Seitenleiste")
                side = _side_text(ui)
                assert "Board Eins" in side and "Board Zwei" in side, side
                assert "Nur Modal" not in side and "Schon zu" not in side, side
        asyncio.run(go())


# ── 7. Antwort-Rundweg: Modal -> Backend -> widgets.json -> Verlauf -> Erzeuger ─
def test_7_antwort_rundweg_ueber_app():
    from textual.widgets import Checkbox, Input, OptionList
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.mods.isaa.extras.icli2.widgets import WidgetModal
    with Betrieb() as b:
        ui = ICli2(b.engine)

        async def go():
            async with ui.run_test(size=(140, 45)) as pilot:
                await _until(pilot, lambda: ui.ready, 60, "READY")
                ui.prompt.text = call_query("widget_show", **FORM)
                await pilot.press("enter")
                await _until(pilot, lambda: isinstance(ui.screen, WidgetModal), 60, "Modal aus dem Agenten-Lauf")
                scr = ui.screen
                await pilot.click(scr.query_one(Input))
                await pilot.press(*"Hamburg")
                await pilot.click(scr.query_one(Checkbox))
                scr.query_one(OptionList).focus()
                await pilot.press("enter")                    # erste Option: hoch
                await pilot.click(scr.query_one(Input))
                await pilot.press("enter")
                await _until(pilot, lambda: len(ui.screen_stack) == 1, 10, "Modal geschlossen")
                await _until(pilot, lambda: any(v.origin == "widget" and v.status != "running"
                                                for v in ui.runs.values()), 60, "Folgelauf des Erzeugers")
                await pilot.pause(0.5)
                from textual.widgets import Static
                return _chat_text(ui), [str(w.render()) for w in ui.chat.query(Static) if w.has_class("note")]

        chat, notes = asyncio.run(go())
        wid, entry = next(iter(b.widgets_json().items()))
        assert entry["values"] == {"ort": "Hamburg", "bahn": True, "prio": "h"}, entry
        line = "Reise planen – Ort: Hamburg, Mit der Bahn: True, Prioritaet: h"
        assert line in chat, chat[-1500:]
        # nur Zeilen, die die UI selbst schreibt (Antwortzeile, Lauf-Kopf); der Antworttext des Agenten
        # darf zitieren, was er bekommen hat (das Ersatz-LLM echot die Nachricht samt id/values)
        answer_lines = [l for l in notes if "Reise planen –" in l]
        assert answer_lines and not any(wid in l or "{" in l for l in answer_lines), answer_lines
        got = [p for p in b.llm.prompts() if "[Widget-Antwort]" in p]
        assert got and "Ort: Hamburg" in got[-1] and '"ort": "Hamburg"' in got[-1], got


# ── 8. Antwort erreicht den ruhenden Erzeuger (anderer Agent, andere Session) ──
def test_8_antwort_startet_ruhenden_erzeuger():
    from toolboxv2.mods.isaa.extras.icli2 import protocol as P
    with Betrieb() as b:
        b.ready()
        _register(b, "alpha")
        wid = _show_via_agent(b, "alpha", "s8", CHOICE, "t8")
        b.events.clear()
        b.engine.send(P.WIDGET_ANSWER, id=wid, values={"value": "gruen"})
        start = b.wait_event(lambda e: e.kind == P.RUN_START and e.data.get("origin") == "widget", 30,
                             "Lauf des Erzeugers")
        assert (start.data["agent"], start.data["session"]) == ("alpha", "s8"), start.data
        b.wait_event(lambda e: e.kind == P.RUN_END and e.run == start.run, 60, "Ende des Erzeuger-Laufs")
        prompt = next(r for r in b.llm.requests if "[Widget-Antwort]" in r["user"])
        assert "Farbe waehlen – gruen" in prompt["user"] and wid in prompt["user"]
        assert b.widgets_json()[wid]["values"] == {"value": "gruen"}


# ── 9. Persistenz: Neustart zeigt dieselben gepinnten Widgets ───────────────
def test_9_neustart_zeigt_gepinnte_widgets():
    from toolboxv2.mods.isaa.extras.icli2.app import ICli2
    from toolboxv2.tests.a_util import reset_app_singleton
    titles = ("Plan A", "Plan B", "Plan C")
    with Betrieb() as b:
        b.ready()
        for i, t in enumerate(titles):
            _show_via_agent(b, "self", "default", _pinned(t), f"t9-{i}")
        b.stop_engine()
        asyncio.run(b.app.a_exit())
        reset_app_singleton(force_exit=True)
        b.start_engine()                                 # neue App, gleicher Temp-Ordner
        ui = ICli2(b.engine)

        async def go():
            async with ui.run_test(size=(140, 45)) as pilot:
                await _until(pilot, lambda: ui.ready, 60, "READY nach Neustart")
                await pilot.press("f2")
                await _until(pilot, lambda: all(t in _side_text(ui) for t in titles), 20,
                             "drei gepinnte Widgets nach Neustart")
        asyncio.run(go())


# ── 10. Betriebstest: icli2 als eigener Prozess wie der Starter ─────────────
def test_10_icli2_prozess_widget_rundweg():
    script = Path(__file__).with_name("icli2_widget_prozess.py")
    with Betrieb.llm_only() as llm_env:
        env = {**os.environ, **llm_env}
        proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                              timeout=TIMEOUT - 10, cwd=env["TB_DATA_DIR"])
    out = proc.stdout + proc.stderr
    marks = [l for l in out.splitlines() if l.startswith("BETRIEB ")]
    assert "BETRIEB modal" in marks, out[-3000:]
    assert any(m.startswith("BETRIEB werte ") and '"ort": "Kiel"' in m for m in marks), marks
    assert "BETRIEB folgelauf done" in marks, marks
    assert "BETRIEB ende" in marks, marks
