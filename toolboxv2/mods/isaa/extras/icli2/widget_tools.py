# -*- coding: utf-8 -*-
"""Widget-Tools fuer ALLE Agenten (kein self-Guard - Markin-Beschluss) + Bindung der icli2-Tools.

Erzeugt/verwaltet Widgets in der icli2 ueber die Backend-Registry
(Persistenz widgets.json, Antworten routen an den Erzeuger). Referenz-Struktur:
self_tool.py (SelfToolMixin-Muster), Registrierungsmuster icli_web
_icli_web_tool.py:register_icli_web_tools (add_tools-Dict, idempotent).

Bindung: register_icli2_tools(agent, backend, name) haengt die Tools an den Tool-Manager eines
Agenten. Jedes Tool kennt den Agenten, an den es gebunden ist; waehrend des Aufrufs steht
(agent, session) in TOOL_CALLER - daraus lesen die Mixins den Erzeuger bzw. Aufrufer
(ein Ort fuer den Dispatch, auch fuer die self_*-Tools).
"""
from __future__ import annotations

import contextvars
from typing import Any

from toolboxv2.mods.isaa.extras.icli2.widget_spec import validate_spec

#: (agent, session) des Agenten, der gerade ein icli2-Tool aufruft; None = Aufruf aus der UI/Backend
TOOL_CALLER: "contextvars.ContextVar[tuple[str, str] | None]" = contextvars.ContextVar(
    "icli2_tool_caller", default=None)

WIDGET_TOOL_NAMES = ("widget_show", "widget_update", "widget_close", "widget_pin", "widget_state")
SELF_TOOL_NAMES = ("self_set_setting", "self_get_setting", "self_get_agent_config",
                   "self_set_agent_config", "self_list_agents", "self_agent_start", "self_agent_stop",
                   "self_agent_pause", "self_agent_resume", "self_agent_status", "self_agent_obs")


def _uid(agent: str) -> str:
    import secrets
    return f"{agent}-{secrets.token_hex(4)}"


def caller_of(backend: Any) -> tuple[str, str]:
    """Aufrufender (agent, session): aus dem Tool-Aufruf, sonst die aktive Auswahl der UI."""
    return TOOL_CALLER.get() or (getattr(backend, "agent", ""), getattr(backend, "session", "default"))


def widget_event(widget_id: str, entry: dict, *, open_modal: bool = False) -> dict:
    """Payload fuer WIDGET_CREATE/UPDATE (protocol.py): voller Spec + Erzeuger."""
    spec = entry.get("spec") or {}
    return {"id": widget_id, "title": spec.get("title", widget_id), "template": spec.get("template"),
            "mode": entry.get("mode", spec.get("mode", "modal")), "spec": spec,
            "creator": list(entry.get("creator") or []), "values": entry.get("values") or {},
            "open": open_modal}


class WidgetToolsMixin:
    """Mixin auf IsaaBackend: Agent-Tools fuer Widget-Verwaltung."""

    def attach_widget_events(self, emit: Any) -> None:
        """Von der Engine gesetzt: emit(kind, data) schickt ein Widget-Event an die UI."""
        self._widget_emit = emit

    def _emit_widget(self, kind: str, data: dict) -> None:
        emit = getattr(self, "_widget_emit", None)
        if emit is not None:
            emit(kind, data)

    async def widget_show(self, spec: dict) -> dict:
        agent, session = caller_of(self)
        r = validate_spec(spec, creator=(agent, session))
        if not r["ok"]:
            return {"ok": False, "errors": r["errors"]}
        wid = _uid(agent)
        await self.widget_registry_add(wid, r["spec"], creator=(agent, session))
        self._emit_widget("widget_create", widget_event(wid, self.widget_registry_load()[wid]))
        return {"ok": True, "id": wid, "mode": r["spec"].get("mode", "modal"),
                "text": f"Widget \"{spec.get('title', wid)}\" erstellt",
                **({"warnings": r["warnings"]} if r["warnings"] else {})}

    def _own_entry(self, widget_id: str, verb: str) -> tuple[dict, dict | None, str]:
        reg = self.widget_registry_load()
        entry = reg.get(widget_id)
        if entry is None:
            return reg, None, f"unbekanntes Widget: {widget_id}"
        if entry.get("creator") != list(caller_of(self)):
            return reg, None, f"nur der Erzeuger darf {verb}"
        return reg, entry, ""

    async def widget_update(self, widget_id: str, spec: dict) -> dict:
        reg, entry, error = self._own_entry(widget_id, "aktualisieren")
        if entry is None:
            return {"ok": False, "error": error}
        r = validate_spec(spec, creator=caller_of(self))
        if not r["ok"]:
            return {"ok": False, "errors": r["errors"]}
        entry["spec"] = r["spec"]
        entry["mode"] = r["spec"].get("mode", entry.get("mode", "modal"))
        self._widgets_save(reg)
        self._emit_widget("widget_update", widget_event(widget_id, entry))
        return {"ok": True, "id": widget_id, "text": f"Widget \"{r['spec'].get('title')}\" aktualisiert",
                **({"warnings": r["warnings"]} if r["warnings"] else {})}

    async def widget_close(self, widget_id: str) -> dict:
        _, entry, error = self._own_entry(widget_id, "schliessen")
        if entry is None:
            return {"ok": False, "error": error}
        await self.widget_registry_close(widget_id)
        self._emit_widget("widget_close", {"id": widget_id})
        return {"ok": True, "text": f"Widget \"{widget_id}\" geschlossen"}

    async def widget_pin(self, widget_id: str, pinned: bool = True) -> dict:
        reg, entry, error = self._own_entry(widget_id, "pinnen")
        if entry is None:
            return {"ok": False, "error": error}
        entry["mode"] = "pinned" if pinned else "modal"
        self._widgets_save(reg)
        self._emit_widget("widget_update", widget_event(widget_id, entry))
        return {"ok": True, "id": widget_id, "mode": entry["mode"]}

    async def widget_state(self) -> dict:
        reg = self.widget_registry_load()
        panels = {wid: {"title": e["spec"].get("title", wid),
                        "mode": e.get("mode", "modal"),
                        "values": e.get("values", {}),
                        "closed": e.get("closed", False)}
                  for wid, e in reg.items()}
        return {"ok": True, "panels": panels}


# ── Bindung an den Tool-Manager eines Agenten ────────────────────────────────
_SHOW_DOC = (
    "Zeigt dem Nutzer in der icli2 ein interaktives Widget und liefert {ok, id}. Die Antwort des Nutzers "
    "kommt spaeter als neue Nachricht '[Widget-Antwort] ...' mit id und values zu dir zurueck.\n"
    "template: form (fields: [{name, label, type: text|number|textarea|select|checkbox, options?, default?, "
    "required?}]) | choice (options: [{label, value}] 2-12) | info (text) | note (text, nur inline) | progress (text + progress 0-100, nur inline) | chart (values: 2-50 Zahlen, chart_type line|bar, nur inline) | mermaid (text = Mermaid-Quellcode, nur inline).\n"
    "mode: modal (Overlay, Standard) | pinned (Seitenleiste F2) | inline (passive Karte im Chat-Verlauf: "
    "nur template note|progress|chart|mermaid; form/choice/info nur modal|pinned).")


def _session_of(agent: Any, name: str) -> str:
    """Session des laufenden Agenten: Lauf-Kontext der icli2 (SHELL_CONTEXT), sonst agent.active_session."""
    try:
        from toolboxv2.mods.isaa.module import SHELL_CONTEXT
        ctx = SHELL_CONTEXT.get() or {}
    except Exception:
        ctx = {}
    if ctx.get("agent") == name and ctx.get("session"):
        return str(ctx["session"])
    return str(getattr(agent, "active_session", None) or "default")


def _bound(agent: Any, name: str, method: Any, *, self_only: bool = False):
    """Tool-Funktion fuer genau diesen Agenten: setzt TOOL_CALLER, prueft bei self-Tools den Aufrufer."""
    import functools
    import inspect

    @functools.wraps(method)
    async def tool(*args, **kwargs):
        if self_only and name != "self":
            raise PermissionError(f"{method.__name__} ist nur fuer den Agenten 'self' (Aufrufer: {name})")
        token = TOOL_CALLER.set((name, _session_of(agent, name)))
        try:
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            TOOL_CALLER.reset(token)

    return tool


def register_icli2_tools(agent: Any, backend: Any, name: str) -> list[str]:
    """Widget-Tools fuer jeden Agenten, self_*-Tools nur fuer 'self'. Idempotent (Marke am Agenten)."""
    if getattr(agent, "_icli2_tools_backend", None) is backend:
        return []

    def show(title: str, template: str, mode: str = "modal", fields: list | None = None,
             options: list | None = None, text: str | None = None,
             values: list | None = None, chart_type: str | None = None) -> Any:
        spec = {"title": title, "template": template, "mode": mode}
        for key, value in (("fields", fields), ("options", options), ("text", text),
                           ("values", values), ("chart_type", chart_type)):
            if value is not None:
                spec[key] = value
        return backend.widget_show(spec)

    def update(widget_id: str, title: str, template: str, mode: str = "modal", fields: list | None = None,
               options: list | None = None, text: str | None = None,
               values: list | None = None, chart_type: str | None = None) -> Any:
        spec = {"title": title, "template": template, "mode": mode}
        for key, value in (("fields", fields), ("options", options), ("text", text),
                           ("values", values), ("chart_type", chart_type)):
            if value is not None:
                spec[key] = value
        return backend.widget_update(widget_id, spec)

    show.__name__, update.__name__ = "widget_show", "widget_update"
    tools = [
        (show, "widget_show", _SHOW_DOC, False),
        (update, "widget_update", "Widget aendern (nur der Erzeuger). Gleiche Felder wie widget_show.", False),
        (backend.widget_close, "widget_close", "Widget dauerhaft schliessen (nur der Erzeuger).", False),
        (backend.widget_pin, "widget_pin", "Widget in die Seitenleiste pinnen (pinned=False: wieder modal).",
         False),
        (backend.widget_state, "widget_state", "Alle Widgets mit Titel, Modus, Werten und closed-Status.", False),
    ]
    if name == "self":
        for tool_name in SELF_TOOL_NAMES:
            method = getattr(backend, tool_name, None)
            if method is not None:
                doc = (method.__doc__ or tool_name).strip().splitlines()[0]
                tools.append((method, tool_name, f"icli2 (nur self): {doc}", True))
    agent.add_tools([{"tool_func": _bound(agent, name, func, self_only=self_only), "name": tool_name,
                      "description": doc, "category": ["icli2", "ui"], "flags": {"icli2": True}}
                     for func, tool_name, doc, self_only in tools])
    agent._icli2_tools_backend = backend
    return [t[1] for t in tools]
