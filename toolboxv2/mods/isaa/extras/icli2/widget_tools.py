# -*- coding: utf-8 -*-
"""Widget-Tools fuer ALLE Agenten (kein self-Guard - Markin-Beschluss).

Erzeugt/verwaltet Widgets in der icli2 ueber die Backend-Registry
(Persistenz widgets.json, Antworten routen an den Erzeuger). Referenz-Struktur:
self_tool.py (SelfToolMixin-Muster), Registrierungsmuster icli_web
_icli_web_tool.py:register_icli_web_tools (add_tools-Dict, idempotent).
"""
from __future__ import annotations

from toolboxv2.mods.isaa.extras.icli2.widget_spec import validate_spec


def _uid(agent: str) -> str:
    import secrets
    return f"{agent}-{secrets.token_hex(4)}"


class WidgetToolsMixin:
    """Mixin auf IsaaBackend: Agent-Tools fuer Widget-Verwaltung."""

    async def widget_show(self, spec: dict) -> dict:
        agent = getattr(self, "agent", "")
        session = getattr(self, "session", "default")
        r = validate_spec(spec, creator=(agent, session))
        if not r["ok"]:
            return {"ok": False, "errors": r["errors"]}
        wid = _uid(agent)
        await self.widget_registry_add(wid, r["spec"], creator=(agent, session))
        try:                                       # UI anzeigen (falls Engine laeuft)
            engine = getattr(self, "engine", None)
            emit = getattr(engine, "emit_widget", None)
            if emit is not None:
                await emit("create", wid, r["spec"])
        except Exception as e:
            from toolboxv2 import get_logger
            get_logger().warning(f"widget_show emit: {e}")
        return {"ok": True, "id": wid, "mode": r["spec"].get("mode", "modal"),
                "text": f"Widget \"{spec.get('title', wid)}\" erstellt"}

    async def widget_update(self, widget_id: str, spec: dict) -> dict:
        agent = getattr(self, "agent", "")
        session = getattr(self, "session", "default")
        reg = self.widget_registry_load()
        entry = reg.get(widget_id)
        if entry is None:
            return {"ok": False, "error": f"unbekanntes Widget: {widget_id}"}
        if entry.get("creator") != [agent, session]:
            return {"ok": False, "error": "nur der Erzeuger darf aktualisieren"}
        r = validate_spec(spec, creator=(agent, session))
        if not r["ok"]:
            return {"ok": False, "errors": r["errors"]}
        entry["spec"] = r["spec"]
        entry["mode"] = r["spec"].get("mode", entry.get("mode", "modal"))
        self._widgets_save(reg)
        return {"ok": True, "id": widget_id, "text": f"Widget \"{r['spec'].get('title')}\" aktualisiert"}

    async def widget_close(self, widget_id: str) -> dict:
        agent = getattr(self, "agent", "")
        reg = self.widget_registry_load()
        entry = reg.get(widget_id)
        if entry is None:
            return {"ok": False, "error": f"unbekanntes Widget: {widget_id}"}
        if entry.get("creator") != [agent, getattr(self, "session", "default")]:
            return {"ok": False, "error": "nur der Erzeuger darf schliessen"}
        await self.widget_registry_close(widget_id)
        return {"ok": True, "text": f"Widget \"{widget_id}\" geschlossen"}

    async def widget_pin(self, widget_id: str, pinned: bool = True) -> dict:
        agent = getattr(self, "agent", "")
        reg = self.widget_registry_load()
        entry = reg.get(widget_id)
        if entry is None:
            return {"ok": False, "error": f"unbekanntes Widget: {widget_id}"}
        if entry.get("creator") != [agent, getattr(self, "session", "default")]:
            return {"ok": False, "error": "nur der Erzeuger darf pinnen"}
        entry["mode"] = "pinned" if pinned else "modal"
        self._widgets_save(reg)
        return {"ok": True, "id": widget_id, "mode": entry["mode"]}

    async def widget_state(self) -> dict:
        reg = self.widget_registry_load()
        panels = {wid: {"title": e["spec"].get("title", wid),
                        "mode": e.get("mode", "modal"),
                        "values": e.get("values", {}),
                        "closed": e.get("closed", False)}
                  for wid, e in reg.items()}
        return {"ok": True, "panels": panels}
