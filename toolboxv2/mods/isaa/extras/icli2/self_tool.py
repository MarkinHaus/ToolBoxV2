# -*- coding: utf-8 -*-
"""T7 Self-Tool: der `self`-Agent steuert eigene icli2-Settings und die Konfiguration anderer Agenten.

Mixin fuer IsaaBackend (backend.py:58) - analog zu bridges.py/connect.py/features.py.
Guard: alle self_* Tools liefern None, wenn self.agent != "self" (Test 1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: erlaubte eigene Settings: key -> (cast, lo, hi)
_SELF_SETTING_KEYS: dict[str, tuple] = {
    "max_iterations": ("int", 1, 500),
    "fast_llm_model": ("str", None, None),
    "complex_llm_model": ("str", None, None),
    "immediate_offload_ratio": ("float", 0.0, 1.0),
    "heavy_hitter_min_tokens": ("int", 1, 1_000_000),
}

#: erlaubte Keys in agent.json anderer Agenten (Whitelist, Test 5)
_AGENT_CONFIG_KEYS: tuple[str, ...] = (
    "fast_llm_model",
    "complex_llm_model",
    "immediate_offload_ratio",
    "heavy_hitter_min_tokens",
)


class SelfToolMixin:
    """Bringt self_set_setting/self_get_setting/self_get_agent_config/
    self_set_agent_config/self_list_agents auf den IsaaBackend."""

    def _self_ok(self) -> bool:
        """Aufrufer ist 'self': beim Tool-Aufruf der aufrufende Agent (TOOL_CALLER), sonst die UI-Auswahl."""
        from .widget_tools import caller_of
        return caller_of(self)[0] == "self"

    # -- eigene Settings -------------------------------------------------
    def self_set_setting(self, key: str, value: Any) -> dict[str, Any] | None:
        """Setting setzen (Whitelist + Range), persistiert via save_setting (backend.py:338)."""
        if not self._self_ok():
            return None
        spec = _SELF_SETTING_KEYS.get(key)
        if spec is None:
            raise ValueError(f"setting '{key}' ist nicht erlaubt")
        cast, lo, hi = spec
        try:
            if cast == "int":
                value = int(str(value).strip())
            elif cast == "float":
                value = float(str(value).strip())
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key}: '{value}' ist kein {cast}: {e}") from e
        if lo is not None and value < lo:
            raise ValueError(f"{key}: {value} < {lo}")
        if hi is not None and value > hi:
            raise ValueError(f"{key}: {value} > {hi}")
        self.save_setting(key, value)          # merge-atomar nach settings.json
        if hasattr(self, key):
            setattr(self, key, value)          # z.B. self.max_iterations
        return {"text": f"self.{key} = {value!r}", "value": value}

    def self_get_setting(self, key: str) -> dict[str, Any] | None:
        if not self._self_ok():
            return None
        cfg = getattr(self, "_settings", None) or {}
        value = cfg.get(key, getattr(self, key, None))
        return {"text": f"self.{key} = {value!r}", "value": value}

    # -- Konfiguration anderer Agenten ------------------------------------
    def _agent_json_path(self, name: str) -> Path:
        return Path(self.app.data_dir) / "Agents" / name / "agent.json"

    def self_get_agent_config(self, name: str) -> dict[str, Any] | None:
        """agent.json eines Agenten lesen; fehlt sie -> Warn-Dict (Test 4)."""
        if not self._self_ok():
            return None
        path = self._agent_json_path(name)
        if not path.is_file():
            return {
                "level": "warn",
                "text": f"Agent '{name}' hat keine agent.json ({path}) - noch nicht gebaut?",
                "name": name,
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"level": "warn", "text": f"agent.json von '{name}' unlesbar: {e}", "name": name}
        return {"text": f"config von '{name}'", "name": name, "data": data}

    def self_set_agent_config(self, name: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Nur Whitelist-Keys schreiben; illegale Keys -> ValueError VOR jedem Schreiben (atomar, Test 5)."""
        if not self._self_ok():
            return None
        if not isinstance(updates, dict) or not updates:
            raise ValueError("updates muessen ein nicht-leeres dict sein")
        bad = [k for k in updates if k not in _AGENT_CONFIG_KEYS]
        if bad:
            raise ValueError(f"keys {bad} sind nicht erlaubt (whitelist: {list(_AGENT_CONFIG_KEYS)})")
        self._persist_config(name, dict(updates))   # backend.py:500 - merge-atomar
        return {"text": f"agent.json von '{name}' aktualisiert: {sorted(updates)}",
                "name": name, "updated": sorted(updates)}

    # -- Uebersicht --------------------------------------------------------
    def self_list_agents(self) -> dict[str, Any] | None:
        """Alle gebauten Agenten (Ordner MIT agent.json) auflisten (Test 6)."""
        if not self._self_ok():
            return None
        base = Path(self.app.data_dir) / "Agents"
        agents = sorted(
            p.parent.name for p in base.glob("*/agent.json") if p.is_file()
        )
        return {"text": f"{len(agents)} Agenten: {', '.join(agents)}",
                "agents": agents, "count": len(agents)}
