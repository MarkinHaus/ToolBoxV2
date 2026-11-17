"""Nachrichten zwischen UI und Engine.

Bewusst nur JSON-faehige Daten: so kann die Engine spaeter ohne Aenderung
der UI in einen eigenen Prozess wandern (vgl. icli_web/bridge.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Commands: UI -> Engine ───────────────────────────────────────────────────
CHAT = "chat"          # data: query, agent, session[, tag]
STOP = "stop"          # data: run
PAUSE = "pause"        # data: run
LIST = "list"            # data: what (agents|models|sessions|mounts|jobs)[, to=side]
DO = "do"                # data: action + Argumente (siehe backend.do)
SHUTDOWN = "shutdown"

# ── Events: Engine -> UI ─────────────────────────────────────────────────────
READY = "ready"            # data: agent, session, model, complex_model, mounts[, history]
RUN_START = "run_start"    # data: agent, session, query, tag, origin (chat|job)
TEXT = "text"              # data: text            (gebuendelte Content-Deltas)
THINK = "think"            # data: text            (gebuendeltes Reasoning)
TOOL_START = "tool_start"  # data: id, name, summary, reason, detail
TOOL_END = "tool_end"      # data: id, name, ok, summary, detail, secs
STATUS = "status"          # data: msg, iter, max_iter, tokens, tokens_max
RUN_END = "run_end"        # data: status (done|failed|stopped|paused), answer, error
ROWS = "rows"              # data: what, title, rows=[{value, label, hint}]
NOTICE = "notice"          # data: level (info|warn|error), text
BYE = "bye"                # Engine ist sauber beendet


@dataclass(slots=True)
class Command:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Event:
    kind: str
    run: str = ""
    data: dict[str, Any] = field(default_factory=dict)
