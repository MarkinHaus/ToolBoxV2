"""Tool-Argumente und -Ergebnisse lesbar machen. Reine Funktionen, laeuft in der Engine.

Ziel: nie rohes JSON zeigen. Hat ein Aufruf eine Begruendung, steht nur die
in der Kopfzeile; sonst der wichtigste Parameter als "schluessel using wert".
"""
from __future__ import annotations

import ast
import json
from typing import Any

REASON_KEYS = ("reason", "reasoning", "thought", "why", "purpose", "intent",
               "explanation", "rationale", "goal", "description")
# gleiche Prioritaet wie ZenRenderer._on_tool_start
PRIMARY_KEYS = ("command", "cmd", "path", "query", "url", "filename", "file", "task",
                "pattern", "code", "name", "tools", "category")
LINE = 72


def parse(value: Any) -> Any:
    """String -> Struktur, wenn es eine ist (JSON, dann Python-Literal)."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text[:1] in "{[":
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(text)
            except Exception:
                continue
    return value


def _one_line(value: Any, limit: int = LINE) -> str:
    if isinstance(value, dict):
        text = ", ".join(f"{k} {_one_line(v, 24)}" for k, v in value.items())
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_one_line(v, 24) for v in value)
    else:
        text = " ".join(str(value).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def reason_of(args: Any) -> str:
    if isinstance(args, dict):
        for key in REASON_KEYS:
            if isinstance(args.get(key), str) and args[key].strip():
                return _one_line(args[key])
    return ""


def summary(args: Any) -> str:
    """Kopfzeile eines Tool-Aufrufs."""
    args = parse(args)
    reason = reason_of(args)
    if reason:
        return reason
    if not isinstance(args, dict):
        return _one_line(args) if args not in (None, "", [], {}) else ""
    keys = [k for k in PRIMARY_KEYS if k in args] or list(args)
    if not keys:
        return ""
    first = keys[0]
    rest = len(args) - 1
    text = f"{first} using {_one_line(args[first], LINE - len(first) - 12)}"
    return text + (f"  +{rest}" if rest else "")


def detail(value: Any, indent: int = 0) -> str:
    """Mehrzeilige Ansicht ohne Klammern und Anfuehrungszeichen."""
    value = parse(value)
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return f"{pad}(leer)"
        width = min(max(len(str(k)) for k in value), 18)
        lines = []
        for key, item in value.items():
            item = parse(item)
            if isinstance(item, (dict, list, tuple)) and item:
                lines.append(f"{pad}{key}")
                lines.append(detail(item, indent + 1))
            elif isinstance(item, str) and "\n" in item.strip():
                lines.append(f"{pad}{key}")
                lines.extend(f"{pad}  {row}" for row in item.strip().splitlines())
            else:
                lines.append(f"{pad}{str(key):<{width}}  {_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{pad}(leer)"
        lines = []
        for item in value:
            item = parse(item)
            if isinstance(item, (dict, list, tuple)) and item:
                body = detail(item, indent + 1).lstrip()
                lines.append(f"{pad}– {body}")
            else:
                lines.append(f"{pad}– {_scalar(item)}")
        return "\n".join(lines)
    text = "" if value is None else str(value).strip()
    return "\n".join(f"{pad}{row}" for row in text.splitlines()) if text else f"{pad}(leer)"


def _scalar(value: Any) -> str:
    if value is None:
        return "–"
    if value is True:
        return "ja"
    if value is False:
        return "nein"
    if value in ("", [], {}):
        return "(leer)"
    return str(value)


def result_summary(result: Any) -> str:
    """Kurzinfo zum Ergebnis, wenn kein Tool-spezifischer Helfer greift."""
    result = parse(result)
    if isinstance(result, dict):
        if result.get("success", True) is False:
            return _one_line(result.get("error", "fehlgeschlagen"), 48)
        for key in ("message", "stdout", "output", "result", "content"):
            if isinstance(result.get(key), str) and result[key].strip():
                return _one_line(result[key].strip().splitlines()[0], 48)
        return f"{len(result)} Felder"
    if isinstance(result, (list, tuple)):
        return f"{len(result)} Einträge"
    text = str(result or "").strip()
    return _one_line(text.splitlines()[0], 48) if text else ""


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… (+{len(text) - limit} Zeichen)"
