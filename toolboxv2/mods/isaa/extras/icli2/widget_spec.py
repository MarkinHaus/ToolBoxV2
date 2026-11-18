# -*- coding: utf-8 -*-
"""Widget-Spec-Validator fuer die icli2 - reine Funktion, keine IO.

Regelgrundlage (gekuerzt auf TUI-taugliche Templates, kein HTML/JS):
toolboxv2/mods/icli_web/_icli_web_tool.py:347-460 (_VALID_TEMPLATES,
_VALID_FIELD_TYPES, Feld-/Limit-Pruefungen). Wir uebernehmen Struktur
und Grenzen, nicht die Web-Templates.
"""
from __future__ import annotations

_VALID_TEMPLATES = ("form", "choice", "info", "note", "progress")
_VALID_FIELD_TYPES = ("text", "number", "textarea", "select", "checkbox")
_VALID_MODES = ("modal", "pinned", "inline")

MAX_FIELDS = 20
MAX_TITLE = 80
MAX_TEXT = 2000
MAX_LABEL = 80
MIN_OPTIONS, MAX_OPTIONS = 2, 12


def _norm_options(raw):
    options = []
    for o in raw:
        if isinstance(o, str):
            o = {"label": o, "value": o}
        if not isinstance(o, dict) or not o.get("label") or "value" not in o:
            return None
        options.append({"label": str(o["label"]), "value": o["value"]})
    return options


def validate_spec(spec: dict, *, creator: tuple[str, str]) -> dict:
    """Prueft + normalisiert einen Widget-Spec.

    creator: (agent_name, session) - beide nicht-leer, sonst Fehler.
    Rueckgabe: {"ok": bool, "errors": [..], "warnings": [..], "spec": dict|None}
    """
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(spec, dict):
        return {"ok": False, "errors": ["spec muss ein Objekt sein"], "warnings": [], "spec": None}

    if not creator or not creator[0] or not creator[1]:
        errors.append("creator (agent, session) fehlt oder ist leer")

    title = str(spec.get("title") or "")
    if not title:
        errors.append("title fehlt")
    elif len(title) > MAX_TITLE:
        errors.append(f"title > {MAX_TITLE} Zeichen")

    template = spec.get("template")
    if template not in _VALID_TEMPLATES:
        errors.append(f"template ung?ltig: {template!r} (erlaubt: form|choice|info)")

    mode = spec.get("mode", "modal")
    if mode not in _VALID_MODES:
        warnings.append(f"mode {mode!r} unbekannt -> fallback modal")
        mode = "modal"

    if template in ("note", "progress") and mode != "inline":
        errors.append(f"template {template!r} nur mit mode=inline")
    if mode == "inline" and template not in ("note", "progress"):
        errors.append("mode=inline nur mit passiven Templates note/progress (keine Interaktion im Chat)")

    norm: dict = {"title": title, "template": template, "mode": mode}

    if template == "form":
        raw_fields = spec.get("fields")
        if not isinstance(raw_fields, list) or not raw_fields:
            errors.append("form braucht fields (nicht-leere Liste)")
        elif len(raw_fields) > MAX_FIELDS:
            errors.append(f"> {MAX_FIELDS} Felder")
        else:
            fields = []
            for f in raw_fields:
                if not isinstance(f, dict):
                    errors.append("Feld muss ein Objekt sein")
                    continue
                name, label, ftype = f.get("name"), str(f.get("label") or ""), f.get("type")
                if not name:
                    errors.append("Feld ohne name")
                    continue
                if not label:
                    errors.append(f"Feld {name}: label fehlt")
                elif len(label) > MAX_LABEL:
                    errors.append(f"Feld {name}: label > {MAX_LABEL} Zeichen")
                if ftype not in _VALID_FIELD_TYPES:
                    errors.append(f"Feld {name}: Typ {ftype!r} ungueltig")
                    continue
                nf = {"name": name, "label": label, "type": ftype}
                if ftype == "select":
                    opts = _norm_options(f.get("options") or [])
                    if opts is None:
                        errors.append(f"Feld {name}: options ungueltig")
                    elif not (MIN_OPTIONS <= len(opts) <= MAX_OPTIONS):
                        errors.append(f"Feld {name}: select braucht {MIN_OPTIONS}-{MAX_OPTIONS} Optionen")
                    else:
                        nf["options"] = opts
                if ftype == "number":
                    lo, hi = f.get("min"), f.get("max")
                    if lo is not None and not isinstance(lo, (int, float)):
                        errors.append(f"Feld {name}: min nicht numerisch")
                    if hi is not None and not isinstance(hi, (int, float)):
                        errors.append(f"Feld {name}: max nicht numerisch")
                    if isinstance(lo, (int, float)):
                        nf["min"] = lo
                    if isinstance(hi, (int, float)):
                        nf["max"] = hi
                if f.get("default") is not None:
                    nf["default"] = f["default"]
                if f.get("required") is not None:
                    nf["required"] = bool(f["required"])
                fields.append(nf)
            norm["fields"] = fields

    elif template == "choice":
        opts = _norm_options(spec.get("options") or [])
        if opts is None:
            errors.append("options ungueltig")
        elif not (MIN_OPTIONS <= len(opts) <= MAX_OPTIONS + 1) or len(opts or []) > 12:
            errors.append(f"choice braucht {MIN_OPTIONS}-12 Optionen")
        else:
            norm["options"] = opts

    elif template in ("note", "progress"):
        text = spec.get("text")
        if not isinstance(text, str) or not text:
            errors.append("note/progress braucht text")
        elif len(text) > MAX_TEXT:
            errors.append(f"text > {MAX_TEXT} Zeichen")
        else:
            norm["text"] = text
        if template == "progress":
            pct = spec.get("progress", 0)
            if not isinstance(pct, (int, float)) or not (0 <= pct <= 100):
                errors.append("progress braucht Zahl 0-100")
            else:
                norm["progress"] = pct

    elif template == "info":
        text = spec.get("text")
        if not isinstance(text, str) or not text:
            errors.append("info braucht text")
        elif len(text) > MAX_TEXT:
            errors.append(f"text > {MAX_TEXT} Zeichen")
        else:
            norm["text"] = text

    ok = not errors
    return {"ok": ok, "errors": errors, "warnings": warnings, "spec": norm if ok else None}
