# -*- coding: utf-8 -*-
"""Subcommand-Autocomplete fuer die icli2 - reine Vorschlagslogik, textual-frei.

Quellen der statischen Tokens (nur bewiesene Dispatcher-Argumente):
- /plugin  install|confirm|discard|remove|market|markets  (app.py plugin-Dispatch,
           Backend plugin_* + markets)
- /set     Schluessel aus backend.save_setting (T1: max_iterations,
           voice_input_device, voice_output_device, tts_provider, tts_voice)
- /model   target fast|complex|both an 2. Position (backend._do_model,
           Name an 1. Position ist frei)
- /job     fire|pause|remove <job> (app.py job-Dispatch -> backend._do_job);
           jedes andere 1. Token ist der Beginn einer Job-Definition (job_add)

Dynamische Werte (Agenten, Sessions, Jobs, Modelle, ...) liefert die Engine ueber
LIST <art>; DYNAMIC sagt, an welcher Position welche Liste gilt, suggest() verbindet beides.
"""
from __future__ import annotations

SETTINGS_KEYS: tuple[str, ...] = (
    "max_iterations",
    "voice_input_device",
    "voice_output_device",
    "tts_provider",
    "tts_voice",
)

# Befehl -> ((Position, Token-Vorschlaege), ...); Position 0 = 1. Token nach Befehl
SUBCOMMANDS: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] = {
    "/plugin": ((0, ("install", "confirm", "discard", "remove", "market", "markets")),),
    "/set": ((0, SETTINGS_KEYS),),
    "/model": ((1, ("fast", "complex", "both")),),
    "/job": ((0, ("fire", "pause", "remove")),),
}

# Befehl -> ((Position, Listen-Art fuer LIST), ...); Art = backend._rows_<art>
DYNAMIC: dict[str, tuple[tuple[int, str], ...]] = {
    "/agent": ((0, "agents"),),
    "/pack": ((0, "agents"),),
    "/session": ((0, "sessions"),),
    "/model": ((0, "models"),),
    "/feature": ((0, "features"),),
    "/skill": ((0, "skills"),),
    "/mcp": ((0, "mcp"),),
    "/job": ((1, "jobs"),),                      # nach fire|pause|remove
    "/plugin": ((1, "plugins"),),                # nach remove (Name eines installierten Plugins)
}
_DYNAMIC_AFTER = {"/job": {"fire", "pause", "remove"}, "/plugin": {"remove"}}
_SKIP_VALUES = {"*", "+", ""}                    # Sonderzeilen der Picker (alle, neu)



def suggest_args(text: str) -> list[str]:
    """Vorschlaege fuer das letzte Wort einer /-Befehlszeile.

    Nur statische Tokens an definierter Position; freie Argumente (Jobname,
    owner/repo, Modellname) erhalten bewusst keinen Vorschlag.
    Rueckgabe: vollstaendige Tokens mit Leerzeichen (zum Anhaengen).
    """
    if not text.startswith("/"):
        return []
    parts = text.split(" ")
    # trailing empty part = Cursor hinter Space; sonst Teilwort
    word = parts[-1] if parts[-1] else ""
    arg_index = len(parts) - 2          # 0-basiertes Argument nach dem Befehl
    rules = SUBCOMMANDS.get(parts[0], ())
    used = set(parts[1:-1])            # bereits getippte Argumente
    for pos, tokens in rules:
        if pos == arg_index:
            return sorted(t + " " for t in tokens
                          if t.startswith(word) and t not in used)
    return []


def _slot(text: str) -> tuple[str, int, str, list[str]] | None:
    """(Befehl, Argument-Position, angefangenes Wort, Teile) einer einzeiligen /-Zeile mit Argument."""
    if not text.startswith("/") or "\n" in text or " " not in text:
        return None
    parts = text.split(" ")
    return parts[0].lower(), len(parts) - 2, parts[-1], parts


def dynamic_kind(text: str) -> str | None:
    """Welche Liste der Engine fuer das aktuelle Argument gebraucht wird (oder None)."""
    slot = _slot(text)
    if slot is None:
        return None
    cmd, pos, _word, parts = slot
    after = _DYNAMIC_AFTER.get(cmd)
    if after is not None and (pos == 0 or parts[1] not in after):
        return None
    return next((kind for p, kind in DYNAMIC.get(cmd, ()) if p == pos), None)


def _more(cmd: str, pos: int) -> bool:
    """Folgt nach Position pos noch ein vorgeschlagenes Argument?"""
    return any(p == pos + 1 for p, _ in SUBCOMMANDS.get(cmd, ()) + DYNAMIC.get(cmd, ()))


def _fuzzy(needle: str, hay: str) -> bool:
    it = iter(hay.lower())
    return all(ch in it for ch in needle.lower())


def suggest(text: str, rows: list[dict] | None = None, limit: int = 8) -> list[tuple[str, str, str]]:
    """Vorschlaege fuer das aktuelle Argument: (einzufuegende Zeile, Anzeige, Hinweis).

    Feste Tokens aus SUBCOMMANDS, dazu `rows` (Antwort der Engine auf LIST dynamic_kind(text)).
    Treffer am Anfang von Wert oder Anzeige zuerst, dann Teilfolgen ('cdr' -> 'coder').
    Ein schon vollstaendig getippter Wert wird nicht noch einmal angeboten (Enter fuehrt aus).
    """
    slot = _slot(text)
    if slot is None:
        return []
    cmd, pos, word, parts = slot
    head = " ".join(parts[:-1]) + " "
    out = [(head + tok, tok.strip(), "") for tok in suggest_args(text) if tok.strip() != word]
    if rows and dynamic_kind(text):
        tail = " " if _more(cmd, pos) else ""
        low = word.lower()
        cand = [r for r in rows if r.get("value", "") not in _SKIP_VALUES and r["value"] != word]
        first = [r for r in cand if r["value"].lower().startswith(low) or r.get("label", "").lower().startswith(low)]
        rest = [r for r in cand if r not in first and (_fuzzy(word, r["value"]) or _fuzzy(word, r.get("label", "")))]
        out += [(head + r["value"] + tail, r.get("label") or r["value"], r.get("hint", ""))
                for r in first + rest]
    return out[:limit]


def is_exact(text: str, rows: list[dict] | None = None) -> bool:
    """Ist das letzte Wort schon ein gueltiger Wert? Dann fuehrt Enter aus statt zu ergaenzen
    ('/plugin market' ist fertig, obwohl 'markets' auch passt)."""
    slot = _slot(text)
    if slot is None or not slot[2]:
        return False
    word = slot[2]
    if word + " " in suggest_args(text):
        return True
    return bool(rows and dynamic_kind(text) and any(r.get("value") == word for r in rows))
