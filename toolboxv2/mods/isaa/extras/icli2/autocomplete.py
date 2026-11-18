# -*- coding: utf-8 -*-
"""Subcommand-Autocomplete fuer die icli2 - reine Vorschlagslogik, textual-frei.

Quellen der statischen Tokens (nur bewiesene Dispatcher-Argumente):
- /plugin  install|confirm|discard|remove|market|markets  (app.py plugin-Dispatch,
           Backend plugin_* + markets)
- /set     Schluessel aus backend.save_setting (T1: max_iterations,
           voice_input_device, voice_output_device, tts_provider, tts_voice)
- /model   target fast|complex|both an 2. Position (backend._do_model,
           Name an 1. Position ist frei)
- /job     fire|pause existieren, aber 1. Token ist der freie Jobname
           (backend._do_job) -> kein statischer Vorschlag an 1. Position
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
}


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
