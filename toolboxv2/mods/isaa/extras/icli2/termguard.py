"""Das Terminal gehoert waehrend der Laufzeit der TUI nur Textual.

Textual faengt print/sys.stderr waehrend der Laufzeit selbst ab. Logging-Handler, die schon VOR dem
Start angelegt wurden, halten aber noch den echten Terminal-Strom und schreiben an Textual vorbei
ins Bild (Bildfehler, Flackern, Neuzeichnen). quiet_terminal() schaltet genau diese Handler fuer die
Laufzeit stumm. Kein eigenes Log: alle Meldungen landen weiter in den uebrigen Handlern des
ToolBox-Loggers (get_logger(): Log-Datei, MobileDB, ...).
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Iterator


def _terminal_streams() -> set[int]:
    return {id(s) for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) if s is not None}


def _terminal(handler: logging.Handler, terminal: set[int]) -> bool:
    """Echte StreamHandler auf dem Terminal (Datei-Handler nie). logging.lastResort und Handler,
    deren `stream` eine Eigenschaft ohne Setter ist, liefern das aktuelle sys.stderr - das faengt
    Textual waehrend der Laufzeit ohnehin ab."""
    if not isinstance(handler, logging.StreamHandler) or isinstance(handler, logging.FileHandler):
        return False
    attr = getattr(type(handler), "stream", None)
    if isinstance(attr, property) and attr.fset is None:
        return False
    return id(getattr(handler, "stream", None)) in terminal


def _all_loggers() -> list[logging.Logger]:
    found = [logging.getLogger()]
    found += [lg for lg in logging.Logger.manager.loggerDict.values() if isinstance(lg, logging.Logger)]
    return found


def _mute(record: logging.LogRecord) -> bool:
    return False


@contextmanager
def quiet_terminal() -> Iterator[list[logging.Handler]]:
    """Terminal-Handler stummschalten; liefert die stummgeschalteten Handler (fuer Tests)."""
    terminal = _terminal_streams()
    muted: list[logging.Handler] = []
    try:
        for logger in _all_loggers():
            for handler in list(logger.handlers):
                if handler not in muted and _terminal(handler, terminal):
                    handler.addFilter(_mute)
                    muted.append(handler)
    except Exception:
        pass                                             # Stummschalten ist nur Komfort - der Start geht vor
    try:
        yield muted
    finally:
        for handler in muted:
            handler.removeFilter(_mute)


def log_path() -> str:
    """Datei des ToolBox-Loggers (fuer Hinweise in der UI), sonst ''."""
    try:
        from toolboxv2 import get_logger
        for handler in get_logger().handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
    except Exception:
        pass
    return ""
