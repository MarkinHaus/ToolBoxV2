"""Das Terminal gehoert waehrend der Laufzeit der TUI nur Textual.

Textual faengt print/sys.stderr waehrend der Laufzeit selbst ab. Logging-Handler, die schon VOR dem
Start angelegt wurden, halten aber noch den echten Terminal-Strom und schreiben an Textual vorbei
ins Bild (Bildfehler, Flackern, Neuzeichnen). log_to_file() lenkt genau diese Handler fuer die
Laufzeit in eine Datei um und stellt sie danach wieder her. Nur Standardbibliothek.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _terminal_streams() -> set[int]:
    return {id(s) for s in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__) if s is not None}


def _movable(handler: logging.Handler, terminal: set[int]) -> bool:
    """Nur echte StreamHandler mit eigenem Terminal-Strom. Ausgenommen: Datei-Handler und Handler,
    deren `stream` eine Eigenschaft ohne Setter ist (z. B. logging._StderrHandler/lastResort: liefert
    immer das aktuelle sys.stderr, das Textual waehrend der Laufzeit ohnehin abfaengt)."""
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


@contextmanager
def log_to_file(path: Path) -> Iterator[list[logging.Handler]]:
    """Terminal-Handler auf `path` umlenken; liefert die umgelenkten Handler (fuer Tests)."""
    terminal = _terminal_streams()
    moved: list[tuple[logging.StreamHandler, object]] = []
    stream = None
    try:
        for logger in _all_loggers():
            for handler in list(logger.handlers):
                if handler in (h for h, _ in moved) or not _movable(handler, terminal):
                    continue
                if stream is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = open(path, "a", encoding="utf-8")
                try:
                    moved.append((handler, handler.setStream(stream)))
                except Exception:                        # unbekannter Handler: lieber ins Terminal als gar nicht starten
                    continue
    except Exception:
        pass                                             # Umlenken ist nur Komfort - der Start geht immer vor
    try:
        yield [h for h, _ in moved]
    finally:
        for handler, old in moved:
            try:
                handler.setStream(old)
            except Exception:
                pass
        if stream is not None:
            stream.close()
