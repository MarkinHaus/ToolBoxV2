"""Voruebergehende Fehler erkennen und wiederholen.

Verbindungsabbrueche (aiohttp ServerDisconnectedError bei wiederverwendeter Keep-Alive-Verbindung,
Connection reset, Timeout) und Server-Stoerungen (HTTP 5xx, 408, 425, 529) verschwinden meist beim
naechsten Versuch. Der Router wiederholt sie deshalb mit kurzem Backoff, bevor er auf ein
Fallback-Modell wechselt oder den Fehler weitergibt. Rate-Limits (429) bleiben Sache des
Rate-Limiters (Warten/Fallback), Anfragefehler (4xx) werden nie wiederholt.
"""
from __future__ import annotations

import asyncio
import os
import random

TRANSIENT_STATUS = {408, 425, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
_TRANSIENT_TEXT = (
    "server disconnected", "serverdisconnected", "connection reset", "connection aborted",
    "connection closed", "remote end closed", "remotedisconnected", "remoteprotocolerror",
    "peer closed", "incomplete chunked", "incompleteread", "payload is not completed",
    "transferencodingerror", "eof occurred", "unexpected eof", "broken pipe",
    "cannot connect to host", "clientconnectorerror", "temporarily unavailable", "timed out",
    "readtimeout", "connecttimeout", "bad gateway", "service unavailable", "gateway timeout",
    "apiconnectionerror", "internalservererror", "overloaded",
)


def _transient_types() -> tuple[type, ...]:
    types: list[type] = [asyncio.TimeoutError, TimeoutError, ConnectionError]
    try:
        import aiohttp
        types += [aiohttp.ClientConnectionError, aiohttp.ClientPayloadError]
    except Exception:
        pass
    try:
        import httpx
        types.append(httpx.TransportError)
    except Exception:
        pass
    return tuple(types)


_TYPES = _transient_types()


def is_transient(error: BaseException) -> bool:
    """True fuer Fehler, die ein erneuter Versuch wahrscheinlich behebt (auch wenn eingepackt)."""
    seen = 0
    current: BaseException | None = error
    while current is not None and seen < 6:
        status = getattr(current, "status_code", None)
        if isinstance(status, int) and status:
            return status in TRANSIENT_STATUS            # HTTP-Status entscheidet (429/4xx: nein)
        if isinstance(current, _TYPES):
            return True
        text = f"{type(current).__name__}: {current}".lower()
        if any(part in text for part in _TRANSIENT_TEXT):
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    return False


def max_retries() -> int:
    try:
        return max(0, int(os.getenv("ISAA_LLM_RETRIES", "3")))
    except ValueError:
        return 3


def backoff(attempt: int) -> float:
    """1 s, 2 s, 4 s, … (hoechstens 10 s) plus etwas Zufall, damit parallele Laeufe sich verteilen."""
    return min(10.0, 2.0 ** (attempt - 1)) + random.uniform(0, 0.4)
