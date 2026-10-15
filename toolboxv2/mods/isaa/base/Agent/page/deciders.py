"""Deciders — typed, measurement-based decisions for the Page engine.

Both implement the ``Decider`` protocol from ``core.py``::

    async def decide(self, state: str, questions: list[dict]) -> dict[str, Any]

A question is ``{key, kind: noul|choice|score, q, options?}``.  The answer dict
maps ``key -> value`` and **omits** a key when there is no usable answer, so the
engine never invents a decision.  Coercion is strict and type-only — never a
natural-language heuristic:

  * ``noul``   -> ``bool`` or the key is omitted (``None`` means "no opinion")
  * ``choice`` -> one of ``options`` or the key is omitted
  * ``score``  -> ``float`` or the key is omitted

Nothing here imports toolboxv2; ``JevDecider`` uses only the standard library
(``urllib``), so this module stays a droppable, dependency-free unit.
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
from typing import Any, Callable


# ------------------------------------------------------------ coercion
def _coerce(kind: str, value: Any, options: list | None) -> Any:
    """Return a typed value for ``kind`` or ``None`` to omit the key.

    Pure type coercion — it never parses meaning out of free text.
    """
    if value is None:
        return None
    if kind == "noul":
        return bool(value)
    if kind == "choice":
        opts = options or []
        return value if value in opts else None
    if kind == "score":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _build(questions: list[dict], raw: Callable[[dict], Any]) -> dict[str, Any]:
    """Apply ``raw(question)`` to each question and keep only typed answers."""
    out: dict[str, Any] = {}
    for q in questions:
        v = _coerce(q.get("kind", "noul"), raw(q), q.get("options"))
        if v is not None:
            out[q["key"]] = v
    return out


# ------------------------------------------------------------ So1Decider (local)
class So1Decider:
    """Local decider: a scoring callable produces each answer.

    ``fn(state, question) -> value`` returns a raw value (``bool``/``str``/
    ``float``/``None``); :func:`_coerce` turns it into the typed answer for the
    question's ``kind``.  The callable may be sync or async.  Keeping the
    judgement in an injected callable means So1 carries no natural-language
    heuristics of its own — the host decides how a value is produced (model,
    ``ast`` metric, bm25 score, …).
    """

    def __init__(self, fn: Callable[[str, dict], Any]):
        self.fn = fn

    async def decide(self, state: str, questions: list[dict]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for q in questions:
            raw = self.fn(state, q)
            if asyncio.iscoroutine(raw):
                raw = await raw
            v = _coerce(q.get("kind", "noul"), raw, q.get("options"))
            if v is not None:
                out[q["key"]] = v
        return out


# ------------------------------------------------------------ JevDecider (HTTP)
class JevDecider:
    """Remote decider over the Jev wire format (OpenJev-compatible).

    Wire contract::

        POST <url>
        { "state": "<state>", "questions": [ {key, kind, q, options?}, ... ] }
        -> 200 { "answers": { "<key>": <value>, ... } }

    Values are coerced to the question's ``kind``; unknown/invalid values drop
    the key.  Network/parse failures propagate (the engine's ``_decide`` wraps
    the call in a timeout + try/except, so a backend problem simply yields no
    hint).  ``opener`` is injectable for testing (defaults to
    ``urllib.request.urlopen``).
    """

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 0.15,
        opener: Callable[..., Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def _call(self, state: str, questions: list[dict]) -> dict[str, Any]:
        body = json.dumps({"state": state, "questions": questions}).encode()
        req = urllib.request.Request(self.url, data=body, headers=self.headers, method="POST")
        with self.opener(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("answers", {}) or {}

    async def decide(self, state: str, questions: list[dict]) -> dict[str, Any]:
        # The blocking HTTP call runs in a worker thread so it never stalls the
        # event loop; the engine still enforces decider_timeout on top.
        answers = await asyncio.to_thread(self._call, state, questions)
        return _build(questions, lambda q: answers.get(q["key"]))
