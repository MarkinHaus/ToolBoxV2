"""Test-Doubles: scriptbares LLM, Dict-Tools, fixer Decider.

ScriptedLLM unterstützt auch die optionale stream()-Methode – damit können
Tests das Streaming-Verhalten der Engine prüfen.
"""
from __future__ import annotations

import asyncio
import itertools
from typing import AsyncIterator, Callable

from .core import Call, Reply
from .store import Fts

_ids = itertools.count(1)


def call(_name: str, **args) -> Call:
    return Call(f"c{next(_ids)}", _name, args)


def reply(*calls: Call, content: str = "") -> Reply:
    return Reply(content, list(calls))


def is_context(messages: list[dict]) -> bool:
    return str(messages[-1].get("content", "")).startswith("CONTEXT STEP")


class ScriptedLLM:
    """Action-Aufrufe poppen aus `actions`; Kontext-Aufrufe gehen an `context(messages)`.
    Jeder Prompt landet in `.calls`.

    stream_chunks: Wenn gesetzt, gibt stream() diese Strings als Chunks zurück statt
    complete() zu nutzen.  Damit testen, dass die Engine Chunks live yieldet.
    """

    def __init__(
        self,
        actions=(),
        context: Callable = lambda m: Reply(),
        context_delay: float = 0,
        stream_chunks: list[list[str]] | None = None,
    ):
        self.actions = list(actions)
        self.context = context
        self.delay = context_delay
        self.calls: list[list[dict]] = []
        # stream_chunks[i] wird bei der i-ten Action als Streaming-Antwort geliefert.
        # None = kein Streaming (nutzt complete()).
        self._stream_chunks: list[list[str]] | None = (
            list(stream_chunks) if stream_chunks else None
        )
        self._stream_idx = 0

    async def complete(self, messages: list[dict], tools: list[dict]) -> Reply:
        self.calls.append(messages)
        if is_context(messages):
            await asyncio.sleep(self.delay)
            return self.context(messages)
        return self.actions.pop(0) if self.actions else Reply("(script empty)")

    async def stream(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncIterator[str]:
        """Streaming-Pfad: liefert Chunks aus stream_chunks[i].

        Wenn keine stream_chunks konfiguriert sind, wird complete() als Fallback
        genutzt (die Engine versucht stream() zuerst weil es existiert, aber ein
        leerer Iterator führt zu einem leeren Reply – daher muss der Caller
        beim ScriptedLLM explizit stream_chunks setzen).
        """
        self.calls.append(messages)
        if is_context(messages):
            # Kontext-Schritte streamen nicht – complete() wird gerufen
            # (Engine nutzt stream() nur für mode=="A")
            return

        if self._stream_chunks and self._stream_idx < len(self._stream_chunks):
            chunks = self._stream_chunks[self._stream_idx]
            self._stream_idx += 1
            for ch in chunks:
                yield ch
            return

        # Kein stream_chunks gesetzt: komplette Reply als einzelnen Chunk
        r = self.actions.pop(0) if self.actions else Reply("(script empty)")
        if r.content:
            yield r.content
        # Tool-Calls als JSON-Block anhängen damit _parse_streamed sie findet
        if r.calls:
            import json
            block = json.dumps(
                [{"id": c.id, "name": c.name, "args": c.args} for c in r.calls]
            )
            yield f"\n```json\n{block}\n```"


class NonStreamingLLM(ScriptedLLM):
    """Wie ScriptedLLM, aber ohne stream()-Methode – testet den complete()-Pfad."""

    # stream() einfach nicht definieren reicht nicht, weil Python Methoden erbt.
    # Deshalb überschreiben und AttributeError werfen.
    def stream(self, *args, **kwargs):  # type: ignore[override]
        raise AttributeError("stream not supported")


class DictTools:
    def __init__(
        self,
        funcs: dict[str, Callable],
        docs: dict[str, str] | None = None,
    ):
        self.funcs = funcs
        self.index = Fts()
        for name in funcs:
            self.index.add(name, f"{name} {(docs or {}).get(name, '')}")
        self.docs = docs or {}

    async def call(self, name: str, args: dict):
        out = self.funcs[name](**args)
        return await out if asyncio.iscoroutine(out) else out

    def search(self, query: str, k: int) -> list[tuple[str, str]]:
        if not query:
            return [(name, self.docs.get(name, "")) for name in self.funcs][:k]
        return [
            (name, self.docs.get(name, ""))
            for name, _ in self.index.search(query, k)
        ]

    def close(self) -> None:
        """P2.12: release the in-memory Fts SQLite handle (no ResourceWarning)."""
        close = getattr(self.index, "close", None)
        if callable(close):
            close()


class FixedDecider:
    def __init__(self, answer: Callable[[dict], object], delay: float = 0):
        self.answer = answer
        self.delay = delay

    async def decide(self, state: str, questions: list[dict]) -> dict:
        await asyncio.sleep(self.delay)
        return {q["key"]: self.answer(q) for q in questions}
