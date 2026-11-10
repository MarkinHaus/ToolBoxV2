"""Cactus Needle 3 adapter — a LOCAL, on-device tool-calling model.

Unlike the other adapters in this package, Needle is not an HTTP endpoint.
It is Cactus Compute's `cactus-needle` Python package: a sub-1 MB inference
engine plus the `needle3.cact` weights, both fetched once from Hugging Face
and cached on first use. The model only emits structured tool calls (a
grammar-constrained JSON object), never free text — exactly what this router's
`CompletionResult.tool_calls` represents.

Design goals (minimal-invasive, "just name the model"):
  * Naming a model ``needle/needle3`` routes here via the ``needle`` prefix
    registered in ``adapters/setup.py``.
  * Auto-install: ``cactus-needle`` is ``pip install``-ed lazily on first use
    if it is not importable. Nothing is imported at module load time, so the
    rest of ISAA never pays for it.
  * Auto-setup: the engine + weights download themselves on the first
    ``needle.Needle(...)`` construction; there is nothing else to build.

The response contract (see https://cactuscompute.com/blog/needle-python-docs):
    {"type": "call"|"respond", "function_calls": [{"name", "arguments"}],
     "suppressed_calls": [...], "reasoning": str, "confidence": float|None, ...}

The model emits no free-text answer, but its short ``reasoning`` (the derivation
of each argument / why it refused) is surfaced as ``CompletionResult.content``
so downstream context keeps it; the full response is always in ``raw``.

Because a Needle agent is bound to ONE toolset ("a new toolset is a new
instance") and the native engine is process-global, not thread-safe and
thread-affine, ALL Needle work runs on ONE dedicated worker thread fed by a
queue (see ``_NeedleWorker``). Every call — from any thread or event loop —
is enqueued and executed sequentially on that single thread, which keeps the
native engine bound to one thread and never re-entered concurrently.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, AsyncIterator

from ..adapter import ProviderAdapter
from ..types import (
    CompletionResult, StreamChunk, EmbedResult, ToolCallData, UsageData,
)
from ..stream_metrics import StreamMetrics

if TYPE_CHECKING:  # pragma: no cover
    import aiohttp


class _NeedleWorker:
    """One long-lived thread that owns the Needle engine and serves calls via a
    queue. The native engine is process-global, not thread-safe and binds to the
    thread that first initialises it — so every call must run on this one thread.

    A single worker is shared process-wide (all adapter instances use it) so the
    agent loop, skill-match and narrator can never re-enter the engine.
    """

    def __init__(self, import_needle):
        self._import_needle = import_needle
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="needle-worker",
                                        daemon=True)
        self._thread.start()
        # engine state, owned exclusively by the worker thread
        self._agent = None
        self._key = None

    def _run(self):
        while True:
            job = self._q.get()
            if job is None:
                break
            needle_tools, system, weights, query, max_new_tokens, fut = job
            try:
                agent = self._get_agent(needle_tools, system, weights)
                try:
                    agent.reset()  # independent turn; keep the engine/toolset loaded
                except AttributeError:
                    pass
                resp = agent.complete(query, max_new_tokens)
                fut.get_loop().call_soon_threadsafe(fut.set_result, resp)
            except Exception as e:  # deliver the error to the awaiting coroutine
                fut.get_loop().call_soon_threadsafe(fut.set_exception, e)

    def _get_agent(self, needle_tools, system, weights):
        """Build the single Needle instance ONCE and reuse it forever.

        The native engine is one-per-process: a second ``needle_init`` (for a
        new toolset) corrupts the resident engine and raises ``needle_init
        failed``. So the adapter is IDEMPOTENT — it binds exactly one instance
        lazily on the first call and reuses it for every subsequent call,
        regardless of later toolset/system changes. Needle's own tool-retrieval
        head selects the relevant tools per query from the catalogue it was
        built with, so a stable instance still serves varying requests; and a
        request whose tool isn't in that catalogue simply yields the empty call
        (handled by the caller). Runs ONLY on the single worker thread.
        """
        if self._agent is not None:
            return self._agent
        needle = self._import_needle()
        kwargs: dict = {}
        if needle_tools is not None:
            kwargs["tools"] = needle_tools
        if system:
            kwargs["system"] = system
        if weights:
            kwargs["weights"] = weights
        self._agent = needle.Needle(**kwargs)
        self._key = _toolset_fingerprint(needle_tools, system)
        return self._agent

    async def complete(self, needle_tools, system, weights, query, max_new_tokens):
        """Enqueue a call and await its result (resolved from the worker thread)."""
        import asyncio
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._q.put((needle_tools, system, weights, query, max_new_tokens, fut))
        return await fut


def _toolset_fingerprint(needle_tools: list | None, system: str) -> str:
    """Stable key for the Needle instance cache (toolset + system facts)."""
    try:
        return json.dumps([needle_tools, system], sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr((needle_tools, system))


class CactusComputeAdapter(ProviderAdapter):
    """Local on-device Needle 3 model via the ``cactus-needle`` package.

    Usage (through the router):
        router.register("needle", CactusComputeAdapter())
        await router.complete("needle/needle3", messages, tools=tools)
    """

    #: pip requirement installed on demand.
    PACKAGE = "cactus-needle"

    # One process-wide worker thread owns the (thread-affine, non-thread-safe)
    # native engine. Shared by every adapter instance; created lazily.
    _WORKER: "_NeedleWorker | None" = None
    _WORKER_LOCK = threading.Lock()

    def __init__(self, base_url: str = "local://needle", default_headers: dict | None = None):
        # base_url is unused (local engine) but kept for ProviderAdapter API.
        super().__init__(base_url=base_url, default_headers=default_headers)
        self._needle_mod = None                 # cached `needle` module

    def _worker(self) -> "_NeedleWorker":
        cls = CactusComputeAdapter
        if cls._WORKER is None:
            with cls._WORKER_LOCK:
                if cls._WORKER is None:
                    cls._WORKER = _NeedleWorker(self._import_needle)
        return cls._WORKER

    # ------------------------------------------------------------------
    # Lazy import + auto-install
    # ------------------------------------------------------------------
    def _raw_import(self):
        """Import the `needle` module (no install). Separated for testing."""
        import importlib
        return importlib.import_module("needle")

    def _pip_install(self):
        """Install the cactus-needle package into the current interpreter."""
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", self.PACKAGE]
        )

    def _import_needle(self):
        """Return the `needle` module, installing cactus-needle on first miss."""
        if self._needle_mod is not None:
            return self._needle_mod
        try:
            self._needle_mod = self._raw_import()
        except ImportError:
            self._pip_install()
            self._needle_mod = self._raw_import()
        return self._needle_mod

    # ------------------------------------------------------------------
    # tools (OpenAI schema) -> needle schema
    # ------------------------------------------------------------------
    @staticmethod
    def _to_needle_tools(tools: list | None) -> list | None:
        """Convert OpenAI tool definitions to Needle's flat JSON schema.

        Needle accepts a flat ``{"name", "description", "parameters"}`` dict.
        OpenAI wraps it in ``{"type": "function", "function": {...}}`` — unwrap.
        Already-flat dicts pass through. ``triggers`` (Needle-specific) are kept.
        """
        if not tools:
            return None
        out: list = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                entry = {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                }
                if "triggers" in fn:
                    entry["triggers"] = fn["triggers"]
            else:
                entry = dict(t)
            out.append(entry)
        return out or None

    # ------------------------------------------------------------------
    # messages -> query / system facts
    # ------------------------------------------------------------------
    @staticmethod
    def _content_to_text(content) -> str:
        """Flatten OpenAI message content (str or list of parts) to text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, str):
                    parts.append(p)
            return "\n".join(parts)
        return str(content)

    @classmethod
    def _extract_query(cls, messages: list) -> str:
        """The query is the last user message (Needle is single-turn per call)."""
        for msg in reversed(messages or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return cls._content_to_text(msg.get("content"))
        # Fallback: last message of any role
        if messages:
            return cls._content_to_text(messages[-1].get("content"))
        return ""

    @classmethod
    def _extract_system(cls, messages: list) -> str:
        """Concatenate system messages into Needle's ``system`` facts string."""
        facts = [
            cls._content_to_text(m.get("content")).strip()
            for m in (messages or [])
            if isinstance(m, dict) and m.get("role") == "system"
        ]
        return "; ".join(f for f in facts if f)

    # ------------------------------------------------------------------
    # Response -> CompletionResult
    # ------------------------------------------------------------------
    @staticmethod
    def _map_response(resp: dict, model: str) -> CompletionResult:
        """Map a Needle response dict to a CompletionResult."""
        resp = resp or {}
        calls = resp.get("function_calls") or []
        tool_calls = [
            ToolCallData(
                id=f"needle_{i}",
                name=c.get("name", ""),
                arguments=c.get("arguments", {}) or {},
            )
            for i, c in enumerate(calls)
            if isinstance(c, dict)
        ]
        has_calls = bool(tool_calls)
        finish_reason = "tool_calls" if has_calls else "stop"
        # Needle emits no free-text answer, but it does return a short
        # `reasoning` (its derivation of each argument / why it refused). Pass
        # that through as `content` so downstream context keeps it. Empty
        # reasoning stays None (a bare refusal); callers must handle the empty
        # case either way. The full response is always in `raw`.
        reasoning = resp.get("reasoning")
        content = reasoning if isinstance(reasoning, str) and reasoning.strip() else None
        return CompletionResult(
            content=content,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=UsageData(),  # the local engine reports no token usage here
            model=model,
            raw=resp,
        )

    # ------------------------------------------------------------------
    # ProviderAdapter API
    # ------------------------------------------------------------------
    async def complete(self, session: 'aiohttp.ClientSession', api_key: str,
                       model: str, messages: list, tools: list | None,
                       **kwargs) -> CompletionResult:
        needle_tools = self._to_needle_tools(tools)
        system = self._extract_system(messages)
        query = self._extract_query(messages)
        weights = kwargs.get("weights")
        max_new_tokens = int(kwargs.get("max_tokens") or 512)

        # A Needle call with NO tools can only ever return the empty call (it has
        # nothing to route to). Answer it WITHOUT touching the native engine:
        # binding the engine with 0 tools and then re-initialising it for a real
        # (tooled) call is the sequence that triggers `needle_init failed`.
        # Short-circuiting keeps the engine's FIRST bind the real tooled one.
        if not needle_tools:
            return CompletionResult(
                content=None, tool_calls=None, finish_reason="stop",
                usage=UsageData(), model=model,
                raw={"type": "respond", "function_calls": [],
                     "reasoning": "no tools declared, returning the empty call"},
            )

        # All Needle work runs on the single worker thread (see _NeedleWorker):
        # enqueue + await. This keeps the thread-affine, non-thread-safe native
        # engine owned by exactly one thread, no matter which caller (React loop,
        # skill-match, narrator) or event loop issues the request.
        resp = await self._worker().complete(
            needle_tools, system, weights, query, max_new_tokens)
        return self._map_response(resp, model)

    async def stream(self, session: 'aiohttp.ClientSession', api_key: str,
                     model: str, messages: list, tools: list | None,
                     metrics: StreamMetrics | None = None,
                     **kwargs) -> AsyncIterator[StreamChunk]:
        """Needle has no token stream; emit the full result as one chunk."""
        import time
        if metrics:
            metrics.t_start = time.perf_counter()
            metrics.t_first_token = metrics.t_start

        result = await self.complete(session, api_key, model, messages, tools, **kwargs)

        # Pass Needle's reasoning through as a content chunk (context), then the
        # grammar-guaranteed tool calls.
        if result.content:
            yield StreamChunk(content=result.content)
        if result.tool_calls:
            from ..types import ToolCallDelta
            for i, tc in enumerate(result.tool_calls):
                yield StreamChunk(
                    tool_call_delta=ToolCallDelta(
                        index=i,
                        id=tc.id,
                        name=tc.name,
                        arguments_delta=json.dumps(tc.arguments),
                    ),
                )

        if metrics:
            metrics.t_end = time.perf_counter()
            metrics.chunk_count += 1

        yield StreamChunk(finish_reason=result.finish_reason, usage=result.usage)

    async def embed(self, session: 'aiohttp.ClientSession', api_key: str,
                    model: str, texts: list[str], **kwargs) -> EmbedResult:
        # Needle exposes `agent.embed(text)` for tool-retrieval, but that is a
        # per-toolset retrieval head, not a general embedding endpoint.
        raise NotImplementedError(
            "CactusComputeAdapter does not support general embeddings; "
            "Needle's embed() is a per-toolset retrieval head."
        )
