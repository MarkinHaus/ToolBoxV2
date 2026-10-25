"""V3 drop-in adapter: expose a :class:`~.core.PageEngine` behind the public
API of ``toolboxv2.mods.isaa.base.Agent.execution_engine.ExecutionEngine`` (V3).

PLAN rule: ``core.py`` imports nothing from toolboxv2.  Integration code lives in
``isaa.py`` and here.  This module itself imports nothing from toolboxv2 at
import time either, so it can be unit-tested isolated (iso_pkg method); the
only toolboxv2 touch points are lazy and optional (``get_interrupted_runs``).

What V3 callers (icli / ``FlowAgent.a_run`` / ``a_stream``) rely on:

    execute(query, session_id, ..., get_ctx=False) -> str | (str, ctx)
    execute_stream(query, session_id, ...)         -> (gen_fn, ctx)
    resume(execution_id, content="", stream=False) -> str | (gen_fn, ctx)
    pause(execution_id)   -> ctx | None
    cancel(execution_id)  -> bool
    list_executions()     -> list[dict]
    get_execution(id)     -> ctx | None
    get_interrupted_runs()-> list
    get_stats()           -> dict
    ctx.working_history, ctx.session_id, ctx.status, ctx.run_id

The stream generator yields the same chunk *types* as V3 so an unchanged client
keeps working:  status, iteration_start, content, reasoning, tool_start,
tool_result, final_answer, paused, cancelled, done.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from .core import Config, PageEngine


# ================================================================== context shim
@dataclass
class ExecutionContext:
    """Minimal V3-compatible context.

    ``working_history`` is a live view onto the page engine's ``context_view``
    (what V3 called working_history) — a system prompt, the rendered page, and
    the raw tail.  ``run_id`` / ``session_id`` / ``status`` match V3 semantics.
    """

    engine: "PageEngine"
    session_id: str = ""
    query: str = ""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "running"          # running | paused | completed | cancelled | max_iterations
    last_response: str = ""
    resume_count: int = 0
    current_iteration: int = 0
    tools_used: list = field(default_factory=list)
    # V3 context_overview analytics fields (page engine has no skill/dynamic-tool
    # pre-pass, so these stay empty — context_overview reads them defensively).
    matched_skills: list = field(default_factory=list)
    dynamic_tools: list = field(default_factory=list)
    context_config: Any = None
    # Cooperative control: the stream loop checks these each iteration.
    _pause: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _cancel: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def working_history(self) -> list[dict]:
        try:
            return self.engine.context_view(self.session_id)
        except Exception:
            return []

    def get_dynamic_tool_names(self) -> list[str]:
        return []


# ================================================================== adapter
class PageExecutionEngine:
    """Adapter presenting a :class:`PageEngine` as a V3 ExecutionEngine."""

    def __init__(self, engine: PageEngine, agent: Any | None = None):
        self.engine = engine
        self.agent = agent
        self._active_executions: dict[str, ExecutionContext] = {}
        self._session_last_run: dict[str, str] = {}

    # ponytail: delegates to agent.session_manager.skills_manager; real fix is
    # guarding icli.py unguarded callers. Remove when icli is fixed.
    # Restore pre-patch state: copy compat_v3.py.orig-backup over compat_v3.py.
    @property
    def skills_manager(self):
        return getattr(getattr(self.agent, "session_manager", None), "skills_manager", None)

    # ------------------------------------------------------------ helpers
    def _new_ctx(self, query: str, session_id: str) -> ExecutionContext:
        ctx = ExecutionContext(self.engine, session_id=session_id, query=query)
        self._active_executions[ctx.run_id] = ctx
        self._session_last_run[session_id] = ctx.run_id
        return ctx

    @property
    def _agent_name(self) -> str:
        agent = self.agent
        try:
            return agent.amd.name  # type: ignore[union-attr]
        except Exception:
            return "page"

    def _enrich(self, chunk: dict, ctx: ExecutionContext) -> dict:
        """Add the V3 enrichment fields icli/a_stream expect on every chunk."""
        chunk.setdefault("agent", self._agent_name)
        chunk.setdefault("iter", ctx.current_iteration)
        chunk.setdefault("iteration", ctx.current_iteration)
        chunk.setdefault("is_sub", False)
        chunk.setdefault("run_id", ctx.run_id)
        try:
            s = self.engine.session(ctx.session_id)
            chunk.setdefault("tokens_used", self.engine.prompt_tokens(s))
        except Exception:
            chunk.setdefault("tokens_used", 0)
        chunk.setdefault("tokens_max", self.engine.cfg.page_cap)
        chunk.setdefault("node_name", chunk.get("mode", "page"))
        chunk.setdefault("phase", chunk.get("mode", "action"))
        return chunk

    def _map_chunk(self, chunk: dict, ctx: ExecutionContext) -> dict | None:
        """Translate a page event into a V3-shaped chunk (or drop it)."""
        t = chunk.get("type")
        # Page emits content_chunk per token; V3 clients listen for "content".
        if t == "content_chunk":
            chunk = {"type": "content", "chunk": chunk.get("chunk", "")}
        elif t == "content":
            chunk = {"type": "content", "chunk": chunk.get("chunk", "")}
        if chunk.get("type") == "iteration_start":
            ctx.current_iteration = chunk.get("iteration", ctx.current_iteration)
        return self._enrich(dict(chunk), ctx)

    # ------------------------------------------------------------ stream core
    async def _drive(
        self, ctx: ExecutionContext, query: str, media: tuple
    ) -> AsyncIterator[dict]:
        """Run the page engine stream, mapping chunks + honoring pause/cancel.

        The underlying ``PageEngine.stream`` runs as a task so pause/cancel can
        interrupt a *running* loop: we stop pulling and cancel the task, then
        surface a V3 ``paused`` / ``cancelled`` + ``done`` chunk.
        """
        agen = self.engine.stream(query, ctx.session_id, media)
        agen_task: asyncio.Task | None = None
        final_answer = ""
        success = True
        try:
            while True:
                if ctx._cancel.is_set():
                    ctx.status = "cancelled"
                    final_answer = "Execution cancelled."
                    success = False
                    yield self._enrich(
                        {"type": "cancelled", "run_id": ctx.run_id,
                         "answer": final_answer}, ctx)
                    break
                if ctx._pause.is_set():
                    ctx.status = "paused"
                    final_answer = (
                        f"Run paused at iteration {ctx.current_iteration}. "
                        f"Use resume() to continue."
                    )
                    success = False
                    yield self._enrich(
                        {"type": "paused", "run_id": ctx.run_id,
                         "answer": final_answer}, ctx)
                    break

                agen_task = asyncio.ensure_future(agen.__anext__())
                try:
                    raw = await agen_task
                except StopAsyncIteration:
                    break

                mapped = self._map_chunk(raw, ctx)
                if mapped is None:
                    continue
                if mapped["type"] == "final_answer":
                    final_answer = mapped.get("answer", "")
                    success = mapped.get("success", True)
                elif mapped["type"] == "done":
                    final_answer = mapped.get("final_answer", final_answer)
                    success = mapped.get("success", success)
                yield mapped
                if raw.get("type") == "done":
                    # The page stream finished on its own.
                    ctx.status = "completed"
                    ctx.last_response = final_answer
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # Cooperative cancellation / client disconnect: terminal, re-raise.
            if ctx.status == "running":
                ctx.status = "cancelled"
            raise
        except Exception as e:
            # ANY stream error must leave a TERMINAL status so the next request
            # never wrongly enters the sidecar (which keys on status=="running").
            ctx.status = "failed"
            final_answer = f"Execution failed: {e}"
            success = False
            yield self._enrich(
                {"type": "error", "error": str(e), "run_id": ctx.run_id}, ctx)
            yield self._enrich(
                {"type": "done", "success": False, "final_answer": final_answer}, ctx)
            return
        finally:
            # Close the underlying async generator (cancels any in-flight await).
            if agen_task is not None and not agen_task.done():
                agen_task.cancel()
                try:
                    await agen_task
                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                    pass
            await agen.aclose()

        # Reached only on pause/cancel break: emit a terminal done chunk so
        # clients that wait for "done" never hang.
        ctx.last_response = final_answer
        yield self._enrich(
            {"type": "done", "success": success, "final_answer": final_answer}, ctx)

    # ------------------------------------------------------------ public API
    async def execute(
        self,
        query: str,
        session_id: str = "default",
        max_iterations: int | None = None,
        ctx: ExecutionContext | None = None,
        get_ctx: bool = False,
        media: tuple = (),
        **_: Any,
    ) -> "str | tuple[str, ExecutionContext]":
        ctx = ctx or self._new_ctx(query, session_id)
        if max_iterations:
            self.engine.cfg.max_steps = max_iterations
        final = ""
        async for chunk in self._drive(ctx, query, tuple(media)):
            if chunk["type"] == "final_answer":
                final = chunk.get("answer", final)
            elif chunk["type"] == "done":
                final = chunk.get("final_answer", final) or final
        if ctx.status == "running":
            ctx.status = "completed"
        ctx.last_response = final
        return (final, ctx) if get_ctx else final

    async def execute_stream(
        self,
        query: str,
        session_id: str = "default",
        max_iterations: int | None = None,
        ctx: ExecutionContext | None = None,
        media: tuple = (),
        **_: Any,
    ) -> "tuple[Callable[[ExecutionContext], AsyncIterator[dict]], ExecutionContext]":
        ctx = ctx or self._new_ctx(query, session_id)
        if max_iterations:
            self.engine.cfg.max_steps = max_iterations
        media_t = tuple(media)

        async def gen(run_ctx: ExecutionContext) -> AsyncIterator[dict]:
            yield self._enrich(
                {"type": "status", "status_msg": "start",
                 "run_id": run_ctx.run_id}, run_ctx)
            async for chunk in self._drive(run_ctx, query, media_t):
                yield chunk

        return gen, ctx

    async def resume(
        self,
        execution_id: str,
        max_iterations: int | None = None,
        content: str = "",
        stream: bool = False,
        **_: Any,
    ) -> "str | tuple[Callable, ExecutionContext]":
        ctx = self._active_executions.get(execution_id)
        if ctx is None:
            return f"Error: Execution {execution_id} is not resumable (not found)"
        ctx.status = "running"
        ctx.resume_count += 1
        ctx._pause.clear()
        ctx._cancel.clear()
        query = content or ctx.query
        ctx.query = query
        if stream:
            return await self.execute_stream(
                query, ctx.session_id, max_iterations=max_iterations, ctx=ctx)
        return await self.execute(
            query, ctx.session_id, max_iterations=max_iterations, ctx=ctx)

    async def pause(self, execution_id: str) -> ExecutionContext | None:
        ctx = self._active_executions.get(execution_id)
        if ctx is None:
            return None
        ctx._pause.set()
        ctx.status = "paused"
        return ctx

    async def cancel(self, execution_id: str) -> bool:
        ctx = self._active_executions.get(execution_id)
        if ctx is None:
            return False
        ctx._cancel.set()
        ctx.status = "cancelled"
        if self._session_last_run.get(ctx.session_id) == execution_id:
            self._session_last_run.pop(ctx.session_id, None)
        return True

    def clear_session(self, session_id: str) -> None:
        """V3-compatible clear: reset the PAGE context for ``session_id`` and
        drop this session's run bookkeeping. Delegates the actual page reset to
        :meth:`PageEngine.clear_session`."""
        self.engine.clear_session(session_id)
        rid = self._session_last_run.pop(session_id, None)
        if rid:
            self._active_executions.pop(rid, None)
        # drop any other active ctx bound to this session
        for rid in [r for r, c in self._active_executions.items()
                    if getattr(c, "session_id", None) == session_id]:
            self._active_executions.pop(rid, None)

    def list_executions(self) -> list[dict]:
        return [
            {
                "run_id": rid,
                "session_id": ctx.session_id,
                "query": (ctx.query[:50] + "...") if len(ctx.query) > 50 else ctx.query,
                "status": ctx.status,
                "iteration": ctx.current_iteration,
                "tools_used": len(ctx.tools_used),
                "dynamic_tools": ctx.get_dynamic_tool_names(),
            }
            for rid, ctx in self._active_executions.items()
        ]

    def get_execution(self, execution_id: str) -> ExecutionContext | None:
        return self._active_executions.get(execution_id)

    def get_interrupted_runs(self) -> list:
        """Cold-resumable runs.  V3 reads these from the obs layer; without an
        agent/obs we fall back to the page engine's on-disk session catalog."""
        agent = self.agent
        obs = getattr(agent, "obs", None) if agent is not None else None
        if obs is not None:
            try:
                return obs.get_interrupted_runs()
            except Exception:
                return []
        return []

    def get_stats(self) -> dict:
        return {
            "engine": "page",
            "agent_name": self._agent_name,
            "active_executions": len(self._active_executions),
            "sessions": list(self.engine.sessions.keys()),
        }

    # ------------------------------------------------------------ V3 context_overview surface
    # ``FlowAgent.context_overview`` (the V3 /context X-Ray) calls these private
    # engine methods directly.  On a page-engine agent we delegate to the page
    # engine's native ``context_breakdown`` (P1.8), whose ``xray`` already maps
    # page+tail into the exact V3 categories — so /context works unchanged and
    # shows the REAL page+tail state instead of crashing / "History=0".
    @staticmethod
    def _ctx_session(ctx: Any) -> str:
        return getattr(ctx, "session_id", None) or "default"

    def _peek_page(self, sid: str) -> dict:
        """Return the committed page dict for a session WITHOUT acquiring the OS
        session lock.  ``/context`` is read-only and may run while the session is
        already open (held by the live run), so we must never re-lock here.

        Order: (1) the already-loaded in-memory session, (2) the on-disk
        ``state.json`` snapshot, (3) an empty page.  Returns the page dict only.
        """
        s = self.engine.sessions.get(sid)
        if s is not None:
            return s.page
        try:
            import json as _json
            p = self.engine.root / sid / "state.json"
            if p.exists():
                state = _json.loads(p.read_text("utf-8"))
                page = state.get("page")
                if page:
                    return page
        except Exception:
            pass
        from .core import _empty_page
        return _empty_page()

    def _build_static_system_prompt(self, ctx: Any, session: Any = None) -> str:
        """The stable prompt the page agent always sees: static prompt + P1 plan
        + P3 helpers (the P1.8 "System Prompt Total" composition)."""
        page = self._peek_page(self._ctx_session(ctx))
        parts = [self.engine.static]
        plan = "\n".join(self.engine._box_line(b) for b in page.get("boxes", []))
        if plan:
            parts.append(f"## P1 Task plan\n{plan}")
        p3 = page.get("slots", {}).get("P3", "")
        if p3:
            parts.append(f"## P3 Helpers\n{p3}")
        return "\n\n".join(parts)

    def _build_dynamic_system_prompt(self, ctx: Any) -> str:
        """The volatile page body: P2 context, P5 chat, P4 code space, PS stubs."""
        page = self._peek_page(self._ctx_session(ctx))
        slots = page.get("slots", {})
        parts = []
        for k in ("P2", "P5", "P4", "PS"):
            body = slots.get(k, "")
            if body:
                parts.append(f"## {k}\n{body}")
        return "\n\n".join(parts)

    def _get_tool_definitions(self, ctx: Any = None) -> list[dict]:
        """The page tool schema (identical in every mode), litellm-shaped."""
        from .core import TOOL_SCHEMA
        return list(TOOL_SCHEMA)

    def _get_max_context_tokens(self, ctx: Any = None) -> int:
        return self.engine.cfg.page_cap

    def _calculate_context_load(self, ctx: Any = None) -> int:
        """Effective token load = the X-Ray Total (System + Tools + Working + Media).

        Uses ``context_breakdown`` when the session is already loaded (no lock);
        otherwise estimates from the on-disk page snapshot so ``/context`` works
        even while a run holds the session lock."""
        sid = self._ctx_session(ctx)
        try:
            if sid in self.engine.sessions:
                return self.engine.context_breakdown(sid)["xray"]["Total"]
            # Lock-free estimate from the on-disk page + static prompt.
            page = self._peek_page(sid)
            tok = self.engine.tokens
            total = tok(self.engine.static)
            for body in page.get("slots", {}).values():
                total += tok(body or "")
            import json as _json
            from .core import TOOL_SCHEMA
            total += tok(_json.dumps(TOOL_SCHEMA, default=str))
            return total
        except Exception:
            return 0

    def _calculate_tool_relevance(self, ctx: Any = None, query: str = "") -> None:
        """Best-effort no-op: the page engine has no tool-relevance pre-pass
        (its tool schema is static).  Present only so context_overview's
        try/except pre-step does not explode."""
        return None

    def _preload_skill_tools(self, ctx: Any = None, query: str = "") -> None:
        """Best-effort no-op: skills are loaded via ``activate_skill`` / prep
        hooks, not a context_overview pre-pass."""
        return None

    def context_breakdown(self, session: str | None = None) -> dict:
        """Direct passthrough to the native page breakdown (CLI /page context)."""
        return self.engine.context_breakdown(session or "default")

    async def close(self) -> None:
        for ctx in self._active_executions.values():
            ctx._cancel.set()
        self._active_executions.clear()
        self._session_last_run.clear()
        await self.engine.close()

    async def wait_all_pending_finalizes(self) -> None:
        """V3-compatible shutdown hook (``FlowAgent.close`` calls this).

        The page engine's "pending finalize" is the per-session background fold
        (``Session.bg``) that may run after ``final_answer``.  Await every
        outstanding one so shutdown is clean; swallow cancellation/errors so
        close never raises.
        """
        import contextlib
        tasks = [
            s.bg
            for s in getattr(self.engine, "sessions", {}).values()
            if getattr(s, "bg", None) is not None and not s.bg.done()
        ]
        for t in tasks:
            with contextlib.suppress(Exception):
                await t
