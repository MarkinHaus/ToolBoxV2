"""Engine: fuehrt alles aus, was dauern oder blockieren kann.

Laeuft in einem eigenen Thread mit eigenem asyncio-Loop. Ein blockierender
Tool-Call haelt hoechstens diesen Loop an, nie die UI.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Protocol

from . import protocol as P

FLUSH_SECONDS = 0.04      # Content-Deltas werden auf ~25 Events/s gebuendelt
RESULT_LIMIT = 4000       # Zeichen pro Tool-Ergebnis, die zur UI gehen
ARGS_LIMIT = 2000


class Backend(Protocol):
    """Was die Engine von der Agenten-Seite braucht (siehe backend.py)."""

    async def boot(self) -> dict[str, Any]: ...
    async def agents(self) -> list[dict[str, Any]]: ...
    async def use(self, agent: str) -> dict[str, Any]: ...
    async def pack(self, agent: str) -> str: ...
    def stream(self, agent: str, session: str, query: str) -> AsyncIterator[dict]: ...
    async def is_running(self, agent: str, session: str) -> bool: ...
    async def stop(self, agent: str, session: str) -> None: ...
    async def pause(self, agent: str, session: str) -> bool: ...
    async def close(self) -> None: ...


def _clip(value: Any, limit: int) -> str:
    if isinstance(value, (dict, list)):
        try:
            value = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except Exception:
            value = str(value)
    text = "" if value is None else str(value)
    if len(text) > limit:
        text = text[:limit] + f"\n… (+{len(text) - limit} Zeichen)"
    return text


def _succeeded(result: Any) -> bool:
    """Tools melden Erfolg im JSON-Ergebnis ({"success": bool}), wie in der alten icli."""
    if isinstance(result, str) and result.lstrip().startswith("{"):
        try:
            result = json.loads(result)
        except Exception:
            return True
    return result.get("success", True) is not False if isinstance(result, dict) else True


class _Run:
    __slots__ = ("id", "agent", "session", "task", "buf", "kind", "timer", "ended")

    def __init__(self, run_id: str, agent: str, session: str):
        self.id, self.agent, self.session = run_id, agent, session
        self.task: asyncio.Task | None = None
        self.buf: list[str] = []
        self.kind = P.TEXT
        self.timer: asyncio.TimerHandle | None = None
        self.ended = ""          # von aussen gesetzter Endstatus (stopped/paused)


class Engine:
    """Fassade fuer die UI. Alle oeffentlichen Methoden sind thread-sicher
    und kehren sofort zurueck."""

    def __init__(self, backend: Backend):
        self._backend = backend
        self._events: deque[P.Event] = deque()          # append/popleft sind atomar
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inbox: asyncio.Queue[P.Command] | None = None
        self._runs: dict[str, _Run] = {}
        self._counter = 0
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._main, name="icli2-engine", daemon=True)
        self.beat = time.monotonic()                    # Herzschlag des Engine-Loops

    # ── UI-Seite ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._thread.start()
        self._started.wait()

    def send(self, kind: str, **data: Any) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._inbox.put_nowait, P.Command(kind, data))
        except RuntimeError:
            pass                                         # Loop wurde gerade beendet

    def poll(self, limit: int = 200) -> list[P.Event]:
        """Bis zu `limit` Events abholen. Blockiert nie."""
        out: list[P.Event] = []
        events = self._events
        while events and len(out) < limit:
            out.append(events.popleft())
        return out

    def lag(self) -> float:
        """Sekunden seit dem letzten Herzschlag (>1 = Engine-Loop blockiert)."""
        return time.monotonic() - self.beat

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout)

    # ── Engine-Thread ────────────────────────────────────────────────────────
    def _emit(self, kind: str, run: str = "", **data: Any) -> None:
        self._events.append(P.Event(kind, run, data))

    def _main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._inbox = asyncio.Queue()
        self._started.set()
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            finally:
                self._loop.close()
                self._emit(P.BYE)

    async def _heartbeat(self) -> None:
        while True:
            self.beat = time.monotonic()
            await asyncio.sleep(0.25)

    async def _serve(self) -> None:
        heart = asyncio.create_task(self._heartbeat())
        boot = asyncio.create_task(self._boot())
        try:
            while True:
                cmd = await self._inbox.get()
                if cmd.kind == P.SHUTDOWN:
                    break
                try:
                    await self._dispatch(cmd, boot)
                except Exception as e:                   # ein Command darf die Engine nie toeten
                    self._emit(P.NOTICE, level="error", text=f"{cmd.kind}: {type(e).__name__}: {e}")
        finally:
            boot.cancel()
            for run in list(self._runs.values()):
                run.ended = "stopped"
                if run.task:
                    run.task.cancel()
            await asyncio.gather(*(r.task for r in self._runs.values() if r.task),
                                 return_exceptions=True)
            try:
                await asyncio.wait_for(self._backend.close(), 10)
            except Exception as e:
                self._emit(P.NOTICE, level="warn", text=f"close: {e}")
            heart.cancel()

    async def _boot(self) -> None:
        try:
            self._emit(P.READY, **await self._backend.boot())
        except Exception as e:
            self._emit(P.NOTICE, level="error", text=f"Start fehlgeschlagen: {type(e).__name__}: {e}")

    async def _dispatch(self, cmd: P.Command, boot: asyncio.Task) -> None:
        d = cmd.data
        if cmd.kind == P.CHAT:
            self._counter += 1
            run = _Run(f"r{self._counter}", d["agent"], d["session"])
            self._runs[run.id] = run
            run.task = asyncio.create_task(self._chat(run, d["query"], boot))
        elif cmd.kind in (P.AGENTS, P.USE, P.PACK):
            # kann dauern (Agent bauen, Paket laden): eigener Task, Inbox bleibt frei
            asyncio.create_task(self._admin(cmd, boot))
        elif cmd.kind in (P.STOP, P.PAUSE):
            run = self._runs.get(d.get("run", ""))
            if run is None:
                return
            if cmd.kind == P.PAUSE and await self._backend.pause(run.agent, run.session):
                run.ended = "paused"
                return                                   # Stream endet selbst mit 'paused'
            run.ended = "stopped"
            if run.task:
                run.task.cancel()

    async def _admin(self, cmd: P.Command, boot: asyncio.Task) -> None:
        try:
            await asyncio.shield(boot)
            if cmd.kind == P.AGENTS:
                self._emit(P.AGENT_LIST, agents=await self._backend.agents())
            elif cmd.kind == P.USE:
                self._emit(P.READY, **await self._backend.use(cmd.data["agent"]))
            elif cmd.kind == P.PACK:
                self._emit(P.NOTICE, level="info", text=await self._backend.pack(cmd.data["agent"]))
        except Exception as e:
            self._emit(P.NOTICE, level="error", text=f"{cmd.kind}: {type(e).__name__}: {e}")

    # ── Ein Lauf ─────────────────────────────────────────────────────────────
    def _flush(self, run: _Run) -> None:
        if run.timer:
            run.timer.cancel()
            run.timer = None
        if run.buf:
            self._emit(run.kind, run.id, text="".join(run.buf))
            run.buf.clear()

    def _push(self, run: _Run, kind: str, text: str) -> None:
        if not text:
            return
        if kind != run.kind:
            self._flush(run)
            run.kind = kind
        run.buf.append(text)
        if run.timer is None:
            run.timer = self._loop.call_later(FLUSH_SECONDS, self._flush, run)

    async def _chat(self, run: _Run, query: str, boot: asyncio.Task) -> None:
        status, answer, error = "done", "", ""
        started: dict[str, float] = {}
        stream = None
        try:
            await asyncio.shield(boot)
            steer = await self._backend.is_running(run.agent, run.session)
            self._emit(P.RUN_START, run.id, agent=run.agent, session=run.session,
                       query=query, steer=steer)
            stream = self._backend.stream(run.agent, run.session, query)
            async for chunk in stream:
                if chunk.get("_sub_agent_id"):
                    continue                             # Sub-Agenten: eigener Schritt
                t = chunk.get("type", "")
                if t == "content":
                    self._push(run, P.TEXT, chunk.get("chunk", ""))
                    continue
                if t == "reasoning":
                    self._push(run, P.THINK, chunk.get("chunk", ""))
                    continue
                self._flush(run)
                if t == "tool_start":
                    tid = str(chunk.get("id") or chunk.get("name", "?"))
                    started[tid] = time.monotonic()
                    args = chunk.get("args", chunk.get("input", chunk.get("arguments", "")))
                    self._emit(P.TOOL_START, run.id, id=tid, name=chunk.get("name", "?"),
                               args=_clip(args, ARGS_LIMIT))
                elif t in ("tool_result", "tool_error"):
                    tid = str(chunk.get("id") or chunk.get("name", "?"))
                    secs = time.monotonic() - started.pop(tid, time.monotonic())
                    result = chunk.get("result", chunk.get("error", ""))
                    self._emit(P.TOOL_END, run.id, id=tid, name=chunk.get("name", "?"),
                               ok=t == "tool_result" and _succeeded(result),
                               result=_clip(result, RESULT_LIMIT), secs=round(secs, 2))
                elif t == "final_answer":
                    answer = str(chunk.get("answer", chunk.get("content", "")) or "")
                elif t == "paused":
                    status = "paused"
                elif t == "error":
                    status, error = "failed", str(chunk.get("message", chunk.get("chunk", "")))
                elif t == "done" and chunk.get("success") is False and status == "done":
                    status = "failed"
                if chunk.get("status_msg") or chunk.get("iter") is not None:
                    self._emit(P.STATUS, run.id, msg=chunk.get("status_msg", ""),
                               iter=chunk.get("iter"), max_iter=chunk.get("max_iter"),
                               tokens=chunk.get("tokens_used"), tokens_max=chunk.get("tokens_max"))
        except asyncio.CancelledError:
            status = run.ended or "stopped"
            try:
                await asyncio.shield(self._backend.stop(run.agent, run.session))
            except BaseException:
                pass
        except Exception as e:
            status, error = "failed", f"{type(e).__name__}: {e}"
        finally:
            if stream is not None and hasattr(stream, "aclose"):
                try:
                    await asyncio.shield(stream.aclose())
                except BaseException:
                    pass
            self._flush(run)
            self._runs.pop(run.id, None)
            self._emit(P.RUN_END, run.id, status=status, answer=answer, error=error)
