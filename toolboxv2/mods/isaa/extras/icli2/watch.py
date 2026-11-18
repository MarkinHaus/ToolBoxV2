"""Laeufe im Blick behalten: Hintergrund-Befehle der Shell, eigene Sessions fuer Jobs, /runs.

- Hintergrund-Befehl (shell background=True) endet -> Meldung an den Lauf, der ihn gestartet hat:
  laeuft er noch, kommt sie als Nachricht in seinen Kontext (wie eine Nachsteuerung); ist er fertig
  oder hat er die Nachricht nicht mehr gelesen, startet ein Folgelauf in derselben Session.
- Jobs laufen in einer eigenen Session job-<name> mit eigenen Shell-Sessions: der Chat wird nie zur
  Nachsteuerung eines Jobs, und Job und Chat teilen sich keine Shell.
- /runs: alle Ausfuehrungen der gebauten Agenten, mit dem Hinweis, ob in der icli2 wirklich noch
  ein Lauf dazu lebt; haengende lassen sich beenden oder zuruecksetzen.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

FOLLOWUP_GIVE_UP = 6 * 3600        # so lange wartet die Zustellung hoechstens auf das Laufende
_SLUG = re.compile(r"[^A-Za-z0-9_-]+")


def _row(value: str, label: str = "", hint: str = "") -> dict[str, str]:
    return {"value": value, "label": label or value, "hint": hint}


def bg_message(event: dict) -> str:
    """Text fuer den Agenten (englisch wie die Tool-Hinweise der Shell)."""
    return (f"[BACKGROUND COMMAND FINISHED] shell session_id={event['session_id']} · "
            f"exit_code={event['exit_code']} · {event['seconds']:.0f}s\n"
            f"Command: {event['command']}\n"
            f"Last output:\n{event['tail'] or '(none)'}\n"
            f"Evaluate the result and continue the task.")


class WatchMixin:
    """Mixin fuer IsaaBackend; braucht self.isaa, self._spawn, self._tasks."""

    _live: Any = None                                    # Engine: lebende Laeufe [(agent, session, id, origin)]
    _notice: Any = None                                  # Engine: Hinweis in die UI
    _loop: asyncio.AbstractEventLoop | None = None

    def attach_engine(self, live: Any, notice: Any) -> None:
        self._live, self._notice = live, notice

    # ── Jobs: eigene Session ─────────────────────────────────────────────────
    def _job_sessions(self) -> set[str]:
        if not hasattr(self, "_job_session_set"):
            self._job_session_set: set[str] = set()
        return self._job_session_set

    def job_session(self, job: Any) -> str:
        """Gesetzte Session des Jobs, sonst job-<name> statt der Chat-Session 'default'."""
        session = str(getattr(job, "session_id", "") or "").strip()
        if not session or session == "default":
            slug = _SLUG.sub("-", str(getattr(job, "name", "") or "")).strip("-")[:40]
            session = f"job-{slug or getattr(job, 'job_id', 'run')}"
        self._job_sessions().add(session)
        return session

    def shell_enter(self, agent: str, session: str) -> None:
        """Fuer den laufenden Agenten-Lauf: wem gehoeren Shell-Aufrufe (Meldung, Trennung)."""
        try:
            from toolboxv2.mods.isaa.module import SHELL_CONTEXT
        except Exception:
            return
        scope = session if session in self._job_sessions() else ""
        SHELL_CONTEXT.set({"agent": agent, "session": session, "scope": scope})

    # ── Hintergrund-Befehle ──────────────────────────────────────────────────
    def watch_boot(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            from toolboxv2.mods.isaa.module import add_background_listener
            add_background_listener(self._bg_listener)
        except Exception:
            pass

    def watch_close(self) -> None:
        try:
            from toolboxv2.mods.isaa.module import remove_background_listener
            remove_background_listener(self._bg_listener)
        except Exception:
            pass

    def _bg_listener(self, event: dict) -> None:
        """Aus dem Beobachter-Thread der Shell: in den Engine-Loop weiterreichen."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._keep, self.bg_done(event))
        except RuntimeError:                             # Loop wurde gerade beendet
            pass

    def _keep(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._kept)

    def _kept(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            error = task.exception()
            if self._notice:
                self._notice(f"Meldung des Hintergrund-Befehls nicht zugestellt: {type(error).__name__}: {error}",
                             "warn")

    def _alive(self, agent: str, session: str) -> bool:
        live = self._live() if self._live else []
        return any(a == agent and s == session for a, s, _id, _origin in live)

    def _instance(self, agent: str) -> Any:
        return self.isaa.config.get(f"agent-instance-{agent}") if self.isaa is not None else None

    def _running_ctx(self, agent: str, session: str) -> Any:
        """Kontext des laufenden Laufs - nur wenn in der icli2 auch wirklich einer lebt."""
        instance = self._instance(agent)
        if instance is None or not self._alive(agent, session):
            return None
        engine = instance._get_execution_engine()
        ctx = engine._active_executions.get(engine._session_last_run.get(session) or "")
        return ctx if ctx is not None and ctx.status == "running" else None

    async def bg_done(self, event: dict) -> None:
        agent, session = event.get("agent", ""), event.get("session", "")
        ok = event.get("exit_code") == 0
        if self._notice:
            self._notice(f"{'✓' if ok else '✗'} Hintergrund-Befehl in {event['session_id']} fertig "
                         f"(Exit {event['exit_code']}, {event['seconds']:.0f}s): {event['command'][:80]}",
                         "info" if ok else "warn")
        if not agent or not session or self._spawn is None:
            return                                       # nicht aus einem Agenten-Lauf gestartet
        text = bg_message(event)
        ctx = self._running_ctx(agent, session)
        if ctx is None:
            await self._spawn(agent, session, text, kind="shell")
            return
        message = {"role": "user", "content": text}
        ctx.working_history.append(message)              # wie die Nachsteuerung (_sidecar_ping)
        await self._followup(agent, session, ctx, message, text)

    async def _followup(self, agent: str, session: str, ctx: Any, message: dict, text: str) -> None:
        """Hat der Lauf die Meldung nicht mehr gelesen (endete vorher), Folgelauf starten."""
        start = time.monotonic()
        while ctx.status == "running" and self._alive(agent, session):
            if time.monotonic() - start > FOLLOWUP_GIVE_UP:
                return
            await asyncio.sleep(1.0)
        history = list(ctx.working_history)
        pos = next((i for i, m in enumerate(history) if m is message), None)
        if pos is None:
            return                                       # Verlauf uebernommen/verdichtet: gelesen
        if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in history[pos + 1:]):
            await self._spawn(agent, session, text, kind="shell")

    # ── /runs ────────────────────────────────────────────────────────────────
    async def _rows_runs(self):
        rows = []
        live = self._live() if self._live else []
        for key in list(self.isaa.config):
            if not str(key).startswith("agent-instance-"):
                continue
            agent = key[len("agent-instance-"):]
            try:
                engine = self.isaa.config[key]._get_execution_engine()
                executions = engine.list_executions()
            except Exception:
                continue
            for ex in executions:
                session, status = ex.get("session_id", ""), ex.get("status", "")
                alive = any(a == agent and s == session for a, s, _i, _o in live)
                stuck = status == "running" and not alive
                mark = "⚠ hängt" if stuck else ("läuft" if alive else status)
                rows.append((0 if stuck else 1 if alive else 2, _row(
                    f"{agent}|{ex['run_id']}", f"{agent} · {session} · {mark}",
                    f"Schritt {ex.get('iteration', '?')} · {ex.get('query', '')}"
                    + ("" if alive or status != "running" else " · kein lebender Lauf in der icli2"))))
        rows.sort(key=lambda r: r[0])
        title = "Läufe der Agenten" if rows else "Keine Läufe – nichts hängt"
        return title, [r for _o, r in rows]

    async def _do_run(self, run: str, op: str) -> dict[str, Any]:
        agent, _, run_id = run.partition("|")
        instance = self._instance(agent)
        if instance is None:
            raise LookupError(f"Agent {agent} ist nicht gebaut")
        engine = instance._get_execution_engine()
        ctx = engine.get_execution(run_id)
        if ctx is None:
            raise LookupError(f"Lauf {run_id} gibt es nicht mehr")
        if op == "stop":
            await instance.cancel_execution(run_id)
            return {"text": f"Lauf {run_id} von {agent} beendet ({ctx.session_id})"}
        if op == "reset":                                # bleibt fortsetzbar, ist aber nicht mehr "running"
            ctx.status = "paused"
            return {"text": f"Lauf {run_id} von {agent} zurückgesetzt – die nächste Nachricht setzt fort"}
        raise ValueError(f"Unbekannte Aktion {op}")
