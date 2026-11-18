# -*- coding: utf-8 -*-
"""T9 Orchestrierung (Mixin-Erweiterung): self steuert andere Agenten (stop/pause/resume/status)
und liest deren OBS (run summaries + live actions). Reine Datei- und Bestands-APIs - kein Spawn,
kein neuer Prozess (isaa bietet keine Spawn-API, siehe PLAN T7-Entscheidung).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _ok_name(name: str) -> str:
    if not _AGENT_NAME_RE.fullmatch(name or ""):
        raise ValueError(f"Agent-Name '{name}' ist ungueltig (erlaubt: A-Z a-z 0-9 _ -)")
    return name


class SelfOrchestratorMixin:
    """Bringt self_agent_stop/pause/resume/status/obs auf den IsaaBackend (nur self, Test 7)."""

    def _self_ok(self) -> bool:
        """Gleiche Pruefung wie SelfToolMixin._self_ok (Aufrufer aus TOOL_CALLER)."""
        from .widget_tools import caller_of
        return caller_of(self)[0] == "self"

    async def _agent_instance(self, name: str):
        return await self._icli2_agent(_ok_name(name))

    # -- steuern (async: engine-API ist async) -----------------------------
    async def self_agent_stop(self, name: str, run_id: str = "") -> dict[str, Any] | None:
        """Stoppt run_id (oder ALLE laufenden Runs) des Agenten via engine.cancel."""
        if not self._self_ok():
            return None
        inst = await self._agent_instance(name)
        runs = inst._get_execution_engine().list_executions()
        targets = [r["run_id"] for r in runs if not run_id or r["run_id"] == run_id]
        stopped = 0
        for rid in targets:
            if await inst.cancel_execution(rid):
                stopped += 1
        return {"text": f"{name}: {stopped}/{len(targets)} Run(s) gestoppt",
                "agent": name, "stopped": stopped > 0, "run_ids": targets}

    async def self_agent_pause(self, name: str, run_id: str = "") -> dict[str, Any] | None:
        """Pausiert run_id (oder den letzten aktiven Run) des Agenten."""
        if not self._self_ok():
            return None
        inst = await self._agent_instance(name)
        runs = inst._get_execution_engine().list_executions()
        targets = [r["run_id"] for r in runs if not run_id or r["run_id"] == run_id]
        paused = 0
        for rid in targets:
            if await inst.pause_execution(rid):
                paused += 1
        return {"text": f"{name}: {paused}/{len(targets)} Run(s) pausiert",
                "agent": name, "paused": paused > 0, "run_ids": targets}

    async def self_agent_resume(self, name: str, run_id: str,
                                max_iterations: int = 30) -> dict[str, Any] | None:
        """Setzt pausierten Run fort (nur solange die Engine des Agenten lebt)."""
        if not self._self_ok():
            return None
        inst = await self._agent_instance(_ok_name(name))
        final = await inst.resume_execution(run_id, max_iterations=max_iterations)
        return {"text": f"{name}: Run {run_id} fortgesetzt", "agent": name,
                "run_id": run_id, "final": final}

    async def self_agent_status(self, name: str) -> dict[str, Any] | None:
        """Listet aktive/pausierte Runs des Agenten (engine.list_executions)."""
        if not self._self_ok():
            return None
        inst = await self._agent_instance(_ok_name(name))
        runs = inst._get_execution_engine().list_executions()
        return {"text": f"{name}: {len(runs)} aktive/pausierte Run(s)", "agent": name, "runs": runs}

    # -- einsehen (sync: reine OBS-Dateien) --------------------------------
    def self_agent_obs(self, name: str, run_id: str = "",
                       tail: int = 20) -> dict[str, Any] | None:
        """OBS des Agenten lesen. Ohne run_id: Run-Liste (_index.json).
        Mit run_id: Summary (run_<id>.json) + letzte `tail` Aktionen (live_<id>.jsonl).

        Hinweis: OBS deckt ALLE Agenten ab (auch andere Prozesse), live-Einsehen
        laufender fremder Runs nur innerhalb derselben isaa-Instanz (siehe status).
        """
        if not self._self_ok():
            return None
        obs = Path(self.app.data_dir) / "Agents" / _ok_name(name) / "obs"
        if not obs.is_dir():
            return {"text": f"{name}: kein OBS gefunden (Agent noch nie gelaufen)",
                    "level": "warn", "agent": name}
        if not run_id:
            idx = obs / "_index.json"
            runs: list[str] = []
            if idx.is_file():
                import json
                try:
                    runs = json.loads(idx.read_text(encoding="utf-8")).get("runs", [])
                except (OSError, ValueError):
                    runs = []
            return {"text": f"{name}: {len(runs)} archivierte Run(s)", "agent": name, "runs": runs}
        import json
        summary_p = obs / f"run_{run_id}.json"
        if not summary_p.is_file():
            return {"text": f"{name}: run_{run_id}.json existiert nicht", "level": "warn",
                    "agent": name, "run_id": run_id}
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
        actions: list[dict] = []
        live_p = obs / f"live_{run_id}.jsonl"
        if live_p.is_file():
            with live_p.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()[-tail:]  # niemals ganze 40MB-Datei laden
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    actions.append(json.loads(ln))
                except ValueError:
                    actions.append({"raw": ln[:200]})
        return {"text": f"{name}/{run_id}: {summary.get('total_iterations', '?')} Iter, "
                        f"{summary.get('tool_call_count', '?')} Tools, success={summary.get('success')}",
                "agent": name, "run_id": run_id, "summary": summary, "actions": actions}

    # -- starten (In-Process: a_stream im Hintergrund-Task) ----------------
    async def self_agent_start(self, name: str, query: str,
                               session: str = "") -> dict[str, Any] | None:
        """Startet den Agenten mit query als Hintergrund-Run (gleiche isaa-Instanz).
        Nur solange der Agent hier laeuft; Ergebnis landet in _orch_results + OBS."""
        if not self._self_ok():
            return None
        import asyncio as _aio
        from uuid import uuid4
        inst = await self._agent_instance(name)
        session = session or f"orch_{uuid4().hex[:8]}"
        if not hasattr(self, "_orch_results"):
            self._orch_results = {}
        if not hasattr(self, "_orch_tasks"):
            self._orch_tasks = {}
        results = self._orch_results
        tasks = self._orch_tasks

        async def _consume():
            n = 0
            try:
                async for _ in inst.a_stream(query=query, session_id=session):
                    n += 1
                results[session] = {"ok": True, "chunks": n}
            except Exception as e:  # laut, nicht schlucken - Ergebnis sichtbar machen
                results[session] = {"ok": False, "error": str(e)[:300]}

        tasks[session] = _aio.create_task(_consume())
        return {"text": f"{name} gestartet (session {session})", "agent": name, "session": session}
