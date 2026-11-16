"""Adapter auf den isaa-Mod. Einzige Stelle der icli 2.0 mit isaa-Importen.

Nutzt ausschliesslich vorhandene Funktionen:
  isaa.get_agent / get_agent_builder / register_agent   (mods/isaa/module.py)
  isaa.save_agent / load_agent                           (Paket .tar.gz inkl. Tools)
  agent.a_stream / pause_execution / cancel_execution    (base/Agent/flow_agent.py)
Nachsteuern im Lauf uebernimmt a_stream selbst (Sidecar-Steering), sobald
fuer dieselbe Session schon ein Lauf aktiv ist.

Agenten-Quellen (werden bei jeder Abfrage neu gelesen, nie gecacht):
  live  im laufenden isaa registriert            config["agents-name-list"]
  disk  von register_agent gespeichert           <data_dir>/Agents/<name>/agent.json
  pack  eigene Tools, automatisch gepackt        <data_dir>/Agents/<name>/<name>.tar.gz
        (isaa._auto_pack beim Bauen, isaa._restore_packed_tools beim Laden)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator


class IsaaBackend:
    def __init__(self, app: Any, agent: str = "self", session: str = "default"):
        self.app = app
        self.agent = agent
        self.session = session
        self.isaa: Any = None
        self.max_iterations = int(os.getenv("DEFAULT_MAX_ITERATIONS", 30))

    # ── Agenten ──────────────────────────────────────────────────────────────
    def _agents_dir(self) -> Path:
        return Path(self.app.data_dir) / "Agents"

    def pack_path(self, agent: str) -> Path:
        return self.isaa.agent_pack_path(agent)

    def _info(self, agent: Any) -> dict[str, Any]:
        return {"agent": self.agent, "session": self.session,
                "model": getattr(agent.amd, "fast_llm_model", "?")}

    async def boot(self) -> dict[str, Any]:
        self.isaa = self.app.get_mod("isaa")             # schwerer Import: hier, nicht in der UI
        if self.agent == "self" and "agent-instance-self" not in self.isaa.config:
            # gleiche Ausstattung wie die alte icli (icli.py:_init_self_agent)
            builder = self.isaa.get_agent_builder(
                name="self", add_base_tools=True, with_dangerous_shell=True)
            await self.isaa.register_agent(builder)
        return await self.use(self.agent)

    async def agents(self) -> list[dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        root = self._agents_dir()
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "agent.json").exists():
                    found[d.name] = {"name": d.name, "state": "disk", "pack": False}
                if d.is_dir() and self.pack_path(d.name).exists():
                    found.setdefault(d.name, {"name": d.name, "state": "disk"})["pack"] = True
        for name in self.isaa.config.get("agents-name-list", []):
            entry = found.setdefault(name, {"name": name, "pack": False})
            built = f"agent-instance-{name}" in self.isaa.config
            entry["state"] = "live" if built else entry.get("state", "registered")
        return list(found.values())

    async def use(self, agent: str) -> dict[str, Any]:
        """Agent laden und aktiv setzen. isaa.get_agent baut ihn aus der gespeicherten
        Konfiguration und haengt eigene Tools aus dem Auto-Paket wieder an."""
        if not any(a["name"] == agent for a in await self.agents()):
            raise LookupError(f"Agent '{agent}' ist nicht registriert")
        instance = await self.isaa.get_agent(agent)
        self.agent = agent
        return self._info(instance)

    async def pack(self, agent: str) -> str:
        ok, result = await self.isaa.save_agent(agent, str(self.pack_path(agent)))
        if not ok:
            raise RuntimeError(str(result))
        skipped = [t.name for t in result.non_serializable_tools]
        text = f"{agent} gepackt: {len(result.serializable_tools)} Tools gesichert"
        return text + (f", nicht serialisierbar: {', '.join(skipped)}" if skipped else "")

    # ── Laeufe ───────────────────────────────────────────────────────────────
    async def stream(self, agent: str, session: str, query: str) -> AsyncIterator[dict]:
        instance = await self.isaa.get_agent(agent)
        async for chunk in instance.a_stream(
            query=query, session_id=session, max_iterations=self.max_iterations,
        ):
            yield chunk

    async def _run_id(self, agent: str, session: str):
        instance = await self.isaa.get_agent(agent)
        engine = instance._get_execution_engine()
        return instance, engine, engine._session_last_run.get(session)

    async def is_running(self, agent: str, session: str) -> bool:
        _, engine, run_id = await self._run_id(agent, session)
        ctx = engine.get_execution(run_id) if run_id else None
        return bool(ctx is not None and ctx.status == "running")

    async def stop(self, agent: str, session: str) -> None:
        instance, _, run_id = await self._run_id(agent, session)
        if run_id:
            await instance.cancel_execution(run_id)

    async def pause(self, agent: str, session: str) -> bool:
        instance, _, run_id = await self._run_id(agent, session)
        return bool(run_id and await instance.pause_execution(run_id))

    def tool_meta(self, name: str, result: dict) -> str:
        # vorhandener Helfer der alten icli fuer bekannte Tools (vfs_*, shell, ...)
        from toolboxv2.mods.isaa.extras.zen.zen_renderer import ZenRenderer
        return ZenRenderer._extract_tool_meta(name, result) or ""

    async def close(self) -> None:
        pass          # app.a_exit() ruft der Flow-Starter nach dem UI-Ende auf
