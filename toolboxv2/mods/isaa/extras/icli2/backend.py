"""Adapter auf den isaa-Mod. Einzige Stelle der icli 2.0 mit isaa-Importen.

Nutzt ausschliesslich vorhandene Funktionen:
  isaa.get_agent / get_agent_builder / register_agent   (mods/isaa/module.py)
  agent.a_stream / pause_execution / cancel_execution    (base/Agent/flow_agent.py)
Nachsteuern im Lauf uebernimmt a_stream selbst (Sidecar-Steering), sobald
fuer dieselbe Session schon ein Lauf aktiv ist.
"""
from __future__ import annotations

import os
from typing import Any, AsyncIterator


class IsaaBackend:
    def __init__(self, app: Any, agent: str = "self", session: str = "default"):
        self.app = app
        self.agent = agent
        self.session = session
        self.isaa: Any = None
        self.max_iterations = int(os.getenv("DEFAULT_MAX_ITERATIONS", 30))

    async def boot(self) -> dict[str, Any]:
        self.isaa = self.app.get_mod("isaa")             # schwerer Import: hier, nicht in der UI
        if self.agent == "self" and "agent-instance-self" not in self.isaa.config:
            # gleiche Ausstattung wie die alte icli (icli.py:_init_self_agent)
            builder = self.isaa.get_agent_builder(
                name="self", add_base_tools=True, with_dangerous_shell=True)
            await self.isaa.register_agent(builder)
        agent = await self.isaa.get_agent(self.agent)
        return {
            "agents": list(self.isaa.config.get("agents-name-list", [])),
            "agent": self.agent,
            "session": self.session,
            "model": getattr(agent.amd, "fast_llm_model", "?"),
        }

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

    async def close(self) -> None:
        pass          # app.a_exit() ruft der Flow-Starter nach dem UI-Ende auf
