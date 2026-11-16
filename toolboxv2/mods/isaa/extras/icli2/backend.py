"""Adapter auf den isaa-Mod. Einzige Stelle der icli 2.0 mit isaa-Importen.

Nutzt ausschliesslich vorhandene Funktionen:
  isaa.get_agent / get_agent_builder / register_agent / save_agent   (mods/isaa/module.py)
  agent.a_stream / pause_execution / cancel_execution                 (base/Agent/flow_agent.py)
  agent.session_manager, session.vfs.mount / unmount / sync_all       (session_manager, vfs_v2)
Nachsteuern im Lauf uebernimmt a_stream selbst (Sidecar-Steering).

Zwei generische Eingaenge fuer die UI, damit neue Bereiche ohne Aenderung an
Engine und Protokoll dazukommen:
  rows(what)        -> Zeilen fuer einen Picker
  do(action, ...)   -> Aktion ausfuehren, liefert {"ready": {...}, "text": "..."}

Agenten-Quellen (bei jeder Abfrage neu gelesen, nie gecacht):
  live  im laufenden isaa registriert            config["agents-name-list"]
  disk  von register_agent gespeichert           <data_dir>/Agents/<name>/agent.json
  pack  eigene Tools, automatisch gepackt        <data_dir>/Agents/<name>/<name>.tar.gz
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

AGENT_STATE = {"live": "geladen", "registered": "registriert", "disk": "gespeichert"}
HISTORY_KEEP = 500


def _row(value: str, label: str = "", hint: str = "") -> dict[str, str]:
    return {"value": value, "label": label or value, "hint": hint}


class IsaaBackend:
    def __init__(self, app: Any, agent: str = "self", session: str = "default"):
        self.app = app
        self.agent = agent
        self.session = session
        self.isaa: Any = None
        self.max_iterations = int(os.getenv("DEFAULT_MAX_ITERATIONS", 30))
        self._models: list[dict[str, str]] | None = None

    # ── Zustand ──────────────────────────────────────────────────────────────
    def _agents_dir(self) -> Path:
        return Path(self.app.data_dir) / "Agents"

    def _history_file(self) -> Path:
        return Path(self.app.appdata) / "icli2" / "history.txt"

    async def _agent(self, name: str | None = None):
        return await self.isaa.get_agent(name or self.agent)

    async def _session(self):
        return await (await self._agent()).session_manager.get_or_create(self.session)

    async def _ready(self) -> dict[str, Any]:
        agent = await self._agent()
        session = await self._session()
        vfs = getattr(session, "vfs", None)
        return {
            "agent": self.agent, "session": self.session,
            "model": getattr(agent.amd, "fast_llm_model", "?"),
            "complex_model": getattr(agent.amd, "complex_llm_model", "?"),
            "mounts": len(getattr(vfs, "mounts", {}) or {}),
        }

    async def boot(self) -> dict[str, Any]:
        self.isaa = self.app.get_mod("isaa")             # schwerer Import: hier, nicht in der UI
        if self.agent == "self" and "agent-instance-self" not in self.isaa.config:
            # gleiche Ausstattung wie die alte icli (icli.py:_init_self_agent)
            builder = self.isaa.get_agent_builder(
                name="self", add_base_tools=True, with_dangerous_shell=True)
            await self.isaa.register_agent(builder)
        ready = await self._ready()
        ready["history"] = await asyncio.to_thread(self._read_history)
        return ready

    # ── Picker-Zeilen ────────────────────────────────────────────────────────
    async def rows(self, what: str) -> dict[str, Any]:
        title, rows = await getattr(self, f"_rows_{what}")()
        return {"title": title, "rows": rows}

    async def _agents(self) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        root = self._agents_dir()
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "agent.json").exists():
                    found[d.name] = {"state": "disk", "pack": self.isaa.agent_pack_path(d.name).exists()}
        for name in self.isaa.config.get("agents-name-list", []):
            entry = found.setdefault(name, {"pack": False})
            built = f"agent-instance-{name}" in self.isaa.config
            entry["state"] = "live" if built else entry.get("state", "registered")
        return found

    async def _rows_agents(self):
        rows = [_row(name, hint=AGENT_STATE.get(a["state"], a["state"])
                     + (" · eigene Tools" if a["pack"] else "")
                     + (" · aktiv" if name == self.agent else ""))
                for name, a in (await self._agents()).items()]
        return "Agent wechseln", rows

    def _load_models(self) -> list[dict[str, str]]:
        """Modell-Register der alten icli lesen, ohne icli.py zu importieren
        (der Import wuerde Audio/Omni laden). Dazu die Gateway-Modelle."""
        found: dict[str, str] = {}
        try:
            import toolboxv2.flows.isaa as flows_isaa
            source = (Path(flows_isaa.__file__).parent / "icli.py").read_text(encoding="utf-8")
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "MODEL_MAPPING":
                    found.update(ast.literal_eval(node.value))
                    break
        except Exception as e:
            self.app.logger.warning(f"icli2 model registry: {e}")
        try:
            from toolboxv2.mods.isaa.base.IntelligentRateLimiter import gateway
            found.update({m: f"gateway/{m}" for m in gateway.get_available_models()})
        except Exception:
            pass
        return [_row(full, alias, full) for alias, full in sorted(found.items())]

    async def _rows_models(self):
        if self._models is None:
            self._models = await asyncio.to_thread(self._load_models)
        agent = await self._agent()
        current = [_row(m, m, f"aktuell {kind}") for kind, m in
                   (("schnell", agent.amd.fast_llm_model), ("komplex", agent.amd.complex_llm_model))]
        return "Modell wählen", current + self._models

    async def _rows_sessions(self):
        manager = (await self._agent()).session_manager
        names = list(dict.fromkeys([self.session, *manager.list_sessions()]))
        rows = [_row("+", "+ neue Session", "leerer Kontext")]
        rows += [_row(n, hint="aktiv" if n == self.session else "") for n in names]
        return "Session wechseln", rows

    async def _rows_mounts(self):
        vfs = (await self._session()).vfs
        rows = [_row(path, path, f"{mount.local_path} · Enter hängt aus und speichert")
                for path, mount in vfs.mounts.items()]
        if rows:
            rows.append(_row("*", "alle speichern", "geänderte Dateien auf Platte schreiben"))
        return ("Mounts" if rows else "Keine Mounts – /mount <ordner>"), rows

    # ── Aktionen ─────────────────────────────────────────────────────────────
    async def do(self, action: str, **args: Any) -> dict[str, Any]:
        return await getattr(self, f"_do_{action}")(**args)

    async def _do_agent(self, name: str) -> dict[str, Any]:
        if name not in await self._agents():
            raise LookupError(f"Agent '{name}' ist nicht registriert")
        await self._agent(name)                          # baut ihn, haengt gepackte Tools an
        self.agent = name
        return {"ready": await self._ready()}

    async def _do_model(self, name: str, target: str = "both") -> dict[str, Any]:
        agent = await self._agent()
        fields = [f for f, t in (("fast_llm_model", "fast"), ("complex_llm_model", "complex"))
                  if target in (t, "both")]
        if not fields:
            raise ValueError("Ziel muss fast, complex oder both sein")
        for field in fields:
            setattr(agent.amd, field, name)
        await asyncio.to_thread(self._persist_config, self.agent, dict.fromkeys(fields, name))
        return {"ready": await self._ready()}

    def _persist_config(self, agent: str, updates: dict[str, Any]) -> None:
        """Wie icli.py:_tool_update_agent_config - Werte in agent.json nachziehen."""
        path = self._agents_dir() / agent / "agent.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.update(updates)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if agent in self.isaa.agent_data:
            self.isaa.agent_data[agent].update(updates)

    async def _do_session(self, name: str) -> dict[str, Any]:
        if name == "+":
            name = datetime.now().strftime("s-%m%d-%H%M%S")
        self.session = name
        await self._session()
        return {"ready": await self._ready()}

    async def _do_session_clear(self) -> dict[str, Any]:
        """Wie '/session clear' der alten icli: Page-Engine ueber clear_session,
        sonst Verlauf und Lauf-Referenzen der Session leeren."""
        agent = await self._agent()
        session = await self._session()
        engine = agent._get_execution_engine()
        if hasattr(engine, "clear_session"):
            engine.clear_session(self.session)
        else:
            session.clear_history()
            run_id = engine._session_last_run.pop(self.session, None)
            if run_id:
                engine._active_executions.pop(run_id, None)
        return {"text": f"Session {self.session} geleert"}

    async def _do_mount(self, path: str, at: str = "") -> dict[str, Any]:
        local = Path(path).expanduser().resolve()
        if not local.is_dir():
            raise FileNotFoundError(f"Kein Ordner: {local}")
        vfs = (await self._session()).vfs
        at = at or ("/project" if "/project" not in vfs.mounts else f"/{local.name}")
        result = await asyncio.to_thread(vfs.mount, local_path=str(local), vfs_path=at)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "Mount fehlgeschlagen")
        return {"ready": await self._ready(),
                "text": f"{local} → {at} · {result.get('files_indexed', '?')} Dateien"}

    async def _do_unmount(self, at: str) -> dict[str, Any]:
        vfs = (await self._session()).vfs
        if at == "*":
            result = await asyncio.to_thread(vfs.sync_all)
            return {"text": f"gespeichert: {len(result.get('synced', []))} Dateien"}
        result = await asyncio.to_thread(vfs.unmount, at, save_changes=True)
        if not result.get("success"):
            raise RuntimeError(result.get("error") or "Aushängen fehlgeschlagen")
        saved = len(result.get("files_saved", []))
        return {"ready": await self._ready(), "text": f"{at} ausgehängt · {saved} Dateien gespeichert"}

    async def _do_pack(self, name: str = "") -> dict[str, Any]:
        name = name or self.agent
        ok, result = await self.isaa.save_agent(name, str(self.isaa.agent_pack_path(name)))
        if not ok:
            raise RuntimeError(str(result))
        skipped = [t.name for t in result.non_serializable_tools]
        text = f"{name} gepackt: {len(result.serializable_tools)} Tools gesichert"
        return {"text": text + (f", nicht serialisierbar: {', '.join(skipped)}" if skipped else "")}

    # ── Eingabe-History (eigene Datei der icli2) ─────────────────────────────
    def _read_history(self) -> list[str]:
        path = self._history_file()
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines()[-HISTORY_KEEP:] if line.strip()]

    def _append_history(self, text: str) -> None:
        path = self._history_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    async def _do_remember(self, text: str) -> dict[str, Any]:
        await asyncio.to_thread(self._append_history, text)
        return {}

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
