"""Adapter auf den isaa-Mod. Einzige Stelle der icli 2.0 mit isaa-Importen.

Nutzt ausschliesslich vorhandene Funktionen:
  isaa.get_agent / get_agent_builder / register_agent / save_agent   (mods/isaa/module.py)
  agent.a_stream / pause_execution / cancel_execution                 (base/Agent/flow_agent.py)
  agent.session_manager, session.vfs.mount / unmount / sync_all       (session_manager, vfs_v2)
  isaa.job_scheduler (list/add/pause/resume/remove, add_dream_job)    (extras/jobs)
  isaa.export_agent_network / import_agent_network                    (mods/isaa/module.py)
  ExecutionContext.to_checkpoint / from_checkpoint                    (execution_engine.py)
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
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from .bridges import BridgeMixin     # Schritt 7: Discord/Telegram ueber die unveraenderten Erweiterungen
from .connect import ConnectMixin    # Schritt 6: MCP-Server und Plugins
from .plugin_install import PluginInstallMixin   # Plugins installieren, entfernen, Marketplace
from .features import FeatureMixin   # Schritt 6: Features und Skills
from .vfsview import VfsViewMixin    # /vfs: Dateien ansehen, Pfad-Vervollstaendigung
from .cloudm import CloudMMixin      # Schritt 8: Login, ToolBox-Funktionen, LiveSync-Shares
from .voice import VoiceMixin        # Schritt 9: Sprache (Audio-Importe erst bei Gebrauch)
from .self_tool import SelfToolMixin  # T7: self-Agent steuert icli2-Settings + Agent-Config
from .watch import WatchMixin
from .self_orchestrator import SelfOrchestratorMixin  # T9: self steuert andere Agenten + liest OBS

AGENT_STATE = {"live": "geladen", "registered": "registriert", "disk": "gespeichert"}
HISTORY_KEEP = 500
AUDIO_TAG = "#audio"
RECAP_MESSAGES = 4
RESUMABLE = ("paused", "max_iterations")
JOB_HELP = ("Zeitplan fehlt. Beispiele:  /job alle 30m name: auftrag   ·   /job um 09:00 name: auftrag"
            "   ·   /job start name: auftrag")


def _row(value: str, label: str = "", hint: str = "") -> dict[str, str]:
    return {"value": value, "label": label or value, "hint": hint}



_AGENT_MODEL_KEYS = ("fast_llm_model", "complex_llm_model")

class IsaaBackend(WatchMixin, SelfToolMixin, SelfOrchestratorMixin, FeatureMixin, PluginInstallMixin, ConnectMixin, BridgeMixin, CloudMMixin, VoiceMixin, VfsViewMixin):
    def __init__(self, app: Any, agent: str = "self", session: str = "default"):
        self.app = app
        self.agent = agent
        self.session = session
        self.isaa: Any = None
        self.max_iterations = int(os.getenv("DEFAULT_MAX_ITERATIONS", 30))
        self._spawn: Any = None                          # Engine.spawn, gesetzt ueber attach()
        self._jobs: Any = None
        self._job_callback: Any = None
        self._unrestored: dict[str, dict] = {}           # pausierte Laeufe, die nicht ladbar waren
        self._tasks: set[asyncio.Task] = set()

    def attach(self, spawn: Any) -> None:
        self._spawn = spawn

    # ── Zustand ──────────────────────────────────────────────────────────────
    def _agents_dir(self) -> Path:
        return Path(self.app.data_dir) / "Agents"

    def _state_dir(self) -> Path:
        return Path(self.app.appdata) / "icli2"

    def _history_file(self) -> Path:
        return self._state_dir() / "history.txt"

    def _networks_dir(self) -> Path:
        return self._agents_dir() / "_networks"

    async def _agent(self, name: str | None = None):
        return await self.isaa.get_agent(name or self.agent)

    async def _session(self):
        return await (await self._agent()).session_manager.get_or_create(self.session)

    async def _ready(self, recap: bool = False) -> dict[str, Any]:
        agent = await self._agent()
        session = await self._session()
        vfs = getattr(session, "vfs", None)
        return {
            **({"recap": self._recap(session)} if recap else {}),
            "agent": self.agent, "session": self.session,
            "model": getattr(agent.amd, "fast_llm_model", "?"),
            "complex_model": getattr(agent.amd, "complex_llm_model", "?"),
            "mounts": len(getattr(vfs, "mounts", {}) or {}),
            "user": await self._cloud_user(),
            "features": self._features_active(),
        }

    async def _switch(self, agent: str | None = None, session: str | None = None) -> dict[str, Any]:
        """Agent/Session wechseln; schlaegt das Laden fehl, bleibt der alte Zustand."""
        old = (self.agent, self.session)
        self.agent, self.session = agent or self.agent, session or self.session
        try:
            return await self._ready(recap=True)
        except BaseException:
            self.agent, self.session = old
            raise

    async def boot(self) -> dict[str, Any]:
        self.isaa = self.app.get_mod("isaa")             # schwerer Import: hier, nicht in der UI
        if self.agent == "self" and "agent-instance-self" not in self.isaa.config:
            # gleiche Ausstattung wie die alte icli (icli.py:_init_self_agent)
            builder = self.isaa.get_agent_builder(
                name="self", add_base_tools=True, with_dangerous_shell=True)
            await self.isaa.register_agent(builder)
        await asyncio.to_thread(self._load_settings)
        self.watch_boot()                                # Meldungen von Hintergrund-Befehlen der Shell
        ready = await self._ready()
        ready["debug"] = bool(getattr(self.app, "debug", False) or os.getenv("ICLI2_DEBUG"))
        ready["history"] = await asyncio.to_thread(self._read_history)
        # darf den Start nie verhindern; Discord/Telegram werden erst bei Gebrauch geladen
        for step in (self._restore_paused, self._start_jobs, self._plugins_boot):
            try:
                await step()
            except Exception as e:
                self.app.logger.warning(f"icli2 {step.__name__}: {e}")
        return ready

    @staticmethod
    def _recap(session: Any) -> list[tuple[str, str]]:
        """Letzte Nachrichten einer Session als (rolle, kurztext)."""
        out = []
        for message in session.get_history(RECAP_MESSAGES):
            role, content = str(message.get("role", "")), message.get("content", "")
            if isinstance(content, list):                # Teile: nur Text zeigen
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            text = " ".join(str(content).split())
            if text and role[:1] in ("u", "a"):
                out.append(("user" if role[:1] == "u" else "assistant",
                            text[:160] + ("…" if len(text) > 160 else "")))
        return out

    # ── Jobs ─────────────────────────────────────────────────────────────────
    async def _start_jobs(self) -> None:
        """Scheduler wie in der alten icli starten; Job-Laeufe gehen durch die Engine
        und erscheinen dadurch als normale Laeufe in der UI."""
        self._jobs = getattr(self.isaa, "job_scheduler", None)
        if self._jobs is None:
            return
        self._job_callback = self._jobs._fire_callback
        self._jobs._fire_callback = self._fire_job
        await self._jobs.start()
        await self._jobs.fire_lifecycle("on_cli_start")
        await self._jobs.fire_missed_jobs()

    async def _fire_job(self, job: Any) -> str:
        stream = self._dream_stream(job) if job.query == "__dream__" else None
        return await self._spawn(job.agent_name, self.job_session(job), f"{job.name}: {job.query}",
                                 kind="job", stream=stream)

    async def _dream_stream(self, job: Any) -> AsyncIterator[dict]:
        """Dream-Jobs laufen ueber den vorhandenen isaa-Callback (a_dream)."""
        yield {"type": "status", "status_msg": "Dream läuft"}
        report = await self._job_callback(job)
        yield {"type": "final_answer", "answer": str(report)}

    async def _rows_jobs(self):
        if self._jobs is None:
            return "Kein Job-Scheduler verfügbar", []
        rows = []
        for job in self._jobs.list_jobs():
            state = "läuft" if job.job_id in getattr(self._jobs, "_firing", ()) else job.status
            rows.append(_row(job.job_id, job.name,
                             f"{job.trigger.trigger_type} · {state} · {job.last_result or '–'} · "
                             f"{job.run_count}x · {job.agent_name}"))
        return ("Jobs" if rows else "Keine Jobs – /job alle 30m name: auftrag  oder  /dream"), rows

    async def _do_job(self, job: str, op: str) -> dict[str, Any]:
        entry = self._jobs.get_job(job) or next(        # /job fire <name> geht auch mit dem Namen
            (j for j in self._jobs.list_jobs() if j.name == job), None)
        if entry is None:
            raise LookupError(f"Job {job} nicht gefunden")
        job = entry.job_id
        if op == "fire":
            task = asyncio.ensure_future(self._jobs._fire_job(entry))   # wie module.py
            self._tasks.add(task)
            task.add_done_callback(self._job_done)
            return {"text": f"Job {entry.name} gestartet"}
        if op == "pause":
            paused = entry.status == "paused"
            (self._jobs.resume_job if paused else self._jobs.pause_job)(job)
            return {"text": f"Job {entry.name} {'fortgesetzt' if paused else 'pausiert'}"}
        if op == "remove":
            self._jobs.remove_job(job)
            return {"text": f"Job {entry.name} gelöscht"}
        raise ValueError(f"Unbekannte Job-Aktion {op}")

    def _job_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.app.logger.warning(f"icli2 job: {task.exception()}")

    async def _do_job_add(self, spec: str) -> dict[str, Any]:
        """'/job alle 30m name: auftrag' | '/job um 09:00 name: auftrag' | '/job start name: auftrag'"""
        from toolboxv2.mods.isaa.extras.jobs import JobDefinition, TriggerConfig
        match = re.match(r"(?:(alle)\s+(\d+)\s*([mh])|(um)\s+(\d{1,2}):(\d{2})|(start))\s+([^:]+):\s*(.+)",
                         spec, re.S | re.I)
        if not match:
            raise ValueError(JOB_HELP)
        every, amount, unit, at, hour, minute, _start, name, query = match.groups()
        if (every and int(amount) < 1) or (at and (int(hour) > 23 or int(minute) > 59)):
            raise ValueError(JOB_HELP)
        if every:
            trigger = TriggerConfig(trigger_type="on_interval",
                                    interval_seconds=int(amount) * (60 if unit.lower() == "m" else 3600))
        elif at:
            trigger = TriggerConfig(trigger_type="on_cron", cron_expression=f"{int(minute)} {int(hour)} * * *")
        else:
            trigger = TriggerConfig(trigger_type="on_cli_start")
        job = JobDefinition(job_id=uuid.uuid4().hex[:8], name=name.strip(), agent_name=self.agent,
                            query=query.strip(), trigger=trigger, session_id=self.session,
                            created_at=datetime.now().isoformat())
        self._jobs.add_job(job)
        return {"text": f"Job {job.name} angelegt ({trigger.trigger_type}, Agent {self.agent})"}

    async def _do_dream(self) -> dict[str, Any]:
        await self._agent()
        if any(j.query == "__dream__" and j.agent_name == self.agent for j in self._jobs.list_jobs()):
            return {"text": f"Dream-Job für {self.agent} gibt es schon (siehe /job)", "level": "warn"}
        self._jobs.add_dream_job(self.agent)             # wie '/job dream create': nachts 03:00
        return {"text": f"Dream-Job für {self.agent} angelegt (täglich 03:00)"}

    # ── Pausierte Laeufe ueber einen Neustart retten ─────────────────────────
    def _resume_file(self) -> Path:
        return self._state_dir() / "resume.json"

    def _collect_paused(self) -> dict[str, dict]:
        found = {}
        for key, agent in list(self.isaa.config.items()):
            if not str(key).startswith("agent-instance-"):
                continue
            engine = getattr(agent, "_execution_engine_cache", None)
            for run_id, ctx in list(getattr(engine, "_active_executions", {}).items()):
                if getattr(ctx, "status", "") in RESUMABLE:
                    found[run_id] = {"agent_name": key[len("agent-instance-"):],
                                     "checkpoint": ctx.to_checkpoint()}
        return found

    def _save_paused(self) -> None:
        path = self._resume_file()
        found = {**self._unrestored, **self._collect_paused()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(found, default=str), encoding="utf-8")
        os.replace(tmp, path)                            # nie eine halbe Datei hinterlassen

    async def _restore_paused(self) -> None:
        """Wie icli.py:_restore_resumable_executions - Kontexte als 'paused' zurueck in
        die Engine; die naechste Nachricht in der Session setzt sie fort."""
        path = self._resume_file()
        if not path.exists():
            return
        from toolboxv2.mods.isaa.base.Agent.execution_engine import ExecutionContext
        pending = json.loads(await asyncio.to_thread(path.read_text, "utf-8"))
        for run_id, entry in pending.items():
            try:
                agent = await self._agent(entry.get("agent_name", "self"))
                engine = agent._get_execution_engine()
                if run_id in engine._active_executions:
                    continue
                ctx = ExecutionContext.from_checkpoint(entry.get("checkpoint", {}))
                ctx.status = "paused"
                engine._active_executions[ctx.run_id] = ctx
                engine._session_last_run.setdefault(ctx.session_id, ctx.run_id)
            except Exception as e:                       # Eintrag behalten, naechster Start versucht es erneut
                self._unrestored[run_id] = entry
                self.app.logger.warning(f"icli2 resume {run_id}: {e}")

    # ── Picker-Zeilen ────────────────────────────────────────────────────────
    async def rows(self, what: str, arg: str | None = None) -> dict[str, Any]:
        method = getattr(self, f"_rows_{what}")
        title, rows = await (method() if arg is None else method(arg))
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
        nets = self._networks_dir()
        if nets.is_dir():
            rows += [_row(f"net:{f}", f"⧉ {f.name[:-7]}", "Agent-Network laden")
                     for f in sorted(nets.glob("*.tar.gz"))]
        return "Agent wechseln", rows

    async def _rows_models(self):
        """Bei jedem Oeffnen neu gelesen (models.py): Custom-Router/Gateways zuerst, dann das
        Register der alten icli. Freie Namen gehen immer ueber '/model <praefix>/<name>'."""
        from . import models
        agent = await self._agent()
        current = [(agent.amd.fast_llm_model, "aktuell schnell"), (agent.amd.complex_llm_model, "aktuell komplex")]
        rows = await asyncio.to_thread(models.all_models, current)
        try:                                             # Praefixe des Routers dieses Agenten
            prefixes = sorted(getattr(agent.router, "_adapters", {}) or {})
        except Exception:
            prefixes = []
        return "Modell wählen" + (f" · Router: {', '.join(prefixes)}" if prefixes else ""), rows

    # ── Einstellungen und Kontext ────────────────────────────────────────────
    def _settings_file(self) -> Path:
        return self._state_dir() / "settings.json"

    def _load_settings(self) -> None:
        try:
            data = json.loads(self._settings_file().read_text(encoding="utf-8"))
        except Exception:
            data = {}                                   # keine Datei / kaputt: Standardwerte
        self.max_iterations = int(data.get("max_iterations", getattr(self, "max_iterations", 5)))
        self._settings = data

    def save_setting(self, key: str, value) -> None:
        """Schluessel in settings.json mergen (atomar via tmp+replace)."""
        import os as _os
        path = self._settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data[key] = value
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        _os.replace(tmp, path)
        if not isinstance(getattr(self, "_settings", None), dict):
            self._settings = {}
        self._settings[key] = value

    def get_voice_devices(self) -> tuple:
        """Gespeicherte Geraete (in, out) oder (None, None)."""
        cfg = getattr(self, "_settings", None) or {}
        return cfg.get("voice_input_device"), cfg.get("voice_output_device")

    def get_tts_config(self) -> tuple:
        """(tts_provider, tts_voice), Defaults wie alte icli (groq_tts/autumn)."""
        cfg = getattr(self, "_settings", None) or {}
        return cfg.get("tts_provider", "groq_tts"), cfg.get("tts_voice", "autumn")

    async def _rows_settings(self):
        user = await self._cloud_user()
        return "Einstellungen", [
            _row("set:max_iterations", "Max. Iterationen", f"{self.max_iterations} · Enter zum Ändern"),
            _row("context", "Kontext", "Auslastung der Session aufschlüsseln"),
            _row("login", "Anmeldung", f"angemeldet als {user}" if user else "nicht angemeldet"),
            _row("credentials", "Zugangsdaten", "API-Schlüssel verwalten · kommt"),
            _row("paths", "Datenordner", str(self.app.data_dir)),
        ]

    async def _do_set(self, key: str = "", value: str = "") -> dict[str, Any]:
        # Agent-Config (nach dem Builder) - geht in agent.json via _persist_config
        if key in _AGENT_MODEL_KEYS:
            return await self._set_agent_value(key, value, cast=str)
        if key == "immediate_offload_ratio":
            return await self._set_agent_value(key, value, cast=float, lo=0.0, hi=1.0)
        if key == "heavy_hitter_min_tokens":
            return await self._set_agent_value(key, value, cast=int, lo=1)
        if key != "max_iterations":
            raise ValueError("Einstellbar: max_iterations, fast_llm_model, "
                             "complex_llm_model, immediate_offload_ratio, "
                             "heavy_hitter_min_tokens  (Beispiel: /set max_iterations 50)")
        if not value.isdigit() or not 1 <= int(value) <= 500:
            raise ValueError("max_iterations braucht eine Zahl von 1 bis 500")
        self.max_iterations = int(value)
        await asyncio.to_thread(self.save_setting, "max_iterations", self.max_iterations)
        return {"text": f"Max. Iterationen: {self.max_iterations} (gilt ab der nächsten Nachricht)"}

    async def _set_agent_value(self, key: str, value: str, *, cast, lo=None, hi=None) -> dict[str, Any]:
        path = self._agents_dir() / self.agent / "agent.json"
        try:
            num = cast(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{key} braucht einen gueltigen Wert ({cast.__name__}): {value}") from e
        if lo is not None and num < lo or hi is not None and num > hi:
            raise ValueError(f"{key} muss zwischen {lo} und {hi} liegen: {value}")
        if not path.exists():
            return {"text": f"Agent {self.agent!r} ist nicht gebaut (keine agent.json unter "
                            f"data/Agents/{self.agent}) - zuerst /agent bauen", "level": "warn"}
        await asyncio.to_thread(self._persist_config, self.agent, {key: num})
        return {"text": f"{key} = {num} (in agent.json von {self.agent})"}

    async def _do_credentials(self) -> dict[str, Any]:
        return {"text": "Zugangsdaten verwalten kommt in einem späteren Schritt – bis dahin über .env", "level": "warn"}

    async def _do_paths(self) -> dict[str, Any]:
        return {"text": f"Daten    {self.app.data_dir}\nZustand  {self._state_dir()}\nAgenten  {self._agents_dir()}"}

    async def _rows_context(self):
        """Kontext-Aufschluesselung fuer die Seitenleiste (schmal), ueber das vorhandene
        agent.context_overview wie '/context' der alten icli."""
        agent = await self._agent()
        data = await agent.context_overview(self.session, print_visual=False)
        used, limit = int(data.get("t_effective") or data.get("t_total") or 0), int(data.get("limit") or 0) or 1

        def bar(part: int, whole: int, width: int) -> str:
            fill = min(width, round(width * part / whole)) if whole else 0
            return "━" * fill + "─" * (width - fill)

        lines = [f"{bar(used, limit, 20)} {100 * used / limit:.1f} %",
                 f"{used / 1000:.1f}k von {limit / 1000:.0f}k · {data.get('model', '?')}"]
        for title, part in (("Aufteilung", data.get("breakdown") or {}),
                            ("System im Detail", data.get("system_details") or {})):
            top = max([int(v or 0) for v in part.values()] or [1]) or 1
            lines += ["", title]
            lines += [f"{str(k)[:19]:<19}{int(v or 0):>6} {bar(int(v or 0), top, 7)}" for k, v in part.items()]
        meta = data.get("meta") or {}
        lines += ["", *[f"{str(k)[:24]:<24}{v:>9}" for k, v in meta.items()]]
        return "Kontext", [_row(str(i), line or " ") for i, line in enumerate(lines)]

    async def _rows_sessions(self):
        manager = (await self._agent()).session_manager
        # der echte SessionManager liefert auch leere Namen (None) - die sind nicht waehlbar
        names = list(dict.fromkeys(str(n) for n in [self.session, *manager.list_sessions()] if n))
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
        if name.startswith("net:"):
            return await self._do_network_load(name[4:])
        if name not in await self._agents():
            raise LookupError(f"Agent '{name}' ist nicht registriert")
        return {"ready": await self._switch(agent=name)}   # baut ihn, haengt gepackte Tools an

    # ── Agent-Networks ───────────────────────────────────────────────────────
    async def _do_network_save(self) -> dict[str, Any]:
        """Aktiven Agenten samt gebundener Agenten als ein Archiv sichern."""
        agent = await self._agent()
        partners = list(getattr(getattr(agent, "bind_manager", None), "bindings", {}) or {})
        names = [self.agent, *[p for p in partners if p != self.agent]]
        path = self._networks_dir() / f"{self.agent}.tar.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, message = await self.isaa.export_agent_network(names, str(path), entry_agent=self.agent)
        if not ok:
            raise RuntimeError(message)
        return {"text": f"Network {self.agent} gesichert: {', '.join(names)}"}

    async def _do_network_load(self, path: str) -> dict[str, Any]:
        file = Path(path).expanduser()
        if not file.exists():
            file = self._networks_dir() / (path if path.endswith(".tar.gz") else f"{path}.tar.gz")
        if not file.exists():
            raise FileNotFoundError(f"Network nicht gefunden: {path}")
        agents, warnings = await self.isaa.import_agent_network(str(file))
        if not agents:
            raise RuntimeError("; ".join(warnings) or "Network leer")
        entry = file.name[:-7] if file.name[:-7] in agents else next(iter(agents))
        text = f"Network geladen: {', '.join(agents)}"
        return {"ready": await self._switch(agent=entry),
                "text": text + (f" · {len(warnings)} Warnungen" if warnings else "")}

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
        return {"ready": await self._switch(session=name)}

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
        self.shell_enter(agent, session)                 # Shell weiss, zu welchem Lauf sie gehoert
        # "#audio" am Ende: nur diese Antwort wird gesprochen (kein Dauer-Schalter)
        speak = query.rstrip().endswith(AUDIO_TAG)
        if speak:
            query = query.rstrip()[:-len(AUDIO_TAG)].rstrip()
            speak = await self._voice_begin()
            if not speak:
                yield {"type": "status", "status_msg": "Sprachausgabe nicht verfügbar – Antwort nur als Text"}
        query = self._skill_wrap(query)                  # aktive Skills voranstellen (wie alte icli)
        try:
            async for chunk in instance.a_stream(
                query=query, session_id=session, max_iterations=self.max_iterations,
            ):
                if speak and chunk.get("type") == "content" and not chunk.get("_sub_agent_id"):
                    self._voice_feed(chunk.get("chunk", ""))
                yield chunk
        finally:
            if speak:
                self._voice_flush()

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
        """Pausierte Laeufe sichern, Scheduler wie die alte icli beenden.
        app.a_exit() ruft der Flow-Starter nach dem UI-Ende auf."""
        if self.isaa is None:
            return
        self.watch_close()
        await self._voice_close()
        await self._bridges_close()
        try:
            await asyncio.to_thread(self._save_paused)
        except Exception as e:
            self.app.logger.warning(f"icli2 save paused: {e}")
        if self._jobs is not None and self._job_callback is not None:
            for step in (lambda: self._jobs.fire_lifecycle("on_cli_exit"), self._jobs.stop):
                try:
                    await step()
                except Exception as e:
                    self.app.logger.warning(f"icli2 job shutdown: {e}")
            self._jobs._fire_callback = self._job_callback
