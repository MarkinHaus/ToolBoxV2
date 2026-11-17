"""MCP-Server und Claude-Code-Plugins der icli 2.0 (in IsaaBackend eingemischt, toolboxv2-Importe lazy).

Genutzt: MCPSessionManager.get_session_with_timeout/_cleanup_session, _register_mcp_session,
agent.tool_manager.unregister, skills_manager.import_skill. Speicherorte wie die alte icli:
  MCP:     <data_dir>/Agents/<agent>/agent.json -> mcp.servers[{name, command, args, env, transport[, url]}]
  Plugins: <appdata>/icli/plugins/<plugin>/.claude-plugin/plugin.json (+ skills/, .mcp.json)
Neu: <appdata>/icli/plugins_disabled.json (deaktivierte Plugins; die alte icli laedt immer alle).
Plugin-Hooks werden hier nicht ausgefuehrt (kein Host-Lebenszyklus im Backend).
"""
from __future__ import annotations

import asyncio
import json
import shlex
from datetime import datetime
from pathlib import Path

MCP_BEISPIEL = "Beispiel:  fs npx @modelcontextprotocol/server-filesystem /pfad   oder  web http://localhost:8000/mcp"


# ------------------------------------------------- Plugin-Parser (1:1 aus icli.py)
class ClaudeCodePluginParser:
    """Parse Claude Code plugin layout (.claude-plugin/plugin.json + skills/*/SKILL.md)."""
    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[dict, str]:
        parts = text.split("---", 2)
        if not text.startswith("---") or len(parts) < 3:
            return {}, text.strip()
        meta: dict = {}
        try:
            import yaml
            loaded = yaml.safe_load(parts[1])
            meta = loaded if isinstance(loaded, dict) else {}
        except Exception:
            pass
        return meta, parts[2].strip()

    @classmethod
    def parse_skill_md(cls, file_path: Path) -> dict:
        meta, body = cls._parse_frontmatter(file_path.read_text(encoding="utf-8", errors="replace"))
        return {"metadata": meta, "instruction": body}

    @staticmethod
    def to_skill_data(plugin_name: str, skill_id: str, parsed: dict) -> dict:
        meta = parsed.get("metadata", {}) or {}
        raw = meta.get("triggers", [])
        if isinstance(raw, str):
            raw = [t.strip() for t in raw.split(",") if t.strip()]
        name = str(meta.get("name", skill_id))
        return {"id": f"plugin_{plugin_name}_{skill_id}", "name": name,
                "triggers": list(dict.fromkeys([*raw, skill_id, name])),
                "instruction": parsed.get("instruction", ""), "tools_used": [], "tool_groups": [],
                "source": "plugin", "confidence": 0.85, "activation_threshold": 0.6,
                "success_count": 0, "failure_count": 0,
                "created_at": datetime.now().isoformat(), "last_used": None}


PLUGIN_SCAN_EXT = {".sh", ".bash", ".zsh", ".py", ".js", ".ts", ".mjs",
                   ".rb", ".pl", ".ps1", ".json", ".md", ".txt", ""}
PLUGIN_DANGER_PATTERNS = [
    ("danger", "pipe-to-shell (remote code exec)", r"(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh\b"),
    ("danger", "destructive delete", r"\brm\s+-[a-z]*r[a-z]*f?\s+(/|~|\$HOME|\*)"),
    ("danger", "credential / secret access", r"(\.ssh/|id_rsa|\.aws/credentials|\.env\b|/Cookies|wallet\.dat|\.netrc|keychain)"),
    ("danger", "permission-bypass flag", r"(--dangerously-skip-permissions|disableAllHooks|bypassPermissions)"),
    ("danger", "base64 decode then exec", r"base64\s+(-d|--decode)[^\n]*\|\s*(ba)?sh\b"),
    ("warn", "dynamic code execution", r"\beval\s*\(|\bexec\s*\(|new\s+Function\s*\(|os\.system\s*\(|child_process|shell\s*=\s*True|pickle\.loads"),
    ("warn", "outbound network call", r"\b(nc|netcat|Invoke-WebRequest)\b|requests\.(get|post)\s*\(|\bfetch\s*\(|\b(curl|wget)\b"),
    ("warn", "persistence / scheduling", r"\b(crontab|launchctl|systemctl\s+enable|schtasks|reg\s+add)\b"),
]


def scan_plugin_dir(root: Path, max_bytes: int = 200_000) -> list[dict]:
    """Grep plugin files for dangerous patterns. Returns findings (best-effort)."""
    import re
    compiled = [(lvl, lbl, re.compile(rx, re.I)) for lvl, lbl, rx in PLUGIN_DANGER_PATTERNS]
    findings: list[dict] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git/" in p.as_posix() or p.suffix.lower() not in PLUGIN_SCAN_EXT:
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            for lvl, lbl, rx in compiled:
                if rx.search(line):
                    findings.append({"level": lvl, "file": rel, "line": i,
                                     "label": lbl, "snippet": line.strip()[:80]})
    return findings


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_plugin(folder: Path) -> dict | None:
    if not folder.is_dir() or not (folder / ".claude-plugin" / "plugin.json").exists():
        return None
    manifest = _read_json(folder / ".claude-plugin" / "plugin.json")
    entry = {"name": manifest.get("name") or folder.name, "manifest": manifest, "root": folder, "skills": {}, "mcp": {}}
    sd = folder / "skills"
    files = list(sd.glob("**/SKILL.md")) if sd.exists() else [f for f in [folder / "SKILL.md"] if f.exists()]
    for sf in files:
        try:
            parsed = ClaudeCodePluginParser.parse_skill_md(sf)
            entry["skills"][parsed["metadata"].get("name") or sf.parent.name] = parsed
        except Exception:
            continue
    try:
        raw = _read_json(folder / ".mcp.json")
        srv = raw.get("mcpServers", raw) if isinstance(raw, dict) else {}
        entry["mcp"] = {k: v for k, v in srv.items() if isinstance(v, dict)}
    except Exception:
        pass
    return entry


def _short(text: str, n: int = 60) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


class ConnectMixin:
    def _agent_json(self) -> Path:
        return Path(self.app.data_dir) / "Agents" / self.agent / "agent.json"

    def _mcp_servers_cfg(self) -> list[dict]:
        p = self._agent_json()
        servers = ((_read_json(p).get("mcp") or {}).get("servers") or []) if p.exists() else []
        return [s for s in servers if isinstance(s, dict) and s.get("name")]

    @staticmethod
    def _mcp_mgr(agent):
        if not getattr(agent, "_mcp_session_manager", None):
            from toolboxv2.mods.isaa.extras.mcp_session_manager import MCPSessionManager
            agent._mcp_session_manager = MCPSessionManager()
        return agent._mcp_session_manager

    async def _mcp_connect(self, agent, name: str, cfg: dict) -> int:
        from toolboxv2.mods.isaa.extras.mcp_session_manager import _register_mcp_session
        mgr = self._mcp_mgr(agent)
        session = await mgr.get_session_with_timeout(name, cfg)
        if session is None:
            raise RuntimeError(f"MCP {name}: Verbindung fehlgeschlagen - Befehl/URL pruefen")
        return await _register_mcp_session(agent, session, name, mgr)

    async def _mcp_disconnect(self, agent, name: str) -> None:
        await self._mcp_mgr(agent)._cleanup_session(name)
        for t in [t for t in list(agent.tool_manager._registry) if t.startswith(f"{name}_")]:
            agent.tool_manager.unregister(t)

    async def _rows_mcp(self) -> tuple[str, list[dict]]:
        agent = await self._agent()
        mgr = getattr(agent, "_mcp_session_manager", None)
        live = set(getattr(mgr, "sessions", None) or {})
        cfgs = {s["name"]: s for s in await asyncio.to_thread(self._mcp_servers_cfg)}
        cfgs.update({n: {"name": n} for n in live - set(cfgs)})  # verbunden, aber nicht in agent.json
        rows = []
        for n, c in sorted(cfgs.items()):
            hint = _short(c.get("url") or " ".join([str(c.get("command", "")), *map(str, c.get("args") or [])]).strip())
            if n in live:
                hint = f"{hint}  {len((mgr.capabilities_cache.get(n) or {}).get('tools', {}))} Tools".strip()
            rows.append({"value": n, "label": ("✓ " if n in live else "· ") + n, "hint": hint})
        return f"MCP-Server ({self.agent})", rows

    async def _do_mcp(self, name: str) -> dict:
        agent = await self._agent()
        if name in self._mcp_mgr(agent).sessions:
            await self._mcp_disconnect(agent, name)
            return {"text": f"MCP {name} getrennt"}
        cfg = next((s for s in await asyncio.to_thread(self._mcp_servers_cfg) if s["name"] == name), None)
        if cfg is None:
            raise ValueError(f"MCP-Server '{name}' ist nicht konfiguriert")
        n = await self._mcp_connect(agent, name, {k: v for k, v in cfg.items() if k != "name"})
        return {"text": f"MCP {name} verbunden ({n} Tools)"}

    async def _do_mcp_add(self, spec: str) -> dict:
        try:
            parts = shlex.split(spec or "")
        except ValueError:
            parts = []
        if len(parts) < 2:
            raise ValueError(f"Ungueltige Angabe. {MCP_BEISPIEL}")
        name, rest = parts[0], parts[1:]
        if len(rest) == 1 and rest[0].startswith(("http://", "https://")):
            cfg = {"command": "", "args": [], "transport": "sse" if "sse" in rest[0] else "http", "env": {}, "url": rest[0]}
        else:
            cfg = {"command": rest[0], "args": rest[1:], "transport": "stdio", "env": {}}
        n = await self._mcp_connect(await self._agent(), name, cfg)
        if await asyncio.to_thread(self._mcp_persist, name, cfg):
            return {"text": f"MCP {name} verbunden ({n} Tools), in agent.json gespeichert"}
        return {"text": f"MCP {name} verbunden ({n} Tools), agent.json fehlt - nicht gespeichert", "level": "warn"}

    def _mcp_persist(self, name: str, cfg: dict) -> bool:
        p = self._agent_json()
        if not p.exists():
            return False
        data = _read_json(p)
        mcp = data.setdefault("mcp", {"enabled": True, "servers": []})
        mcp["servers"] = [s for s in mcp.setdefault("servers", []) if s.get("name") != name] + [{"name": name, **cfg}]
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True

    def _plugins_dir(self) -> Path:
        return Path(self.app.appdata) / "icli" / "plugins"

    def _dfile(self) -> Path:
        return Path(self.app.appdata) / "icli" / "plugins_disabled.json"

    def _plugins_disabled(self) -> set[str]:
        try:
            return set(_read_json(self._dfile()))
        except Exception:
            return set()

    def _plugins_scan(self) -> dict[str, dict]:
        found, d = {}, self._plugins_dir()
        for folder in sorted(d.iterdir()) if d.exists() else []:
            try:
                e = _read_plugin(folder)
            except Exception as ex:
                self.app.logger.warning(f"Plugin {folder.name}: plugin.json defekt ({ex})")
                continue
            if e:
                found[e["name"]] = e
        return found

    def _plugins_set_disabled(self, name: str, off: bool) -> None:
        cur, f = self._plugins_disabled(), self._dfile()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(sorted(cur | {name} if off else cur - {name})), encoding="utf-8")

    @property
    def _plugin_active(self) -> dict[str, dict]:
        return self.__dict__.setdefault("_plugin_active_d", {})

    async def _rows_plugins(self) -> tuple[str, list[dict]]:
        rows = []
        for n, e in (await asyncio.to_thread(self._plugins_scan)).items():
            hint = f"{len(e['skills'])} Skills, {len(e['mcp'])} MCP-Server"
            if not e["skills"] and not e["mcp"]:
                hint = str(e["manifest"].get("description") or hint)
            rows.append({"value": n, "label": ("✓ " if n in self._plugin_active else "· ") + n, "hint": _short(hint)})
        return "Plugins", rows

    async def _plugin_activate(self, entry: dict) -> str:
        # Skills + MCP-Server (<plugin>_<server>) wie _load_all_plugins
        name, agent = entry["name"], await self._agent()
        sm = agent._get_execution_engine().skills_manager
        for sid, parsed in entry["skills"].items():
            sm.import_skill(ClaudeCodePluginParser.to_skill_data(name, sid, parsed), overwrite=True)
        servers = []
        for srv, raw in entry["mcp"].items():
            cfg = dict(raw)
            if "url" in cfg:
                cfg.setdefault("transport", "sse" if "sse" in str(cfg["url"]) else "http")
            else:
                cfg = {"transport": "stdio", "args": [], "env": {}, **cfg}
            try:
                await self._mcp_connect(agent, f"{name}_{srv}", cfg)
                servers.append(f"{name}_{srv}")
            except Exception as ex:
                self.app.logger.warning(f"Plugin {name}: MCP {srv} fehlgeschlagen ({ex})")
        self._plugin_active[name] = {"skills": [f"plugin_{name}_{s}" for s in entry["skills"]], "mcp": servers}
        return f"{len(entry['skills'])} Skills, {len(servers)} MCP-Server"

    async def _do_plugin(self, name: str) -> dict:
        if name in self._plugin_active:  # deaktivieren: Skills entfernen, MCP trennen
            agent, st = await self._agent(), self._plugin_active.pop(name)
            sm = agent._get_execution_engine().skills_manager
            for sid in st["skills"]:
                sm.skills.pop(sid, None)
            for srv in st["mcp"]:
                await self._mcp_disconnect(agent, srv)
            await asyncio.to_thread(self._plugins_set_disabled, name, True)
            return {"text": f"Plugin {name} deaktiviert"}
        found = await asyncio.to_thread(self._plugins_scan)
        if name not in found:
            raise ValueError(f"Plugin '{name}' nicht gefunden in {self._plugins_dir()}")
        info = await self._plugin_activate(found[name])
        await asyncio.to_thread(self._plugins_set_disabled, name, False)
        return {"text": f"Plugin {name} aktiviert ({info})"}

    async def _plugins_boot(self) -> None:
        try:
            disabled = await asyncio.to_thread(self._plugins_disabled)
            for n, e in (await asyncio.to_thread(self._plugins_scan)).items():
                if n in disabled or n in self._plugin_active:
                    continue
                try:
                    await self._plugin_activate(e)
                except Exception as ex:
                    self.app.logger.warning(f"Plugin {n}: Laden fehlgeschlagen ({ex})")
        except Exception as ex:
            self.app.logger.warning(f"Plugins: Start fehlgeschlagen ({ex})")
