"""Plugin-Installation der icli 2.0 (in dieselbe Host-Klasse wie ConnectMixin eingemischt).

Spiegelt die alte icli (_plugin_install_from_git, _plugin_browse, /plugin marketplace ...), aber ohne
Rueckfrage im Terminal: install -> Staging -> (bei Funden) confirm/discard. Es wird nichts aus dem
geklonten Repository ausgefuehrt oder importiert; Plugin-Hooks laufen nicht.
Marketplaces: <appdata>/icli/plugin_marketplaces.json = JSON-Liste von Quellen (wie die alte icli).
Wie die alte icli: Plugin = Ordner mit .claude-plugin/plugin.json (auch verschachtelt im Repo),
Name aus plugin.json, Quelle auch als 'name@marketplace'. Anders: es wird nie ueberschrieben.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_REMOTE = ("https://", "http://", "git@", "ssh://", "file://")


def _check_name(name: str) -> str:
    """Name pruefen (nur Buchstaben, Ziffern, - _ . ; nicht am Anfang ein - oder .)."""
    if not isinstance(name, str) or ".." in name or "/" in name or "\\" in name or not _NAME_RE.match(name):
        raise ValueError(f"Ungueltiger Plugin-Name: {name!r}")
    return name


def plugin_name_from_source(source: str) -> str:
    """Name aus einer Quelle ableiten (wie die alte icli): letzter Pfadteil ohne ".git"."""
    s = (source or "").strip().rstrip("/\\")
    if not s:
        raise ValueError("Keine Quelle angegeben")
    last = re.split(r"[/\\:]", s)[-1]
    if last.endswith(".git"):
        last = last[:-4]
    return _check_name(last)


def to_git_url(source: str) -> str:
    """"owner/repo" -> GitHub-URL; URLs, git@ und lokale Pfade bleiben unveraendert."""
    s = source.strip()
    if s.startswith(_REMOTE) or s.startswith(("/", ".", "~")) or os.path.exists(s):
        return s
    if re.match(r"^[\w.-]+/[\w.-]+$", s):
        return f"https://github.com/{s.strip('/')}.git"
    return s


def _symlink_findings(root: Path) -> list[dict]:
    """Symlinks, die aus dem Plugin hinauszeigen, als Funde melden."""
    out, base = [], root.resolve()
    for cur, dirs, files in os.walk(root, followlinks=False):
        for n in dirs + files:
            p = Path(cur) / n
            if p.is_symlink() and not p.resolve().is_relative_to(base):
                out.append({"level": "danger", "file": p.relative_to(root).as_posix(), "line": 0,
                            "label": "Symlink zeigt ausserhalb des Plugins", "snippet": os.readlink(p)[:80]})
    return out


def _report(root: Path) -> tuple[str, bool]:
    """Sicherheitsbericht (wie _plugin_security_report) -> (Text, gibt_es_Funde)."""
    from .connect import scan_plugin_dir
    findings = scan_plugin_dir(root) + _symlink_findings(root)
    lines, hook_events = [], []
    hp = root / "hooks" / "hooks.json"
    if hp.exists():
        try:
            hook_events = list(json.loads(hp.read_text(encoding="utf-8")).keys())
        except Exception:
            hook_events = ["<unlesbar>"]
    if hook_events:
        lines.append(f"! Hooks fuer: {', '.join(hook_events)} (werden hier nicht ausgefuehrt)")
    if (root / ".mcp.json").exists():
        lines.append("! Enthaelt .mcp.json - externer MCP-Server verbindet sich automatisch")
    for f in findings:
        if f["level"] == "danger":
            lines.append(f"GEFAHR {f['file']}:{f['line']} - {f['label']}  {f['snippet']}")
    for f in findings:
        if f["level"] != "danger":
            lines.append(f"- {f['file']}:{f['line']} - {f['label']}")
    return "\n".join(lines), bool(lines)


def _find_root(stage: Path) -> Path | None:
    """Plugin-Wurzel wie die alte icli: Ordner mit .claude-plugin/plugin.json, oben oder verschachtelt.
    Symlinks werden dabei nicht verfolgt."""
    if (stage / ".claude-plugin" / "plugin.json").is_file():
        return stage
    for cur, dirs, _files in os.walk(stage, followlinks=False):
        dirs.sort()
        if Path(cur).name == ".claude-plugin" and (Path(cur) / "plugin.json").is_file():
            return Path(cur).parent
    return None


def _manifest_name(root: Path) -> str:
    """Name aus plugin.json (sonst Ordnername), geprueft wie jeder andere Name."""
    try:
        manifest = json.loads((root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"plugin.json ist defekt: {e}")
    return _check_name(str((manifest.get("name") if isinstance(manifest, dict) else None) or root.name))


def _fetch_market(src: str) -> dict:
    """marketplace.json laden (URL, file://, lokale Datei/Ordner), Zeitlimit 8 s."""
    import urllib.request
    raw = "https://raw.githubusercontent.com/{}/HEAD/.claude-plugin/marketplace.json"
    if src.startswith("file://") and not src.endswith(".json"):
        src = src.rstrip("/") + "/.claude-plugin/marketplace.json"
    elif src.startswith(("http://", "https://")) and not src.endswith(".json"):
        s = src.split("github.com/", 1)[-1] if "github.com/" in src else src
        s = s.rstrip("/")
        src = raw.format(s[:-4] if s.endswith(".git") else s)
    elif not src.startswith("file://"):
        if src.endswith(".json") or os.path.exists(src):
            p = Path(src)
            p = p if p.is_file() else p / ".claude-plugin" / "marketplace.json"
            return json.loads(p.read_text(encoding="utf-8"))
        src = raw.format(src.rstrip("/"))
    with urllib.request.urlopen(src, timeout=8) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _plugin_source(source, base=None) -> str | None:
    """Marketplace-Eintrag `source` -> Installationsquelle (wie _source_to_git_url der alten icli).

    Relative Pfade ("./plugins/x") werden gegen `base` (Wurzel des lokalen
    Marketplace-Ordners) aufgeloest; ohne base bleiben sie ungueltig (wie alte icli).
    """
    if isinstance(source, str):
        if source.startswith(("./", "../")):
            if base is None:
                return None
            root = Path(base).resolve()
            try:
                resolved = (root / source).resolve()
                resolved.relative_to(root)          # Containment: nicht aus der Wurzel raus
            except ValueError:
                return None
            return str(resolved)
        return source
    if isinstance(source, dict):
        if source.get("source") == "github" and source.get("repo"):
            return f"https://github.com/{source['repo']}"
        if source.get("source") in ("git", "url", "git-subdir") and source.get("url"):
            return source["url"]
    return None


def _short(t: str, n: int = 60) -> str:
    return t if len(t) <= n else t[:n - 1] + "…"


class PluginInstallMixin:
    def _staging_dir(self) -> Path:
        return self._plugins_dir() / ".staging"

    def _markets_file(self) -> Path:
        return Path(self.app.appdata) / "icli" / "plugin_marketplaces.json"

    def _markets_load(self) -> list:
        try:
            return list(json.loads(self._markets_file().read_text(encoding="utf-8")))
        except Exception:
            return []

    def _markets_save(self, markets: list) -> None:
        f = self._markets_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(markets, indent=2), encoding="utf-8")

    def _safe_path(self, base: Path, name: str) -> Path:
        """Name pruefen und Zielpfad innerhalb von base sicherstellen."""
        _check_name(name)
        p = (base / name).resolve()
        if not p.is_relative_to(base.resolve()) or p == base.resolve():
            raise ValueError(f"Pfad ausserhalb des Plugin-Ordners: {name!r}")
        return p

    async def _resolve_market_ref(self, ref: str) -> str:
        """'name@marketplace' (wie die alte icli) -> Installationsquelle aus dem Marketplace."""
        name, _, market = ref.partition("@")
        for src in await asyncio.to_thread(self._markets_load):
            try:
                data = await asyncio.to_thread(_fetch_market, src)
            except Exception:
                continue
            if market and data.get("name") != market:
                continue
            # Relative Pfad-Eintraege nur gegen eine lokale Marketplace-Wurzel aufloesbar
            base = src if os.path.exists(src) and not src.startswith("file://") else None
            for pl in data.get("plugins", []):
                if pl.get("name") == name:
                    found = _plugin_source(pl.get("source", {}), base=base)
                    if not found:
                        hint = " (relativer Pfad nur bei lokalem Marketplace aufloesbar)" \
                            if base is None else ""
                        raise ValueError(f"Quelle von '{name}' wird nicht unterstuetzt{hint}")
                    return found
        raise ValueError(f"'{ref}' in keinem gespeicherten Marketplace gefunden")

    async def _do_plugin_install(self, source: str) -> dict:
        """Klonen ins Staging, pruefen; ohne Funde direkt installieren, sonst confirm/discard."""
        source = (source or "").strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+", source):
            source = await self._resolve_market_ref(source)
        if source.startswith("-") or "::" in source:
            raise ValueError("Ungueltige Quelle")
        stage_name = plugin_name_from_source(source)
        stage = self._safe_path(self._staging_dir(), stage_name)
        url = to_git_url(source)
        await asyncio.to_thread(shutil.rmtree, stage, True)
        stage.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "file:https:http:ssh"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "clone", "--depth", "1", "--", url, str(stage), env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError("git wurde nicht gefunden (PATH)")
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            await asyncio.to_thread(shutil.rmtree, stage, True)
            raise RuntimeError("git clone: Zeitlimit von 120 s ueberschritten")
        if proc.returncode != 0:
            await asyncio.to_thread(shutil.rmtree, stage, True)
            lines = err.decode(errors="replace").strip().splitlines()
            raise RuntimeError(f"git clone fehlgeschlagen: {lines[-1] if lines else 'unbekannter Fehler'}")
        await asyncio.to_thread(shutil.rmtree, stage / ".git", True)
        try:
            root, name = await asyncio.to_thread(self._inspect_stage, stage)
        except Exception:
            await asyncio.to_thread(shutil.rmtree, stage, True)
            raise
        text, found = await asyncio.to_thread(_report, root)
        if not found:
            return await self._do_plugin_confirm(stage_name)
        rename = f" (Name laut plugin.json: {name})" if name != stage_name else ""
        return {"level": "warn", "text": f"Sicherheitsbericht {stage_name}{rename}:\n{text}\n"
                f"Zum Installieren:  /plugin confirm {stage_name}     Verwerfen:  /plugin discard {stage_name}"}

    def _inspect_stage(self, stage: Path) -> tuple[Path, str]:
        """Plugin-Wurzel und endgueltigen Namen bestimmen; schon installiert -> ValueError."""
        root = _find_root(stage)
        if root is None:
            raise ValueError("Kein Claude-Code-Plugin: .claude-plugin/plugin.json fehlt")
        name = _manifest_name(root)
        if self._safe_path(self._plugins_dir(), name).exists():
            raise ValueError(f"Plugin '{name}' ist bereits installiert")
        return root, name

    async def _do_plugin_confirm(self, name: str) -> dict:
        """Gepruefte Kopie aus dem Staging uebernehmen und aktivieren."""
        stage = self._safe_path(self._staging_dir(), name)
        if not stage.is_dir():
            raise ValueError(f"Kein Staging fuer '{name}' vorhanden")
        root, final = await asyncio.to_thread(self._inspect_stage, stage)
        dest = self._safe_path(self._plugins_dir(), final)
        await asyncio.to_thread(shutil.move, str(root), str(dest))
        await asyncio.to_thread(shutil.rmtree, stage, True)          # Rest des Repos (bei verschachtelter Wurzel)
        entry = next((e for e in (await asyncio.to_thread(self._plugins_scan)).values()
                      if Path(e["root"]).resolve() == dest), None)
        if entry is None:
            return {"level": "warn", "text": f"Plugin {final} installiert, aber nicht lesbar - siehe Log"}
        info = await self._plugin_activate(entry)
        await asyncio.to_thread(self._plugins_set_disabled, entry["name"], False)
        return {"text": f"Plugin {entry['name']} installiert: {info}"}

    async def _do_plugin_discard(self, name: str) -> dict:
        stage = self._safe_path(self._staging_dir(), name)
        if not stage.is_dir():
            raise ValueError(f"Kein Staging fuer '{name}' vorhanden")
        await asyncio.to_thread(shutil.rmtree, stage, True)
        return {"text": f"Staging '{name}' verworfen"}

    async def _do_plugin_remove(self, name: str) -> dict:
        dest = self._safe_path(self._plugins_dir(), name)
        if not dest.is_dir():
            raise ValueError(f"Plugin '{name}' ist nicht installiert")
        for e in (await asyncio.to_thread(self._plugins_scan)).values():
            if Path(e["root"]).resolve() == dest and e["name"] in self._plugin_active:
                await self._do_plugin(e["name"])  # deaktivieren (Skills/MCP entfernen)
                await asyncio.to_thread(self._plugins_set_disabled, e["name"], False)
        await asyncio.to_thread(shutil.rmtree, dest, True)
        return {"text": f"Plugin {name} entfernt"}

    async def _rows_market(self) -> tuple[str, list[dict]]:
        rows = []
        installed = set(await asyncio.to_thread(self._plugins_scan))
        pdir = self._plugins_dir()
        for src in self._markets_load():
            try:
                data = await asyncio.to_thread(_fetch_market, src)
                mname = str(data.get("name", src))
                for pl in data.get("plugins", []):
                    pn, inst = str(pl.get("name", "?")), _plugin_source(pl.get("source", {}))
                    done = pn in installed or (pdir / pn).exists()
                    desc = str(pl.get("description", "") or "")
                    hint = _short(f"{mname} · {desc}" if desc else mname)
                    if not inst:
                        hint = _short(f"{mname} · Quelle nicht unterstuetzt")
                    rows.append({"value": inst or "", "label": ("✓ " if done else "") + pn, "hint": hint})
            except Exception as ex:
                rows.append({"value": "", "label": src, "hint": _short(f"Nicht erreichbar: {ex}")})
        if not rows:
            return "Keine Marketplaces gespeichert – /plugin market owner/repo", rows
        return "Marketplace – Enter installiert", rows

    async def _do_market_add(self, source: str) -> dict:
        source = (source or "").strip()
        if not source or source.startswith("-"):
            raise ValueError("Beispiel:  /plugin market owner/repo")
        markets = await asyncio.to_thread(self._markets_load)
        if source not in markets:
            await asyncio.to_thread(self._markets_save, markets + [source])
        return {"text": f"Marketplace hinzugefuegt: {source}"}

    async def _do_market_remove(self, source: str) -> dict:
        markets = await asyncio.to_thread(self._markets_load)
        await asyncio.to_thread(self._markets_save, [m for m in markets if m != source])
        return {"text": f"Marketplace entfernt: {source}"}

    async def _rows_markets(self) -> tuple[str, list[dict]]:
        rows = [{"value": m, "label": m, "hint": "Enter entfernt"} for m in self._markets_load()]
        return ("Marketplaces" if rows else "Keine Marketplaces gespeichert – /plugin market owner/repo"), rows
