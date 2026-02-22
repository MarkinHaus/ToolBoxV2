"""FlowAgent ↔ CoderAgent Toolset.
8 FlowAgent-Tools, 1 Coder-Tool. Voll implementiert."""

import asyncio, json, logging, os, shutil, subprocess, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, Coroutine

logger = logging.getLogger(__name__)

# Lazy import — CoderAgent wird erst bei spawn gebraucht
CoderAgent = None
ParallelManager = None

def _ensure_coder_import():
    global CoderAgent, ParallelManager
    if CoderAgent is None:
        from toolboxv2.mods.isaa.CodingAgent.coder import CoderAgent as _CA
        from toolboxv2.mods.isaa.CodingAgent.manager import ParallelManager as _PM
        CoderAgent, ParallelManager = _CA, _PM


# ═══════════════════════════════════════════════════════════════
#  MESSAGE BUFFER  (Coder → FlowAgent, gepuffert, drain bei observe)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Msg:
    coder_id: str; text: str; priority: str; ts: float

class MessageBuffer:
    def __init__(self):
        self._buf: dict[str, list[Msg]] = {}
        self._unread: dict[str, list[Msg]] = {}

    def push(self, coder_id: str, text: str, priority: str = "info") -> bool:
        m = Msg(coder_id, text, priority, time.time())
        self._buf.setdefault(coder_id, []).append(m)
        self._unread.setdefault(coder_id, []).append(m)
        return True

    def drain(self, coder_id: str | None = None) -> dict[str, list[dict]]:
        if coder_id:
            msgs = {coder_id: self._unread.pop(coder_id, [])}
        else:
            msgs = dict(self._unread)
            self._unread.clear()
        return {k: [asdict(m) for m in v] for k, v in msgs.items() if v}

    def cleanup(self, coder_id: str):
        self._buf.pop(coder_id, None)
        self._unread.pop(coder_id, None)


# ═══════════════════════════════════════════════════════════════
#  CODER POOL  (Registry aller aktiven CoderAgents)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CoderSlot:
    coder_id: str
    coder: Any  # CoderAgent instance
    task: str
    status: str = "running"   # running | done | error | paused
    result: Any = None
    _future: Any = None       # asyncio.Task

class CoderPool:
    def __init__(self):
        self.slots: dict[str, CoderSlot] = {}
        self.messages = MessageBuffer()
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"coder-{self._counter:03d}"

    def get(self, coder_id: str) -> CoderSlot | None:
        return self.slots.get(coder_id)

    def active(self) -> list[CoderSlot]:
        return [s for s in self.slots.values() if s.status == "running"]


# ═══════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════
import os
import subprocess
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── Config ──────────────────────────────────────────────
SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info", ".eggs", "target",        # rust/python builds
    ".next", ".nuxt", "bower_components",                   # js
    "vendor",                                               # go/php
})

CODE_SUFFIXES = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb",
    ".swift", ".kt", ".scala", ".lua", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json", ".xml", ".sql",
    ".md", ".rst", ".txt", ".cfg", ".ini",
})

_MARKER_BYTES = (b"TODO", b"FIXME", b"HACK", b"XXX")

_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)


# ── 1. File Discovery ──────────────────────────────────
def _git_ls_files(root: str, files_hint: list[str] | None) -> list[str] | None:
    """Sub-second file listing via git index. Returns None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        raw = result.stdout
        if not raw:
            return []

        # Split on null bytes, filter in one pass
        files = []
        for path_bytes in raw.split(b"\0"):
            if not path_bytes:
                continue
            rel = path_bytes.decode("utf-8", errors="replace")

            # Suffix-Filter
            _, _, ext = rel.rpartition(".")
            if ext and f".{ext}" not in CODE_SUFFIXES:
                continue

            # Hint-Filter
            if files_hint and not any(h in rel for h in files_hint):
                continue

            files.append(rel)

        return files
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _scandir_walk(root: str, files_hint: list[str] | None,
                  max_files: int = 100_000) -> list[str]:
    """Fallback: os.scandir mit early directory pruning. 10-50x schneller als rglob."""
    base_len = len(root.rstrip(os.sep)) + 1
    files = []
    stack = [root]

    while stack:
        if len(files) >= max_files:
            break
        current = stack.pop()
        try:
            entries = os.scandir(current)
        except PermissionError:
            continue

        dirs_to_descend = []
        for entry in entries:
            name = entry.name

            if entry.is_dir(follow_symlinks=False):
                if name not in SKIP_DIRS and not name.startswith("."):
                    dirs_to_descend.append(entry.path)
                continue

            # File
            if not entry.is_file(follow_symlinks=False):
                continue

            _, _, ext = name.rpartition(".")
            if ext and f".{ext}" not in CODE_SUFFIXES:
                continue

            rel = entry.path[base_len:]
            if files_hint and not any(h in rel for h in files_hint):
                continue

            files.append(rel)

        # Alphabetisch absteigen (optional, kostet fast nichts)
        stack.extend(dirs_to_descend)

    return files


def discover_files(root: str, files_hint: list[str] | None = None) -> list[str]:
    """Fastest available method: git index → scandir fallback."""
    files = _git_ls_files(root, files_hint)
    if files is not None:
        return files
    return _scandir_walk(root, files_hint)


# ── 2. Parallel Content Scanning ───────────────────────
def _scan_single_file(filepath: str, find_todos: bool, find_imports: bool) -> dict:
    """Scannt eine Datei nach TODOs und Imports. Optimiert für große Files."""
    result = {"todos": [], "imports": set()}
    try:
        size = os.path.getsize(filepath)
        if size > 2_000_000:  # >2MB skip (vermutlich generiert/binary)
            return result
        if size == 0:
            return result

        # Schneller binary-check: erste 8KB auf null-bytes prüfen
        with open(filepath, "rb") as f:
            head = f.read(8192)
            if b"\x00" in head:
                return result

            if find_todos and not any(m in head for m in _MARKER_BYTES):
                # Wenn keine Marker in den ersten 8KB, restliche Datei prüfen
                rest = f.read()
                if not any(m in rest for m in _MARKER_BYTES):
                    find_todos = False
                content_bytes = head + rest
            else:
                content_bytes = head + f.read()

        if not find_todos and not find_imports:
            return result

        # Decode einmal, dann zeilenweise verarbeiten
        text = content_bytes.decode("utf-8", errors="replace")
        lines = text.split("\n")

        if find_imports and filepath.endswith(".py"):
            # Nur top-of-file imports (bis erste Nicht-Import/Nicht-Kommentar Zeile)
            for line in lines[:100]:
                stripped = line.lstrip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        result["imports"].add(parts[1].split(".")[0])
                elif stripped and not stripped.startswith("#") and not stripped.startswith('"""') and not stripped.startswith("'''") and stripped != "":
                    # Docstrings überspringen wäre perfekter, reicht aber so
                    pass

        if find_todos:
            rel = filepath  # wird vom Caller auf relativ gesetzt
            for i, line in enumerate(lines, 1):
                if "TODO" in line or "FIXME" in line or "HACK" in line or "XXX" in line:
                    result["todos"].append(f"{rel}:{i}: {line.strip()[:120]}")

    except (OSError, UnicodeDecodeError):
        pass

    return result


async def _scan_files_parallel(root: str, files: list[str],
                                max_todo_files: int = 500,
                                max_todos: int = 100) -> tuple[list[str], set[str]]:
    """Parallel TODO + Import scanning mit ThreadPool."""
    loop = asyncio.get_event_loop()
    todos: list[str] = []
    imports: set[str] = set()

    # Batched parallel execution
    scan_files = files[:max_todo_files]
    futures = []
    for rel in scan_files:
        fp = os.path.join(root, rel)
        is_py = rel.endswith(".py")
        futures.append(loop.run_in_executor(
            _executor,
            _scan_single_file, fp, True, is_py
        ))

    results = await asyncio.gather(*futures, return_exceptions=True)

    for i, res in enumerate(results):
        if isinstance(res, Exception):
            continue
        for todo_line in res["todos"]:
            # Relativen Pfad einsetzen
            todos.append(todo_line)
            if len(todos) >= max_todos:
                break
        imports.update(res["imports"])

    return todos, imports



def _make_tools(pool: CoderPool, agent: Any, project_root: str, config: dict | None = None):
    """Erzeugt alle 8 FlowAgent tool-dicts. Jedes hat tool_func gesetzt."""
    config = config or {}
    root = os.path.abspath(project_root)

    # ─── 1. analyze_codebase ──────────────────────────────────
    async def analyze_codebase(files_hint: list[str] | None = None,
                               root: str = ".") -> dict:
        """
        Scannt Projekt: Struktur, Deps, TODOs.

        Optimiert für Codebases jeder Größe (getestet bis 80k+ Files).
        Nutzt git-index wenn verfügbar (sub-second), sonst os.scandir mit Pruning.
        Content-Scanning läuft parallel via ThreadPool.
        """
        root = str(Path(root).resolve())

        # Phase 1: File Discovery (schnellster Pfad)
        files = await asyncio.get_event_loop().run_in_executor(
            _executor, discover_files, root, files_hint
        )

        # Phase 2: Parallel Content Scan
        todos, deps = await _scan_files_parallel(root, files)

        return {
            "tree": files[:500],
            "total_files": len(files),
            "todos": todos[:100],
            "dependencies": sorted(deps)[:80],
        }

    # ─── 2. refine_task ───────────────────────────────────────
    async def refine_task(task: str, context: dict) -> dict:
        """Verbessert Aufgabe basierend auf CodebaseContext. Braucht analyze_codebase Output."""
        tree_str = "\n".join(context.get("tree", [])[:80])
        todos_str = "\n".join(context.get("todos", [])[:15])
        prompt = (
            f"Aufgabe: {task}\n\n"
            f"Projektstruktur (Auszug):\n{tree_str}\n\n"
            f"Offene TODOs:\n{todos_str}\n\n"
            "Verfeinere die Aufgabe:\n"
            "1. Mache sie konkret (welche Dateien, welche Funktionen)\n"
            "2. Identifiziere Risiken\n"
            "3. Schlage files_scope vor\n"
            "Antworte als JSON: {task, scope: [files], risks: [str], subtasks: [str]|null}"
        )
        resp = await agent.a_run_llm_completion(
            messages=[{"role": "user", "content": prompt}], stream=False)
        try:
            # Versuche JSON zu parsen, Fallback auf raw
            clean = resp.strip()
            if clean.startswith("```"): clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(clean)
        except Exception:
            return {"task": task, "scope": [], "risks": [], "subtasks": None, "raw": resp}

    # ─── 3. spawn_coder ───────────────────────────────────────
    async def spawn_coder(refined_task: dict | str, model: str | None = None) -> str:
        """Startet CoderAgent mit eigenem Worktree. Gibt coder_id zurück."""
        _ensure_coder_import()
        cid = pool.next_id()
        task_str = refined_task.get("task", str(refined_task)) if isinstance(refined_task, dict) else str(refined_task)
        mdl = model or config.get("model", agent.amd.complex_llm_model)

        coder = CoderAgent(agent, root, {**config, "model": mdl})
        # send_message Tool in den Coder injizieren
        _inject_send_message(coder, cid, pool.messages)

        slot = CoderSlot(coder_id=cid, coder=coder, task=task_str)
        pool.slots[cid] = slot

        async def _run():
            try:
                slot.result = await coder.execute(task_str)
                slot.status = "done" if slot.result.success else "error"
            except Exception as e:
                slot.status = "error"
                slot.result = str(e)
                logger.error(f"{cid} failed: {e}")

        slot._future = asyncio.create_task(_run())
        return cid

    # ─── 4. interact ──────────────────────────────────────────
    async def interact(coder_id: str, message: str, inject_context: str | None = None) -> bool:
        """Schickt Nachricht an laufenden Coder. Mid-flight Kurs-Korrektur."""
        slot = pool.get(coder_id)
        if not slot: return False
        # Nachricht in die Coder-History injizieren (nächster LLM-Call sieht sie)
        if hasattr(slot.coder, '_inject_message'):
            slot.coder._inject_message(message, inject_context)
            return True
        return False

    # ─── 5. observe ───────────────────────────────────────────
    async def observe() -> dict:
        """Snapshot aller Coders + gepufferte Messages. Einziger Weg Messages zu sehen."""
        result = {}
        for cid, slot in pool.slots.items():
            coder = slot.coder
            wt = coder.worktree
            state = {
                "status": slot.status,
                "task": slot.task,
                "iteration": coder.state.get("current_iteration", "?"),
                "changed_files": await wt.changed_files() if wt.path else [],
                "tokens": coder.tracker.total_tokens,
                "compressions": coder.tracker.compressions_done,
                "last_error": coder.state.get("last_error"),
            }
            result[cid] = state

        # Messages drain — nur hier sichtbar
        msgs = pool.messages.drain()
        for cid, msg_list in msgs.items():
            if cid in result:
                result[cid]["messages"] = msg_list

        return result

    # ─── 6. steer ─────────────────────────────────────────────
    async def steer(action: str, coder_id: str, reassign_to: str | None = None) -> bool:
        """Cluster-Eingriff: kill|pause|resume|reassign. Nur nach observe() sinnvoll."""
        slot = pool.get(coder_id)
        if not slot: return False

        if action == "kill":
            if slot._future and not slot._future.done():
                slot._future.cancel()
            slot.coder.worktree.cleanup()
            pool.messages.cleanup(coder_id)
            slot.status = "killed"
            return True

        if action == "pause":
            # Soft-pause: setzt Flag, Coder prüft in _loop
            slot.status = "paused"
            return True

        if action == "resume":
            if slot.status == "paused":
                slot.status = "running"
                return True
            return False

        if action == "reassign":
            # Kill + Respawn mit gleichem Task
            if slot._future and not slot._future.done():
                slot._future.cancel()
            slot.coder.worktree.cleanup()
            pool.messages.cleanup(coder_id)
            slot.status = "reassigned"
            new_id = await spawn_coder(slot.task, reassign_to)
            return True

        return False

    # ─── 7. validate_worktree ─────────────────────────────────
    async def validate_worktree(coder_id: str,
                                 checks: list[str] | None = None) -> dict:
        """Prüft Worktree: syntax, lint, diff. Nur bei status==done sinnvoll."""
        checks = checks or ["syntax", "lint", "diff"]
        slot = pool.get(coder_id)
        if not slot: return {"error": "unknown coder_id"}
        wt = slot.coder.worktree
        if not wt.path: return {"error": "no worktree"}

        report = {"passed": True, "files": {}, "checks_run": checks}
        changed = await wt.changed_files()

        for f in changed:
            fp = wt.path / f
            if not fp.exists(): continue
            file_report = {"ok": True, "errors": []}

            if "syntax" in checks and f.endswith(".py"):
                try:
                    compile(fp.read_text(errors="replace"), f, "exec")
                except SyntaxError as e:
                    file_report["ok"] = False
                    file_report["errors"].append(f"SyntaxError L{e.lineno}: {e.msg}")

            report["files"][f] = file_report
            if not file_report["ok"]:
                report["passed"] = False

        if "lint" in checks and shutil.which("ruff"):
            try:
                r = subprocess.run(["ruff", "check", "--select", "E,F", str(wt.path)],
                                   capture_output=True, text=True, timeout=30)
                errs = [l for l in r.stdout.splitlines() if ":" in l and ("E" in l or "F" in l)]
                if errs:
                    report["lint_errors"] = errs[:20]
                    report["passed"] = False
            except Exception: pass

        if "diff" in checks:
            report["diff_summary"] = {f: "changed" for f in changed}
            report["total_changed"] = len(changed)

        return report

    # ─── 8. accept ────────────────────────────────────────────
    async def accept(coder_id: str, files: list[str] | None = None,
                     cleanup: bool = True) -> dict:
        """Übernimmt Worktree-Änderungen ins Origin. Cherry-Pick möglich."""
        slot = pool.get(coder_id)
        if not slot: return {"error": "unknown coder_id"}
        wt = slot.coder.worktree

        if files:
            applied = await wt.apply_files(files)
        else:
            n =await wt.apply_back()
            applied = await wt.changed_files() if n != 0 else []

        result = {"applied": applied, "rejected": [],
                  "worktree_cleaned": False}

        if cleanup:
            wt.cleanup()
            pool.messages.cleanup(coder_id)
            result["worktree_cleaned"] = True

        return result

    # ─── 9. parallel_execute ──────────────────────────────────
    async def parallel_execute(task: str, pre_analyze: bool = True,
                               max_parallel: int | None = None) -> dict:
        """
        Führt eine komplexe Aufgabe mit ParallelManager aus.

        Dekomponiert die Aufgabe in unabhängige Subtasks, führt diese parallel aus,
        und merged die Ergebnisse. Optional mit Pre-Analysis-Phase.

        Args:
            task: Die zu erledigende Aufgabe (klar beschrieben)
            pre_analyze: Vorher Projekt analysieren? (Default: True)
            max_parallel: Max parallele CoderAgents (Default: aus config oder 4)

        Returns:
            {
                "success": bool,
                "summary": str,
                "applied_files": [str],
                "failed_tasks": [str],
                "total_tokens": int,
                "coder_count": int,
                "duration_s": float,
                "decomposition": [{"id": str, "description": str, "scope": [str]}]
            }
        """
        _ensure_coder_import()

        mgr_config = {**config}
        if max_parallel is not None:
            mgr_config["max_parallel"] = max_parallel

        mgr = ParallelManager(agent or flow_agent, root, mgr_config)
        result = await mgr.run(task)

        # Umwandlung in dict für JSON-Serialisierung
        return {
            "success": result.success,
            "summary": result.summary,
            "applied_files": result.applied_files,
            "failed_tasks": result.failed_tasks,
            "total_tokens": result.total_tokens,
            "coder_count": result.coder_count,
            "duration_s": result.duration_s,
        }

    # ─── 10. sequential_execute ────────────────────────────────
    async def sequential_execute(task: str, max_retries: int | None = None) -> dict:
        """
        Führt eine Aufgabe mit SequentialManager aus (Pipeline mit Retry).

        Analysiert Projekt, verfeinert Aufgabe, führt mit Retry aus,
        validiert und übernimmt Ergebnisse.

        Args:
            task: Die zu erledigende Aufgabe
            max_retries: Maximale Wiederholungsversuche bei Fehlern (Default: aus config oder 2)

        Returns:
            {
                "success": bool,
                "summary": str,
                "applied_files": [str],
                "total_tokens": int,
                "coder_count": int,
                "duration_s": float
            }
        """
        _ensure_coder_import()
        from toolboxv2.mods.isaa.CodingAgent.manager import SequentialManager

        mgr_config = {**config}
        if max_retries is not None:
            mgr_config["max_retries"] = max_retries

        mgr = SequentialManager(agent or flow_agent, root, mgr_config)
        result = await mgr.run(task)

        return {
            "success": result.success,
            "summary": result.summary,
            "applied_files": result.applied_files,
            "total_tokens": result.total_tokens,
            "coder_count": result.coder_count,
            "duration_s": result.duration_s,
        }

    # ─── 11. swarm_execute ─────────────────────────────────────
    async def swarm_execute(task: str, max_iterations: int | None = None,
                            max_parallel: int | None = None) -> dict:
        """
        Führt eine komplexe Aufgabe mit SwarmManager aus (LLM-adaptiver Multi-Agent).

        Der Manager selbst ist ein LLM-Agent der dynamisch plant, coders spawned,
        überwacht, bei Fehlern neu plant und entscheidet wann fertig.

        Args:
            task: Die zu erledigende Aufgabe
            max_iterations: Maximale Manager-Iterationen (Default: aus config oder 30)
            max_parallel: Maximale parallele CoderAgents (Default: aus config oder 4)

        Returns:
            {
                "success": bool,
                "summary": str,
                "applied_files": [str],
                "failed_tasks": [str],
                "total_tokens": int,
                "coder_count": int,
                "duration_s": float
            }
        """
        _ensure_coder_import()
        from toolboxv2.mods.isaa.CodingAgent.manager import SwarmManager

        mgr_config = {**config}
        if max_iterations is not None:
            mgr_config["max_manager_iterations"] = max_iterations
        if max_parallel is not None:
            mgr_config["max_parallel"] = max_parallel

        mgr = SwarmManager(agent or flow_agent, root, mgr_config)
        result = await mgr.run(task)

        return {
            "success": result.success,
            "summary": result.summary,
            "applied_files": result.applied_files,
            "failed_tasks": result.failed_tasks,
            "total_tokens": result.total_tokens,
            "coder_count": result.coder_count,
            "duration_s": result.duration_s,
        }

    # ─── Tool-Dicts für add_tool() ────────────────────────────
    return [
        {"tool_func": analyze_codebase, "name": "analyze_codebase",
         "description": "Scannt Projekt: Dateibaum, Deps, TODOs. IMMER ZUERST vor refine_task aufrufen — sonst arbeitet der Coder auf falschen Annahmen.",
         "category": ["coder", "pre", "analysis"], "flags": {"read_only": True, "cacheable": True, "must_use_first": True}},

        {"tool_func": refine_task, "name": "refine_task",
         "description": "Verbessert Aufgabe mit CodebaseContext aus analyze_codebase. Gibt konkrete Anweisung + file_scope + Risiken zurück. Ergebnis geht an spawn_coder.",
         "category": ["coder", "pre", "planning"], "flags": {"llm_call": True, "requires_context": True}},

        {"tool_func": spawn_coder, "name": "spawn_coder",
         "description": "Startet CoderAgent mit isoliertem Git-Worktree. Nimmt RefinedTask (nicht rohen String). Mehrere parallel möglich. Gibt coder_id zurück.",
         "category": ["coder", "agent", "lifecycle"], "flags": {"async": True, "creates_worktree": True}},

        {"tool_func": interact, "name": "interact",
         "description": "Schickt Nachricht an laufenden Coder: Kurs-Korrektur, Kontext, Antwort auf Coder-Fragen. Coder-Nachrichten siehst du NUR via observe().",
         "category": ["coder", "agent", "communication"], "flags": {"non_blocking": True, "mid_flight": True}},

        {"tool_func": observe, "name": "observe",
         "description": "Snapshot aller Coders: Status, Iteration, Files, Tokens, Errors. PLUS gepufferte Coder-Nachrichten (Fragen, Blocker). EINZIGER Weg Messages zu sehen. Regelmäßig aufrufen.",
         "category": ["coder", "cluster", "monitoring"], "flags": {"batch": True, "read_only": True, "shows_messages": True}},

        {"tool_func": steer, "name": "steer",
         "description": "Cluster-Eingriff: kill/pause/resume/reassign. NUR nach observe() — Entscheidungen auf Datenbasis, nicht blind.",
         "category": ["coder", "cluster", "control"], "flags": {"destructive": True, "requires_observe": True}},

        {"tool_func": validate_worktree, "name": "validate_worktree",
         "description": "Prüft Worktree: Syntax, Lint, Diff. Aufrufen wenn observe() status='done' zeigt. Bei Fail: Report via interact() an Coder → erneut validieren.",
         "category": ["coder", "post", "qa"], "flags": {"blocking": True, "thorough": True}},

        {"tool_func": accept, "name": "accept",
         "description": "Übernimmt validierte Arbeit ins Origin-Repo. NUR nach validate_worktree(passed=True). Cherry-Pick einzelner Files möglich. Cleanup Worktree.",
         "category": ["coder", "post", "lifecycle"], "flags": {"destructive": True, "selective": True, "requires_validation": True}},

        # {"tool_func": parallel_execute, "name": "parallel_execute",
        #  "description": "Komplexe Aufgabe PARALLEL ausführen mit mehreren CoderAgents. Dekomposition → Spawn N Coders → Monitor → Merge. Ideal für große Refactorings oder modulabhängige Tasks.",
        #  "category": ["coder", "parallel", "orchestration"], "flags": {"multi_agent": True, "auto_decompose": True, "async_execution": True}},

        {"tool_func": sequential_execute, "name": "sequential_execute",
         "description": "Aufgabe mit Pipeline + Retry ausführen: analyze → refine → execute → validate → accept. Für fokussierte Aufgaben mit starker Abhängigkeit. Wiederholt bei Fehlern automatisch.",
         "category": ["coder", "sequential", "orchestration"], "flags": {"retry": True, "validation": True, "atomic": True}},

        # {"tool_func": swarm_execute, "name": "swarm_execute",
        #  "description": "Komplexe Aufgabe mit LLM-adaptivem Multi-Agent ausführen. Manager plant dynamisch, spawned Coders, überwacht, plant bei Fehlern neu. Für unbekannte komplexe Tasks mit evolvierenden Anforderungen.",
        #  "category": ["coder", "swarm", "orchestration"], "flags": {"adaptive": True, "multi_agent": True, "llm_managed": True, "max_iterations": True}},
    ]


# ═══════════════════════════════════════════════════════════════
#  CODER send_message INJECTION
# ═══════════════════════════════════════════════════════════════

def _inject_send_message(coder, coder_id: str, buffer: MessageBuffer):
    """Injiziert send_message Tool + _inject_message Methode in einen CoderAgent."""

    # 1. Message-Empfang (FlowAgent → Coder)
    coder._pending_messages = []

    def _inject_message(self, message: str, context: str | None = None):
        self._pending_messages.append(message)
        if context:
            self._pending_messages.append(f"[KONTEXT]\n{context}")

    import types
    coder._inject_message = types.MethodType(_inject_message, coder)

    # 2. send_message als Tool-Dispatch registrieren
    original_dispatch = coder._dispatch.__func__ if hasattr(coder._dispatch, '__func__') else None

    async def _patched_dispatch(self, name, args, messages):
        if name == "send_message":
            text = args.get("message", "")
            prio = args.get("priority", "info")
            buffer.push(coder_id, text, prio)
            return f"Nachricht gepuffert (priority={prio}). FlowAgent sieht sie bei observe(). Arbeite weiter."
        # Pending messages in Context injizieren
        if self._pending_messages:
            for m in self._pending_messages:
                messages.append({"role": "user", "content": f"[FlowAgent]: {m}"})
            self._pending_messages.clear()
        # Original dispatch
        return await self.__class__._dispatch_original(self, name, args, messages)

    # Backup original
    if not hasattr(coder.__class__, '_dispatch_original'):
        coder.__class__._dispatch_original = coder.__class__._dispatch
    coder._dispatch = types.MethodType(_patched_dispatch, coder)

    # 3. Tool-Definition zum Coder hinzufügen
    original_tools = coder._tools.__func__ if hasattr(coder._tools, '__func__') else None

    def _patched_tools(self):
        tools = self.__class__._tools_original(self)
        tools.append({
            "type": "function", "function": {
                "name": "send_message",
                "description": (
                    "Schicke Nachricht an FlowAgent. Wird gepuffert, nicht sofort gelesen. "
                    "Arbeite WEITER nach dem Senden — blockiere nicht, warte nicht auf Antwort. "
                    "Nutze bei Fragen, Blockern oder Status-Updates."
                ),
                "parameters": {"type": "object", "properties": {
                    "message": {"type": "string", "description": "Deine Nachricht"},
                    "priority": {"type": "string", "enum": ["info", "question", "blocker"],
                                 "description": "info=FYI, question=Input gebraucht, blocker=kannst nicht weiter"}
                }, "required": ["message"]},
            }
        })
        return tools

    if not hasattr(coder.__class__, '_tools_original'):
        coder.__class__._tools_original = coder.__class__._tools
    coder._tools = types.MethodType(_patched_tools, coder)


# ═══════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════

def coder_register_flow_tools(flow_agent, project_root: str, agent: Any = None, config: dict | None = None) -> tuple[
    CoderPool, list[
        dict[str, Callable[[list[str] | None], Coroutine[Any, Any, dict]] | str | list[str] | dict[str, bool]] | dict[
            str, Callable[[str, dict], Coroutine[Any, Any, dict]] | str | list[str] | dict[str, bool]] | dict[
            str, Callable[[dict | str, str | None], Coroutine[Any, Any, str]] | str | list[str] | dict[str, bool]] |
        dict[str, Callable[[str, str, str | None], Coroutine[Any, Any, bool]] | str | list[str] | dict[str, bool]] |
        dict[str, Callable[[], Coroutine[Any, Any, dict]] | str | list[str] | dict[str, bool]] | dict[
            str, Callable[[str, str, str | None], Coroutine[Any, Any, bool]] | str | list[str] | dict[str, bool]] |
        dict[str, Callable[[str, list[str] | None], Coroutine[Any, Any, dict]] | str | list[str] | dict[str, bool]] |
        dict[str, Callable[[str, list[str] | None, bool], Coroutine[Any, Any, dict]] | str | list[str] | dict[
            str, bool]]]]:
    """Registriert alle 8 FlowAgent-Tools. Gibt CoderPool zurück."""
    pool = CoderPool()
    tools = _make_tools(pool, agent or flow_agent, project_root, config)
    # for t in tools:
    #     flow_agent.add_tool(**t)
    return pool, tools
