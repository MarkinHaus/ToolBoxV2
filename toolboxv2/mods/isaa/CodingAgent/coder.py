"""CoderAgent v4 – 50-statt-500 Edition.
Real git worktree, timeout-safe bash, proper validation, minimal CLI."""

import asyncio, datetime, json, logging, os, platform, re, shlex, shutil, subprocess, tempfile, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Optional Imports ---
try:
    from toolboxv2.mods.isaa.base.AgentUtils import detect_shell
except ImportError:
    def detect_shell() -> Tuple[str, str]:
        return ("cmd.exe", "/c") if platform.system() == "Windows" else ("/bin/sh", "-c")

try: import litellm
except ImportError: litellm = None

# --- Data ---

@dataclass
class EditBlock:
    file_path: str; search: str; replace: str

@dataclass
class CoderResult:
    success: bool; message: str; changed_files: List[str]; history: List[dict]
    memory_saved: bool = False; tokens_used: int = 0; compressions_done: int = 0

@dataclass
class ExecutionReport:
    timestamp: str; task: str; changed_files: List[str]; success: bool
    key_decisions: List[str] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    summary: str = ""
    patterns_learned: list | None = None

    def to_context_str(self) -> str:
        return (f"[{self.timestamp}] {'✓' if self.success else '✗'} {self.task}\n"
                f"  Files: {', '.join(self.changed_files) or '-'}  {self.summary}")

# --- Token Management ---

def _ctx_limit(model: str) -> int:
    if litellm:
        try:
            info = litellm.get_model_info(model)
            if lim := (info.get("max_input_tokens") or info.get("max_tokens")): return int(lim)
        except Exception: pass
    return 8_192

def _count_tokens(messages: List[dict], model: str) -> int:
    if litellm:
        try: return litellm.token_counter(model=model, messages=messages)
        except Exception: pass
    return max(1, sum(len(str(m.get("content", ""))) // 4 for m in messages))


class TokenTracker:
    def __init__(self, model: str, agent=None):
        self.model, self.agent = model, agent
        self.limit = _ctx_limit(model)
        self.threshold = int(self.limit * 0.70)
        self.total_tokens = self.compressions_done = 0

    def needs_compression(self, messages: List[dict]) -> bool:
        self.total_tokens = _count_tokens(messages, self.model)
        return self.total_tokens >= self.threshold

    async def compress(self, messages: List[dict]) -> List[dict]:
        if len(messages) <= 6: return messages

        system_msg, task_msg = messages[0], messages[1]
        recent_history = messages[-4:]
        middle_history = messages[2:-4]

        # 1. "Latest Version" Dictionary für Datei-Kontext
        # Key: Dateipfad, Value: Letzter gelesener Inhalt
        file_contexts = {}

        # Regex zum Extrahieren von Pfaden aus Edits und Read-Outputs
        re_edit = re.compile(r"~~~edit:(.+?)~~~")
        re_read_header = re.compile(r"--- (.+?) \(\d+-\d+/\d+\) ---")

        for msg in middle_history:
            content = msg.get("content", "") or ""

            # A) Wenn der Assistant editiert, ist der alte Lese-Kontext ungültig
            if msg.get("role") == "assistant":
                for path in re_edit.findall(content):
                    path = path.strip()
                    if path in file_contexts:
                        del file_contexts[path]  # Alten Stand löschen

            # B) Wenn ein Tool-Output Code liefert, speichere ihn als aktuellsten Stand
            elif msg.get("role") == "tool":
                match = re_read_header.search(content)
                if match:
                    path = match.group(1).strip()
                    # Wir speichern den Content (begrenzt auf 50 Zeilen für die Senke)
                    lines = content.splitlines()
                    snippet = "\n".join(lines[:55])  # Header + 50 Zeilen + Footer
                    if len(lines) > 55: snippet += "\n... [gekürzt in Zusammenfassung] ..."
                    file_contexts[path] = snippet

        # 2. Zusammenfassung generieren
        summary_text = await self._summarize(middle_history)

        # 3. History neu aufbauen
        new_history = [system_msg, task_msg]

        # Zusammenfassung der Schritte
        new_history.append({
            "role": "user",
            "content": f"### VERLAUF-RECAP:\n{summary_text}"
        })

        # Sinking: Nur die AKTUELLSTEN Versionen der Dateien einfügen
        if file_contexts:
            context_blob = "\n\n".join(file_contexts.values())
            new_history.append({
                "role": "system",
                "content": f"### AKTUELLER DATEI-KONTEXT (Letzter Stand):\n{context_blob}"
            })

        new_history.extend(recent_history)
        self.compressions_done += 1
        return new_history

    def usage_ratio(self, messages: List[dict]) -> float:
        self.total_tokens = _count_tokens(messages, self.model)
        return self.total_tokens / self.limit

    async def _summarize(self, msgs: List[dict]) -> str:
        if not self.agent: return "(Zusammenfassung nicht möglich)"
        try:
            prompt = (
                "Fasse die bisherigen Aktionen zusammen. Antworte extrem kurz.\n"
                "Fokus: Welche Dateien wurden editiert? Welche Probleme traten auf?\n"
                "Ignoriere den Dateiinhalt selbst, fasse nur die TÄTIGKEIT zusammen."
            )
            # Nur die Essenz der Nachrichten zur Zusammenfassung schicken
            stream = "\n".join([f"{m['role']}: " + (m.get('content', '')[:200]) for m in msgs])
            return await self.agent.a_run_llm_completion(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": stream}],
                stream=False
            ) or "..."
        except Exception:
            return "Dialog komprimiert."


# --- Memory ---

class ExecutionMemory:
    def __init__(self, root: str):
        self.path = Path(root) / ".coder_memory.json"
        self.reports: List[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try: self.reports = json.loads(self.path.read_text()).get("reports", [])
            except Exception: self.reports = []

    def add(self, report: ExecutionReport):
        self.reports = (self.reports + [asdict(report)])[-10:]
        self.path.write_text(json.dumps({"reports": self.reports}, indent=2))

    def get_context(self) -> Optional[str]:
        if not self.reports: return None
        return "\n".join(ExecutionReport(**r).to_context_str() for r in self.reports[-3:])


# --- Smart File Reader ---

async def smart_read_file(
    path: str, start: Optional[int], end: Optional[int],
    worktree: Path, agent=None, messages: list = None, model: str = "", query: str = ""
) -> str:
    messages = messages or []
    target = (worktree / path).resolve()
    if not target.exists(): return f"Error: {path} not found."
    if b'\x00' in target.read_bytes()[:512]: return f"Binary: {path}. Use bash+xxd."

    content = await asyncio.to_thread(target.read_text, encoding="utf-8", errors="replace")
    lines = content.splitlines()
    total = len(lines)

    # Explicit range always honored
    if start or end:
        return _fmt(path, lines, start or 1, end or total)

    usage = _count_tokens(messages, model) / _ctx_limit(model) if model else 0

    # MODE 1: Direct read — files ≤600 lines ALWAYS shown in full if context allows
    if usage < 0.60:
        if total <= 600:
            return _fmt(path, lines, 1, total)
        # Large file: show first+last 150 lines with gap indicator
        head = _fmt(path, lines, 1, 150)
        tail = _fmt(path, lines, total - 149, total)
        return f"{head}\n\n... [{total - 300} lines omitted — use read_file with start_line/end_line] ...\n\n{tail}"

    # MODE 2: LLM extraction (60-85%)
    if usage < 0.85 and agent:
        # Still show structure so agent knows WHERE to look
        skeleton = "\n".join(
            f"{i+1}|{l}" for i, l in enumerate(lines)
            if l.strip().startswith(("def ", "class ", "import ", "from ", "if __"))
            or (query and any(t in l for t in query.split()[:3]))
        )
        resp = await agent.a_run_llm_completion(
            messages=[{"role": "system", "content": "Extrahiere relevante Abschnitte mit Zeilennummern."},
                      {"role": "user", "content": f"File: {path} ({total} lines)\nQuery: {query}\n\nStructure:\n{skeleton}\n\nFull (first 4000 lines):\n{content[:150_000]}"}],
            stream=False)
        return f"[Extracted from {path} ({total} lines)]\n{resp}"

    # MODE 3: Critical — grep + skeleton
    terms = query.split()[:3] if query else ["def ", "class "]
    hits = [f"{i+1}|{l}" for i, l in enumerate(lines) for t in terms if t in l][:100]
    if not hits: return f"[Critical] {path}: {total} lines, no matches for {terms}. Use read_file with line range."
    if agent:
        resp = await agent.a_run_llm_completion(
            messages=[{"role": "system", "content": "Kontext kritisch. Analysiere Chunks."},
                      {"role": "user", "content": "\n".join(hits)}], stream=False)
        return f"[Critical extract from {path}]\n{resp}"
    return "\n".join(hits)


def _fmt(path: str, lines: list, start: int, end: int) -> str:
    s, e = max(0, start - 1), min(len(lines), end)
    body = "\n".join(f"{i+1:5d}|{l}" for i, l in enumerate(lines[s:e], start=s))
    return f"--- {path} ({start}-{e}/{len(lines)}) ---\n{body}"


# --- Git Worktree (Real Implementation + Fallback) ---
def _file_hash_md5(p: Path) -> str:
    """
    Berechnet den MD5-Hash einer Datei effizient in 4KB-Blöcken.
    Dies ist der Motor für die O(1) Vergleichslogik im Worktree.
    """
    import hashlib
    if not p.exists() or p.is_dir():
        return ""

    hash_md5 = hashlib.md5()
    try:
        with open(p, "rb") as f:
            # Liest die Datei in Chunks, um RAM-Overhead bei großen Files zu vermeiden
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (OSError, IOError) as e:
        logger.error(f"Hash-Fehler bei {p}: {e}")
        return f"error-{uuid.uuid4().hex[:4]}"  # Eindeutiger Hash bei Fehler erzwingt Diff


class GitWorktree:
    def __init__(self, root: str):
        self.origin_root = Path(root)
        self._is_git = (self.origin_root / ".git").exists()
        self._branch = f"coder-{uuid.uuid4().hex[:8]}"
        self.path: Optional[Path] = None

    @property
    def worktree_path(self) -> Optional[Path]:
        """Compat alias for _cmd_coder integration."""
        return self.path

    def setup(self):
        if self.path and self.path.exists(): return
        if self._is_git:
            self.path = Path(tempfile.mkdtemp(prefix="coder_wt_"))
            subprocess.run(
                ["git", "worktree", "add", "-b", self._branch, str(self.path)],
                cwd=self.origin_root, check=True, capture_output=True)
        else:
            self.path = Path(tempfile.mkdtemp(prefix="coder_cp_"))
            shutil.copytree(self.origin_root, self.path, dirs_exist_ok=True)

    def commit(self, msg: str):
        if self._is_git and self.path:
            subprocess.run(["git", "add", "."], cwd=self.path, capture_output=True)
            subprocess.run(["git", "commit", "-m", msg, "--allow-empty"],
                           cwd=self.path, capture_output=True)

    async def _compute_hashes_parallel(self, files: list[Path]) -> dict[str, str]:
        """Compute MD5 hashes for multiple files in parallel using ThreadPoolExecutor."""
        import concurrent.futures
        import hashlib

        def _hash_single(p: Path) -> tuple[str, str]:
            try:
                return (str(p), _file_hash_md5(p))
            except Exception:
                return (str(p), None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(_hash_single, f) for f in files]
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                path, h = fut.result()
                if h is not None:
                    results[path] = h
            return results

    async def apply_back(self) -> int:
        """Merge worktree changes back to origin. Returns changed file count (-1 = git merge)."""
        if not self.path: return 0
        if self._is_git:
            self.commit("pre-merge checkpoint")
            subprocess.run(["git", "merge", self._branch, "--no-edit"],
                           cwd=self.origin_root, capture_output=True, check=True)
            return -1

        # Parallel hash-based comparison
        src_files = [f for f in self.path.rglob("*")
                     if not f.is_dir() and ".git" not in f.parts]
        dst_files = [(self.origin_root / f.relative_to(self.path)) for f in src_files]

        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel(
            [d for d in dst_files if d.exists()])

        count = 0
        for src, dst in zip(src_files, dst_files):
            rel = src.relative_to(self.path)
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
        return count

    def cleanup(self):
        if not self.path: return
        if self._is_git:
            subprocess.run(["git", "worktree", "remove", str(self.path), "--force"],
                           cwd=self.origin_root, capture_output=True)
            subprocess.run(["git", "branch", "-D", self._branch],
                           cwd=self.origin_root, capture_output=True)
        else:
            shutil.rmtree(self.path, ignore_errors=True)
        self.path = None

    def rollback(self, files: list[str] | None = None):
        """Reset worktree to origin state. If files given, only those."""
        if not self.path: return
        if files:
            for f in files:
                src = self.origin_root / f
                dst = self.path / f
                if src.exists():
                    shutil.copy2(src, dst)
                elif dst.exists():
                    dst.unlink()
        elif self._is_git:
            subprocess.run(["git", "checkout", "."], cwd=self.path, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=self.path, capture_output=True)
        else:
            shutil.rmtree(self.path, ignore_errors=True)
            shutil.copytree(self.origin_root, self.path, dirs_exist_ok=True)

    async def apply_files(self, files: list[str]) -> list[str]:
        """Cherry-pick: apply only specific files back to origin. Returns applied list."""
        if not self.path: return []
        if not files: return []

        src_paths = [self.path / f for f in files]
        dst_paths = [self.origin_root / f for f in files]

        # Filter existing sources
        existing_pairs = [(s, d, f) for s, d, f in zip(src_paths, dst_paths, files) if s.exists()]
        if not existing_pairs:
            return []

        src_list, dst_list, file_list = zip(*existing_pairs)
        src_hashes = await self._compute_hashes_parallel(list(src_list))
        dst_hashes = await self._compute_hashes_parallel(
            [d for d in dst_list if d.exists()])

        applied = []
        for src, dst, f in zip(src_list, dst_list, file_list):
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                applied.append(f)
        return applied

    async def changed_files(self) -> list[str]:
        """List files that differ between worktree and origin."""
        if not self.path: return []
        if self._is_git:
            r = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                               cwd=self.path, capture_output=True, text=True)
            # Also untracked
            u = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                               cwd=self.path, capture_output=True, text=True)
            return [f for f in (r.stdout + u.stdout).splitlines() if f.strip()]

        # Parallel hash-based comparison
        src_files = [f for f in self.path.rglob("*")
                     if not f.is_dir() and ".git" not in f.parts]
        dst_files = [(self.origin_root / f.relative_to(self.path)) for f in src_files]

        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel(
            [d for d in dst_files if d.exists()])

        changed = []
        for src, dst in zip(src_files, dst_files):
            rel = str(src.relative_to(self.path))
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                changed.append(rel)
        return changed

    async def sync_from_origin(self) -> list[str]:
        """Sync files changed in origin since worktree was created.
        Returns list of files that were updated. Does NOT overwrite
        files the agent has already modified in the worktree."""
        if not self.path: return []
        agent_changed = set(await self.changed_files())

        # Get origin files excluding agent-changed files
        src_files = [f for f in self.origin_root.rglob("*")
                     if not f.is_dir() and ".git" not in f.parts
                     and str(f.relative_to(self.origin_root)) not in agent_changed]

        if not src_files:
            return []

        dst_files = [(self.path / f.relative_to(self.origin_root)) for f in src_files]

        # Parallel hash-based comparison
        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel(
            [d for d in dst_files if d.exists()])

        synced = []
        for src, dst in zip(src_files, dst_files):
            rel = str(src.relative_to(self.origin_root))
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                synced.append(rel)
        return synced


# --- Core Agent ---

class CoderAgent:
    SYSTEM_PROMPT = (
        "Du bist ein Elite Coding-Agent.\n"
        "REGELN:\n"
        "1. ARCHITEKTUR ZUERST: Nutze 'memory_recall', um Zusammenhänge zu verstehen, BEVOR du Code schreibst.\n"
        "2. LIES Dateien (read_file) BEVOR du editierst. Niemals blind raten!\n"
        "3. Änderungen NUR via Edit-Blöcke.\n"
        "4. Beende mit [DONE] wenn fertig.\n"
        "5. Schreibe VOR jedem Edit einen <thought>...</thought> Block, der Folgendes enthält:\n"
        "   - Was du in der Datei gesehen hast und wie es zur Architektur passt.\n"
        "   - Warum du diese Änderung machst.\n"
        "   - (Optional aber empfohlen) Deinen Plan als Checkliste, damit das System deinen Status tracken kann:\n"
        "     - [ ] Offener Teilschritt\n"
        "     - [x] Erledigter Teilschritt\n"
        "FORMAT:\n"
        "<thought>\n"
        "Ich passe das Login an.\n"
        "Plan:\n"
        "- [x] config.py lesen\n"
        "- [ ] auth.py ändern\n"
        "- [ ] test_auth.py updaten\n"
        "</thought>\n"
        "~~~edit:pfad/datei.py~~~\n<<<<<<< SEARCH\nalter code\n=======\nneuer code\n>>>>>>> REPLACE\n~~~end~~~"
    )

    def __init__(self, agent, project_root: str, config: dict = None):
        self.agent = agent
        self.root = project_root
        self.config = config or {}
        self.model = self.config.get("model", "gpt-4o")
        self.run_tests = self.config.get("run_tests", False)
        self.bash_timeout = self.config.get("bash_timeout", 300)
        self.memory = ExecutionMemory(project_root)
        self.tracker = TokenTracker(self.model, agent)
        self.worktree = GitWorktree(project_root)

        self.state = {
            "plan": [],
            "done": [],
            "current_file": "None",
            "last_error": None
        }

    async def execute(self, task: str) -> CoderResult:
        self.worktree.setup()

        # Sync any external changes before starting
        synced = await self.worktree.sync_from_origin()

        sys_msg = self.SYSTEM_PROMPT
        if prev := self.memory.get_context():
            sys_msg += f"\n\n## VERLAUF:\n{prev}"

        task_full = task
        if synced:
            task_full += f"\n\n[INFO: {len(synced)} file(s) synced from origin: {', '.join(synced[:5])}]"

        # --- NEU: Auto-Memory Fetch (Lazy Mapping) ---
        if self.agent and hasattr(self.agent, "arun_function"):
            try:
                # Holt automatisch den Architektur-Kontext für die Aufgabe aus dem Memory
                arch_context = await self.agent.arun_function("memory_recall", query=task, search_type="auto")
                if arch_context and "No results" not in arch_context:
                    task_full = f"### SYSTEM ARCHITEKTUR-KONTEXT (Aus Memory):\n{arch_context}\n\n### DEINE AUFGABE:\n{task_full}"
            except Exception as e:
                logger.warning(f"Auto-Memory fetch failed: {e}")
        # ----------------------------------------------

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": task_full}]
        try:
            edits, thought = await self._loop(messages)
            results = self._apply_edits(edits)
            errors = await self._validate([r["file_path"] for r in results if r["success"]])

            if errors or any(not r["success"] for r in results):
                edits, repair_thought = await self._repair(messages, results, errors)
                self._apply_edits(edits)
                if repair_thought:
                    thought = f"{thought}\n[Repair] {repair_thought}"

            self.worktree.commit(task)
        except Exception as e:
            logger.error(f"Execute failed: {e}")
            return CoderResult(False, str(e), [], messages)

        report = ExecutionReport(
            datetime.datetime.now().isoformat(), task,
            [e.file_path for e in edits], True,
            summary=thought[:500] if thought else "")
        self.memory.add(report)

        return CoderResult(True, "Done", [e.file_path for e in edits], messages,
                           memory_saved=True, tokens_used=self.tracker.total_tokens,
                           compressions_done=self.tracker.compressions_done)

    async def _loop(self, messages: list) -> Tuple[List[EditBlock], str]:
        """Core execution loop with Anti-Loop protection."""
        tools = self._tools()
        thought = ""
        recent_actions = []  # Speichert Signaturen der letzten Tool-Calls
        max_iters = 25

        for iteration in range(max_iters):
            # 1. State-Update in den Prompt injizieren
            # Wir hängen eine "System-Status" Nachricht an, die den Agenten zentriert
            status_update = (
                f"### AKTUELLE ITERATION: {iteration + 1}/{max_iters}\n"
                f"**STATUS:**\n"
                f"- Datei im Fokus: {self.state['current_file']}\n"
                f"- Erledigt: {', '.join(self.state['done']) or 'Nichts'}\n"
                f"- Offen: {', '.join(self.state['plan']) or 'Unbekannt (bitte Plan definieren)'}\n"
            )

            if iteration == max_iters - 1:
                status_update += "\n⚠️ LETZTE ITERATION! Beende deine Arbeit oder fasse exakt zusammen, was noch fehlt!"

            # Wir fügen den Status als flüchtige User-Nachricht ein (nicht dauerhaft in History)
            messages += [{"role": "user", "content": status_update}]

            # 1. Sync & Compression
            if iteration > 0 and iteration % 3 == 0:
                synced = await self.worktree.sync_from_origin()
                if synced:
                    messages.append(
                        {"role": "user", "content": f"[SYNC] Dateien vom System aktualisiert: {', '.join(synced)}"})

            if self.tracker.needs_compression(messages):
                messages[:] = await self.tracker.compress(messages)

            # 2. LLM Call
            resp = await self.agent.a_run_llm_completion(
                messages=messages, tools=tools, stream=False, get_response_message=True)

            content = resp.content or ""
            tool_calls = resp.tool_calls
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if thought_match := re.search(r"<thought>(.*?)</thought>", content, re.DOTALL):
                thought = thought_match.group(1).strip()
                self._update_internal_state(thought)

            # 3. Tool Execution & Loop-Check
            if tool_calls:
                for tc in tool_calls:
                    try:
                        # Deterministische Signatur: Name + sortierte Argumente
                        args_data = json.loads(tc.function.arguments)
                        sig = f"{tc.function.name}:{json.dumps(args_data, sort_keys=True)}"

                        if sig in recent_actions:
                            # Loop erkannt -> Fehler zurückgeben statt ausführen
                            res = f"FEHLER: Du wiederholst dich! Den Aufruf '{sig}' hast du bereits getätigt. Ändere deine Strategie oder lies die Datei neu, falls du denkst, sie hätte sich geändert."
                            logger.warning(f"Loop blocked: {sig}")
                        else:
                            recent_actions.append(sig)
                            if tc.function.name == "read_file":
                                self.state["current_file"] = args.get("path")
                            if len(recent_actions) > 8: recent_actions.pop(0)  # Gedächtnis-Fenster
                            res = await self._dispatch(tc.function.name, args_data, messages)

                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                    except Exception as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Fehler: {e}"})
                continue

            # 4. Finalizing
            if blocks := self._parse_edits(content):
                return blocks, thought
            if "[DONE]" in content:
                return [], thought

        return [], thought

    def _update_internal_state(self, thought: str):
        """Versucht den Plan des Agenten aus seinen Gedanken zu lesen."""
        # Suche nach Zeilen wie "- [ ]" oder "Plan: ..."
        lines = thought.splitlines()
        new_plan = []
        for l in lines:
            l = l.strip()
            if l.startswith("- [ ]"):
                new_plan.append(l[5:].strip())
            elif l.startswith("- [x]"):
                done_item = l[5:].strip()
                if done_item not in self.state["done"]: self.state["done"].append(done_item)

        if new_plan:
            self.state["plan"] = new_plan

    async def _repair(self, messages: list, failed: list, errors: list) -> Tuple[List[EditBlock], str]:
        """Force re-read of failed files before retry."""
        lines = ["Fehler bei Edits. LIES DIE DATEIEN NEU bevor du es nochmal versuchst:"]

        force_read_files = set()
        for p in failed:
            if not p["success"]:
                lines.append(f"- {p['file_path']}: {p['error']}")
                force_read_files.add(p["file_path"])
        for e in errors:
            lines.append(f"- {e}")

        if force_read_files:
            lines.append(f"\nPflicht: Rufe read_file auf für: {', '.join(force_read_files)}")
            lines.append("Dann korrigiere den SEARCH-Block basierend auf dem ECHTEN Dateiinhalt.")

        messages.append({"role": "user", "content": "\n".join(lines)})
        return await self._loop(messages)

    def _apply_edits(self, edits: List[EditBlock]) -> list:
        """ATOMIC MULTI-FILE TRANSACTION: Validate ALL, then write ALL."""
        import difflib
        results = []
        my_module = Path(__file__).resolve()
        pending = []  # (target, content, is_self, original_backup)

        for e in edits:
            target = (self.worktree.path / e.file_path).resolve()
            is_self = target.samefile(my_module) if target.exists() and my_module.exists() else False

            try:
                # --- New file ---
                if not e.search.strip():
                    pending.append((target, e.replace, is_self, None))
                    results.append({"file_path": e.file_path, "success": True})
                    continue

                if not target.exists():
                    results.append({"file_path": e.file_path, "success": False, "error": "File missing"})
                    continue

                txt = target.read_text(encoding="utf-8", errors="replace")
                backup = txt if is_self else None

                # --- 1. Exact match ---
                if e.search in txt:
                    new_txt = txt.replace(e.search, e.replace, 1)

                # --- 2. Stripped-line match (whitespace tolerance) ---
                elif (idx := self._fuzzy_find(txt, e.search)) is not None:
                    src_lines = txt.splitlines(keepends=True)
                    s_lines = e.search.splitlines()
                    new_lines = src_lines[:idx] + [e.replace + "\n"] + src_lines[idx + len(s_lines):]
                    new_txt = "".join(new_lines)

                else:
                    results.append({"file_path": e.file_path, "success": False,
                                    "error": "SEARCH not found (exact + fuzzy)"})
                    continue

                # Self-edit: validate BEFORE transaction
                if is_self and e.file_path.endswith(".py"):
                    try:
                        compile(new_txt, e.file_path, "exec")
                    except SyntaxError as se:
                        results.append({"file_path": e.file_path, "success": False,
                                        "error": f"Self-edit blocked: SyntaxError L{se.lineno}: {se.msg}"})
                        continue

                pending.append((target, new_txt, is_self, backup))
                results.append({"file_path": e.file_path, "success": True})

            except Exception as ex:
                results.append({"file_path": e.file_path, "success": False, "error": str(ex)})

        # === ATOMIC WRITE PHASE: All-or-nothing ===
        written = []
        try:
            for target, content, is_self, backup in pending:
                self._atomic_write(target, content)
                written.append((target, backup))
        except BaseException as commit_err:
            # Rollback all written files
            for target, backup in written:
                if backup is not None:
                    try:
                        self._atomic_write(target, backup)
                    except:
                        pass
            # Mark all as failed
            for r in results:
                if r["success"]:
                    r["success"] = False
                    r["error"] = f"Transaction failed: {commit_err}"
            raise

        return results

    @staticmethod
    def _atomic_write(path: Path, content: str):
        """Write via tempfile + os.replace for crash safety."""
        import tempfile as _tf
        fd, tmp = _tf.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp, path)
        except BaseException:
            os.close(fd) if not os.get_inheritable(fd) else None
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _fuzzy_find(txt: str, search: str, threshold: float = 0.85) -> int | None:
        """Find SEARCH block in txt with whitespace tolerance.
        Returns start line index or None."""
        import difflib
        src_lines = [l.strip() for l in txt.splitlines()]
        search_lines = [l.strip() for l in search.splitlines()]
        if not search_lines:
            return None
        window = len(search_lines)
        best_ratio, best_idx = 0.0, None
        for i in range(len(src_lines) - window + 1):
            candidate = src_lines[i:i + window]
            ratio = difflib.SequenceMatcher(None, candidate, search_lines).ratio()
            if ratio > best_ratio:
                best_ratio, best_idx = ratio, i
        return best_idx if best_ratio >= threshold else None

    async def _dispatch(self, name: str, args: dict, messages: list) -> str:
        if name == "read_file":
            return await smart_read_file(
                args["path"], args.get("start_line"), args.get("end_line"),
                self.worktree.path, self.agent, messages, self.model, args.get("query", ""))
        if name == "bash": return await self._run_bash(args["command"])
        if name == "grep": return await self._run_grep(args["pattern"])
        if name == "memory_recall":
            if self.agent and hasattr(self.agent, "arun_function"):
                return await self.agent.arun_function("memory_recall", query=args["query"],
                                                      search_type=args.get("search_type", "auto"))
            return "Fehler: Memory-System nicht verfügbar."
        return f"Unknown tool: {name}"

    async def _run_bash(self, cmd: str) -> str:
        shell, flag = detect_shell()
        try:
            proc = await asyncio.create_subprocess_exec(
                shell, flag, cmd,
                cwd=str(self.worktree.path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.bash_timeout)
        except asyncio.TimeoutError:
            try: proc.kill()
            except Exception: pass
            return f"ERROR: Timeout after {self.bash_timeout}s"
        except NotImplementedError:
            def _sync():
                return subprocess.run(
                    [shell, flag, cmd], cwd=str(self.worktree.path),
                    capture_output=True, timeout=self.bash_timeout).stdout
            try:
                return (await asyncio.to_thread(_sync)).decode(errors="replace")
            except subprocess.TimeoutExpired:
                return f"ERROR: Timeout after {self.bash_timeout}s"
        return (out + err).decode(errors="replace")

    async def _run_grep(self, pattern: str) -> str:
        """Cross-platform grep: rg > git grep > findstr/grep."""
        quoted = shlex.quote(pattern) if platform.system() != "Windows" else pattern
        if shutil.which("rg"):
            return await self._run_bash(f"rg -n {quoted} .")
        if (self.worktree.path / ".git").exists() or (self.worktree.path / ".git").is_file():
            return await self._run_bash(f"git grep -n {quoted}")
        if platform.system() == "Windows":
            # findstr is always available on Windows, /S=recursive /N=line numbers
            return await self._run_bash(f'findstr /S /N /C:"{pattern}" *')
        return await self._run_bash(f"grep -rn {quoted} .")

    def _tools(self) -> list:
        return [
            {"type": "function", "function": {"name": "read_file", "description": "Read file content",
                                              "parameters": {"type": "object", "properties": {
                                                  "path": {"type": "string", "description": "Relative file path"},
                                                  "start_line": {"type": "integer",
                                                                 "description": "Start line (1-indexed)"},
                                                  "end_line": {"type": "integer",
                                                               "description": "End line (inclusive)"},
                                                  "query": {"type": "string",
                                                            "description": "Search context for smart extraction"}},
                                                             "required": ["path"]}}},
            {"type": "function", "function": {"name": "bash", "description": "Run shell command in worktree",
                                              "parameters": {"type": "object", "properties": {
                                                  "command": {"type": "string",
                                                              "description": "Shell command to execute"}},
                                                             "required": ["command"]}}},
            {"type": "function", "function": {"name": "grep", "description":
                "Search pattern in codebase (uses rg/git-grep/findstr). Install ripgrep for best speed: "
                "winget install BurntSushi.ripgrep.MSVC",
                                              "parameters": {"type": "object", "properties": {
                                                  "pattern": {"type": "string",
                                                              "description": "Search pattern (literal string)"}},
                                                             "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "memory_recall", "description":
                "Ruft Architektur-Wissen, Datei-Zusammenhänge und Projekt-Regeln ab. Nutze dies, um Abhängigkeiten zu verstehen.",
                                              "parameters": {"type": "object", "properties": {
                                                  "query": {"type": "string",
                                                            "description": "Suchbegriff (z.B. 'Login System' oder 'database.py')"},
                                                  "search_type": {"type": "string",
                                                                  "enum": ["auto", "concept", "relations"],
                                                                  "description": "Art der Suche"}
                                              }, "required": ["query"]}}}
        ]

    def _parse_edits(self, text: str) -> List[EditBlock]:
        """State-machine parser — immune to content containing markers.
        Markers must be EXACT standalone lines. Content is captured verbatim."""
        IDLE, IN_SEARCH, IN_REPLACE = 0, 1, 2
        state = IDLE
        path = ""
        search_lines: list[str] = []
        replace_lines: list[str] = []
        blocks: list[EditBlock] = []

        for raw_line in text.split("\n"):
            stripped = raw_line.strip()

            if state == IDLE:
                m = re.match(r"^~~~edit:(.+?)~~~$", stripped)
                if m:
                    path = m.group(1).strip()
                    search_lines, replace_lines = [], []
                    state = IN_SEARCH  # next expect <<<<<<< SEARCH, but we skip it
                continue  # skip non-block lines in IDLE

            if state == IN_SEARCH:
                if stripped == "<<<<<<< SEARCH":
                    continue  # opening marker, skip
                if stripped == "=======":
                    state = IN_REPLACE
                    continue
                search_lines.append(raw_line)

            elif state == IN_REPLACE:
                if stripped == ">>>>>>> REPLACE":
                    continue  # closing marker, skip
                if stripped == "~~~end~~~":
                    blocks.append(EditBlock(
                        file_path=path,
                        search="\n".join(search_lines),
                        replace="\n".join(replace_lines),
                    ))
                    state = IDLE
                    continue
                replace_lines.append(raw_line)

        return blocks

    async def _validate(self, changed_files: List[str]) -> List[str]:
        if not changed_files: return []
        errors = []
        wt = str(self.worktree.path)
        if shutil.which("ruff"):
            res = await self._run_bash(f"ruff check --select E,F {shlex.quote(wt)}")
            err_lines = [l for l in res.splitlines() if re.match(r'.+:\d+:\d+: [EF]\d+', l)]
            if err_lines: errors.append("Lint:\n" + "\n".join(err_lines[:20]))
        if self.run_tests and shutil.which("pytest"):
            res = await self._run_bash(f"pytest {shlex.quote(wt)} -x --tb=short -q")
            if "failed" in res.lower(): errors.append(f"Tests:\n{res[:800]}")
        return errors


# --- Minimal CLI ---

"""
# A flow agent tool

a dict with

tool_func
name
description
category
flags

def add_tool(
        self,
        tool_func: Callable,
        name: str | None = None,
        description: str | None = None,
        category: list[str] | str | None = None,
        flags: dict[str, bool] | None = None,
        **kwargs,
    ):

a minimal tools set 5 tools

"""

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CoderAgent v4 – Standalone")
    p.add_argument("task", help="Coding task")
    p.add_argument("-p", "--project", default=".", help="Project root (default: cwd)")
    p.add_argument("-m", "--model", default="gpt-4o", help="LLM model")
    p.add_argument("--run-tests", action="store_true", help="Run pytest after edits")
    p.add_argument("--timeout", type=int, default=300, help="Bash timeout in seconds")
    p.add_argument("--apply", action="store_true", help="Auto-apply changes to repo")
    args = p.parse_args()

    async def _main():
        agent = None
        coder = CoderAgent(agent, os.path.abspath(args.project),
                           {"model": args.model, "run_tests": args.run_tests,
                            "bash_timeout": args.timeout})
        result = await coder.execute(args.task)
        status = '✓' if result.success else '✗'
        print(f"{status} {result.message} ({result.tokens_used} tokens, {result.compressions_done} compressions)")
        for f in result.changed_files: print(f"  → {f}")
        if args.apply and result.success:
            n = await coder.worktree.apply_back()
            print(f"Applied to origin" + (f" ({n} files)" if n >= 0 else " (git merge)"))
        coder.worktree.cleanup()

    asyncio.run(_main())
