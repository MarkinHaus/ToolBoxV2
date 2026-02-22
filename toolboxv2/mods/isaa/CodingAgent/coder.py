"""CoderAgent v4 – 50-statt-500 Edition.
Real git worktree, timeout-safe bash, proper validation, minimal CLI.

FIXES v4.1:
- BUG1: commit() silent failure → auto-config + error logging + apply_back fallback
- BUG2: _parse_edits() truncated blocks → incomplete block rescue + agent re-loop
"""

import asyncio, datetime, json, logging, os, platform, re, shlex, shutil, subprocess, tempfile, uuid
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import sys  # <--- Hinzufügen falls fehlt

# === FIX: WINDOWS UTF-8 OUTPUT ===
if sys.platform == "win32":
    # Zwingt stdout/stderr auf UTF-8, um Abstürze bei Emojis (✔, ⚠) zu verhindern
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')
# =================================
from toolboxv2 import get_logger
logger = get_logger()

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

        file_contexts = {}
        re_edit = re.compile(r"~~~edit:(.+?)~~~")
        re_read_header = re.compile(r"--- (.+?) \(\d+-\d+/\d+\) ---")

        for msg in middle_history:
            content = msg.get("content", "") or ""
            if msg.get("role") == "assistant":
                for path in re_edit.findall(content):
                    path = path.strip()
                    if path in file_contexts:
                        del file_contexts[path]
            elif msg.get("role") == "tool":
                match = re_read_header.search(content)
                if match:
                    path = match.group(1).strip()
                    lines = content.splitlines()
                    snippet = "\n".join(lines[:55])
                    if len(lines) > 55: snippet += "\n... [gekürzt in Zusammenfassung] ..."
                    file_contexts[path] = snippet

        summary_text = await self._summarize(middle_history)
        new_history = [system_msg, task_msg]
        new_history.append({"role": "user", "content": f"### VERLAUF-RECAP:\n{summary_text}"})

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

# --- Helper for Windows-safe subprocess calls ---

def _safe_run(cmd: list, cwd=None, capture_output=True, check=False, timeout=None, text=True):
    """
    Windows-safe subprocess.run that handles encoding errors gracefully.

    On Windows, subprocess with text=True can fail with cp1252 errors when
    binary data is in output. This function uses bytes and manual decoding.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            check=check,
            timeout=timeout
        )

        # Decode output manually with error handling
        if text:
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

            # Create a new namespace with decoded strings
            class SafeResult:
                def __init__(self, r, out, err):
                    self.returncode = r.returncode
                    self.stdout = out
                    self.stderr = err
                    self.args = r.args
                    self.check_returncode = r.check_returncode if hasattr(r, 'check_returncode') else None

            return SafeResult(result, stdout, stderr)

        return result
    except Exception as e:
        logger.debug(f"subprocess.run failed: {e}")
        raise


async def smart_read_file(
    path: str, start: Optional[int], end: Optional[int],
    worktree: Path, agent=None, messages: list = None, model: str = "", query: str = ""
) -> str:
    messages = messages or []
    target = (worktree / path).resolve()
    if not target.exists(): return f"Error: {path} not found."
    with open(target, "rb") as _bf:
        if b'\x00' in (_bf.read(512) or b''): return f"Binary: {path}. Use bash+xxd."

    content = await asyncio.to_thread(target.read_text, encoding="utf-8", errors="replace")
    lines = content.splitlines()
    total = len(lines)

    if start is not None or end is not None:
        return _fmt(path, lines, start if start is not None else 1, end if end is not None else total)

    usage = _count_tokens(messages, model) / _ctx_limit(model) if model else 0

    if usage < 0.60:
        if total <= 600:
            return _fmt(path, lines, 1, total)
        head = _fmt(path, lines, 1, 150)
        tail = _fmt(path, lines, total - 149, total)
        return f"{head}\n\n... [{total - 300} lines omitted — use read_file with start_line/end_line] ...\n\n{tail}"

    if usage < 0.85 and agent:
        skeleton = "\n".join(
            f"{i+1}|{l}" for i, l in enumerate(lines)
            if l.strip().startswith(("def ", "class ", "import ", "from ", "if __"))
            or (query and any(t in l for t in query.split()[:3]))
        )
        max_chars = min(32_000, int((_ctx_limit(model) * 0.4) * 4))
        resp = await agent.a_run_llm_completion(
            messages=[{"role": "system", "content": "Extrahiere relevante Abschnitte mit Zeilennummern."},
                      {"role": "user", "content": f"File: {path} ({total} lines)\nQuery: {query}\n\nStructure:\n{skeleton}\n\nContent (capped):\n{content[:max_chars]}"}],
            stream=False)
        return f"[Extracted from {path} ({total} lines)]\n{resp}"

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
    import hashlib
    if not p.exists() or p.is_dir():
        return ""
    hash_md5 = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (OSError, IOError) as e:
        logger.error(f"Hash-Fehler bei {p}: {e}")
        return f"error-{uuid.uuid4().hex[:4]}"


class GitWorktree:
    SCAN_EXCLUDES = frozenset({
        ".git", "node_modules", ".venv", "venv", "__pycache__", ".tox",
        "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".eggs", "*.egg-info", ".next", ".nuxt", "target", "out",
        "coverage", ".coverage", "htmlcov", ".idea", ".vscode",
    })

    def __init__(self, root: str):
        self.origin_root = Path(root).resolve()  # Resolve to absolute path
        self._is_git, self._git_root = self._detect_git()
        if self._git_root:
            self._git_root = self._git_root.resolve()
        self._branch = f"coder-{uuid.uuid4().hex[:8]}"
        self.path: Optional[Path] = None
        self._wt_root: Optional[Path] = None
        self._last_sync_time: float = 0.0
        self._is_subfolder_mode = False  # Merken ob wir im Subfolder-Mode sind

    def _detect_git(self) -> Tuple[bool, Optional[Path]]:
        try:
            # FIX: Use bytes + manual decode to avoid Windows cp1252 errors
            r = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.origin_root, capture_output=True, timeout=5)
            if r.returncode == 0:
                # Decode stdout manually with error handling
                stdout = r.stdout.decode("utf-8", errors="replace").strip()
                if stdout:
                    return True, Path(stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, NotADirectoryError):
            pass
        except UnicodeDecodeError:
            # Fallback for binary data in output
            pass
        for parent in [self.origin_root] + list(self.origin_root.parents):
            if (parent / ".git").exists():
                return True, parent
        return False, None

    @property
    def worktree_path(self) -> Optional[Path]:
        return self.path

    def _list_tracked_files(self, root: Path) -> List[Path]:
        """List tracked files, respecting the project root boundary."""
        if self._is_git and self._is_subfolder_mode:
            # In subfolder mode: use git ls-files but filter to only files within our project root
            try:
                # Get all tracked files from git
                r = subprocess.run(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=self._git_root, capture_output=True, timeout=15)
                if r.returncode == 0:
                    stdout = r.stdout.decode("utf-8", errors="replace")
                    # Filter to only files within our origin_root
                    result = []
                    for f in stdout.splitlines():
                        if not f.strip():
                            continue
                        full_path = self._git_root / f
                        # Only include files that are within or equal to origin_root
                        try:
                            full_path.resolve()
                            origin_resolved = self.origin_root.resolve()
                            # Check if full_path is within origin_root
                            if full_path == origin_resolved or str(full_path).startswith(str(origin_resolved)):
                                if full_path.is_file():
                                    # Return path relative to origin_root for proper copying
                                    result.append(full_path)
                        except (OSError, ValueError):
                            pass
                    return result
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("git ls-files failed, falling back to filtered rglob")
            except UnicodeDecodeError:
                logger.warning("git ls-files output had encoding issues, using fallback")

        if self._is_git:
            try:
                # FIX: Use bytes + manual decode to avoid Windows cp1252 errors
                r = subprocess.run(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=root, capture_output=True, timeout=15)
                if r.returncode == 0:
                    stdout = r.stdout.decode("utf-8", errors="replace")
                    return [root / f for f in stdout.splitlines() if f.strip() and (root / f).is_file()]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("git ls-files failed, falling back to filtered rglob")
            except UnicodeDecodeError:
                logger.warning("git ls-files output had encoding issues, using fallback")
        return [f for f in root.rglob("*")
                if not f.is_dir() and not (self.SCAN_EXCLUDES & set(f.parts))]

    def setup(self):
        if self.path and self.path.exists(): return
        import time
        self._last_sync_time = time.time()

        # === FIX: Prüfe ob origin_root == git_root (genau das Repo-Root) ===
        # Wenn origin_root ein Subfolder ist, IMMER gefilterte Kopie nutzen
        # weil git worktree immer das gesamte Repo klont
        use_git_worktree = False  # Default: keine git worktree

        if self._is_git and self._git_root:
            # Nur git worktree nutzen wenn origin_root EXAKT das git_root ist
            if self.origin_root.resolve() == self._git_root.resolve():
                use_git_worktree = True
                logger.info(f"[SETUP] Project is at git root, can use worktree")
            else:
                # origin_root ist ein Subfolder → gefilterte Kopie nutzen
                logger.info(f"[SETUP] Subfolder detected: {self.origin_root}")
                logger.info(f"[SETUP] Git root: {self._git_root}")
                logger.info(f"[SETUP] Using filtered copy (avoids cloning entire repo)")

        if use_git_worktree:
            # git worktree vom git_root erstellen
            self._wt_root = Path(tempfile.mkdtemp(prefix="coder_wt_"))
            try:
                existing = _safe_run(
                    ["git", "branch", "--list", self._branch],
                    cwd=self.origin_root, timeout=5)
                if existing.stdout.strip():
                    self._branch = f"coder-{uuid.uuid4().hex[:8]}"

                wt_list = _safe_run(
                    ["git", "worktree", "list", "--porcelain"],
                    cwd=self.origin_root, timeout=5)
                if "locked" in wt_list.stdout:
                    subprocess.run(["git", "worktree", "prune"],
                                   cwd=self.origin_root, capture_output=True, timeout=5)

                subprocess.run(
                    ["git", "worktree", "add", "-b", self._branch, str(self._wt_root)],
                    cwd=self._git_root,  # WICHTIG: vom git_root ausführen
                    capture_output=True, timeout=30)

                self.path = self._wt_root
                logger.info(f"[SETUP] git worktree = {self.path}")
                logger.info(f"[SETUP] origin       = {self.origin_root}")

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.warning(f"git worktree add failed ({e}), falling back to copytree")
                self._is_git = False
                if self._wt_root.exists():
                    shutil.rmtree(self._wt_root, ignore_errors=True)
                self._wt_root = Path(tempfile.mkdtemp(prefix="coder_cp_"))
                self.path = self._wt_root
                self._copy_filtered(self.origin_root, self.path)
        else:
            # IMMER gefilterte Kopie - nur den spezifischen Projektordner
            self._wt_root = Path(tempfile.mkdtemp(prefix="coder_cp_"))
            self.path = self._wt_root
            self._copy_filtered(self.origin_root, self.path)

            # FIX: Merke dass wir im "subfolder mode" sind für apply_back
            self._is_subfolder_mode = True
            # FIX: WICHTIG - Setze _is_git=False damit commit() nicht versucht git commands zu nutzen
            self._is_git = False

            logger.info(f"[SETUP] filtered copy = {self.path}")
            logger.info(f"[SETUP] origin        = {self.origin_root}")
            logger.info(f"[SETUP] file count    = {len(list(self.path.rglob('*')))}")

    def _copy_filtered(self, src_root: Path, dst_root: Path):
        files = self._list_tracked_files(src_root)
        for f in files:
            # Handle both absolute paths (from subfolder mode) and relative paths
            try:
                if f.is_absolute():
                    # File is absolute path - calculate relative to origin_root
                    rel = f.relative_to(self.origin_root)
                else:
                    # File is relative to src_root
                    rel = f.relative_to(src_root)
            except ValueError:
                # File is not relative to expected root, skip it
                logger.debug(f"Skipping file outside project root: {f}")
                continue

            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)

    # ===================================================================
    # FIX 1: commit() — Error checking + auto-config + return status
    # ===================================================================
    def commit(self, msg: str) -> bool:
        if not (self._is_git and self.path):
            return False
        for key, fallback in [("user.name", "CoderAgent"), ("user.email", "coder@local")]:
            check = _safe_run(["git", "config", key], cwd=self.path, timeout=5)
            if check.returncode != 0 or not check.stdout.strip():
                subprocess.run(["git", "config", key, fallback], cwd=self.path, capture_output=True)

        r_add = _safe_run(["git", "add", "."], cwd=self.path, timeout=10)
        if r_add.returncode != 0:
            logger.error(f"[COMMIT] git add failed: {r_add.stderr.strip()}")
            return False

        # LOG: Was wird committed?
        staged = _safe_run(["git", "diff", "--cached", "--name-only"],
                           cwd=self.path, timeout=5)
        staged_files = [f for f in staged.stdout.splitlines() if f.strip()]
        logger.info(f"[COMMIT] worktree={self.path}")
        logger.info(f"[COMMIT] staged {len(staged_files)} file(s): {staged_files}")

        if not staged_files:
            logger.warning("[COMMIT] Nothing staged — git add may have failed silently")

        r_commit = _safe_run(["git", "commit", "-m", msg, "--allow-empty"],
                              cwd=self.path, timeout=15)
        if r_commit.returncode != 0:
            stderr = r_commit.stderr.strip()
            if "nothing to commit" not in stderr and "nothing added" not in stderr:
                logger.error(f"[COMMIT] git commit failed: {stderr}")
                return False
            logger.info(f"[COMMIT] nothing to commit (ok)")
        else:
            logger.info(f"[COMMIT] committed: {r_commit.stdout.strip()[:200]}")
        return True

    async def _compute_hashes_parallel(self, files: list[Path]) -> dict[str, str]:
        import concurrent.futures
        def _hash_single(p: Path) -> tuple[str, str]:
            try: return (str(p), _file_hash_md5(p))
            except Exception: return (str(p), None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as executor:
            futures = [executor.submit(_hash_single, f) for f in files]
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                path, h = fut.result()
                if h is not None:
                    results[path] = h
            return results

    # ===================================================================
    # FIX 1b: apply_back() — Verifikation + Fallback auf Datei-Copy
    # ===================================================================
    async def apply_back(self) -> int:
        """Merge worktree changes back to origin. Returns changed file count (-1 = git merge)."""
        if not self.path:
            return 0

        if self._is_git:
            commit_ok = self.commit("pre-merge checkpoint")

            # CHECK: Gibt es überhaupt was zu mergen?
            try:
                diff_check = _safe_run(
                    ["git", "diff", "--stat", "HEAD", self._branch],
                    cwd=self.origin_root, timeout=10)
                has_changes = bool(diff_check.stdout.strip())
            except (subprocess.TimeoutExpired, FileNotFoundError):
                has_changes = False

            if not has_changes:
                # Branch hat keine Änderungen relativ zu HEAD
                # → Fallback auf direkte Datei-Kopie
                logger.warning("Git branch has no diff vs HEAD — falling back to file copy")
                return await self._apply_back_copy()

            try:
                subprocess.run(
                    ["git", "merge", self._branch, "--no-edit"],
                    cwd=self.origin_root, capture_output=True, check=True, timeout=30)
                return -1
            except subprocess.CalledProcessError as e:
                # Get stderr from result
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else "unknown"
                logger.error(f"git merge failed: {stderr}")
                logger.warning("Falling back to file copy after merge failure")
                # Abort the failed merge
                subprocess.run(["git", "merge", "--abort"],
                               cwd=self.origin_root, capture_output=True)
                return await self._apply_back_copy()

        return await self._apply_back_copy()

    async def _apply_back_copy(self) -> int:
        """Fallback: Hash-basierte Datei-Kopie."""
        src_files = self._list_tracked_files(self.path)
        dst_files = [(self.origin_root / f.relative_to(self.path)) for f in src_files]
        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel([d for d in dst_files if d.exists()])
        count = 0
        copied = []
        for src, dst in zip(src_files, dst_files):
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
                copied.append(str(dst.relative_to(self.origin_root)))
        logger.info(f"[APPLY-COPY] copied {count} file(s) to {self.origin_root}: {copied}")
        return count

    def cleanup(self):
        if not self.path: return
        # Das echte Worktree-Root für git worktree remove
        wt_root = getattr(self, '_wt_root', self.path)
        if self._is_git:
            subprocess.run(["git", "worktree", "remove", str(wt_root), "--force"],
                           cwd=self.origin_root, capture_output=True)
            subprocess.run(["git", "branch", "-D", self._branch],
                           cwd=self.origin_root, capture_output=True)
        else:
            # Non-git: Rename → instant, Background-Thread räumt auf
            import time, threading
            trash = self._wt_root.with_name(f".trash_{int(time.time())}_{self._wt_root.name}")
            try:
                self._wt_root.rename(trash)
                threading.Thread(target=shutil.rmtree, args=(trash,),
                                 kwargs={"ignore_errors": True}, daemon=True).start()
            except OSError:
                # Rename fehlgeschlagen (cross-device etc.) → tracked files einzeln
                for f in self._list_tracked_files(self._wt_root):
                    try:
                        f.unlink()
                    except OSError:
                        pass
        self.path = None
        self._wt_root = None

    async def rollback(self, files: list[str] | None = None):
        if not self.path: return
        if files:
            for f in files:
                src = self.origin_root / f
                dst = self.path / f
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                elif dst.exists():
                    dst.unlink()
        elif self._is_git:
            subprocess.run(["git", "checkout", "."], cwd=self.path, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=self.path, capture_output=True)
        else:
            changed = await self.changed_files()
            for f in changed:
                src = self.origin_root / f
                dst = self.path / f
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                elif dst.exists():
                    dst.unlink()
            wt_files = set(str(f.relative_to(self.path)) for f in self._list_tracked_files(self.path))
            origin_files = set(str(f.relative_to(self.origin_root)) for f in self._list_tracked_files(self.origin_root))
            for new_file in (wt_files - origin_files):
                target = self.path / new_file
                if target.exists():
                    target.unlink()

    async def apply_files(self, files: list[str]) -> list[str]:
        if not self.path: return []
        if not files: return []

        src_paths = [self.path / f for f in files]
        dst_paths = [self.origin_root / f for f in files]
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
        if not self.path: return []
        if self._is_git:
            r = _safe_run(["git", "diff", "--name-only", "HEAD"], cwd=self.path, timeout=10)
            u = _safe_run(["git", "ls-files", "--others", "--exclude-standard"], cwd=self.path, timeout=10)
            return [f for f in (r.stdout + u.stdout).splitlines() if f.strip()]

        src_files = self._list_tracked_files(self.path)
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

    async def sync_from_origin(self, sync_enabled: bool = True, sync_interval: float = 30.0) -> list[str]:
        import time
        if not self.path or not sync_enabled:
            return []

        now = time.time()
        if self._last_sync_time and (now - self._last_sync_time) < sync_interval:
            return []

        agent_changed = set(await self.changed_files())

        if self._is_git:
            try:
                r = _safe_run(["git", "diff", "--name-only"], cwd=self.origin_root, timeout=10)
                u = _safe_run(["git", "ls-files", "--others", "--exclude-standard"], cwd=self.origin_root, timeout=10)
                candidates = [f.strip() for f in (r.stdout + u.stdout).splitlines()
                              if f.strip() and f.strip() not in agent_changed]
                src_files = [self.origin_root / f for f in candidates if (self.origin_root / f).is_file()]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                src_files = []
        else:
            src_files = [f for f in self._list_tracked_files(self.origin_root)
                         if str(f.relative_to(self.origin_root)) not in agent_changed
                         and f.stat().st_mtime > self._last_sync_time]

        self._last_sync_time = now

        if not src_files:
            return []

        dst_files = [(self.path / f.relative_to(self.origin_root)) for f in src_files]

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
        "3. Änderungen NUR via Edit-Blöcke. BENUTZE NIEMALS 'bash' mit 'cat' oder 'echo' um Code zu schreiben!"
        " du must Edit-Blöcke verwenden für (Datein erstellen, Datein schriben, Datein berabeiten)\n"
        "   VERBOTEN: Nutze NIEMALS 'bash', 'cat', 'echo' oder 'printf' um Dateien zu erstellen oder zu ändern!\n"
        "   Das System erkennt Änderungen nur über Edit-Blöcke.\n"
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
        self.model = self.config.get("model", agent.amd.complex_llm_model)
        self.run_tests = self.config.get("run_tests", False)
        self.bash_timeout = self.config.get("bash_timeout", 300)
        self.sync_enabled = self.config.get("sync_enabled", True)
        self.sync_interval = self.config.get("sync_interval", 30.0)
        self.stream_enabled = self.config.get("stream",
                                              os.getenv("CODER_STREAM", "false").lower() == "true")
        self.log_handler = self.config.get("log_handler", None)
        self.stream_callback = self.config.get("stream_callback", self._default_stream_handler)
        self.debug_mode = (
            os.getenv("CODER_DEBUG", "false").lower() == "true" or
            os.getenv("AGENT_VERBOSE", "false").lower() == "true"
        )

        self.memory = ExecutionMemory(project_root)
        self.tracker = TokenTracker(self.model, agent)
        self.worktree = GitWorktree(project_root)

        self.state = {
            "plan": [],
            "done": [],
            "current_file": "None",
            "last_error": None
        }

    def _get_folder_structure(self, max_depth: int = 2) -> str:
        """
        Sammelt die Ordnerstruktur bis zu max_depth Ebenen.
        Gibt einen formatierten String zurück für den System-Prompt.
        """
        if not self.worktree.path or not self.worktree.path.exists():
            return "# Keine Workspace-Info verfügbar"

        try:
            root = self.worktree.path
            lines = [f"# Projektstruktur (max {max_depth} Ebenen):"]
            lines.append(f"# Root: {root}")
            lines.append("")

            # Ordner und Dateien sammeln
            items = []

            # Erste Ebene - Directories und Files
            for item in sorted(root.iterdir()):
                if item.name.startswith('.'):
                    continue

                rel_path = item.relative_to(root)

                if item.is_file():
                    size = item.stat().st_size
                    size_str = f" ({size}b)" if size < 1024 else f" ({size//1024}KB)"
                    lines.append(f"📄 {rel_path}{size_str}")
                elif item.is_dir():
                    lines.append(f"📁 {rel_path}/")

                    # Zweite Ebene - Dateien im Ordner
                    try:
                        for sub_item in sorted(item.iterdir())[:50]:  # Limit zu 50 pro Ordner
                            if sub_item.name.startswith('.'):
                                continue

                            sub_rel = sub_item.relative_to(root)

                            if sub_item.is_file():
                                size = sub_item.stat().st_size
                                size_str = f" ({size}b)" if size < 1024 else f" ({size//1024}KB)"
                                lines.append(f"   📄 {sub_item.name}{size_str}")
                            elif sub_item.is_dir():
                                lines.append(f"   📁 {sub_item.name}/")
                    except PermissionError:
                        lines.append(f"   (Zugriff verweigert)")

            return "\n".join(lines)

        except Exception as e:
            return f"# Fehler beim Lesen der Struktur: {e}"

    @staticmethod
    def _default_stream_handler(chunk: str):
        print(chunk, end="", flush=True)

    def _log(self, section: str, content: str, color: str = "white"):
        if not self.debug_mode: return
        C = {
            "cyan": "\033[96m", "yellow": "\033[93m", "green": "\033[92m",
            "red": "\033[91m", "grey": "\033[90m", "bold": "\033[1m", "reset": "\033[0m"
        }
        c_code = C.get(color, C["reset"])
        header = f"{C['bold']}[{section}]{C['reset']}"

        # === FIX: Safe Print für Windows ===
        try:
            print(f"{c_code}{header} {content}{C['reset']}", flush=True)
        except UnicodeEncodeError:
            # Fallback: Wenn UTF-8 fehlschlägt, ersetze Emojis durch '?' oder Ascii
            clean_content = content.encode('ascii', 'replace').decode('ascii')
            print(f"{c_code}{header} {clean_content}{C['reset']}", flush=True)
        # ===================================

        if self.log_handler:
            try:
                self.log_handler(section, content)
            except Exception:
                pass

    async def execute(self, task: str) -> CoderResult:
        self.worktree.setup()

        synced = await self.worktree.sync_from_origin(
            sync_enabled=self.sync_enabled, sync_interval=0)

        sys_msg = self.SYSTEM_PROMPT

        # FIX: Ordnerstruktur automatisch einfügen (2-level)
        folder_structure = self._get_folder_structure(max_depth=2)
        sys_msg += f"\n\n{folder_structure}\n"

        if prev := self.memory.get_context():
            sys_msg += f"\n\n## VERLAUF:\n{prev}"

        task_full = task
        if synced:
            task_full += f"\n\n[INFO: {len(synced)} file(s) synced from origin: {', '.join(synced[:5])}]"

        if self.agent and hasattr(self.agent, "arun_function"):
            try:
                arch_context = await self.agent.arun_function("memory_recall", query=task, search_type="auto")
                if arch_context and "No results" not in arch_context:
                    task_full = f"### SYSTEM ARCHITEKTUR-KONTEXT (Aus Memory):\n{arch_context}\n\n### DEINE AUFGABE:\n{task_full}"
            except Exception as e:
                logger.warning(f"Auto-Memory fetch failed: {e}")
                import traceback
                traceback.print_exc()

        messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": task_full}]

        # FIX: Track ALL changed files across loop + repair, not just last edits
        all_changed_files: List[str] = []

        try:
            edits, thought = await self._loop(messages)
            results = self._apply_edits(edits)
            all_changed_files.extend(r["file_path"] for r in results if r["success"])

            errors = await self._validate([r["file_path"] for r in results if r["success"]])

            if errors or any(not r["success"] for r in results):
                repair_edits, repair_thought = await self._repair(messages, results, errors)
                repair_results = self._apply_edits(repair_edits)
                all_changed_files.extend(r["file_path"] for r in repair_results if r["success"])
                if repair_thought:
                    thought = f"{thought}\n[Repair] {repair_thought}"

            self.worktree.commit(task)
        except Exception as e:
            logger.error(f"Execute failed: {e}")
            return CoderResult(False, str(e), [], messages)

        # Deduplicate
        all_changed_files = list(dict.fromkeys(all_changed_files))

        report = ExecutionReport(
            datetime.datetime.now().isoformat(), task,
            all_changed_files, True,
            summary=thought[:500] if thought else "")
        self.memory.add(report)

        return CoderResult(True, "Done", all_changed_files, messages,
                           memory_saved=True, tokens_used=self.tracker.total_tokens,
                           compressions_done=self.tracker.compressions_done)

    async def _loop(self, messages: list) -> Tuple[List[EditBlock], str]:
        """Core execution loop with Anti-Loop protection and Enhanced Debugging."""
        tools = self._tools()
        thought = ""
        recent_actions = []
        self.max_iters = 25
        iteration = -1
        _transient_status_idx = None
        # FIX: Sammle alle Edits über alle Iterationen (für incomplete block recovery)
        accumulated_edits: List[EditBlock] = []

        while iteration <= self.max_iters:
            iteration += 1
            self._log("LOOP", f"Iteration {iteration + 1}/{self.max_iters:} | Tokens: {self.tracker.total_tokens}", "cyan")

            if _transient_status_idx is not None and _transient_status_idx < len(messages):
                messages.pop(_transient_status_idx)

            status_update = (
                f"### AKTUELLE ITERATION: {iteration + 1}/{self.max_iters:}\n"
                f"**STATUS:**\n"
                f"- Datei im Fokus: {self.state['current_file']}\n"
                f"- Erledigt: {', '.join(self.state['done']) or 'Nichts'}\n"
                f"- Offen: {', '.join(self.state['plan']) or 'Unbekannt'}\n"
            )
            if iteration == self.max_iters - 1:
                status_update += "\n⚠️ LETZTE ITERATION! Beende deine Arbeit oder fasse exakt zusammen, was noch fehlt!"

            messages.append({"role": "user", "content": status_update})
            _transient_status_idx = len(messages) - 1

            if iteration > 0:
                synced = await self.worktree.sync_from_origin(
                    sync_enabled=self.sync_enabled, sync_interval=self.sync_interval)
                if synced:
                    msg = f"[SYNC] Dateien vom System aktualisiert: {', '.join(synced)}"
                    self._log("SYNC", msg, "grey")
                    messages.append({"role": "user", "content": msg})

            if self.tracker.needs_compression(messages):
                self._log("MEMORY", "Compressing context...", "yellow")
                messages[:] = await self.tracker.compress(messages)

            self._log("LLM", "Waiting for response...", "grey")

            content = ""
            tool_calls = []

            if self.stream_enabled:
                self._log("STREAM", "Streaming started...", "cyan")
                full_tool_calls_json = {}

                response_gen = await self.agent.a_run_llm_completion(
                    messages=messages, tools=tools, stream=True, true_stream=True
                )

                async for chunk in response_gen:
                    delta_content = ""
                    delta_tool_calls = None

                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        delta_content = getattr(delta, "content", "") or ""
                        delta_tool_calls = getattr(delta, "tool_calls", None)
                    elif isinstance(chunk, dict):
                        delta_content = chunk.get("content", "")
                    elif isinstance(chunk, str):
                        delta_content = chunk

                    if delta_content:
                        content += delta_content
                        if self.stream_callback:
                            try:
                                if asyncio.iscoroutinefunction(self.stream_callback):
                                    await self.stream_callback(delta_content)
                                else:
                                    self.stream_callback(delta_content)
                            except Exception:
                                pass

                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            idx = tc.index
                            if idx not in full_tool_calls_json:
                                full_tool_calls_json[idx] = {"id": tc.id, "name": "", "args": ""}
                            if hasattr(tc, "id") and tc.id:
                                full_tool_calls_json[idx]["id"] = tc.id
                            if hasattr(tc, "function"):
                                if tc.function.name:
                                    full_tool_calls_json[idx]["name"] += tc.function.name
                                if tc.function.arguments:
                                    full_tool_calls_json[idx]["args"] += tc.function.arguments

                if self.stream_callback == self._default_stream_handler:
                    print()

                if full_tool_calls_json:
                    from types import SimpleNamespace
                    for idx in sorted(full_tool_calls_json.keys()):
                        data = full_tool_calls_json[idx]
                        if data["name"] or data["args"]:
                            tc_obj = SimpleNamespace(
                                id=data["id"],
                                function=SimpleNamespace(
                                    name=data["name"],
                                    arguments=data["args"]
                                )
                            )
                            tool_calls.append(tc_obj)

            else:
                resp = await self.agent.a_run_llm_completion(
                    messages=messages, tools=tools, stream=False, get_response_message=True, model=self.model)
                content = resp.content or ""
                tool_calls = resp.tool_calls or []

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if thought_match := re.search(r"<thought>(.*?)</thought>", content, re.DOTALL):
                current_thought = thought_match.group(1).strip()
                self._update_internal_state(current_thought)
                new_thought_part = current_thought.replace(thought, "").strip()
                if new_thought_part:
                    self._log("THOUGHT", new_thought_part, "yellow")
                thought = current_thought
            elif content.strip():
                self._log("MSG", content[:300] + ("..." if len(content) > 300 else ""), "yellow")

            # 3. Tool Execution & Loop-Check
            if tool_calls:
                for tc in tool_calls:
                    try:
                        args_data = json.loads(tc.function.arguments)
                        sig = f"{tc.function.name}:{json.dumps(args_data, sort_keys=True)}"

                        self._log("TOOL CALL", f"{tc.function.name}({json.dumps(args_data, indent=None)})", "green")

                        if sig in recent_actions:
                            res = f"FEHLER: Loop erkannt! '{sig}' wurde bereits ausgeführt."
                            logger.warning(f"Loop blocked: {sig}")
                            self._log("LOOP-GUARD", f"Blocked repetition: {sig}", "red")
                        else:
                            recent_actions.append(sig)
                            if tc.function.name == "read_file":
                                self.state["current_file"] = args_data.get("path")
                            if len(recent_actions) > 8: recent_actions.pop(0)

                            res = await self._dispatch(tc.function.name, args_data, messages)

                        display_res = res if len(res) < 500 else (
                                res[:200] + f"\n... [+{len(res) - 200} chars] ...\n" + res[-200:])
                        self._log("TOOL RESULT", display_res, "grey")

                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": res})
                    except Exception as e:
                        err_msg = f"Fehler: {e}"
                        self._log("TOOL ERROR", err_msg, "red")
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": err_msg})
                continue

            # ===========================================================
            # FIX 2: Finalizing — mit incomplete block handling
            # ===========================================================
            blocks, incomplete = self._parse_edits(content)

            # Incomplete Blöcke: sofort anwenden + Agent zum Weitermachen auffordern
            if incomplete:
                self._log("ACTION",
                          f"⚠ {len(incomplete)} INCOMPLETE edit block(s) detected. Applying partial content.",
                          "yellow")
                self._apply_edits(incomplete)
                accumulated_edits.extend(incomplete)

                truncated_files = [b.file_path for b in incomplete]
                messages.append({
                    "role": "user",
                    "content": (
                        f"⚠️ WARNUNG: {len(incomplete)} Edit-Block(s) waren unvollständig "
                        f"(dein Output wurde abgeschnitten)!\n"
                        f"Betroffene Dateien: {', '.join(truncated_files)}\n"
                        f"Der bisherige Content wurde gespeichert. Bitte:\n"
                        f"1. Lies die Datei(en) mit read_file\n"
                        f"2. Vervollständige den fehlenden Content mit einem neuen Edit-Block\n"
                        f"3. Benutze den SEARCH-Block mit den letzten ~5 Zeilen des bisherigen Contents"
                    )
                })

                if blocks:
                    # Vollständige Blöcke auch anwenden
                    accumulated_edits.extend(blocks)
                # Loop weiterlaufen lassen damit Agent vervollständigen kann
                continue

            if blocks:
                all_blocks = accumulated_edits + blocks
                self._log("ACTION", f"Found {len(blocks)} complete edit blocks. Stopping loop.", "green")
                return all_blocks, thought

            if "[DONE]" in content:
                if accumulated_edits:
                    # Agent sagt DONE, hat aber vorher incomplete blocks produziert
                    self._log("DONE", f"Agent done. {len(accumulated_edits)} blocks from earlier iterations included.", "green")
                    return accumulated_edits, thought
                self._log("DONE", "Agent marked task as complete.", "green")
                return [], thought

        self._log("EXIT", "Max iterations reached without explicit done.", "red")
        return accumulated_edits, thought

    def _update_internal_state(self, thought: str):
        lines = thought.splitlines()
        new_plan = []
        newly_done = []

        for l in lines:
            l = l.strip()
            if l.startswith("- [ ]"):
                task = l[5:].strip()
                new_plan.append(task)
            elif l.startswith("- [x]"):
                done_item = l[5:].strip()
                if done_item not in self.state["done"]:
                    self.state["done"].append(done_item)
                    newly_done.append(done_item)

        if new_plan:
            self.state["plan"] = new_plan

        if self.debug_mode:
            if newly_done:
                for item in newly_done:
                    self._log("PLAN UPDATE", f"✔ Erledigt: {item}", "green")
                    self.max_iters += 2
            if len(new_plan) != len(self.state.get("last_plan_snapshot", [])):
                self._log("PLAN UPDATE", f"Neue Agenda: {len(new_plan)} offene Punkte", "cyan")
                self.state["last_plan_snapshot"] = new_plan

    async def _repair(self, messages: list, failed: list, errors: list) -> Tuple[List[EditBlock], str]:
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
        """ATOMIC TRANSACTION with IO Logging."""
        results = []
        my_module = Path(__file__).resolve()
        pending = []

        self._log("IO", f"Starting Transaction for {len(edits)} files...", "cyan")

        for e in edits:
            target = (self.worktree.path / e.file_path).resolve()
            is_self = target.samefile(my_module) if target.exists() and my_module.exists() else False

            try:
                if not e.search.strip():
                    new_bytes = len(e.replace.encode('utf-8'))
                    self._log("IO-CALC", f"[NEW] {e.file_path} (+{new_bytes} bytes)", "green")
                    pending.append((target, e.replace, None))
                    results.append({"file_path": e.file_path, "success": True})
                    continue

                if not target.exists():
                    self._log("IO-ERR", f"{e.file_path} not found!", "red")
                    results.append({"file_path": e.file_path, "success": False, "error": "File missing"})
                    continue

                txt = target.read_text(encoding="utf-8", errors="replace")
                original_bytes = len(txt.encode('utf-8'))

                if e.search in txt:
                    match_type = "EXACT"
                    new_txt = txt.replace(e.search, e.replace, 1)
                elif (idx := self._fuzzy_find(txt, e.search)) is not None:
                    match_type = "FUZZY"
                    src_lines = txt.splitlines(keepends=True)
                    s_lines = e.search.splitlines()
                    new_lines = src_lines[:idx] + [e.replace + "\n"] + src_lines[idx + len(s_lines):]
                    new_txt = "".join(new_lines)
                else:
                    self._log("IO-ERR", f"SEARCH block not found in {e.file_path}", "red")
                    results.append({"file_path": e.file_path, "success": False,
                                    "error": "SEARCH not found (exact + fuzzy)"})
                    continue

                new_bytes = len(new_txt.encode('utf-8'))
                byte_diff = new_bytes - original_bytes
                sign = "+" if byte_diff > 0 else ""

                self._log("IO-CALC",
                          f"[{match_type}] {e.file_path} | {original_bytes} -> {new_bytes} bytes ({sign}{byte_diff})",
                          "yellow" if match_type == "FUZZY" else "green")

                if is_self and e.file_path.endswith(".py"):
                    try:
                        compile(new_txt, e.file_path, "exec")
                    except SyntaxError as se:
                        self._log("IO-FAIL", f"Self-edit SyntaxError: {se}", "red")
                        results.append({"file_path": e.file_path, "success": False,
                                        "error": f"Self-edit blocked: SyntaxError L{se.lineno}"})
                        continue

                pending.append((target, new_txt, txt))
                results.append({"file_path": e.file_path, "success": True})

            except Exception as ex:
                self._log("IO-EXC", str(ex), "red")
                results.append({"file_path": e.file_path, "success": False, "error": str(ex)})

        written = []
        try:
            for target, new_content, original_content in pending:
                self._atomic_write(target, new_content)
                written.append((target, original_content))
            self._log("IO-COMMIT", "All files written successfully.", "green")
        except BaseException as commit_err:
            self._log("IO-ROLLBACK", f"Transaction failed: {commit_err}. Rolling back...", "red")
            for target, original in written:
                try:
                    if original is not None:
                        self._atomic_write(target, original)
                    elif target.exists():
                        target.unlink()
                except Exception:
                    pass
            for r in results:
                if r["success"]:
                    r["success"] = False
                    r["error"] = f"Transaction failed: {commit_err}"
            raise

        return results

    @staticmethod
    def _atomic_write(path: Path, content: str):
        import tempfile as _tf
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tf.mkstemp(dir=path.parent, suffix=".tmp")
        fd_closed = False
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            fd_closed = True
            os.replace(tmp, path)
        except BaseException:
            if not fd_closed:
                try: os.close(fd)
                except OSError: pass
            try: os.unlink(tmp)
            except OSError: pass
            raise

    @staticmethod
    def _fuzzy_find(txt: str, search: str, threshold: float = 0.85) -> int | None:
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
        result_output = (out + err).decode(errors="replace").strip()
        if not result_output:
            return "Command executed (no output)."
        return result_output

    async def _run_grep(self, pattern: str) -> str:
        quoted = shlex.quote(pattern) if platform.system() != "Windows" else pattern
        if shutil.which("rg"):
            return await self._run_bash(f"rg -n {quoted} .")
        if self.worktree._is_git:
            return await self._run_bash(f"git grep -n {quoted}")
        if platform.system() == "Windows":
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

    # ===================================================================
    # FIX 2: _parse_edits() — Rettet unvollständige Blöcke
    # ===================================================================
    def _parse_edits(self, text: str) -> Tuple[List[EditBlock], List[EditBlock]]:
        """State-machine parser. Returns (complete_blocks, incomplete_blocks)."""
        IDLE, IN_SEARCH, IN_REPLACE = 0, 1, 2
        state = IDLE
        path = ""
        search_lines: list[str] = []
        replace_lines: list[str] = []
        blocks: list[EditBlock] = []
        incomplete: list[EditBlock] = []

        if self.debug_mode and "~~~edit:" in text:
            self._log("PARSER", f"Scanning {len(text)} chars for edit blocks...", "grey")

        for raw_line in text.split("\n"):
            stripped = raw_line.strip()

            if state == IDLE:
                m = re.match(r"^~~~edit:(.+?)~~~$", stripped)
                if m:
                    path = m.group(1).strip()
                    search_lines, replace_lines = [], []
                    state = IN_SEARCH
                    self._log("PARSER", f"Start Block: {path}", "cyan")
                continue

            if state == IN_SEARCH:
                if stripped == "<<<<<<< SEARCH":
                    continue
                if stripped == "=======":
                    state = IN_REPLACE
                    continue
                if stripped.endswith("=======") and len(stripped) > 7:
                    # Zeile VOR dem Separator gehört noch zum SEARCH
                    prefix = raw_line.split("=======")[0]
                    if prefix.strip():
                        search_lines.append(prefix)
                    state = IN_REPLACE
                    self._log("PARSER", f"⚠ Inline separator detected in SEARCH block", "yellow")
                    continue
                search_lines.append(raw_line)

            elif state == IN_REPLACE:
                if stripped == ">>>>>>> REPLACE":
                    continue
                if stripped == "~~~end~~~":
                    s_count = len(search_lines)
                    r_count = len(replace_lines)
                    diff = r_count - s_count
                    diff_sign = "+" if diff > 0 else ""

                    self._log("PARSER",
                              f"End Block: {path} | Search: {s_count} lines | Replace: {r_count} lines | Delta: {diff_sign}{diff}",
                              "yellow")

                    blocks.append(EditBlock(
                        file_path=path,
                        search="\n".join(search_lines),
                        replace="\n".join(replace_lines),
                    ))
                    state = IDLE
                    continue
                if ">>>>>>> REPLACE" in stripped:
                    prefix = raw_line.split(">>>>>>> REPLACE")[0]
                    if prefix.strip():
                        replace_lines.append(prefix)
                    continue
                    # FIX: Inline ~~~end~~~
                if stripped.endswith("~~~end~~~") and len(stripped) > 9:
                    prefix = raw_line.split("~~~end~~~")[0]
                    if prefix.strip():
                        replace_lines.append(prefix)
                    # Block fertig
                    s_count = len(search_lines)
                    r_count = len(replace_lines)
                    diff = r_count - s_count
                    diff_sign = "+" if diff > 0 else ""
                    self._log("PARSER",
                              f"End Block: {path} | Search: {s_count} lines | Replace: {r_count} lines | Delta: {diff_sign}{diff}",
                              "yellow")
                    blocks.append(EditBlock(
                        file_path=path,
                        search="\n".join(search_lines),
                        replace="\n".join(replace_lines),
                    ))
                    state = IDLE
                    continue
                replace_lines.append(raw_line)

        # === FIX: Unvollständigen Block retten statt still droppen ===
        if state != IDLE and path:
            is_new_file = not "\n".join(search_lines).strip()
            has_content = bool(replace_lines)

            if is_new_file and has_content:
                # Neuer File mit Content → retten (häufigstes Truncation-Szenario)
                self._log("PARSER",
                          f"⚠ INCOMPLETE new-file block rescued: {path} | {len(replace_lines)} lines (truncated by LLM)",
                          "yellow")
                incomplete.append(EditBlock(
                    file_path=path,
                    search="",
                    replace="\n".join(replace_lines),
                ))
            elif state == IN_REPLACE and has_content:
                # Edit mit SEARCH+REPLACE aber abgebrochen → auch retten
                self._log("PARSER",
                          f"⚠ INCOMPLETE edit block rescued: {path} | Search: {len(search_lines)} | Replace: {len(replace_lines)} (truncated)",
                          "yellow")
                incomplete.append(EditBlock(
                    file_path=path,
                    search="\n".join(search_lines),
                    replace="\n".join(replace_lines),
                ))
            else:
                self._log("PARSER",
                          f"✗ DROPPED incomplete block: {path} (state={state}, no usable content)",
                          "red")

        return blocks, incomplete

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

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CoderAgent v4.1 – Standalone")
    p.add_argument("task", help="Coding task")
    p.add_argument("-p", "--project", default=".", help="Project root (default: cwd)")
    p.add_argument("-m", "--model", default="gpt-4o", help="LLM model")
    p.add_argument("--run-tests", action="store_true", help="Run pytest after edits")
    p.add_argument("--timeout", type=int, default=300, help="Bash timeout in seconds")
    p.add_argument("--apply", action="store_true", help="Auto-apply changes to repo")
    p.add_argument("--stream", action="store_true", help="Stream output to terminal")
    args = p.parse_args()

    async def _main():
        agent = None
        coder = CoderAgent(agent, os.path.abspath(args.project),
                           {"model": args.model, "stream": args.stream, "run_tests": args.run_tests,
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
