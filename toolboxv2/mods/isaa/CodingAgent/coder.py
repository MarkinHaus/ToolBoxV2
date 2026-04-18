"""CoderAgent v5 – Multi-Agent Orchestration Edition.

Architecture:
  execute(task)
    ├─ Worktree setup (unchanged)
    ├─ _ensure_agents()  ← lazy init: planner, coder(s), validator
    ├─ VFS mount: worktree → /project for all agents
    ├─ PLANNER PHASE  → structured subtasks
    ├─ CODER PHASE    → parallel on non-overlapping file sets
    ├─ VALIDATOR PHASE → issues → fix loop (max 2)
    └─ POST-EXECUTE   → commit, report, memory

External interface unchanged: CoderAgent(agent, root, config).execute(task) → CoderResult
"""

import asyncio, datetime, json, logging, os, platform, re, shlex, shutil, subprocess, tempfile, uuid
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import sys

from toolboxv2.utils.extras.Style import print_prompt

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')

from toolboxv2 import get_logger, Spinner

logger = get_logger()

try:
    from toolboxv2.mods.isaa.base.AgentUtils import detect_shell
except ImportError:
    def detect_shell() -> Tuple[str, str]:
        return ("cmd.exe", "/c") if platform.system() == "Windows" else ("/bin/sh", "-c")

try: import litellm
except ImportError: litellm = None

from toolboxv2.mods.isaa.base.Agent.lsp_manager import LSPManager, DiagnosticSeverity


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES (unchanged)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# TOKEN MANAGEMENT (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def _ctx_limit(model: str) -> int:
    if litellm:
        try:
            info = litellm.get_model_info(model)
            if lim := (info.get("max_input_tokens") or info.get("max_tokens")): return int(lim)
        except Exception: pass
    return 200_000

def _count_tokens(messages: List[dict], model: str) -> int:
    if litellm:
        try: return litellm.token_counter(model=model, messages=messages)
        except Exception: pass
    return max(1, sum(len(str(m.get("content", ""))) // 4 for m in messages))


class TokenTracker:
    def __init__(self, model: str, agent=None, limit=0.65):
        self.model, self.agent = model, agent
        self.limit = _ctx_limit(model.split('/')[-1])
        self.threshold = int(self.limit * limit)
        self.total_tokens = self.compressions_done = 0

    def needs_compression(self, messages: List[dict]) -> bool:
        self.total_tokens = _count_tokens(messages, self.model)
        return self.total_tokens >= self.threshold

    async def compress(self, messages: List[dict]) -> List[dict]:
        if len(messages) <= 6: return messages

        messages = messages.copy()
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
                    if len(lines) > 55: snippet += "\n... [gekürzt] ..."
                    file_contexts[path] = snippet

        summary_text = await self._summarize(middle_history)
        new_history = [system_msg, task_msg]
        new_history.append({"role": "user", "content": f"### VERLAUF-RECAP:\n{summary_text}"})

        if file_contexts:
            context_blob = "\n\n".join(file_contexts.values())
            new_history.append({
                "role": "system",
                "content": f"### AKTUELLER DATEI-KONTEXT:\n{context_blob}"
            })

        new_history.extend(recent_history)
        sanitized_history = []
        in_tool_block = False
        for msg in new_history:
            role = msg.get("role")
            if role == "tool":
                prev = sanitized_history[-1] if sanitized_history else None
                is_valid = (in_tool_block or
                            (prev and prev.get("role") == "assistant" and prev.get("tool_calls")))
                if is_valid:
                    in_tool_block = True
                    sanitized_history.append(msg)
                else:
                    sanitized_history.append({
                        "role": "user",
                        "content": f"[Tool-Ergebnis]\n{msg.get('content', '')}"
                    })
                continue
            in_tool_block = False
            sanitized_history.append(msg)

        self.compressions_done += 1
        return sanitized_history

    def usage_ratio(self, messages: List[dict]) -> float:
        self.total_tokens = _count_tokens(messages, self.model)
        return self.total_tokens / self.limit

    async def _summarize(self, msgs: List[dict]) -> str:
        if not self.agent: return "(Zusammenfassung nicht möglich)"
        try:
            prompt = (
                "Fasse die bisherigen Aktionen zusammen. Antworte extrem kurz.\n"
                "Fokus: Welche Dateien wurden editiert? Welche Probleme traten auf?"
            )
            stream = "\n".join([f"{m['role']}: " + (m.get('content', '')[:200]) for m in msgs])
            return await self.agent.a_run_llm_completion(
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": stream}],
                stream=False
            ) or "..."
        except Exception:
            return "Dialog komprimiert."


# ═══════════════════════════════════════════════════════════════════════
# EXECUTION MEMORY (unchanged)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# HELPERS (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def _safe_run(cmd: list, cwd=None, capture_output=True, check=False, timeout=None, text=True):
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=capture_output, check=check, timeout=timeout)
        if text:
            stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            class SafeResult:
                def __init__(self, r, out, err):
                    self.returncode = r.returncode; self.stdout = out; self.stderr = err
                    self.args = r.args
            return SafeResult(result, stdout, stderr)
        return result
    except Exception as e:
        logger.debug(f"subprocess.run failed: {e}")
        raise


def _file_hash_md5(p: Path) -> str:
    import hashlib
    if not p.exists() or p.is_dir(): return ""
    hash_md5 = hashlib.md5()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except (OSError, IOError) as e:
        logger.error(f"Hash-Fehler bei {p}: {e}")
        return f"error-{uuid.uuid4().hex[:4]}"


def text_to_block(text: str, max_chars: int = 300) -> list[str]:
    result = []
    for line in text.splitlines():
        words = line.split()
        if not words:
            result.append("")
            continue
        current_line = ""
        for word in words:
            while len(word) > max_chars:
                if current_line:
                    result.append(current_line)
                    current_line = ""
                result.append(word[:max_chars])
                word = word[max_chars:]
            if not word: continue
            if current_line and len(current_line) + 1 + len(word) > max_chars:
                result.append(current_line)
                current_line = word
            else:
                current_line = f"{current_line} {word}" if current_line else word
        if current_line:
            result.append(current_line)
    return result


# ═══════════════════════════════════════════════════════════════════════
# SMART FILE READER (unchanged)
# ═══════════════════════════════════════════════════════════════════════

async def smart_read_file(
    path: str, start: Optional[int], end: Optional[int],
    worktree: Path, agent=None, messages: list = None, model: str = "", query: str = "", lsp_manager: LSPManager = None
) -> str:
    messages = messages or []
    target = (worktree / path).resolve()
    if not target.exists(): return f"Error: {path} not found."
    with open(target, "rb") as _bf:
        if b'\x00' in (_bf.read(512) or b''): return f"Binary: {path}. Use bash+xxd."

    content = await asyncio.to_thread(target.read_text, encoding="utf-8", errors="replace")
    lines = text_to_block(content)
    total = len(lines)

    if lsp_manager:
        ext = Path(path).suffix.lower().replace(".", "")
        lang_id = {"py": "python", "js": "javascript", "ts": "typescript", "tsx": "typescriptreact"}.get(ext, ext)
        if lang_id:
            try:
                diags = await lsp_manager.get_diagnostics(path, content, lang_id)
                if diags:
                    diag_strs = [d.to_display_string(lines) for d in diags
                                 if d.severity in [DiagnosticSeverity.ERROR, DiagnosticSeverity.WARNING]][:5]
                    if diag_strs:
                        lines += "\n\n### LSP Diagnostics:\n" + "\n".join(diag_strs)
            except Exception as e:
                logger.debug(f"LSP Error ignored: {e}")

    if start is not None or end is not None:
        return _fmt(path, lines, start if start is not None else 1, end if end is not None else total)

    usage = _count_tokens(messages, model) / _ctx_limit(model) if model else 0

    if usage < 0.60:
        if total <= 600:
            return _fmt(path, lines, 1, total)
        head = _fmt(path, lines, 1, 150)
        tail = _fmt(path, lines, total - 149, total)
        return f"{head}\n\n... [{total - 300} lines omitted] ...\n\n{tail}"

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
    if not hits: return f"[Critical] {path}: {total} lines, no matches for {terms}."
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


# ═══════════════════════════════════════════════════════════════════════
# GIT WORKTREE (unchanged — full class kept for completeness)
# ═══════════════════════════════════════════════════════════════════════

class GitWorktree:
    SCAN_EXCLUDES = frozenset({
        ".git", "node_modules", ".venv", "venv", "__pycache__", ".tox",
        "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".eggs", "*.egg-info", ".next", ".nuxt", "target", "out",
        "coverage", ".coverage", "htmlcov", ".idea", ".vscode",
    })

    def __init__(self, root: str):
        self.origin_root = Path(root).resolve()
        self.state_file = self.origin_root / ".coder_worktree.json"
        self._is_git, self._git_root = self._detect_git()
        if self._git_root:
            self._git_root = self._git_root.resolve()
        self._branch = f"coder-{uuid.uuid4().hex[:8]}"
        self.path: Optional[Path] = None
        self._wt_root: Optional[Path] = None
        self._last_sync_time: float = 0.0
        self._is_subfolder_mode = False

    def _detect_git(self) -> Tuple[bool, Optional[Path]]:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=self.origin_root, capture_output=True, timeout=5)
            if r.returncode == 0:
                stdout = r.stdout.decode("utf-8", errors="replace").strip()
                if stdout:
                    return True, Path(stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, NotADirectoryError):
            pass
        except UnicodeDecodeError:
            pass
        for parent in [self.origin_root] + list(self.origin_root.parents):
            if (parent / ".git").exists():
                return True, parent
        return False, None

    @property
    def worktree_path(self) -> Optional[Path]:
        return self.path

    def _list_tracked_files(self, root: Path) -> List[Path]:
        if self._is_git and self._is_subfolder_mode:
            try:
                r = subprocess.run(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=self._git_root, capture_output=True, timeout=15)
                if r.returncode == 0:
                    stdout = r.stdout.decode("utf-8", errors="replace")
                    result = []
                    origin_str = str(self.origin_root.resolve()) + os.sep
                    for f in stdout.splitlines():
                        if not f.strip(): continue
                        full_path = (self._git_root / f).resolve()
                        fp_str = str(full_path)
                        if fp_str.startswith(origin_str) or fp_str == origin_str.rstrip(os.sep):
                            result.append(full_path)
                    return result
            except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeDecodeError):
                logger.warning("git ls-files failed, falling back to walk")

        if self._is_git:
            try:
                r = subprocess.run(
                    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                    cwd=root, capture_output=True, timeout=15)
                if r.returncode == 0:
                    stdout = r.stdout.decode("utf-8", errors="replace")
                    return [root / f for f in stdout.splitlines() if f.strip() and (root / f).is_file()]
            except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeDecodeError):
                logger.warning("git ls-files failed, falling back to walk")
        return self._list_tracked_files_walk(root)

    def _list_tracked_files_walk(self, root: Path) -> List[Path]:
        result = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.SCAN_EXCLUDES and not d.endswith(".egg-info")]
            for fname in filenames:
                result.append(Path(dirpath) / fname)
        return result

    def setup(self):
        if self.path and self.path.exists(): return
        import time
        self._last_sync_time = time.time()

        if self.state_file.exists():
            try:
                saved_data = json.loads(self.state_file.read_text())
                saved_path = Path(saved_data.get("worktree_path", ""))
                if saved_path.exists():
                    self.path = saved_path
                    self._wt_root = saved_path
                    self._is_subfolder_mode = saved_data.get("subfolder_mode", False)
                    if not (self.path / ".git").exists():
                        self._is_git = False
                    logger.info(f"[SETUP] Resumed worktree: {self.path}")
                    return
            except Exception as e:
                logger.warning(f"[SETUP] Failed to resume: {e}")

        use_git_worktree = False
        if self._is_git and self._git_root:
            if self.origin_root.resolve() == self._git_root.resolve():
                use_git_worktree = True

        if use_git_worktree:
            self._wt_root = Path(tempfile.mkdtemp(prefix="coder_wt_"))
            try:
                existing = _safe_run(["git", "branch", "--list", self._branch], cwd=self.origin_root, timeout=5)
                if existing.stdout.strip():
                    self._branch = f"coder-{uuid.uuid4().hex[:8]}"
                wt_list = _safe_run(["git", "worktree", "list", "--porcelain"], cwd=self.origin_root, timeout=5)
                if "locked" in wt_list.stdout:
                    subprocess.run(["git", "worktree", "prune"], cwd=self.origin_root, capture_output=True, timeout=5)
                subprocess.run(
                    ["git", "worktree", "add", "-b", self._branch, str(self._wt_root)],
                    cwd=self._git_root, capture_output=True, timeout=30)
                self.path = self._wt_root
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                self._is_git = False
                if self._wt_root.exists():
                    shutil.rmtree(self._wt_root, ignore_errors=True)
                self._wt_root = Path(tempfile.mkdtemp(prefix="coder_cp_"))
                self.path = self._wt_root
                self._copy_filtered(self.origin_root, self.path)
        else:
            self._is_subfolder_mode = True
            self._is_git = False
            self._wt_root = Path(tempfile.mkdtemp(prefix="coder_cp_"))
            self.path = self._wt_root
            self._copy_filtered(self.origin_root, self.path)

        try:
            self.state_file.write_text(json.dumps({
                "worktree_path": str(self.path.resolve()),
                "subfolder_mode": self._is_subfolder_mode,
                "branch": self._branch
            }))
        except Exception as e:
            logger.warning(f"Could not save worktree state: {e}")

    def _copy_filtered(self, src_root: Path, dst_root: Path):
        files = self._list_tracked_files(src_root)
        for f in files:
            try:
                rel = f.relative_to(self.origin_root) if f.is_absolute() else f.relative_to(src_root)
            except ValueError:
                continue
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        return len(files)

    def commit(self, msg: str) -> bool:
        if not (self._is_git and self.path): return False
        for key, fallback in [("user.name", "CoderAgent"), ("user.email", "coder@local")]:
            check = _safe_run(["git", "config", key], cwd=self.path, timeout=5)
            if check.returncode != 0 or not check.stdout.strip():
                subprocess.run(["git", "config", key, fallback], cwd=self.path, capture_output=True)
        r_add = _safe_run(["git", "add", "."], cwd=self.path, timeout=10)
        if r_add.returncode != 0: return False
        staged = _safe_run(["git", "diff", "--cached", "--name-only"], cwd=self.path, timeout=5)
        staged_files = [f for f in staged.stdout.splitlines() if f.strip()]
        if not staged_files: return True
        r_commit = _safe_run(["git", "commit", "-m", msg, "--allow-empty"], cwd=self.path, timeout=15)
        return r_commit.returncode == 0 or "nothing to commit" in r_commit.stderr

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
                if h is not None: results[path] = h
            return results

    async def apply_back(self) -> int:
        if not self.path: return 0
        if self._is_git:
            self.commit("pre-merge checkpoint")
            try:
                diff_check = _safe_run(["git", "diff", "--stat", "HEAD", self._branch], cwd=self.origin_root, timeout=10)
                if not diff_check.stdout.strip():
                    return await self._apply_back_copy()
                subprocess.run(["git", "merge", self._branch, "--no-edit"],
                               cwd=self.origin_root, capture_output=True, check=True, timeout=30)
                return -1
            except subprocess.CalledProcessError:
                subprocess.run(["git", "merge", "--abort"], cwd=self.origin_root, capture_output=True)
                return await self._apply_back_copy()
        return await self._apply_back_copy()

    async def _apply_back_copy(self) -> int:
        src_files = self._list_tracked_files(self.path)
        dst_files = [(self.origin_root / f.relative_to(self.path)) for f in src_files]
        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel([d for d in dst_files if d.exists()])
        count = 0
        for src, dst in zip(src_files, dst_files):
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                count += 1
        return count

    def cleanup(self):
        if not self.path: return
        wt_root = getattr(self, '_wt_root', self.path)
        if self._is_git:
            subprocess.run(["git", "worktree", "remove", str(wt_root), "--force"],
                           cwd=self.origin_root, capture_output=True)
            subprocess.run(["git", "branch", "-D", self._branch],
                           cwd=self.origin_root, capture_output=True)
        else:
            import time, threading
            trash = self._wt_root.with_name(f".trash_{int(time.time())}_{self._wt_root.name}")
            try:
                self._wt_root.rename(trash)
                threading.Thread(target=shutil.rmtree, args=(trash,),
                                 kwargs={"ignore_errors": True}, daemon=True).start()
            except OSError:
                shutil.rmtree(self._wt_root, ignore_errors=True)
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

    async def apply_files(self, files: list[str]) -> list[str]:
        if not self.path or not files: return []
        src_paths = [self.path / f for f in files]
        dst_paths = [self.origin_root / f for f in files]
        existing_pairs = [(s, d, f) for s, d, f in zip(src_paths, dst_paths, files) if s.exists()]
        if not existing_pairs: return []
        src_list, dst_list, file_list = zip(*existing_pairs)
        src_hashes = await self._compute_hashes_parallel(list(src_list))
        dst_hashes = await self._compute_hashes_parallel([d for d in dst_list if d.exists()])
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
            try:
                r = _safe_run(["git", "diff", "--name-only", "HEAD"], cwd=self.path, timeout=20)
                u = _safe_run(["git", "ls-files", "--others", "--exclude-standard"], cwd=self.path, timeout=20)
                return [f for f in (r.stdout + u.stdout).splitlines() if f.strip()]
            except subprocess.TimeoutExpired:
                pass
        src_files = self._list_tracked_files(self.path)
        dst_files = [(self.origin_root / f.relative_to(self.path)) for f in src_files]
        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel([d for d in dst_files if d.exists()])
        changed = []
        for src, dst in zip(src_files, dst_files):
            rel = str(src.relative_to(self.path))
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                changed.append(rel)
        return changed

    async def sync_from_origin(self, sync_enabled: bool = True, sync_interval: float = 30.0) -> list[str]:
        import time
        if not self.path or not sync_enabled: return []
        now = time.time()
        if self._last_sync_time and (now - self._last_sync_time) < sync_interval: return []
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
            src_files = []
            for f in self._list_tracked_files(self.origin_root):
                rel = str(f.relative_to(self.origin_root))
                if rel in agent_changed: continue
                try:
                    if f.stat().st_mtime > self._last_sync_time:
                        src_files.append(f)
                except OSError:
                    continue
        self._last_sync_time = now
        if not src_files: return []
        dst_files = [(self.path / f.relative_to(self.origin_root)) for f in src_files]
        src_hashes = await self._compute_hashes_parallel(src_files)
        dst_hashes = await self._compute_hashes_parallel([d for d in dst_files if d.exists()])
        synced = []
        for src, dst in zip(src_files, dst_files):
            if not dst.exists() or src_hashes.get(str(src)) != dst_hashes.get(str(dst)):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                synced.append(str(src.relative_to(self.origin_root)))
        return synced


# ═══════════════════════════════════════════════════════════════════════
# SUB-AGENT SYSTEM PROMPTS (NEW)
# ═══════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """\
Du bist der PLANNER-Agent eines Multi-Agent-Coding-Systems.

AUFGABE: Analysiere die Codebasis und erstelle einen strukturierten Ausführungsplan.

REGELN:
1. Nutze read_file und grep um die Codebasis zu verstehen BEVOR du planst.
2. Erstelle den Plan via create_plan Tool mit klar definierten Subtasks.
3. Jeder Subtask MUSS spezifizieren: description, files (Liste), priority.
4. Dateien dürfen NICHT zwischen Subtasks überlappen! Jede Datei gehört genau einem Subtask.
5. Du darfst NUR .md Dateien schreiben (via write_md Tool).
6. Schreibe Status-Updates in _coordination.md.
7. Wenn der Task nur 1-2 Dateien betrifft, erstelle nur 1 Subtask.
8. Rufe am Ende ZWINGEND create_plan auf.

FORMAT für create_plan:
[
  {"description": "Was getan werden soll", "files": ["pfad/datei.py"], "priority": "high|normal|low"},
  ...
]
"""

VALIDATOR_SYSTEM = """\
Du bist der VALIDATOR-Agent eines Multi-Agent-Coding-Systems.

AUFGABE: Prüfe die Änderungen der Coder-Agents auf Korrektheit.

REGELN:
1. Lies die geänderten Dateien via read_file.
2. Prüfe auf: Syntaxfehler, Logikfehler, fehlende Imports, kaputte Referenzen.
3. Führe Tests aus wenn verfügbar (run_file oder bash).
4. Melde Ergebnisse via report_issues Tool.
5. Leere Liste = alles OK.
6. Lies _coordination.md für Kontext zum Plan.

FORMAT für report_issues:
[
  {"file": "pfad/datei.py", "line": 42, "severity": "error|warning", "message": "Beschreibung"},
  ...
]
"""


# ═══════════════════════════════════════════════════════════════════════
# INPUT MANAGER (unchanged)
# ═══════════════════════════════════════════════════════════════════════

class InputManager:
    def __init__(self):
        pass

    async def ask(self, text):
        return input(text)


# ═══════════════════════════════════════════════════════════════════════
# CODER AGENT v5 (UPDATED — Multi-Agent Orchestration)
# ═══════════════════════════════════════════════════════════════════════

class CoderAgent:
    SYSTEM_PROMPT = (
        "Du bist ein Elite Coding-Agent.\n"
        "REGELN:\n"
        "1. ARCHITEKTUR ZUERST: Nutze 'memory_recall', um Zusammenhänge zu verstehen, BEVOR du Code schreibst.\n"
        "2. LIES Dateien (read_file) BEVOR du editierst. Niemals blind raten!\n"
        "3. Änderungen NUR via XML Edit-Blöcke. BENUTZE NIEMALS 'bash' mit 'cat' oder 'echo' um Code zu schreiben!\n"
        "   VERBOTEN: 'bash', 'cat', 'echo', 'printf' zum Schreiben von Dateien!\n"
        "   Das System erkennt Änderungen NUR über die unten beschriebenen <edit>-Blöcke.\n"
        "4. BEENDEN: Wenn du alle Aufgaben erledigt hast, rufe ZWINGEND das Tool 'done' auf (oder schreibe [DONE]).\n"
        "5. Schreibe VOR jedem Edit einen <thought>...</thought> Block.\n"
        "6. ISOLIERTER WORKSPACE: Alle Pfade RELATIV zum Projektordner!\n"
        "7. CODE AUSFÜHREN: Nutze IMMER 'run_file' für Skripte/Tests. 'bash' NUR wenn nötig.\n"
        "8. <edit>-Blöcke sind KEIN JSON-Tool! Gib sie als plain Text in deiner Antwort aus.\n"
        "9. Lies _coordination.md für den Gesamtplan und Status der anderen Agents.\n"
        "10. Bearbeite NUR die dir zugewiesenen Dateien!\n\n"
        "FORMAT FÜR ÄNDERUNGEN:\n"
        "<edit path=\"pfad/datei.py\">\n"
        "<search>\n"
        "exakter alter code\n"
        "</search>\n"
        "<replace>\n"
        "neuer code\n"
        "</replace>\n"
        "</edit>\n\n"
        "FORMAT FÜR NEUE DATEIEN (search leer):\n"
        "<edit path=\"pfad/neue_datei.py\">\n"
        "<search>\n"
        "</search>\n"
        "<replace>\n"
        "kompletter neuer Inhalt\n"
        "</replace>\n"
        "</edit>\n\n"
        "ELLIPSIS-SYNTAX für große Blöcke:\n"
        "<edit path=\"pfad/datei.py\">\n"
        "<search>\n"
        "erste Zeilen\n"
        "...\n"
        "letzte Zeilen\n"
        "</search>\n"
        "<replace>\n"
        "kompletter neuer Code\n"
        "</replace>\n"
        "</edit>\n"
    )

    def __init__(self, agent, project_root: str, config: dict = None):
        self.agent = agent
        self.root = project_root
        self.config = config or {}
        self.p_input = InputManager()
        self.lsp = LSPManager(auto_install=True)
        self.use_root_env = self.config.get("use_root_env", True)
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
            os.getenv("CODER_VERBOSE", "false").lower() == "true" or
            os.getenv("AGENT_VERBOSE", "false").lower() == "true" or self.config.get("debug", False)
        )
        self.ask_enabled = self.config.get(
            "ask_enabled", os.getenv("CODER_ASK_ENABLED", "false").lower() == "true" and self.config.get("ask_callback", False)
        )
        self.max_iters = int(os.getenv("DEFAULT_MAX_ITERATIONS", 30)) + 60
        self.print = print

        self.memory = ExecutionMemory(project_root)
        self.tracker = TokenTracker(self.model, agent, limit=self.config.get("compression_limit", 0.7))
        self.worktree = GitWorktree(project_root)

        self.row_chunk_fun = None

        self.state = {
            "plan": [], "done": [], "current_file": "None", "last_error": None
        }

        # ── NEW: Sub-agent tracking ──
        self._sub_agents_ready = False
        self._agent_uid = uuid.uuid4().hex[:6]
        self._planner_name = f"planner_{self._agent_uid}"
        self._coder_names: list[str] = []  # populated in _ensure_agents
        self._validator_name = f"validator_{self._agent_uid}"
        self._current_plan: list[dict] = []
        self._validation_issues: list[dict] = []

    # ─────────────────────────────────────────────────────────────────
    # EXISTING METHODS (unchanged): stream, bash, grep, run_file, etc.
    # ─────────────────────────────────────────────────────────────────

    def _default_stream_handler(self, chunk: str):
        if self.print == print:
            self.print(chunk, end="", flush=True)
        else:
            self.print(chunk)

    def _log(self, section: str, content: str, color: str = "white"):
        if not self.debug_mode: return
        if self.log_handler:
            try: self.log_handler(section, content); return
            except Exception: pass

        C = {
            "cyan": "\033[96m", "yellow": "\033[93m", "green": "\033[92m",
            "red": "\033[91m", "grey": "\033[90m", "bold": "\033[1m", "reset": "\033[0m"
        }
        c_code = C.get(color, C["reset"])
        header = f"{C['bold']}[{section}]{C['reset']}"
        try:
            self.print(f"{c_code}{header} {content}{C['reset']}", flush=True)
        except UnicodeEncodeError:
            clean_content = content.encode('ascii', 'replace').decode('ascii')
            self.print(f"{c_code}{header} {clean_content}{C['reset']}", flush=True)


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
        return result_output or "Command executed (no output)."

    async def _smart_bash(self, cmd: str, messages: list) -> str:
        raw_output = await self._run_bash(cmd)
        if not raw_output.strip() or "ERROR: Timeout" in raw_output:
            return raw_output
        lines = text_to_block(raw_output)
        total = len(lines)
        if total > 4000:
            head = "\n".join(lines[:1500])
            tail = "\n".join(lines[-1500:])
            return (f"GUARD: Output too large ({total} lines). Showing first/last 1500.\n\n"
                    f"{head}\n\n... [{total - 3000} lines omitted] ...\n\n{tail}")
        if total > 6000 and self.agent:
            try:
                head = "\n".join(lines[:4000])
                tail = "\n".join(lines[-4000:])
                input_blob = f"{head}\n\n... [{total - 8000} lines omitted] ...\n\n{tail}"
                filtered = await self.agent.a_run_llm_completion(
                    messages=[
                        {"role": "system", "content": f"Filter shell output of '{cmd}'. Keep errors, success, key info."},
                        {"role": "user", "content": input_blob}
                    ], stream=False)
                if filtered:
                    return f"### BASH OUTPUT (Ranked from {total} lines):\n{filtered}"
            except Exception:
                pass
        return raw_output

    async def _run_file(self, file_path: str, args_str: str, messages: list) -> str:
        target = (self.worktree.path / file_path).resolve()
        if not target.exists():
            return f"FEHLER: Datei {file_path} existiert nicht."
        quoted_path = shlex.quote(file_path) if platform.system() != "Windows" else f'"{file_path}"'
        ext = target.suffix.lower()
        if ext == ".py":
            root = self.worktree.origin_root if self.use_root_env else self.worktree.path
            is_win = platform.system() == "Windows"
            python_subpath = "Scripts/python.exe" if is_win else "bin/python"
            if (root / "uv.lock").exists() and shutil.which("uv"):
                python_bin = "uv run python"
            elif (root / ".venv" / python_subpath).exists():
                python_bin = f'"{(root / ".venv" / python_subpath).absolute()}"'
            elif (root / "venv" / python_subpath).exists():
                python_bin = f'"{(root / "venv" / python_subpath).absolute()}"'
            else:
                python_bin = f'"{sys.executable}"'
            cmd = f"{python_bin} {quoted_path} {args_str}"
        elif ext in [".sh", ".bash"]:
            cmd = f"bash {quoted_path} {args_str}"
        else:
            prefix = "./" if platform.system() != "Windows" and not file_path.startswith("/") else ""
            cmd = f"{prefix}{quoted_path} {args_str}"
        return await self._smart_bash(cmd.strip(), messages)

    async def _smart_grep(self, pattern: str, messages: list) -> str:
        quoted = shlex.quote(pattern) if platform.system() != "Windows" else pattern
        if shutil.which("rg"):
            cmd = f"rg -n {quoted} ."
        elif self.worktree._is_git:
            cmd = f"git grep -n {quoted}"
        else:
            cmd = f"grep -rn {quoted} ."
        raw_output = await self._run_bash(cmd)
        lines = text_to_block(raw_output)
        total = len(lines)
        if total > 2000:
            head = "\n".join(lines[:500])
            tail = "\n".join(lines[-500:])
            return (f"GUARD: Too many matches ({total}). Showing first/last 500.\n\n"
                    f"{head}\n\n... [{total - 1000} lines omitted] ...\n\n{tail}")
        return raw_output

    async def _ask_user(self, question: str) -> str:
        if not getattr(self, "ask_enabled", False):
            return "ERROR: 'ask' tool is disabled. Proceed autonomously."
        custom_ask = self.config.get("ask_callback")
        if custom_ask:
            try:
                if asyncio.iscoroutinefunction(custom_ask):
                    return await custom_ask(question)
                else:
                    return await asyncio.to_thread(custom_ask, question)
            except Exception as e:
                self._log("ASK-ERROR", f"Callback failed: {e}", "red")
        self.print(f"\n\033[96mAgent fragt:\033[0m {question}")
        return input("\033[93m> Antwort:\033[0m ")

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

    # ─────────────────────────────────────────────────────────────────
    # NEW: Edit Block Parsing & Application
    # ─────────────────────────────────────────────────────────────────

    def _parse_edits(self, response: str) -> list[dict]:
        """Parse <edit> blocks from agent response text."""
        edits = []
        # Regex: tolerant of path= or file=, captures content between tags
        pattern = re.compile(
            r'<edit\s+(?:path|file)=["\']([^"\']+)["\']\s*>\s*\n'
            r'<search>\s*\n(.*?)\n</search>\s*\n'
            r'<replace>\s*\n(.*?)\n</replace>\s*\n'
            r'</edit>',
            re.DOTALL
        )
        for match in pattern.finditer(response):
            edits.append({
                "path": match.group(1).strip(),
                "search": match.group(2),
                "replace": match.group(3),
            })
        if not edits:
            # Fallback: looser parse for incomplete blocks
            loose = re.compile(
                r'<edit\s+(?:path|file)=["\']([^"\']+)["\']\s*>.*?'
                r'<search>(.*?)</search>.*?'
                r'<replace>(.*?)</replace>',
                re.DOTALL
            )
            for match in loose.finditer(response):
                edits.append({
                    "path": match.group(1).strip(),
                    "search": match.group(2).strip("\n"),
                    "replace": match.group(3).strip("\n"),
                })
        return edits

    async def _apply_edits(self, edits: list[dict]) -> list[str]:
        """Apply parsed edit blocks to worktree. Returns list of changed file paths."""
        changed = []
        for edit in edits:
            path = edit["path"]
            search = edit["search"]
            replace = edit["replace"]
            target = (self.worktree.path / path).resolve()

            # New file: empty search
            if not search.strip():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(replace, encoding="utf-8")
                changed.append(path)
                self._log("EDIT", f"Created: {path}", "green")
                continue

            if not target.exists():
                self._log("EDIT-ERR", f"File not found: {path}", "red")
                continue

            content = target.read_text(encoding="utf-8", errors="replace")

            # Ellipsis syntax: first_lines ... last_lines
            if "\n...\n" in search:
                parts = search.split("\n...\n", 1)
                start_marker = parts[0].rstrip("\n")
                end_marker = parts[1].lstrip("\n")
                start_idx = content.find(start_marker)
                if start_idx >= 0:
                    end_idx = content.find(end_marker, start_idx + len(start_marker))
                    if end_idx >= 0:
                        end_idx += len(end_marker)
                        content = content[:start_idx] + replace.strip("\n") + content[end_idx:]
                        target.write_text(content, encoding="utf-8")
                        changed.append(path)
                        self._log("EDIT", f"Ellipsis edit: {path}", "green")
                        continue
                self._log("EDIT-ERR", f"Ellipsis markers not found in {path}", "red")
                continue

            # Exact match replacement
            search_clean = search.strip("\n")
            if search_clean in content:
                content = content.replace(search_clean, replace.strip("\n"), 1)
                target.write_text(content, encoding="utf-8")
                changed.append(path)
                self._log("EDIT", f"Modified: {path}", "green")
            else:
                self._log("EDIT-ERR", f"Search block not found in {path}", "red")

        return changed

    # ─────────────────────────────────────────────────────────────────
    # NEW: Sub-Agent Lifecycle
    # ─────────────────────────────────────────────────────────────────

    async def _ensure_agents(self):
        """Lazy-init planner, coder(s), validator via FlowAgentBuilder."""
        if self._sub_agents_ready:
            return

        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")
        host = self  # closure reference

        # ── PLANNER ──
        pb = isaa.get_agent_builder(
            name=self._planner_name, add_base_tools=False, with_dangerous_shell=False
        )
        pb.with_system_message(PLANNER_SYSTEM)
        pb.with_models(self.model)
        self._register_planner_tools(pb, host)
        await isaa.register_agent(pb)

        # ── CODER (1 by default, more spawned if plan has parallel subtasks) ──
        coder_name = f"coder_{self._agent_uid}_0"
        self._coder_names = [coder_name]
        cb = isaa.get_agent_builder(
            name=coder_name, add_base_tools=False, with_dangerous_shell=False
        )
        cb.with_system_message(self.SYSTEM_PROMPT)
        cb.with_models(self.model)
        self._register_coder_tools(cb, host, coder_name)
        await isaa.register_agent(cb)

        # ── VALIDATOR ──
        vb = isaa.get_agent_builder(
            name=self._validator_name, add_base_tools=False, with_dangerous_shell=False
        )
        vb.with_system_message(VALIDATOR_SYSTEM)
        vb.with_models(self.model)
        self._register_validator_tools(vb, host)
        await isaa.register_agent(vb)

        # ── VFS Mount for all agents ──
        for aname in [self._planner_name] + self._coder_names + [self._validator_name]:
            await self._mount_worktree_vfs(aname)

        self._sub_agents_ready = True
        self._log("AGENTS", f"Sub-agents ready: {self._planner_name}, {self._coder_names}, {self._validator_name}", "green")

    async def _spawn_extra_coder(self, index: int) -> str:
        """Spawn an additional coder agent for parallel work."""
        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")

        coder_name = f"coder_{self._agent_uid}_{index}"
        if coder_name in self._coder_names:
            return coder_name

        cb = isaa.get_agent_builder(
            name=coder_name, add_base_tools=False, with_dangerous_shell=False
        )
        cb.with_system_message(self.SYSTEM_PROMPT)
        cb.with_models(self.model)
        self._register_coder_tools(cb, self, coder_name)
        await isaa.register_agent(cb)
        await self._mount_worktree_vfs(coder_name)

        self._coder_names.append(coder_name)
        self._log("AGENTS", f"Spawned extra coder: {coder_name}", "cyan")
        return coder_name

    async def _mount_worktree_vfs(self, agent_name: str):
        """Mount worktree into agent's session VFS at /project."""
        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")
        agent = await isaa.get_agent(agent_name)
        session = await agent.session_manager.get_or_create("default")
        try:
            session.vfs.mount(
                local_path=str(self.worktree.path),
                vfs_path="/project",
                readonly=False,
                auto_sync=True
            )
            self._log("VFS", f"Mounted worktree for {agent_name}", "cyan")
        except Exception as e:
            self._log("VFS-WARN", f"Mount failed for {agent_name}: {e}", "yellow")

    async def cleanup_agents(self):
        """Remove all sub-agents and unmount VFS."""
        if not self._sub_agents_ready:
            return

        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")

        all_names = [self._planner_name] + self._coder_names + [self._validator_name]
        for name in all_names:
            try:
                agent = await isaa.get_agent(name)
                session = await agent.session_manager.get_or_create("default")
                try:
                    session.vfs.unmount("/project", save_changes=False)
                except Exception:
                    pass
                # Agent entfernen — je nach isaa API
                if hasattr(isaa, "unregister_agent"):
                    await isaa.unregister_agent(name)
                elif hasattr(isaa, "remove_agent"):
                    await isaa.remove_agent(name)
                self._log("CLEANUP", f"Removed agent: {name}", "grey")
            except Exception as e:
                logger.debug(f"Cleanup {name}: {e}")

        self._coder_names = []
        self._sub_agents_ready = False
        self._log("CLEANUP", "All sub-agents removed", "green")

    # ─────────────────────────────────────────────────────────────────
    # NEW: Tool Registration per Agent Type
    # ─────────────────────────────────────────────────────────────────

    def _register_planner_tools(self, builder, host: "CoderAgent"):
        """Register planner-specific tools: read, grep, list, create_plan, write_md."""

        async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
            """Read a file from the project. Use start_line/end_line for ranges."""
            return await smart_read_file(
                path, start_line, end_line, host.worktree.path,
                host.agent, [], host.model, ""
            )

        async def grep(pattern: str) -> str:
            """Search for a pattern across the codebase."""
            return await host._smart_grep(pattern, [])

        async def list_files(path: str = ".") -> str:
            """List files and directories at the given path."""
            target = (host.worktree.path / path).resolve()
            if not target.exists():
                return f"Error: {path} not found"
            result = []
            for item in sorted(target.iterdir()):
                if item.name in GitWorktree.SCAN_EXCLUDES or item.name.startswith("."):
                    continue
                prefix = "DIR  " if item.is_dir() else "FILE "
                result.append(f"{prefix}{item.relative_to(host.worktree.path)}")
            return "\n".join(result[:200]) if result else "(empty)"

        async def create_plan(subtasks_json: str) -> str:
            """Create execution plan. Argument: JSON array of subtasks.
            Each subtask: {"description": "...", "files": ["path/a.py"], "priority": "high|normal|low"}
            Files MUST NOT overlap between subtasks!"""
            try:
                subtasks = json.loads(subtasks_json) if isinstance(subtasks_json, str) else subtasks_json
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON — {e}"

            if not isinstance(subtasks, list) or not subtasks:
                return "Error: subtasks must be a non-empty list."

            # Validate no file overlap
            seen_files = set()
            for st in subtasks:
                for f in st.get("files", []):
                    if f in seen_files:
                        return f"Error: File '{f}' assigned to multiple subtasks! Fix overlap."
                    seen_files.add(f)

            host._current_plan = subtasks

            # Write coordination file
            plan_md = "# Execution Plan\n\n"
            for i, st in enumerate(subtasks):
                plan_md += f"## Subtask {i+1}: {st.get('description', '?')}\n"
                plan_md += f"- Files: {', '.join(st.get('files', []))}\n"
                plan_md += f"- Priority: {st.get('priority', 'normal')}\n"
                plan_md += f"- Status: pending\n\n"
            plan_md += "---\n## Status Log\n"

            coord_path = host.worktree.path / "_coordination.md"
            coord_path.write_text(plan_md, encoding="utf-8")

            return f"Plan created: {len(subtasks)} subtask(s). Written to _coordination.md."

        async def write_md(path: str, content: str) -> str:
            """Write a .md file in the worktree. Planner can ONLY write .md files."""
            if not path.endswith(".md"):
                return "Error: Planner can only write .md files."
            target = (host.worktree.path / path).resolve()
            # Safety: must stay within worktree
            try:
                target.relative_to(host.worktree.path)
            except ValueError:
                return "Error: Path escapes worktree."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Written: {path}"

        builder.add_tool(read_file, "read_file", "Read a project file", category=["filesystem"])
        builder.add_tool(grep, "grep", "Search pattern in codebase", category=["search"])
        builder.add_tool(list_files, "list_files", "List directory contents", category=["filesystem"])
        builder.add_tool(create_plan, "create_plan", "Create structured execution plan (JSON)", category=["planning"])
        builder.add_tool(write_md, "write_md", "Write a .md file (planner only)", category=["filesystem"])

    def _register_coder_tools(self, builder, host: "CoderAgent", coder_name: str):
        """Register coder-specific tools: read, grep, bash, run_file, status, done."""

        async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
            """Read a file from the project."""
            return await smart_read_file(
                path, start_line, end_line, host.worktree.path,
                host.agent, [], host.model, ""
            )

        async def grep(pattern: str) -> str:
            """Search for a pattern in the codebase."""
            return await host._smart_grep(pattern, [])

        async def bash(command: str) -> str:
            """Run a shell command in the worktree. Do NOT use for writing files!"""
            return await host._smart_bash(command, [])

        async def run_file(file_path: str, args: str = "") -> str:
            """Run a script/test file in the worktree."""
            return await host._run_file(file_path, args, [])

        async def update_status(status: str) -> str:
            """Append a status line to _coordination.md."""
            coord_path = host.worktree.path / "_coordination.md"
            existing = coord_path.read_text(encoding="utf-8") if coord_path.exists() else ""
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            existing += f"\n[{ts}] {coder_name}: {status}"
            coord_path.write_text(existing, encoding="utf-8")
            return "Status updated."

        async def done() -> str:
            """Signal that all assigned tasks are complete."""
            return "[DONE]"

        builder.add_tool(read_file, "read_file", "Read a project file", category=["filesystem"])
        builder.add_tool(grep, "grep", "Search pattern in codebase", category=["search"])
        builder.add_tool(bash, "bash", "Run shell command (NOT for writing files!)", category=["shell"])
        builder.add_tool(run_file, "run_file", "Run a script or test file", category=["execution"])
        builder.add_tool(update_status, "update_status", "Update coordination status", category=["coordination"])
        builder.add_tool(done, "done", "Signal task completion", category=["control"])

    def _register_validator_tools(self, builder, host: "CoderAgent"):
        """Register validator-specific tools: read, grep, bash, run_file, report_issues."""

        async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
            """Read a file from the project."""
            return await smart_read_file(
                path, start_line, end_line, host.worktree.path,
                host.agent, [], host.model, ""
            )

        async def grep(pattern: str) -> str:
            """Search for a pattern in the codebase."""
            return await host._smart_grep(pattern, [])

        async def bash(command: str) -> str:
            """Run a shell command for testing/linting."""
            return await host._smart_bash(command, [])

        async def run_file(file_path: str, args: str = "") -> str:
            """Run a test or script file."""
            return await host._run_file(file_path, args, [])

        async def report_issues(issues_json: str) -> str:
            """Report validation results. Argument: JSON array.
            Each issue: {"file": "path.py", "line": 42, "severity": "error|warning", "message": "..."}
            Pass empty array [] if everything is OK."""
            try:
                issues = json.loads(issues_json) if isinstance(issues_json, str) else issues_json
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON — {e}"
            host._validation_issues = issues if isinstance(issues, list) else []
            if not host._validation_issues:
                return "Validation passed. No issues found."
            report = "\n".join(
                f"- [{i.get('severity', '?')}] {i.get('file', '?')}:{i.get('line', '?')}: {i.get('message', '?')}"
                for i in host._validation_issues
            )
            return f"Found {len(host._validation_issues)} issue(s):\n{report}"

        builder.add_tool(read_file, "read_file", "Read a project file", category=["filesystem"])
        builder.add_tool(grep, "grep", "Search pattern in codebase", category=["search"])
        builder.add_tool(bash, "bash", "Run shell command for testing", category=["shell"])
        builder.add_tool(run_file, "run_file", "Run a test file", category=["execution"])
        builder.add_tool(report_issues, "report_issues", "Report validation results (JSON)", category=["validation"])

    # ─────────────────────────────────────────────────────────────────
    # NEW: Stream Collector
    # ─────────────────────────────────────────────────────────────────

    async def _collect_stream(self, agent_name: str, query: str, prefix: str = "") -> str:
        """Run agent via a_stream(), collect full response, forward chunks to stream_callback."""
        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")
        agent = await isaa.get_agent(agent_name)

        full_response = []
        async for chunk in agent.a_stream(
            query=query,
            session_id="default",
            max_iterations=self.max_iters,
        ):
            if self.row_chunk_fun:
                self.row_chunk_fun(chunk)
            ctype = chunk.get("type", "")

            if ctype == "content":
                text = chunk.get("chunk", "")
                full_response.append(text)
                if self.stream_enabled and self.stream_callback:
                    tagged = f"[{prefix}] {text}" if prefix else text
                    if asyncio.iscoroutinefunction(self.stream_callback):
                        await self.stream_callback(tagged)
                    else:
                        self.stream_callback(tagged)

            elif ctype == "tool_start":
                self._log(f"{prefix}-TOOL", chunk.get("name", "?"), "cyan")
                if self.log_handler:
                    self.log_handler("TOOL CALL", f"{chunk.get('name', '?')}({chunk.get('args', '')})")

            elif ctype == "tool_result":
                result_text = str(chunk.get("result", ""))[:300]
                self._log(f"{prefix}-RESULT", result_text, "grey")
                if self.log_handler:
                    self.log_handler("TOOL RESULT", result_text)

            elif ctype == "error":
                self._log(f"{prefix}-ERROR", chunk.get("error", "?"), "red")

            elif ctype == "final_answer":
                answer = chunk.get("answer", "")
                if answer:
                    full_response.append(answer)

        return "".join(full_response)

    # ─────────────────────────────────────────────────────────────────
    # NEW: Orchestration Phases
    # ─────────────────────────────────────────────────────────────────

    async def _run_planner(self, task: str) -> list[dict]:
        """Phase 1: Planner analyzes codebase and creates subtask plan."""
        self._current_plan = []

        planner_query = (
            f"Analysiere das Projekt und erstelle einen Plan für folgende Aufgabe:\n\n"
            f"{task}\n\n"
            f"Schritte:\n"
            f"1. Nutze list_files('.') und read_file um die Struktur zu verstehen.\n"
            f"2. Nutze grep um relevante Stellen zu finden.\n"
            f"3. Erstelle den Plan via create_plan Tool.\n"
            f"4. Jeder Subtask braucht: description, files (KEINE Überlappung!), priority.\n"
            f"5. Wenn nur 1-2 Dateien betroffen: 1 Subtask reicht.\n"
        )

        await self._collect_stream(self._planner_name, planner_query, prefix="PLANNER")

        if not self._current_plan:
            self._log("PLANNER", "No plan created, falling back to single subtask", "yellow")
            self._current_plan = [{"description": task, "files": [], "priority": "normal"}]

        return self._current_plan

    async def _run_coders(self, subtasks: list[dict]) -> list[str]:
        """Phase 2: Run coder(s) on subtasks. Parallel if non-overlapping."""
        all_changed: list[str] = []

        # Check if we can parallelize (>1 subtask with non-overlapping files)
        can_parallel = len(subtasks) > 1 and all(st.get("files") for st in subtasks)

        if can_parallel:
            # Ensure enough coder agents
            for i in range(1, len(subtasks)):
                if i >= len(self._coder_names):
                    await self._spawn_extra_coder(i)

            # Run in parallel
            async def _run_single(coder_name: str, subtask: dict) -> list[str]:
                query = self._build_coder_query(subtask)
                response = await self._collect_stream(coder_name, query, prefix=coder_name.upper())
                edits = self._parse_edits(response)
                return await self._apply_edits(edits)

            tasks = []
            for i, st in enumerate(subtasks):
                coder_name = self._coder_names[i % len(self._coder_names)]
                tasks.append(_run_single(coder_name, st))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self._log("CODER-ERR", str(result), "red")
                else:
                    all_changed.extend(result)
        else:
            # Sequential on coder_0
            coder_name = self._coder_names[0]
            for st in subtasks:
                query = self._build_coder_query(st)
                response = await self._collect_stream(coder_name, query, prefix="CODER")
                edits = self._parse_edits(response)
                applied = await self._apply_edits(edits)
                all_changed.extend(applied)

        return list(set(all_changed))

    def _build_coder_query(self, subtask: dict) -> str:
        """Build coder prompt from a subtask."""
        files = subtask.get("files", [])
        desc = subtask.get("description", "")
        files_str = ', '.join(files) if files else "(alle relevanten Dateien)"

        return (
            f"Aufgabe: {desc}\n\n"
            f"Zugewiesene Dateien: {files_str}\n\n"
            f"WICHTIG:\n"
            f"- Lies _coordination.md für den Gesamtplan.\n"
            f"- Bearbeite NUR die dir zugewiesenen Dateien.\n"
            f"- Lies jede Datei mit read_file BEVOR du sie editierst.\n"
            f"- Schreibe Änderungen als <edit>-Blöcke in deiner Antwort.\n"
            f"- Rufe update_status auf wenn du einen Teilschritt abschließt.\n"
            f"- Rufe done() auf wenn du fertig bist.\n"
        )

    async def _run_validator(self, changed_files: list[str]) -> list[dict]:
        """Phase 3: Validator checks changed files for issues."""
        self._validation_issues = []

        if not changed_files:
            return []

        query = (
            f"Validiere die folgenden geänderten Dateien:\n"
            f"{', '.join(changed_files)}\n\n"
            f"Schritte:\n"
            f"1. Lies jede geänderte Datei mit read_file.\n"
            f"2. Prüfe auf Syntaxfehler, fehlende Imports, Logikfehler.\n"
            f"3. Führe vorhandene Tests aus (z.B. run_file tests/...).\n"
            f"4. Lies _coordination.md für Kontext.\n"
            f"5. Melde Ergebnisse via report_issues Tool.\n"
            f"   Leere Liste [] = alles OK.\n"
        )

        await self._collect_stream(self._validator_name, query, prefix="VALIDATOR")
        return self._validation_issues

    # ─────────────────────────────────────────────────────────────────
    # UPDATED: Main Execute (orchestrates planner → coder → validator)
    # ─────────────────────────────────────────────────────────────────

    async def execute(self, task: str) -> CoderResult:
        """
        Execute a coding task using multi-agent orchestration.

        Flow:
            1. Worktree setup
            2. Ensure sub-agents (planner, coder, validator)
            3. PLANNER phase → structured subtasks
            4. CODER phase → parallel edits on non-overlapping files
            5. VALIDATOR phase → issue detection
            6. Fix loop (max 2 rounds)
            7. Post-execute: commit, memory, report
        """
        # ── 1. Worktree Setup ──
        with Spinner(f"Setup Worktree for agent: {self.agent.amd.name}", symbols="b"):
            self.worktree.setup()

        synced = await self.worktree.sync_from_origin(
            sync_enabled=self.sync_enabled, sync_interval=0
        )

        # ── 2. Ensure Sub-Agents ──
        with Spinner("Initializing sub-agents...", symbols="c"):
            await self._ensure_agents()

        # ── 3. Initial Commit ──
        try:
            self.worktree.commit(f"pre-task: {task[:50]}")
        except Exception:
            pass

        # ── Build enriched task ──
        enriched_task = task
        if synced:
            enriched_task += f"\n\n[INFO: {len(synced)} file(s) synced: {', '.join(synced[:5])}]"
        if prev := self.memory.get_context():
            enriched_task += f"\n\n## Previous context:\n{prev}"
        if getattr(self, "ask_enabled", False):
            enriched_task += "\n\n[NOTE: 'ask' tool available for user questions]"

        try:
            # ── 4. PLANNER PHASE ──
            self._log("PHASE", "=== PLANNER ===", "bold")
            subtasks = await self._run_planner(enriched_task)
            self._log("PHASE", f"Plan: {len(subtasks)} subtask(s)", "green")

            # ── 5. CODER PHASE ──
            self._log("PHASE", "=== CODER ===", "bold")
            edit_changed = await self._run_coders(subtasks)
            self._log("PHASE", f"Coder changed: {edit_changed}", "green")

            # ── 6. VALIDATOR PHASE + FIX LOOP ──
            all_changed = edit_changed or await self.worktree.changed_files()

            if all_changed:
                self._log("PHASE", "=== VALIDATOR ===", "bold")
                issues = await self._run_validator(all_changed)

                for fix_round in range(2):
                    if not issues:
                        self._log("VALIDATOR", "All clear!", "green")
                        break

                    self._log("PHASE", f"=== FIX ROUND {fix_round + 1} ===", "yellow")
                    fix_subtasks = []
                    for issue in issues:
                        f = issue.get("file", "")
                        if f:
                            fix_subtasks.append({
                                "description": f"FIX [{issue.get('severity', '?')}]: {issue.get('message', '?')}",
                                "files": [f],
                                "priority": "high",
                            })

                    if not fix_subtasks:
                        break

                    # Deduplicate by file
                    seen = set()
                    deduped = []
                    for st in fix_subtasks:
                        key = tuple(st["files"])
                        if key not in seen:
                            seen.add(key)
                            deduped.append(st)

                    await self._run_coders(deduped)
                    all_changed = await self.worktree.changed_files()
                    issues = await self._run_validator(all_changed)

            # ── 7. POST-EXECUTE ──
            self.worktree.commit(f"task: {task[:50]}")
            final_changed = await self.worktree.changed_files()

            report = ExecutionReport(
                timestamp=datetime.datetime.now().isoformat(),
                task=task[:200],
                changed_files=final_changed or [],
                success=True,
                summary=f"Completed: {len(subtasks)} subtask(s), {len(final_changed or [])} file(s) changed",
            )
            self.memory.add(report)

            return CoderResult(
                success=True,
                message="Done",
                changed_files=final_changed or [],
                history=[],
                memory_saved=True,
                tokens_used=self.tracker.total_tokens,
                compressions_done=self.tracker.compressions_done,
            )

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Execute failed: {error_details}")

            report = ExecutionReport(
                timestamp=datetime.datetime.now().isoformat(),
                task=task[:200],
                changed_files=[],
                success=False,
                errors_encountered=[str(e)],
                summary=f"Failed: {str(e)[:100]}",
            )
            self.memory.add(report)

            return CoderResult(
                success=False,
                message=str(e),
                changed_files=[],
                history=[],
                memory_saved=True,
            )


# ═══════════════════════════════════════════════════════════════════════
# CLI (unchanged)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CoderAgent v5 – Multi-Agent")
    p.add_argument("task", help="Coding task")
    p.add_argument("-p", "--project", default=".", help="Project root")
    p.add_argument("-m", "--model", default="gpt-4o", help="LLM model")
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()

    async def _main():
        from toolboxv2 import get_app
        agent = await get_app(args.project).get_mod("isaa").get_agent("self")
        coder = CoderAgent(agent, os.path.abspath(args.project),
                           {"model": args.model, "stream": args.stream, "run_tests": args.run_tests,
                            "bash_timeout": args.timeout})
        result = await coder.execute(args.task)
        status = '✓' if result.success else '✗'
        print(f"{status} {result.message} ({result.tokens_used} tokens, {result.compressions_done} compressions)")
        for f in result.changed_files: print(f"  -> {f}")
        if args.apply and result.success:
            n = await coder.worktree.apply_back()
            print(f"Applied to origin" + (f" ({n} files)" if n >= 0 else " (git merge)"))
        await coder.cleanup_agents()
        coder.worktree.cleanup()

    asyncio.run(_main())
