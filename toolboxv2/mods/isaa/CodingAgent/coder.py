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
from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs

# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES (unchanged)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoderResult:
    success: bool; message: str; changed_files: List[str]; history: List[dict]
    memory_saved: bool = False; tokens_used: int = 0; compressions_done: int = 0

@dataclass
class ExecutionReport:
    timestamp: str
    task: str
    changed_files: List[str]
    success: bool
    summary: str = ""
    patterns_learned: list | None = None
    final_response: dict = field(default_factory=dict)

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
# PROJECT CONTEXT DETECTION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProjectContext:
    """Detected project metadata. Zero-guessing — all fields based on actual filesystem."""
    mode: str                        # "new" | "existing_project" | "existing_file" | "file_pair" | "subfolder"
    root: Path                       # The actual working root for the worktree
    kind: str                        # "python-uv" | "python-pip" | "node-npm" | "node-bun" | "web-static" | "rust" | "go" | "mixed" | "empty" | "unknown"
    has_git: bool
    has_tests: bool
    test_paths: List[str]            # Detected test dirs/files relative to root
    entry_files: List[str]           # Main entry points relative to root
    package_files: List[str]         # pyproject.toml, package.json, Cargo.toml, etc.
    run_commands: List[str]          # Suggested run commands (e.g. "uv run pytest", "npm test")
    notes: List[str]                 # Human-readable findings for confirmation UI

    def to_summary(self) -> str:
        lines = [
            f"Mode:      {self.mode}",
            f"Kind:      {self.kind}",
            f"Root:      {self.root}",
            f"Git:       {'yes' if self.has_git else 'no'}",
            f"Tests:     {'yes' if self.has_tests else 'no'}"
            + (f" ({', '.join(self.test_paths[:3])})" if self.test_paths else ""),
        ]
        if self.entry_files:
            lines.append(f"Entries:   {', '.join(self.entry_files[:5])}")
        if self.package_files:
            lines.append(f"Packages:  {', '.join(self.package_files)}")
        if self.run_commands:
            lines.append(f"Run:       {' | '.join(self.run_commands[:3])}")
        return "\n".join(lines)


class ProjectDetector:
    """Detects project layout and suggests a working root for the worktree."""

    PYTHON_PACKAGE_FILES = {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "uv.lock", "poetry.lock", "Pipfile"}
    NODE_PACKAGE_FILES = {"package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb", "bun.lock"}
    RUST_FILES = {"Cargo.toml"}
    GO_FILES = {"go.mod"}
    WEB_FILES = {"index.html"}
    TEST_DIR_NAMES = {"tests", "test", "__tests__", "spec"}
    TEST_FILE_PATTERNS = ("test_", "_test.", ".test.", ".spec.", "tests.py")

    @classmethod
    def detect(cls, path: str) -> ProjectContext:
        p = Path(path).expanduser().resolve()

        # ── File vs Directory ──
        if p.is_file():
            return cls._detect_file(p)
        if not p.exists():
            # New project scenario — empty dir that doesn't exist yet
            return ProjectContext(
                mode="new", root=p, kind="empty", has_git=False, has_tests=False,
                test_paths=[], entry_files=[], package_files=[], run_commands=[],
                notes=[f"Path does not exist — will be created: {p}"],
            )
        if not p.is_dir():
            raise ValueError(f"Not a file or directory: {p}")

        return cls._detect_dir(p)

    @classmethod
    def _detect_file(cls, f: Path) -> ProjectContext:
        """Single file handed to coder. Worktree root = parent dir, but scoped to this file."""
        parent = f.parent
        ext = f.suffix.lower()
        kind_map = {".py": "python-pip", ".js": "node-npm", ".ts": "node-npm",
                    ".jsx": "node-npm", ".tsx": "node-npm", ".html": "web-static",
                    ".rs": "rust", ".go": "go"}
        kind = kind_map.get(ext, "unknown")

        # Look for a sibling test file / pair
        pair = cls._find_test_pair(f)
        entries = [f.name]
        if pair:
            entries.append(pair.name)

        return ProjectContext(
            mode="file_pair" if pair else "existing_file",
            root=parent,
            kind=kind,
            has_git=(parent / ".git").exists(),
            has_tests=bool(pair),
            test_paths=[pair.name] if pair else [],
            entry_files=entries,
            package_files=[],
            run_commands=cls._suggest_run_for_file(f, kind),
            notes=[f"Single file: {f.name}"] + ([f"Test pair: {pair.name}"] if pair else []),
        )

    @classmethod
    def _find_test_pair(cls, f: Path) -> Optional[Path]:
        """Given a source file, find its test counterpart in parent dir (shallow scan)."""
        stem = f.stem
        parent = f.parent
        candidates = [
            parent / f"test_{stem}{f.suffix}",
            parent / f"{stem}_test{f.suffix}",
            parent / f"{stem}.test{f.suffix}",
            parent / f"{stem}.spec{f.suffix}",
            parent / "tests" / f"test_{stem}{f.suffix}",
            parent / "test" / f"test_{stem}{f.suffix}",
        ]
        for c in candidates:
            if c.exists() and c != f:
                return c
        # Reverse: if user passed the test file, find source
        for prefix in ("test_", ):
            if stem.startswith(prefix):
                src = parent / f"{stem[len(prefix):]}{f.suffix}"
                if src.exists():
                    return src
        return None

    @classmethod
    def _detect_dir(cls, d: Path) -> ProjectContext:
        """Directory detection — shallow scan of root for markers."""
        try:
            entries = {e.name for e in d.iterdir()}
        except PermissionError:
            entries = set()

        if not entries or entries == {".git"}:
            return ProjectContext(
                mode="new", root=d, kind="empty", has_git=(".git" in entries),
                has_tests=False, test_paths=[], entry_files=[], package_files=[],
                run_commands=[], notes=["Directory is empty — new project mode"],
            )

        kind, pkg_files, run_cmds = cls._classify_kind(d, entries)
        test_paths = cls._find_tests(d, entries)
        entry_files = cls._find_entries(d, entries, kind)
        has_git = (d / ".git").exists() or cls._has_git_ancestor(d)

        # Is this a subfolder of a larger repo?
        is_subfolder = has_git and not (d / ".git").exists()

        return ProjectContext(
            mode="subfolder" if is_subfolder else "existing_project",
            root=d,
            kind=kind,
            has_git=has_git,
            has_tests=bool(test_paths),
            test_paths=test_paths,
            entry_files=entry_files,
            package_files=pkg_files,
            run_commands=run_cmds,
            notes=[f"{len(entries)} top-level entries"]
                  + (["Subfolder of a larger git repo"] if is_subfolder else []),
        )

    @classmethod
    def _classify_kind(cls, d: Path, entries: set) -> tuple[str, list[str], list[str]]:
        pkg_files = []
        run_cmds = []

        py_files = entries & cls.PYTHON_PACKAGE_FILES
        node_files = entries & cls.NODE_PACKAGE_FILES
        rust_files = entries & cls.RUST_FILES
        go_files = entries & cls.GO_FILES
        web_files = entries & cls.WEB_FILES

        if py_files and node_files:
            pkg_files = sorted(py_files | node_files)
            return "mixed", pkg_files, cls._suggest_mixed_cmds(d, py_files, node_files)

        if py_files:
            pkg_files = sorted(py_files)
            uv_mode = "uv.lock" in py_files or shutil.which("uv") is not None and "pyproject.toml" in py_files
            kind = "python-uv" if uv_mode else "python-pip"
            run_cmds = cls._suggest_python_cmds(d, uv_mode, py_files)
            return kind, pkg_files, run_cmds

        if node_files:
            pkg_files = sorted(node_files)
            if "bun.lockb" in node_files or "bun.lock" in node_files:
                kind = "node-bun"; run_cmds = ["bun install", "bun test", "bun run dev"]
            elif "pnpm-lock.yaml" in node_files:
                kind = "node-npm"; run_cmds = ["pnpm install", "pnpm test", "pnpm dev"]
            elif "yarn.lock" in node_files:
                kind = "node-npm"; run_cmds = ["yarn install", "yarn test", "yarn dev"]
            else:
                kind = "node-npm"; run_cmds = ["npm install", "npm test", "npm run dev"]
            return kind, pkg_files, run_cmds

        if rust_files:
            return "rust", sorted(rust_files), ["cargo build", "cargo test", "cargo run"]
        if go_files:
            return "go", sorted(go_files), ["go build ./...", "go test ./...", "go run ."]
        if web_files:
            return "web-static", sorted(web_files), ["python -m http.server 8000"]

        return "unknown", [], []

    @classmethod
    def _suggest_python_cmds(cls, d: Path, uv_mode: bool, pkg_files: set) -> list[str]:
        cmds = []
        if uv_mode:
            cmds.extend(["uv sync", "uv run pytest"])
            if (d / "pyproject.toml").exists():
                cmds.append("uv run python -m <module>")
        else:
            is_win = platform.system() == "Windows"
            venv_py = ".venv/Scripts/python.exe" if is_win else ".venv/bin/python"
            if (d / ".venv").exists() or (d / "venv").exists():
                cmds.append(f"{venv_py} -m pytest")
            else:
                cmds.extend(["python -m venv .venv", "pip install -r requirements.txt"])
            cmds.append("pytest" if "requirements.txt" in pkg_files else "python -m unittest")
        return cmds

    @classmethod
    def _suggest_mixed_cmds(cls, d: Path, py: set, node: set) -> list[str]:
        out = []
        if "uv.lock" in py: out.append("uv run pytest")
        elif py: out.append("pytest")
        if "bun.lockb" in node or "bun.lock" in node: out.append("bun test")
        elif node: out.append("npm test")
        return out

    @classmethod
    def _suggest_run_for_file(cls, f: Path, kind: str) -> list[str]:
        name = f.name
        if kind.startswith("python"):
            # Check if uv available at file's parent
            if shutil.which("uv") and (f.parent / "pyproject.toml").exists():
                return [f"uv run python {name}", f"uv run pytest {name}"]
            return [f"python {name}", f"pytest {name}"]
        if kind.startswith("node"):
            return [f"node {name}", f"npm test -- {name}"]
        if kind == "rust":
            return [f"cargo run --bin {f.stem}"]
        if kind == "web-static":
            return [f"open {name}"]
        return []

    @classmethod
    def _find_tests(cls, d: Path, entries: set) -> list[str]:
        found = []
        for tname in cls.TEST_DIR_NAMES:
            if tname in entries and (d / tname).is_dir():
                found.append(tname + "/")
        # Also scan top-level for test files (shallow, no recursion)
        try:
            for f in d.iterdir():
                if f.is_file() and any(pat in f.name for pat in cls.TEST_FILE_PATTERNS):
                    found.append(f.name)
                    if len(found) >= 10:
                        break
        except PermissionError:
            pass
        return found

    @classmethod
    def _find_entries(cls, d: Path, entries: set, kind: str) -> list[str]:
        candidates = {
            "python-uv": ["main.py", "app.py", "__main__.py", "cli.py", "run.py"],
            "python-pip": ["main.py", "app.py", "__main__.py", "cli.py", "run.py"],
            "node-npm": ["index.js", "index.ts", "main.js", "main.ts", "src/index.js", "src/index.ts"],
            "node-bun": ["index.ts", "index.js", "main.ts", "src/index.ts"],
            "web-static": ["index.html"],
            "rust": ["src/main.rs", "src/lib.rs"],
            "go": ["main.go", "cmd/main.go"],
        }.get(kind, [])
        found = []
        for c in candidates:
            if (d / c).exists():
                found.append(c)
        return found

    @staticmethod
    def _has_git_ancestor(d: Path) -> bool:
        for parent in d.parents:
            if (parent / ".git").exists():
                return True
        return False

class ProjectScaffolder:
    """Creates minimal scaffolding for new projects. Only when user opts in."""

    @staticmethod
    def scaffold(root: Path, kind: str, name: Optional[str] = None) -> list[str]:
        """Create minimal files for given project kind. Returns list of created files."""
        root.mkdir(parents=True, exist_ok=True)
        name = name or root.name
        created = []

        if kind == "python-uv":
            (root / "pyproject.toml").write_text(
                f'[project]\nname = "{name}"\nversion = "0.1.0"\nrequires-python = ">=3.10"\ndependencies = []\n\n'
                f'[tool.uv]\ndev-dependencies = ["pytest"]\n'
            )
            (root / "main.py").write_text(f'def main():\n    print("Hello from {name}")\n\n\nif __name__ == "__main__":\n    main()\n')
            (root / "tests").mkdir(exist_ok=True)
            (root / "tests" / "__init__.py").touch()
            (root / "tests" / "test_main.py").write_text(
                "import unittest\nfrom main import main\n\n\nclass TestMain(unittest.TestCase):\n"
                "    def test_main_runs(self):\n        main()  # smoke test\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            )
            (root / ".gitignore").write_text(".venv/\n__pycache__/\n*.pyc\n.pytest_cache/\n")
            created = ["pyproject.toml", "main.py", "tests/__init__.py", "tests/test_main.py", ".gitignore"]

        elif kind == "python-pip":
            (root / "requirements.txt").write_text("")
            (root / "main.py").write_text(f'def main():\n    print("Hello from {name}")\n\n\nif __name__ == "__main__":\n    main()\n')
            (root / "tests").mkdir(exist_ok=True)
            (root / "tests" / "test_main.py").write_text(
                "import unittest\nfrom main import main\n\n\nclass TestMain(unittest.TestCase):\n"
                "    def test_main_runs(self):\n        main()\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            )
            (root / ".gitignore").write_text(".venv/\nvenv/\n__pycache__/\n*.pyc\n")
            created = ["requirements.txt", "main.py", "tests/test_main.py", ".gitignore"]

        elif kind == "node-npm":
            (root / "package.json").write_text(
                f'{{\n  "name": "{name}",\n  "version": "0.1.0",\n  "type": "module",\n'
                f'  "scripts": {{\n    "start": "node index.js",\n    "test": "node --test"\n  }}\n}}\n'
            )
            (root / "index.js").write_text(f'console.log("Hello from {name}");\n')
            (root / "index.test.js").write_text(
                "import { test } from 'node:test';\nimport assert from 'node:assert';\n\n"
                "test('smoke', () => { assert.ok(true); });\n"
            )
            (root / ".gitignore").write_text("node_modules/\n.env\n")
            created = ["package.json", "index.js", "index.test.js", ".gitignore"]

        elif kind == "node-bun":
            (root / "package.json").write_text(
                f'{{\n  "name": "{name}",\n  "version": "0.1.0",\n  "type": "module",\n'
                f'  "scripts": {{\n    "start": "bun run index.ts",\n    "test": "bun test"\n  }}\n}}\n'
            )
            (root / "index.ts").write_text(f'console.log("Hello from {name}");\n')
            (root / "index.test.ts").write_text(
                'import { test, expect } from "bun:test";\n\ntest("smoke", () => { expect(true).toBe(true); });\n'
            )
            (root / ".gitignore").write_text("node_modules/\n.env\nbun.lockb\n")
            created = ["package.json", "index.ts", "index.test.ts", ".gitignore"]

        elif kind == "web-static":
            (root / "index.html").write_text(
                f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>{name}</title>\n<link rel="stylesheet" href="style.css">\n</head>\n'
                f'<body>\n<h1>{name}</h1>\n<script src="app.js"></script>\n</body>\n</html>\n'
            )
            (root / "style.css").write_text("body { font-family: system-ui; margin: 2rem; }\n")
            (root / "app.js").write_text("console.log('app loaded');\n")
            created = ["index.html", "style.css", "app.js"]

        return created

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

PLANNER_SYSTEM = """
Du bist der PLANNER-Agent eines Multi-Agent-Coding-Systems.

AUFGABE: Analysiere die Codebasis und erstelle einen strukturierten Ausführungsplan.

REGELN:
1. Nutze vfs_shell("get GROUNDED information\'s","ls /project") und vfs_shell("get GROUNDED information\'s","cat /project/...") um die Codebasis zu verstehen BEVOR du planst.
2. Nutze vfs_shell("get GROUNDED information\'s","grep -rn 'pattern' /project") um relevante Stellen zu finden.
3. Erstelle den Plan via add_subtask Tool (einmal pro Subtask aufrufen).
4. Jeder Subtask MUSS spezifizieren: description, files (Komma-getrennt, MINDESTENS 1 Datei), priority.
5. Dateien dürfen NICHT zwischen Subtasks überlappen! Jede Datei gehört genau einem Subtask.
6. Wenn der Task nur 1-2 Dateien betrifft, erstelle nur 1 Subtask.
7. STRICT ZERO-GUESSING: Rate NIEMALS welche Dateien existieren. Nutze vfs_shell('get GROUNDED information\'s','ls') um zu prüfen.
8. Lies JEDE Datei die du in einen Subtask aufnimmst mindestens einmal (cat), damit der Umfang klar ist.
9. Rufe am Ende ZWINGEND finalize_plan auf.
10. Schreibe Status-Updates in _coordination.md via vfs_shell("writing status","echo '...' >> /project/_coordination.md").
"""

VALIDATOR_SYSTEM = """
Du bist der VALIDATOR-Agent eines Multi-Agent-Coding-Systems.

AUFGABE: Prüfe die Änderungen der Coder-Agents auf Korrektheit.

REGELN:
1. Lies die geänderten Dateien via vfs_shell("validation work", "cat /project/...") oder vfs_view.
2. Prüfe auf: Syntaxfehler, Logikfehler, fehlende Imports, kaputte Referenzen.
3. Führe Tests aus wenn verfügbar (run_file oder bash).
4. Melde Ergebnisse via report_issues Tool.
5. Leere Liste [] = alles OK.
6. Lies /project/_coordination.md für Kontext zum Plan.

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
        "1. ARCHITEKTUR ZUERST: Lies /project/_coordination.md um den Plan zu verstehen.\n"
        "2. LIES Dateien via vfs_shell('finding information\'s','cat /project/...') oder vfs_view BEVOR du editierst. Niemals blind raten!\n"
        "3. SCHREIBEN — STRIKT NACH DATEI-GRÖSSE:\n"
        "   - Datei < 40 Zeilen:  vfs_shell('Creating content','write /project/f.py \"content\"')\n"
        "   - Datei >= 40 Zeilen: write_chunk-Protokoll (siehe unten). NIEMALS echo >> für grosse Dateien!\n"
        "   write_chunk Protokoll:\n"
        "     vfs_shell('creating data inital','write_chunk /project/f.js 0 3 \"...block 0...\"')  # erzeugt Datei\n"
        "     vfs_shell('creating data appended','write_chunk /project/f.js 1 3 \"...block 1...\"')  # appended\n"
        "     vfs_shell('creating data finalisiert','write_chunk /project/f.js 2 3 \"...block 2...\"')  # finalisiert\n"
        "   Wenn ein write_chunk Call abbricht: vfs_shell('validation write progress','write_chunk_status /project/f.js') zeigt welche Bloecke fehlen.\n"
        "4. BEENDEN: Wenn du alle Aufgaben erledigt hast, rufe ZWINGEND das Tool 'done' auf (oder schreibe [DONE]).\n"
        "5. VERIFIZIERE JEDEN WRITE: Nach jedem write/write_chunk sofort vfs_shell('validation work', 'cat /project/...') und pruefen dass der Inhalt stimmt.\n"
        "   Wenn Inhalt abweicht: write_chunk_status aufrufen, dann fehlende Bloecke nachschicken.\n"
        "6. ISOLIERTER WORKSPACE: Alle Pfade unter /project/!\n"
        "7. CODE AUSFUEHREN: Nutze IMMER 'run_file' fuer Skripte/Tests. 'bash' NUR wenn noetig.\n"
        "8. Bearbeite NUR die dir zugewiesenen Dateien!\n"
        "9. Nutze vfs_shell('finding informations','grep -rn ...') zum Suchen im Code.\n\n"
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
        self._shared_mount_key: str | None = None

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

    def _write_via_shared_store(self, relative_path: str, content: str) -> None:
        """
        Schreibt Datei über den Shared-Store wenn möglich.
        Fallback: direktes Disk-Write (wie bisher).

        Shared-Writes invalidieren automatisch alle Agent-VFS-Caches —
        kein refresh_mount mehr nötig.
        """
        if self._shared_mount_key is not None:
            try:
                gvfs = get_global_vfs()
                result = gvfs.shared_write(
                    mount_key=self._shared_mount_key,
                    relative_path=relative_path,
                    content=content,
                    local_base=self._shared_worktree_path,
                    author="coder_orchestrator",
                )
                if result.get("success"):
                    return
            except Exception as e:
                self._log("SHARED-WARN", f"Shared write failed: {e}", "yellow")

        # Fallback: direkt auf Disk
        target = (self.worktree.path / relative_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    # ─────────────────────────────────────────────────────────────────
    # NEW: Sub-Agent Lifecycle
    # ─────────────────────────────────────────────────────────────────
    def _emit_swarm_phase(self, phase: str, info: str = ""):
        """Signal swarm phase transition to parent TaskView.
        Chunk OHNE _sub_agent_id → landet auf Parent/Summary."""
        if self.row_chunk_fun:
            self.row_chunk_fun({
                "type": "swarm_phase",
                "swarm_phase": phase,
                "swarm_info": info,
            })

    async def _ensure_agents(self):
        """Lazy-init planner, coder(s), validator via FlowAgentBuilder."""
        if self._sub_agents_ready:
            return

        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")

        # ── Create shared VFS share from worktree ──
        from toolboxv2.mods.isaa.base.patch.power_vfs import get_sharing_manager
        sharing = get_sharing_manager()

        # We need a "source VFS" to create the share from.
        # Use the first agent's VFS as bootstrap, mount worktree, then share it.
        # Instead: create share directly from local path (bypass VFS share API limitation)
        self._shared_worktree_path = str(self.worktree.path)
        gvfs = get_global_vfs()
        self._shared_mount_key = gvfs.register_shared_mount(
            self._shared_worktree_path,
            mount_key=f"worktree-{self._agent_uid}",
            hydrate=True,
        )
        self._log(
            "SHARED",
            f"Worktree shared store: {self._shared_mount_key}",
            "cyan",
        )

        # ── PLANNER ──
        pb = isaa.get_agent_builder(
            name=self._planner_name, add_base_tools=False, with_dangerous_shell=False
        )
        pb.with_system_message(PLANNER_SYSTEM)
        pb.with_models(self.model)
        self._register_planner_tools(pb, self)
        await isaa.register_agent(pb)

        # ── CODER ──
        coder_name = f"coder_{self._agent_uid}_0"
        self._coder_names = [coder_name]
        cb = isaa.get_agent_builder(
            name=coder_name, add_base_tools=False, with_dangerous_shell=False
        )
        cb.with_system_message(self.SYSTEM_PROMPT)
        cb.with_models(self.model)
        self._register_coder_tools(cb, self, coder_name)
        await isaa.register_agent(cb)

        # ── VALIDATOR ──
        vb = isaa.get_agent_builder(
            name=self._validator_name, add_base_tools=False, with_dangerous_shell=False
        )
        vb.with_system_message(VALIDATOR_SYSTEM)
        vb.with_models(self.model)
        self._register_validator_tools(vb, self)
        await isaa.register_agent(vb)

        # ── Mount worktree into ALL agents as shared mount ──
        for aname in [self._planner_name] + self._coder_names + [self._validator_name]:
            await self._mount_shared_worktree(aname)

        self._sub_agents_ready = True
        self._log("AGENTS",
                  f"Sub-agents ready (shared worktree): {self._planner_name}, {self._coder_names}, {self._validator_name}",
                  "green")

    async def _mount_shared_worktree(self, agent_name: str):
        """Mount worktree als Shared-Mount. Alle Agents teilen RAM-State + Disk."""
        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")
        agent = await isaa.get_agent(agent_name)
        session = await agent.session_manager.get_or_create("default")
        try:
            session.vfs.mount(
                local_path=self._shared_worktree_path,
                vfs_path="/project",
                readonly=False,
                auto_sync=True,
            )
            # EBENE 3: VFS-Instanz beim GlobalVFSManager registrieren,
            # damit Cache-Invalidierung bei Shared-Writes funktioniert.
            gvfs = get_global_vfs()
            gvfs.register_vfs(session.vfs)
            self._log(
                "VFS",
                f"Shared mount + store for {agent_name} → {self._shared_worktree_path}",
                "cyan",
            )
        except Exception as e:
            self._log("VFS-WARN", f"Mount failed for {agent_name}: {e}", "yellow")

    async def _refresh_agent_vfs(self, agent_name: str):
        """Refresh agent's VFS mount to pick up changes from other agents."""
        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")
        try:
            agent = await isaa.get_agent(agent_name)
            session = await agent.session_manager.get_or_create("default")
            session.vfs.sync_all()

            session.vfs.unmount("/project", save_changes=True)
            session.vfs.refresh_mount("/project")
            await self._mount_shared_worktree(agent_name)
            self._log("SYNC", f"Refreshed VFS for {agent_name}", "cyan")
        except Exception as e:
            self._log("SYNC-WARN", f"Refresh failed for {agent_name}: {e}", "yellow")

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
                    session.vfs.unmount("/project", save_changes=True)
                except Exception:
                    pass
                try:
                    from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs
                    gvfs = get_global_vfs()
                    key = f"{session.vfs.agent_name}:{session.vfs.session_id}"
                    gvfs._mounted_vfs.pop(key, None)
                except Exception:
                    pass
                # Agent entfernen — je nach isaa API
                if hasattr(isaa, "unregister_agent"):
                    await isaa.unregister_agent(name)
                elif hasattr(isaa, "delete_agent"):
                    await isaa.delete_agent(name)
                self._log("CLEANUP", f"Removed agent: {name}", "grey")
            except Exception as e:
                logger.debug(f"Cleanup {name}: {e}")

        # EBENE 3: Shared-Mount entfernen
        if self._shared_mount_key is not None:
            try:
                from toolboxv2.mods.isaa.base.patch.power_vfs import get_global_vfs
                gvfs = get_global_vfs()
                gvfs.unregister_shared_mount(self._shared_worktree_path)
                self._log(
                    "CLEANUP",
                    f"Shared mount unregistered: {self._shared_mount_key}",
                    "grey",
                )
            except Exception as e:
                logger.debug(f"Unregister shared mount: {e}")
            self._shared_mount_key = None

        self._coder_names = []
        self._sub_agents_ready = False
        self._log("CLEANUP", "All sub-agents removed", "green")

    # ─────────────────────────────────────────────────────────────────
    # NEW: Tool Registration per Agent Type
    # ─────────────────────────────────────────────────────────────────

    def _register_planner_tools(self, builder, host: "CoderAgent"):
        """Register planner-specific tools. VFS tools come from FlowAgent."""

        async def add_subtask(description: str, files: str, priority: str = "normal") -> str:
            """Add a subtask to the execution plan.

            Args:
                description: What needs to be done in this subtask.
                files: Comma-separated list of file paths this subtask will modify.
                      Example: "src/main.py, src/utils.py"
                priority: "high", "normal", or "low"

            Call this once per subtask. Files MUST NOT overlap between subtasks!
            Call finalize_plan when all subtasks are added.
            """
            if priority not in ("high", "normal", "low"):
                return f"Error: priority must be high/normal/low, got '{priority}'"

            file_list = [f.strip() for f in files.split(",") if f.strip()]
            if not file_list:
                return "Error: Subtask must specify at least one file. Use vfs_shell('get GROUNDED information\'s','ls /project') to discover files."

            existing_files = set()
            for st in host._current_plan:
                existing_files.update(st.get("files", []))

            overlap = set(file_list) & existing_files
            if overlap:
                return f"Error: Files already assigned to another subtask: {', '.join(overlap)}"

            subtask = {"description": description, "files": file_list, "priority": priority}
            host._current_plan.append(subtask)

            await self._refresh_agent_vfs(self._planner_name)
            return f"Subtask {len(host._current_plan)} added: {description} ({len(file_list)} files, {priority})"

        async def finalize_plan() -> str:
            """Finalize the execution plan. Call after all subtasks are added via add_subtask.
            This writes _coordination.md and locks the plan."""
            if not host._current_plan:
                return "Error: No subtasks added. Use add_subtask first."

            plan_md = "# Execution Plan\n\n"
            for i, st in enumerate(host._current_plan):
                plan_md += f"## Subtask {i + 1}: {st.get('description', '?')}\n"
                plan_md += f"- Files: {', '.join(st.get('files', []))}\n"
                plan_md += f"- Priority: {st.get('priority', 'normal')}\n"
                plan_md += f"- Status: pending\n\n"
            plan_md += "---\n## Status Log\n"

            coord_path = host.worktree.path / "_coordination.md"
            coord_path.write_text(plan_md, encoding="utf-8")
            await self._refresh_agent_vfs(self._planner_name)
            return f"Plan finalized: {len(host._current_plan)} subtask(s). Written to _coordination.md."

        # NO read_file, grep, list_files, write_md — agents use vfs_shell instead
        builder.add_tool(add_subtask, "add_subtask", "Add a subtask to the plan (call per subtask)",
                         category=["planning"], flags={"system_tool_by_name": True})
        builder.add_tool(finalize_plan, "finalize_plan", "Finalize plan after all subtasks added",
                         category=["planning"], flags={"system_tool_by_name": True})

    def _register_coder_tools(self, builder, host: "CoderAgent", coder_name: str):
        """Register coder-specific tools. VFS tools (read, write, grep, ls) come from FlowAgent."""

        async def run_file(file_path: str, args: str = "") -> str:
            """Run a script/test file in the worktree."""
            res = await host._run_file(file_path, args, [])
            await self._refresh_agent_vfs(coder_name)
            return res

        async def update_status(status: str) -> str:
            """Append a status line to _coordination.md."""
            coord_path = host.worktree.path / "_coordination.md"
            existing = coord_path.read_text(encoding="utf-8") if coord_path.exists() else ""
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            existing += f"\n[{ts}] {coder_name}: {status}"
            coord_path.write_text(existing, encoding="utf-8")
            await self._refresh_agent_vfs(coder_name)
            return "Status updated."

        async def done() -> str:
            """Signal that all assigned tasks are complete."""
            return "[DONE]"

        # NO read_file, grep — agents use vfs_shell("cat ..."), vfs_shell("grep ...") instead
        builder.add_tool(run_file, "run_file", "Run a script or test file only valid for project dir!", category=["execution"],flags={"system_tool_by_name": True} )
        builder.add_tool(update_status, "update_status", "Update coordination status", category=["coordination"],flags={"system_tool_by_name": True} )
        builder.add_tool(done, "done", "Signal task completion", category=["control"],flags={"system_tool_by_name": True} )

    def _register_validator_tools(self, builder, host: "CoderAgent"):
        """Register validator-specific tools. VFS tools come from FlowAgent."""

        async def bash(command: str) -> str:
            """Run a shell command for testing/linting."""
            res = await host._smart_bash(command, [])
            await self._refresh_agent_vfs(self._validator_name)
            return res

        async def run_file(file_path: str, args: str = "") -> str:
            """Run a test or script file."""
            res = await host._run_file(file_path, args, [])
            await self._refresh_agent_vfs(self._validator_name)
            return res

        async def report_issues(issues_json: str) -> str:
            """Report validation results. Argument: JSON array.
            Each issue: {"file": "path.py", "line": 42, "severity": "error|warning", "message": "..."}
            Pass empty array [] if everything is OK."""
            try:
                issues = json.loads(issues_json) if isinstance(issues_json, str) else issues_json
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON - {e}"
            host._validation_issues = issues if isinstance(issues, list) else []
            if not host._validation_issues:
                return "Validation passed. No issues found."
            report = "\n".join(
                f"- [{i.get('severity', '?')}] {i.get('file', '?')}:{i.get('line', '?')}: {i.get('message', '?')}"
                for i in host._validation_issues
            )
            await self._refresh_agent_vfs(self._validator_name)
            return f"Found {len(host._validation_issues)} issue(s):\n{report}"

        # NO read_file, grep — agents use vfs_shell instead
        builder.add_tool(bash, "bash", "Run shell command for testing no vfs ops!", category=["shell"], flags={"system_tool_by_name": True})
        builder.add_tool(run_file, "run_file", "Run a test file", category=["execution"], flags={"system_tool_by_name": True})
        builder.add_tool(report_issues, "report_issues", "Report validation results (JSON)", category=["validation"], flags={"system_tool_by_name": True})

    # ─────────────────────────────────────────────────────────────────
    # NEW: Stream Collector
    # ─────────────────────────────────────────────────────────────────

    async def _collect_stream(self, agent_name: str, query: str, prefix: str = "") -> str:
        """Run agent via a_stream(), collect full response, forward chunks tagged with _sub_agent_id."""
        from toolboxv2 import get_app
        isaa = get_app().get_mod("isaa")
        agent = await isaa.get_agent(agent_name)

        # Sub-Agent Start → Router legt Top-Level-TaskView an
        if self.row_chunk_fun:
            self.row_chunk_fun({
                "type": "swarm_sub_start",
                "_sub_agent_id": agent_name,
                "phase": prefix.lower() or "running",
                "query": query[:100],
                "max_iter": self.max_iters,
            })

        full_response = []
        try:
            await self._refresh_agent_vfs(agent_name)
        except Exception as e:
            print(e)
        try:
            async for chunk in agent.a_stream(
                query=query,
                session_id="default",
                max_iterations=self.max_iters,
            ):
                # ── Jeden Chunk mit Sub-Agent-Tag an Router weitergeben ──
                if self.row_chunk_fun:
                    tagged = dict(chunk)
                    tagged["_sub_agent_id"] = agent_name
                    tagged["_swarm_phase"] = prefix.lower() or "running"
                    self.row_chunk_fun(tagged)

                # ── Logging / stream_callback / final_answer sammeln ──
                ctype = chunk.get("type", "")
                if ctype == "content":
                    text = chunk.get("chunk", "")
                    full_response.append(text)
                    if self.stream_enabled and self.stream_callback:
                        label = f"[{prefix}] {text}" if prefix else text
                        if asyncio.iscoroutinefunction(self.stream_callback):
                            await self.stream_callback(label)
                        else:
                            self.stream_callback(label)

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
        finally:
            # Sub-Agent Done → Router setzt Status, Summary zählt hoch
            if self.row_chunk_fun:
                self.row_chunk_fun({
                    "type": "swarm_sub_done",
                    "_sub_agent_id": agent_name,
                    "phase": prefix.lower() or "running",
                })

        return "".join(full_response)
    # ─────────────────────────────────────────────────────────────────
    # Orchestration Phases
    # ─────────────────────────────────────────────────────────────────
    async def _run_planner(self, task: str) -> list[dict]:
        self._current_plan =[]

        # Phase 1a: Specs & Data Model
        self._log("PLANNER", "Phase 1a: Interface Contract & Data Model generieren", "cyan")
        spec_query = (
            f"Aufgabe:\n{task}\n\n"
            f"PHASE 1a - SYSTEM-ARCHITEKTUR & SPECS:\n"
            f"1. Analysiere das Projekt (vfs_shell 'ls' / 'cat').\n"
            f"2. Erstelle falls nötig /project/_specs/ (vfs_shell 'bash' -> 'mkdir -p /project/_specs').\n"
            f"3. Schreibe einen Interface-Contract als JSON nach /project/_specs/interface_contract.json mit:\n"
            f"   - dom_ids: Alle HTML-Element-IDs\n"
            f"   - css_classes: Alle CSS-Klassen\n"
            f"   - dom_structure: Verschachtelung als Baum\n"
            f"   - state_keys: Key-Namen für JS-State\n"
            f"   - shared_constants: Projektweite Konstanten\n"
            f"4. Schreibe das Datenmodell als Markdown nach /project/_specs/data_model.md.\n"
            f"REGELN: Alle IDs/Klassen/Keys müssen EXPLIZIT definiert sein, keine Duplikate."
        )
        await self._collect_stream(self._planner_name, spec_query, prefix="PLANNER-SPEC")

        # Phase 1b: Task Planning basierend auf Specs
        self._log("PLANNER", "Phase 1b: Subtasks erstellen", "cyan")
        task_query = (
            f"PHASE 1b - PLANUNG BASIEREND AUF SPECS:\n"
            f"1. Lade deinen Contract aus /project/_specs/interface_contract.json (via vfs_shell 'cat').\n"
            f"2. Erstelle Subtasks via add_subtask() Tool.\n"
            f"3. Jeder Subtask MUSS die relevanten IDs/Klassen aus dem Contract in der 'description' referenzieren.\n"
            f"4. Beachte strikte Trennung der Dateien zwischen Subtasks.\n"
            f"5. Rufe am Ende finalize_plan() auf."
        )
        await self._collect_stream(self._planner_name, task_query, prefix="PLANNER-TASKS")

        if not self._current_plan:
            self._log("PLANNER", "No plan created, falling back to single subtask", "yellow")
            self._current_plan = [{"description": task, "files": [], "priority": "normal"}]

        return self._current_plan

    async def _run_coders(self, subtasks: list[dict]) -> list[str]:
        """Phase 2: Master-Coder übernimmt alle Subtasks synchron und delegiert bei Bedarf."""
        if not subtasks:
            return []

        coder_name = self._coder_names[0]
        query = self._build_coder_query(subtasks)

        self._log("MASTER-CODER", f"Starte Bearbeitung von {len(subtasks)} Subtasks", "cyan")
        response = await self._collect_stream(coder_name, query, prefix="MASTER-CODER")

        return [response]


    async def _run_fix_coders(self, fix_subtasks: list[dict]) -> list[str]:
        """Run Master-Fixer to resolve all remaining validation issues."""
        if not fix_subtasks:
            return []

        coder_name = self._coder_names[0]
        query = self._build_fix_query(fix_subtasks)

        self._log("MASTER-FIXER", f"Starte Bearbeitung von {len(fix_subtasks)} Fixes", "yellow")
        response = await self._collect_stream(coder_name, query, prefix="MASTER-FIXER")

        return [response]

    def _build_coder_query(self, subtasks: list[dict]) -> str:
        contract_content = self._load_interface_contract()
        coord_content = ""
        try:
            coord_path = self.worktree.path / "_coordination.md"
            if coord_path.exists():
                coord_content = coord_path.read_text(encoding="utf-8")
        except Exception:
            pass

        tasks_dump = json.dumps(subtasks, indent=2)

        return (
            f"## DU BIST DER MASTER-CODER\n\n"
            f"Hier sind ALLE anstehenden Subtasks für dich:\n{tasks_dump}\n\n"
            f"## INTERFACE CONTRACT (BINDEND):\n"
            f"{contract_content or '(kein Contract generiert)'}\n\n"
            f"## VOLLSTÄNDIGER PLAN (aus _coordination.md):\n"
            f"{coord_content or '(leer)'}\n\n"
            f"## REGELN FÜR DEN MASTER-CODER:\n"
            f"1. Lies ALLE betroffenen Dateien (vfs_shell 'cat') BEVOR du etwas änderst.\n"
            f"2. Kleinere Änderungen (1-3 Zeilen): Mach es direkt selbst via vfs_shell.\n"
            f"3. Große Änderungen / neue Dateien: Nutze spawn_sub_agent() für Code-Generierung, dann schreibe das Ergebnis in die Datei.\n"
            f"4. NACH jedem Subtask: Lies die geänderte Datei neu und prüfe sie.\n"
            f"5. NACH allen Subtasks: Lies ALLE Dateien, prüfe Querverweise (IDs, Klassen, Keys).\n"
            f"6. Behebe gefundene Probleme SOFORT selbst.\n"
            f"7. IDs/Klassen/Keys MÜSSEN exakt dem Interface-Contract entsprechen.\n"
            f"8. Du bist ALLEIN verantwortlich für die Integration. Wenn etwas nicht passt: Repariere es.\n"
            f"9. Rufe done() auf, wenn ALLE Aufgaben erledigt und integriert sind.\n"
            f"10. alle äderungen und bearbitungen validiren nicht bild vertruen fehler in den tools transparent im final answer melden.!\n"
        )

    async def _run_validator(self, changed_files: list[str]) -> list[dict]:
        """Phase 3: Validator prüft gegen Specs und repariert kleine Issues selbst."""
        self._validation_issues = []

        if not changed_files:
            return []

        contract_content = self._load_interface_contract()

        query = (
            f"Validiere die folgenden geänderten Dateien:\n"
            f"{', '.join(changed_files)}\n\n"
            f"## INTERFACE CONTRACT:\n"
            f"{contract_content or '(kein Contract vorhanden)'}\n\n"
            f"Schritte & Regeln:\n"
            f"1. Lies jede geänderte Datei (vfs_shell 'cat').\n"
            f"2. Prüfe auf Syntaxfehler, fehlende Imports, Logikfehler.\n"
            f"3. Prüfe STRIKT auf ID/Klassen-Mismatch gegen den Interface Contract!\n"
            f"4. Führe vorhandene Tests aus (run_file).\n"
            f"5. REPARATUR-MODUS:\n"
            f"   - KLEINE FIXES (Tippfehler, falsche ID, fehlender Import): Repariere SOFORT mit vfs_shell.\n"
            f"   - Lies die reparierte Datei danach neu und VERIFIZIERE den Fix.\n"
            f"   - GROSSE PROBLEME (Architekturfehler): NICHT selbst reparieren.\n"
            f"6. ABSCHLUSS:\n"
            f"   - Rufe report_issues() auf mit NUR NOCH UNGELÖSTEN Problemen.\n"
            f"   - Leere Liste [] = alles OK.\n"
        )

        await self._collect_stream(self._validator_name, query, prefix="VALIDATOR")
        return self._validation_issues

    def _build_fix_query(self, fix_subtasks: list[dict]) -> str:
        contract_content = self._load_interface_contract()
        tasks_dump = json.dumps(fix_subtasks, indent=2)

        return (
            f"## DU BIST DER MASTER-FIXER\n\n"
            f"Der Validator hat folgende ungelöste Probleme gefunden:\n{tasks_dump}\n\n"
            f"## INTERFACE CONTRACT (BINDEND):\n"
            f"{contract_content or '(kein Contract vorhanden)'}\n\n"
            f"## REGELN FÜR DEN FIX MODE:\n"
            f"1. Lies betroffene Dateien mit vfs_shell('get GROUNDED information\'s', 'cat') BEVOR du editierst.\n"
            f"2. Behebe NUR die gemeldeten Probleme. Keine neuen Features oder Refactorings.\n"
            f"3. Kleinere Fixes (1-3 Zeilen): Mach es direkt selbst via vfs_shell.\n"
            f"4. Große Fixes: Nutze spawn_sub_agent() für Code-Generierung, dann speichere das Ergebnis.\n"
            f"5. NACH dem Fix: Lies die Datei neu und VERIFIZIERE, ob das Problem behoben ist.\n"
            f"6. IDs/Klassen/Keys MÜSSEN dem Interface-Contract entsprechen.\n"
            f"7. Rufe SOFORT done() auf, wenn alle Fixes verifiziert und angewendet sind.\n"
        )


    def _load_interface_contract(self) -> str:
        """Lädt den Interface Contract aus Phase 1a falls vorhanden."""
        try:
            contract_path = self.worktree.path / "_specs" / "interface_contract.json"
            if contract_path.exists():
                return contract_path.read_text(encoding="utf-8")
        except Exception:
            pass
        return ""
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
        subtasks = []
        try:
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

            # ── 4. PLANNER PHASE ──
            self._emit_swarm_phase("planning")
            self._log("PHASE", "=== PLANNER ===", "bold")
            subtasks = await self._run_planner(enriched_task)
            self._log("PHASE", f"Plan: {len(subtasks)} subtask(s)", "green")

            # ── 5. CODER PHASE ──
            self._log("PHASE", "=== CODER ===", "bold")
            self._emit_swarm_phase("coding", f"{len(subtasks)} subtasks")
            edit_changed = await self._run_coders(subtasks)
            self._log("PHASE", f"Coder changed: {edit_changed}", "green")

            # ── 6. VALIDATOR PHASE + FIX LOOP ──
            all_changed = edit_changed or await self.worktree.changed_files()

            if all_changed:
                self._log("PHASE", "=== VALIDATOR ===", "bold")
                self._emit_swarm_phase("validating")
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

                    await self._run_fix_coders(deduped)
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
                final_response={
                    "status": "success",
                    "subtasks_planned": len(subtasks),
                    "subtasks": [
                        {"description": st.get("description", ""), "files": st.get("files", [])}
                        for st in subtasks
                    ],
                    "files_changed": final_changed or [],
                    "validation_issues": self._validation_issues or [],
                    "tokens_used": self.tracker.total_tokens,
                    "compressions": self.tracker.compressions_done,
                    "model": self.model,
                    "worktree_path": str(self.worktree.path) if self.worktree.path else None,
                },
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
                summary=f"Failed: {str(e)[:100]}",
                final_response={
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": error_details,
                    "subtasks_planned": len(subtasks) if subtasks else 0,
                    "tokens_used": self.tracker.total_tokens,
                },
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
