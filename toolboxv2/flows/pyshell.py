# toolboxv2/flows/pyshell.py
# IPython-like multiline Python REPL -- no IPython required
# Entry: tb -f pyshell   or   tb -m pyshell

import asyncio
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.shortcuts import set_title
from prompt_toolkit.styles import Style

from toolboxv2 import App, get_app

NAME = "pyshell"

# ─────────────────────────────── optional pygments ────────────────────────────
try:
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.python import PythonLexer
    _LEXER = PygmentsLexer(PythonLexer)
except ImportError:
    _LEXER = None

# ─────────────────────────────── executor import ──────────────────────────────
try:
    from toolboxv2.mods.isaa.CodingAgent.live import MockIPython
except ImportError:
    MockIPython = None


# ─────────────────────────────── completion ───────────────────────────────────
import re
import rlcompleter
import shutil
import subprocess
from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter, PathCompleter
from prompt_toolkit.document import Document as _Doc


class _RlCompleter(Completer):
    def __init__(self, executor):
        self._ex = executor

    def get_completions(self, document, complete_event):
        word = document.get_word_before_cursor(pattern=r"[\w\.]+")
        rc = rlcompleter.Completer(self._ex.get_namespace())
        i, seen = 0, set()
        while True:
            try:
                c = rc.complete(word, i)
            except Exception:
                break
            if c is None:
                break
            c = c.rstrip("(")
            if c not in seen:
                seen.add(c)
                yield Completion(c, start_position=-len(word))
            i += 1


class _ShellCompleter(Completer):
    """Completes !-commands: binary names, --flags, file paths."""

    def __init__(self):
        self._path = PathCompleter()
        self._flag_cache: dict = {}

    def _executables(self):
        exes = set()
        for d in os.environ.get("PATH", "").split(os.pathsep):
            try:
                for f in os.listdir(d):
                    if os.access(os.path.join(d, f), os.X_OK):
                        exes.add(f)
            except OSError:
                pass
        return exes

    def _flags(self, cmd: str) -> list[str]:
        if cmd not in self._flag_cache:
            try:
                out = subprocess.run(
                    [cmd, "--help"], capture_output=True, text=True, timeout=2
                )
                self._flag_cache[cmd] = re.findall(r"-{1,2}[\w][\w-]*", out.stdout + out.stderr)[:60]
            except Exception:
                self._flag_cache[cmd] = []
        return self._flag_cache[cmd]

    def get_completions(self, document, complete_event):
        raw = document.text_before_cursor.lstrip()
        # strip leading !
        text = raw[1:] if raw.startswith("!") else raw
        parts = text.split()
        no_space = not text.endswith(" ")

        if not parts or (len(parts) == 1 and no_space):
            word = parts[0] if parts else ""
            for exe in sorted(self._executables()):
                if exe.startswith(word):
                    yield Completion(exe, start_position=-len(word), display_meta="cmd")
            return

        cmd     = parts[0]
        current = parts[-1] if no_space else ""

        # file/path
        if current.startswith(("/", "./", "~")):
            yield from self._path.get_completions(_Doc(current), complete_event)
            return

        # flags
        for flag in self._flags(cmd):
            if flag.startswith(current):
                yield Completion(flag, start_position=-len(current), display_meta="flag")


class _HybridCompleter(Completer):
    def __init__(self, executor, sig_state: dict):
        self._ex    = executor
        self._rl    = _RlCompleter(executor)
        self._shell = _ShellCompleter()
        self._sig   = sig_state   # shared {"text": ""}

    def get_completions(self, document, complete_event):
        src = document.text_before_cursor

        # ── shell passthrough ──
        if src.lstrip().startswith("!"):
            yield from self._shell.get_completions(document, complete_event)
            return

        # ── jedi: signatures + completions ──
        try:
            import jedi
            ns   = self._ex.get_namespace()
            line = src.count("\n") + 1
            col  = len(src.splitlines()[-1]) if src.splitlines() else 0
            interp = jedi.Interpreter(src, [ns])

            # signature for toolbar
            sigs = interp.get_signatures(line, col)
            if sigs:
                s      = sigs[0]
                params = ", ".join(p.description for p in s.params)
                self._sig["text"] = f"{s.name}({params})"
            else:
                self._sig["text"] = ""

            for c in interp.complete(line, col):
                yield Completion(
                    c.name_with_symbols,
                    start_position=-(len(c.name_with_symbols) - len(c.complete)),
                    display_meta=c.type,
                )
            return
        except Exception:
            self._sig["text"] = ""

        yield from self._rl.get_completions(document, complete_event)

# ══════════════════════════════════════════════════════════════════════════════
#  Fallback executor  (pure stdlib, no venv / VFS overhead)
# ══════════════════════════════════════════════════════════════════════════════
class _SimpleExecutor:
    """Minimal async executor used when MockIPython is unavailable."""

    def __init__(self, app: App):
        self._count = 0
        self.user_ns: dict = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
            "app": app,
        }
        self.output_history: dict = {}

    async def run_cell(self, code: str, live_output: bool = True) -> str:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        import ast

        self._count += 1
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result = None

        try:
            tree = ast.parse(code)
            # split: last expression → eval, rest → exec
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                exec_tree = ast.Module(body=tree.body[:-1], type_ignores=[])
                eval_expr = ast.Expression(body=tree.body[-1].value)
                ast.fix_missing_locations(exec_tree)
                ast.fix_missing_locations(eval_expr)
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    exec(compile(exec_tree, "<exec>", "exec"), self.user_ns)
                    result = eval(compile(eval_expr, "<eval>", "eval"), self.user_ns)
                    if asyncio.iscoroutine(result):
                        result = await result
            else:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    co = compile(tree, "<exec>", "exec")
                    exec(co, self.user_ns)
        except Exception:
            tb = traceback.format_exc()
            if live_output:
                sys.stderr.write(tb)
            stderr_buf.write(tb)

        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()
        self.output_history[self._count] = {"code": code, "stdout": out, "stderr": err, "result": result}

        parts = []
        if result is not None:
            parts.append(str(result))
        if out:
            parts.append(out.rstrip())
        if err:
            parts.append(f"\x1b[31m{err.rstrip()}\x1b[0m")
        return "\n".join(parts)

    def reset(self):
        self.user_ns = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
            "app": get_app(),
        }
        self.output_history.clear()
        self._count = 0

    def get_namespace(self) -> dict:
        return self.user_ns.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  Block / multiline detection
# ══════════════════════════════════════════════════════════════════════════════
_BLOCK_STARTERS = (
    "def ", "async def ", "class ", "if ", "elif ", "else:",
    "for ", "while ", "with ", "try:", "except", "finally:",
    "async for ", "async with ",
)


def _needs_continuation(code: str) -> bool:
    """Return True if the last non-empty line opens a new block."""
    lines = [l for l in code.splitlines() if l.strip()]
    if not lines:
        return False
    last = lines[-1].rstrip()
    if last.endswith(":") or last.endswith("\\"):
        return True
    # open parentheses / brackets / braces
    if code.count("(") > code.count(")"):
        return True
    if code.count("[") > code.count("]"):
        return True
    if code.count("{") > code.count("}"):
        return True
    return False


def _is_complete(code: str) -> bool:
    """
    True when the buffer is ready to execute.
    Empty lines after an indented block = complete.
    """
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        # 'unexpected EOF' or 'expected an indented block' → incomplete
        msg = str(e).lower()
        if "unexpected eof" in msg or "expected an indented" in msg or "was never closed" in msg:
            return False
        return True  # real syntax error → let executor show it


# ══════════════════════════════════════════════════════════════════════════════
#  Magic commands
# ══════════════════════════════════════════════════════════════════════════════
class MagicHandler:
    PREFIX = "%"

    def __init__(self, executor, shell: "PyShell"):
        self.ex = executor
        self.shell = shell

    async def handle(self, line: str) -> bool:
        """Returns True if line was a magic command."""
        stripped = line.strip()
        if not stripped.startswith(self.PREFIX):
            return False

        cmd = stripped[len(self.PREFIX):].split()[0].lower()
        args = stripped[len(self.PREFIX) + len(cmd):].strip()

        if cmd in ("reset", "clear_ns"):
            self.ex.reset()
            print("🔄 Namespace reset.")
        elif cmd in ("history", "hist"):
            self._show_history(args)
        elif cmd in ("ns", "who", "whos"):
            self._show_ns(detailed=(cmd == "whos"))
        elif cmd in ("run", "load"):
            await self._run_file(args)
        elif cmd in ("save",):
            self._save_session(args)
        elif cmd == "timeit":
            await self._timeit(args)
        elif cmd in ("help", "?"):
            self._show_help()
        else:
            print(f"❓ Unknown magic: %{cmd}  (try %help)")
        return True

    def _show_history(self, args: str):
        n = int(args) if args.isdigit() else 20
        items = list(self.ex.output_history.items())[-n:]
        for idx, data in items:
            print(f"\x1b[36mIn [{idx}]:\x1b[0m {data['code'][:80]}")

    def _show_ns(self, detailed=False):
        ns = self.ex.get_namespace()
        skip = {"__name__", "__builtins__", "__file__", "__path__", "app",
                 "auto_install", "modify_code", "open", "PYTHON_EXEC"}
        for k, v in sorted(ns.items()):
            if k.startswith("_") or k in skip:
                continue
            if detailed:
                print(f"  {k:20s} {type(v).__name__:15s} {repr(v)[:60]}")
            else:
                print(f"  {k}")

    async def _run_file(self, path: str):
        p = Path(path.strip())
        if not p.exists():
            print(f"❌ File not found: {p}")
            return
        code = p.read_text(encoding="utf-8")
        print(f"▶  Running {p} ...")
        result = await self.ex.run_cell(code)
        if result:
            print(result)

    def _save_session(self, args: str):
        if hasattr(self.ex, "save_session"):
            name = args.strip() or "pyshell_session"
            self.ex.save_session(name)
            print(f"💾 Session saved: {name}")
        else:
            print("⚠  save_session not available with SimpleExecutor")

    async def _timeit(self, stmt: str):
        import time
        runs = 100
        start = time.perf_counter()
        for _ in range(runs):
            await self.ex.run_cell(stmt, live_output=False)
        elapsed = (time.perf_counter() - start) / runs
        print(f"⏱  {elapsed * 1000:.4f} ms per loop (mean of {runs})")

    @staticmethod
    def _show_help():
        print("""
╔══════════════════════════════════════════════════════╗
║            PyShell  —  Magic Commands                ║
╠══════════════════════════════════════════════════════╣
║  %reset / %clear_ns   Reset the Python namespace     ║
║  %history [n]         Show last n inputs (def. 20)   ║
║  %ns / %who / %whos   List namespace variables       ║
║  %run <file.py>       Execute a Python file          ║
║  %save [name]         Save session (MockIPython only)║
║  %timeit <expr>       Benchmark an expression        ║
║  %help / %?           This help                      ║
╠══════════════════════════════════════════════════════╣
║  Keyboard shortcuts                                  ║
║  Enter          Add line (multiline mode)            ║
║  Alt+Enter / Meta+Enter   Submit cell                ║
║  Ctrl+C         Cancel current input                 ║
║  Ctrl+D         Exit shell                           ║
║  !cmd           Run shell command                    ║
╚══════════════════════════════════════════════════════╝""")


# ══════════════════════════════════════════════════════════════════════════════
#  Main Shell
# ══════════════════════════════════════════════════════════════════════════════
class PyShell:
    BANNER = (
        "\n\x1b[1;36m"
        "  ╔══════════════════════════════════════╗\n"
        "  ║   PyShell  ·  ToolBoxV2              ║\n"
        "  ║   IPython-like REPL  (no IPython)    ║\n"
        "  ║   %help for magic commands           ║\n"
        "  ║   Ctrl+D to exit                     ║\n"
        "  ╚══════════════════════════════════════╝"
        "\x1b[0m\n"
    )

    _STYLE = Style.from_dict({
        "prompt.in":    "#ansigreen bold",
        "prompt.cont":  "#ansiyellow",
        "prompt.num":   "#ansicyan",
        "rprompt":      "#888888",
        "bottom-toolbar": "bg:#1a1a2e #aaaaaa",
    })

    def __init__(self, app: App):
        self.app = app
        self._count = 0

        # Executor
        if MockIPython is not None:
            try:
                self.ex = MockIPython(auto_remove=True)
                self.ex.update_namespace({"app": app})
                self._ex_name = "MockIPython"
            except Exception as e:
                print(f"⚠  MockIPython init failed ({e}), using SimpleExecutor")
                self.ex = _SimpleExecutor(app)
                self._ex_name = "SimpleExecutor"
        else:
            self.ex = _SimpleExecutor(app)
            self._ex_name = "SimpleExecutor"

        self.magic = MagicHandler(self.ex, self)
        self._running = True
        self._session: Optional[PromptSession] = None
        self._sig_state: dict = {"text": ""}

    # ── prompt helpers ────────────────────────────────────────────────────────

    def _in_prompt(self) -> HTML:
        n = self.ex._count + 1 if hasattr(self.ex, "_count") else self._count + 1
        return HTML(f'<prompt.in>In [</prompt.in><prompt.num>{n}</prompt.num><prompt.in>]: </prompt.in>')

    def _cont_prompt(self) -> HTML:
        return HTML('<prompt.cont>   ...: </prompt.cont>')

    def _rprompt(self) -> HTML:
        ns_size = len([k for k in self.ex.user_ns if not k.startswith("_")])
        return HTML(f'<rprompt> {self._ex_name} | ns:{ns_size} </rprompt>')

    def _toolbar(self) -> HTML:
        sig = self._sig_state.get("text", "")
        sig_part = f'  <ansicyan><b>{sig}</b></ansicyan>' if sig else ""
        return HTML(
            f' <b>PyShell</b>{sig_part}  '
            '<ansiyellow>Alt+Enter</ansiyellow>=submit  '
            '<ansiyellow>Ctrl+C</ansiyellow>=cancel  '
            '<ansiyellow>Ctrl+D</ansiyellow>=exit  '
            '<ansiyellow>!cmd</ansiyellow>=shell  '
            '<ansiyellow>%help</ansiyellow>=magics'
        )

    # ── key bindings ──────────────────────────────────────────────────────────

    def _build_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("escape", "enter")  # Alt+Enter  (escape = meta prefix)
        @kb.add("escape", "c-m")   # Alt+Enter on some terminals
        def _submit(event):
            """Force-submit current buffer."""
            event.current_buffer.validate_and_handle()

        @kb.add("enter")
        def _newline_or_submit(event):
            buf = event.current_buffer
            text = buf.text
            cursor_at_end = buf.cursor_position == len(text)
            # submit on empty line after a block, or if AST complete
            if cursor_at_end and text.strip():
                if _is_complete(text) and not _needs_continuation(text):
                    buf.validate_and_handle()
                    return
                # inside open block → insert newline + auto-indent
                lines = text.splitlines()
                last = lines[-1] if lines else ""
                indent = len(last) - len(last.lstrip())
                extra = 4 if last.rstrip().endswith(":") else 0
                buf.insert_text("\n" + " " * (indent + extra))
            elif cursor_at_end and not text.strip():
                # empty line = submit (ends dedented block)
                buf.validate_and_handle()
            else:
                buf.insert_text("\n")

        return kb

    # ── main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        history_path = Path(self.app.data_dir) / "pyshell-history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        self._session = PromptSession(
            history=FileHistory(str(history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            color_depth=ColorDepth.TRUE_COLOR,
            style=self._STYLE,
            lexer=_LEXER,
            multiline=True,
            key_bindings=self._build_bindings(),
            rprompt=self._rprompt,
            bottom_toolbar=self._toolbar,
            mouse_support=False,
            completer=ThreadedCompleter(_HybridCompleter(self.ex, self._sig_state)),
            complete_while_typing=True,
        )

        print(self.BANNER)

        while self._running:
            try:
                code = await self._session.prompt_async(
                    message=self._in_prompt,
                    prompt_continuation=lambda w, l, c: self._cont_prompt(),
                )
            except KeyboardInterrupt:
                print("\n⚠  KeyboardInterrupt — cell cancelled")
                continue
            except EOFError:
                print("\n👋 Bye.")
                break

            code = code.strip()
            if not code:
                continue

            # ── exit ──
            if code in ("exit", "quit", "exit()", "quit()"):
                print("👋 Bye.")
                break

            # ── shell passthrough ──
            if code.startswith("!"):
                os.system(code[1:])
                continue

            # ── magic commands ──
            if await self.magic.handle(code):
                continue

            # ── execute ──
            try:
                self._count += 1
                result = await self.ex.run_cell(code, live_output=True)
                n = self.ex._count if hasattr(self.ex, "_count") else self._count
                if result and str(result).strip():
                    print(f"\x1b[36mOut[{n}]:\x1b[0m {result}")
            except KeyboardInterrupt:
                print("\n⚠  Interrupted")
            except Exception:
                traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  Flow entry point
# ══════════════════════════════════════════════════════════════════════════════

async def run(app: App, args):
    with contextlib.suppress(Exception):
        set_title(f"PyShell  ·  ToolBoxV2 {app.version}")

    shell = PyShell(app)
    await shell.run()

    with contextlib.suppress(Exception):
        set_title("")

    await app.a_exit()
