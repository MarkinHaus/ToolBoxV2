"""
Live Harness — persistent Jupyter-like execution environment for FlowAgent.

Replaces the MockIPython path of executors.py for the `exec_code` tool.

Design goals
------------
1. Real top-level await via ``ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`` (same mechanism
   IPython uses) instead of an AST wrapper function. The wrapper approach made
   every assignment *local* to the wrapper, so state silently vanished between
   cells whenever a cell contained ``await``.
2. Persistent namespace + persistent harness package on disk, keyed by
   (agent, session). Survives process restarts.
3. Two modes:
   - ``write``   → the agent extends its own harness (writes importable modules
                   and registers new tools into its ToolManager)
   - ``execute`` → the agent runs code / calls its own harness + tools
4. Structured result dicts. No string-format round trip, no swallowed errors,
   no LLM call anywhere in this path.

Author: ToolBoxV2 / ISAA
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import inspect
import io
import json
import os
import pickle
import sys
import textwrap
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_NO_RESULT = object()
_RESULT_KEY = "__cell_result__"

# Names that must never be pickled into the persisted namespace snapshot.
_NON_PERSIST = {
    "app", "agent", "tools", "mods", "session", "vfs", "tool",
    "auto_install", "harness", "__builtins__", "__loader__", "__spec__",
}


# =============================================================================
# TOOL DECORATOR (used by the agent inside `write` mode)
# =============================================================================

def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: list[str] | str | None = None,
    flags: dict[str, bool] | None = None,
):
    """Mark a function in a harness module for registration as an agent tool.

    Usage inside `exec_code(mode="write")`::

        @tool(description="Fetch and summarise a URL", category=["web"])
        async def fetch_title(url: str) -> str:
            ...
    """

    def _wrap(fn: Callable) -> Callable:
        fn._tb_tool = {
            "name": name or fn.__name__,
            "description": description or (fn.__doc__ or f"Tool: {fn.__name__}").strip(),
            "category": category or ["harness"],
            "flags": flags or {},
        }
        return fn

    if _func is not None:
        return _wrap(_func)
    return _wrap


# =============================================================================
# PROXIES
# =============================================================================

class AgentToolProxy:
    """Access agent tools from executed code.

    ``await tools.name(...)``  → always correct (recommended).
    ``tools.name(...)``        → sync convenience; blocks on a worker loop.

    The old proxy fired coroutines into a background task in async context and
    returned the Task, so results were lost and failures were invisible.
    """

    def __init__(self, agent):
        self._agent = agent

    def _registry(self) -> dict:
        tm = getattr(self._agent, "tool_manager", None)
        return getattr(tm, "_registry", {}) or {}

    def __getattr__(self, tool_name: str):
        if tool_name.startswith("_"):
            raise AttributeError(tool_name)
        tm = getattr(self._agent, "tool_manager", None)
        fn = tm.get_function(tool_name) if tm else None
        if fn is None:
            raise AttributeError(
                f"Tool '{tool_name}' not found. Available: {sorted(self._registry())[:40]}"
            )
        return _CallableTool(tool_name, fn)

    def __getitem__(self, tool_name: str):
        return self.__getattr__(tool_name)

    def __dir__(self):
        return sorted(self._registry())

    def __repr__(self):
        return f"<tools: {len(self._registry())} available — dir(tools) to list>"


class _CallableTool:
    """Result of ``tools.x``.

    Jupyter semantics: an async tool returns its coroutine, so the natural call
    is ``await tools.x(...)``. A sync tool returns its value directly.
    ``tools.x.sync(...)`` blocks on a worker loop when you cannot await.
    """

    __slots__ = ("_name", "_fn")

    def __init__(self, name: str, fn: Callable):
        self._name = name
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def sync(self, *args, **kwargs):
        res = self._fn(*args, **kwargs)
        if not inspect.isawaitable(res):
            return res
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(res)
        return _run_coro_blocking(res)

    def __repr__(self):
        kind = "async " if asyncio.iscoroutinefunction(self._fn) else ""
        return f"<{kind}tool {self._name}{_safe_sig(self._fn)}>"


def _safe_sig(fn) -> str:
    try:
        return str(inspect.signature(fn))
    except Exception:
        return "(...)"


def _run_coro_blocking(coro, timeout: float | None = None):
    box: dict[str, Any] = {}

    def _runner():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised on caller thread
            box["e"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError("Blocking tool call exceeded timeout; use `await tools.x()`")
    if "e" in box:
        raise box["e"]
    return box.get("v")


class ModProxy:
    """Gated access to ToolBox mods.

    ``mods.DB`` → ``app.get_mod("DB")``. Only mods on the allowlist resolve,
    unless the session is privileged (self / tb_admin), which gets everything.
    """

    def __init__(self, app, allowed: set[str] | None, privileged: bool):
        self._app = app
        self._allowed = allowed or set()
        self._privileged = privileged
        self._cache: dict[str, Any] = {}

    def __getattr__(self, mod_name: str):
        if mod_name.startswith("_"):
            raise AttributeError(mod_name)
        if not self._privileged and mod_name not in self._allowed:
            raise PermissionError(
                f"Mod '{mod_name}' not allowed for this agent. "
                f"Allowed: {sorted(self._allowed) or 'none'}"
            )
        if mod_name not in self._cache:
            self._cache[mod_name] = self._app.get_mod(mod_name)
        return self._cache[mod_name]

    def __getitem__(self, mod_name: str):
        return self.__getattr__(mod_name)

    def __dir__(self):
        if self._privileged:
            try:
                return sorted(self._app.get_all_mods())
            except Exception:
                return []
        return sorted(self._allowed)

    def __repr__(self):
        scope = "all (privileged)" if self._privileged else sorted(self._allowed)
        return f"<mods: {scope}>"


# =============================================================================
# CELL COMPILATION / EXECUTION
# =============================================================================

def compile_cell(code: str) -> tuple[Any, bool]:
    """Compile a notebook cell.

    Returns ``(code_object, captures_result)``. The last bare expression is
    rewritten into an assignment to ``__cell_result__`` so falsy values
    (``0``, ``""``, ``False``, ``[]``) survive — the old path coerced them to
    the literal string ``"stdout"``.
    """
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\n", "\n")
    code = textwrap.dedent(code)

    tree = ast.parse(code)
    captures = False
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id=_RESULT_KEY, ctx=ast.Store())],
            value=last.value,
        )
        ast.fix_missing_locations(tree)
        captures = True

    co = compile(tree, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    return co, captures


@dataclass
class CellResult:
    success: bool
    output: str = ""
    error: str | None = None
    result_repr: str | None = None
    result: Any = None
    execution_count: int = 0
    duration_ms: int = 0
    new_names: list[str] = field(default_factory=list)

    def to_dict(self, include_object: bool = False) -> dict:
        d = {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "result": self.result_repr,
            "execution_count": self.execution_count,
            "duration_ms": self.duration_ms,
        }
        if self.new_names:
            d["new_names"] = self.new_names
        if include_object:
            d["result_object"] = self.result
        return d


# =============================================================================
# LIVE SESSION
# =============================================================================

class LiveSession:
    """Persistent interpreter + harness package for one (agent, session)."""

    _REGISTRY: dict[str, "LiveSession"] = {}
    _REG_LOCK = threading.Lock()

    def __init__(
        self,
        agent=None,
        session_id: str = "default",
        root: Path | None = None,
        privileged: bool | None = None,
        allowed_mods: set[str] | None = None,
        persist: bool = True,
    ):
        self.agent = agent
        self.agent_name = _agent_name(agent)
        self.session_id = session_id or "default"
        self.persist = persist

        if privileged is None:
            privileged = self.agent_name in ("self", "tb_admin")
        self.privileged = bool(privileged)
        self.allowed_mods = set(allowed_mods or set())

        self.root = Path(root) if root else _default_root() / self.agent_name / self.session_id
        self.harness_dir = self.root / "harness"
        self.work_dir = self.root / "work"
        for d in (self.harness_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.user_ns: dict[str, Any] = {}
        self.history: list[dict] = []
        self.execution_count = 0
        self.registered_tools: dict[str, list[str]] = {}   # module -> tool names
        self._lock = asyncio.Lock()
        self._bootstrapped = False

        self._reset_ns()

    # -- registry -----------------------------------------------------------
    @classmethod
    def get(cls, agent=None, session_id: str = "default", **kw) -> "LiveSession":
        key = f"{_agent_name(agent)}::{session_id or 'default'}"
        with cls._REG_LOCK:
            inst = cls._REGISTRY.get(key)
            if inst is None:
                inst = cls(agent=agent, session_id=session_id, **kw)
                cls._REGISTRY[key] = inst
            elif agent is not None:
                inst.agent = agent
                inst.user_ns["agent"] = agent
                inst.user_ns["tools"] = AgentToolProxy(agent)
            return inst

    @classmethod
    def drop(cls, agent=None, session_id: str = "default"):
        key = f"{_agent_name(agent)}::{session_id or 'default'}"
        with cls._REG_LOCK:
            cls._REGISTRY.pop(key, None)

    # -- namespace ----------------------------------------------------------
    def _reset_ns(self):
        ns: dict[str, Any] = {
            "__name__": "__live__",
            "__builtins__": __builtins__,
            "__file__": str(self.work_dir / "cell.py"),
            "tool": tool,
            "HARNESS_DIR": str(self.harness_dir),
            "WORK_DIR": str(self.work_dir),
        }
        if self.agent is not None:
            ns["agent"] = self.agent
            ns["tools"] = AgentToolProxy(self.agent)
        app = _get_app()
        if app is not None:
            ns["mods"] = ModProxy(app, self.allowed_mods, self.privileged)
            if self.privileged:
                ns["app"] = app
        sess = _agent_session(self.agent)
        if sess is not None:
            ns["session"] = sess
            if getattr(sess, "vfs", None) is not None:
                ns["vfs"] = sess.vfs
        try:
            from toolboxv2.mods.isaa.CodingAgent.live import auto_install
            ns["auto_install"] = auto_install
        except Exception:
            pass
        self.user_ns = ns

    # -- bootstrap ----------------------------------------------------------
    def bootstrap(self) -> dict:
        """Load persisted namespace + import every harness module."""
        if self._bootstrapped:
            return {"harness_modules": self.list_harness(), "restored": []}
        self._bootstrapped = True

        if str(self.harness_dir) not in sys.path:
            sys.path.insert(0, str(self.harness_dir))

        restored: list[str] = []
        state_file = self.root / "state.pkl"
        if self.persist and state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    blob = pickle.load(f)
                self.user_ns.update(blob.get("ns", {}))
                self.execution_count = blob.get("execution_count", 0)
                restored = sorted(blob.get("ns", {}))
            except Exception:
                pass

        hist_file = self.root / "history.jsonl"
        if self.persist and hist_file.exists():
            try:
                with open(hist_file, encoding="utf-8") as f:
                    self.history = [json.loads(l) for l in f if l.strip()][-200:]
            except Exception:
                self.history = []

        loaded = []
        for mod in self.list_harness():
            r = self.load_harness_module(mod, register=True)
            if r.get("success"):
                loaded.append(mod)
        return {"harness_modules": loaded, "restored": restored}

    def _save_state(self):
        if not self.persist:
            return
        keep: dict[str, Any] = {}
        for k, v in self.user_ns.items():
            if k in _NON_PERSIST or k.startswith("__"):
                continue
            if inspect.ismodule(v) or inspect.isfunction(v) or inspect.isclass(v):
                continue
            try:
                pickle.dumps(v)
                keep[k] = v
            except Exception:
                continue
        try:
            with open(self.root / "state.pkl", "wb") as f:
                pickle.dump({"ns": keep, "execution_count": self.execution_count}, f)
        except Exception:
            pass

    def _append_history(self, entry: dict):
        self.history.append(entry)
        if not self.persist:
            return
        try:
            with open(self.root / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    # -- execute mode -------------------------------------------------------
    async def execute(
        self,
        code: str,
        timeout: float = 120.0,
        live_output: bool = False,
        max_output_chars: int = 8000,
    ) -> CellResult:
        self.bootstrap()
        async with self._lock:
            return await self._execute_unlocked(code, timeout, live_output, max_output_chars)

    async def _execute_unlocked(self, code, timeout, live_output, max_output_chars) -> CellResult:
        t0 = time.time()
        self.execution_count += 1
        before = set(self.user_ns)

        try:
            co, captures = compile_cell(code)
        except SyntaxError as e:
            res = CellResult(
                success=False,
                error=_format_syntax_error(code, e),
                execution_count=self.execution_count,
                duration_ms=int((time.time() - t0) * 1000),
            )
            self._append_history({"n": self.execution_count, "code": code, "ok": False,
                                  "error": res.error[:500]})
            return res

        out_buf, err_buf = io.StringIO(), io.StringIO()
        self.user_ns.pop(_RESULT_KEY, None)
        result = _NO_RESULT
        error = None

        cwd_before = os.getcwd()
        try:
            with redirect_stdout(_Tee(out_buf, live_output and sys.__stdout__)), \
                 redirect_stderr(_Tee(err_buf, live_output and sys.__stderr__)):
                try:
                    os.chdir(self.work_dir)
                except Exception:
                    pass
                try:
                    ret = eval(co, self.user_ns)  # noqa: S307 - deliberate
                    if inspect.isawaitable(ret):
                        await asyncio.wait_for(ret, timeout=timeout)
                    if captures:
                        result = self.user_ns.pop(_RESULT_KEY, _NO_RESULT)
                except asyncio.TimeoutError:
                    error = f"TimeoutError: cell exceeded {timeout}s"
                except KeyboardInterrupt:
                    error = "KeyboardInterrupt: execution stopped"
                except BaseException as e:  # noqa: BLE001 - notebook semantics
                    error = _format_runtime_error(e)
        finally:
            try:
                os.chdir(cwd_before)
            except Exception:
                pass

        stdout_s = _clip(out_buf.getvalue(), max_output_chars)
        stderr_s = _clip(err_buf.getvalue(), max_output_chars)
        output = stdout_s
        if stderr_s:
            output = (output + "\n[stderr]\n" + stderr_s).strip()

        if result is not _NO_RESULT and result is not None:
            self.user_ns["_"] = result

        res = CellResult(
            success=error is None,
            output=output,
            error=error,
            result_repr=None if result is _NO_RESULT else _clip(repr(result), 2000),
            result=None if result is _NO_RESULT else result,
            execution_count=self.execution_count,
            duration_ms=int((time.time() - t0) * 1000),
            new_names=sorted(n for n in set(self.user_ns) - before if not n.startswith("__")),
        )
        self._append_history({
            "n": self.execution_count, "code": _clip(code, 2000), "ok": res.success,
            "error": (res.error or "")[:500], "ts": time.time(),
        })
        self._save_state()
        return res

    # -- write mode ---------------------------------------------------------
    def list_harness(self) -> list[str]:
        return sorted(
            p.stem for p in self.harness_dir.glob("*.py") if not p.stem.startswith("_")
        )

    def read_harness(self, module: str) -> str | None:
        p = self.harness_dir / f"{_safe_mod(module)}.py"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def write_harness(
        self,
        module: str,
        code: str,
        append: bool = False,
        register: bool = True,
        analyze: bool = True,
    ) -> dict:
        """Persist a harness module, import it, register its @tool functions."""
        self.bootstrap()
        module = _safe_mod(module)
        path = self.harness_dir / f"{module}.py"
        code = textwrap.dedent(code)

        try:
            ast.parse(code if not append else (self.read_harness(module) or "") + "\n" + code)
        except SyntaxError as e:
            return {"success": False, "mode": "write", "module": module,
                    "error": _format_syntax_error(code, e)}

        previous = path.read_text(encoding="utf-8") if path.exists() else None
        if append and previous is not None:
            new_code = previous.rstrip() + "\n\n" + code.lstrip("\n")
        else:
            new_code = code
        if not new_code.lstrip().startswith(("from ", "import ", "#", '"""')):
            new_code = "from toolboxv2.mods.isaa.base.Agent.live_harness import tool\n\n" + new_code
        elif "import tool" not in new_code and "@tool" in new_code:
            new_code = "from toolboxv2.mods.isaa.base.Agent.live_harness import tool\n" + new_code

        path.write_text(new_code, encoding="utf-8")

        load = self.load_harness_module(module, register=register)
        if not load.get("success"):
            # rollback so a broken module can never poison the next bootstrap
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(previous, encoding="utf-8")
            load["rolled_back"] = True
            load["mode"] = "write"
            load["module"] = module
            return load

        out = {
            "success": True,
            "mode": "write",
            "module": module,
            "path": str(path),
            "lines": new_code.count("\n") + 1,
            "exports": load.get("exports", []),
            "registered_tools": load.get("registered_tools", []),
            "import_as": f"import {module}",
        }
        if analyze:
            rep = _static_analyze(path)
            if rep:
                out["static_analysis"] = rep
        self._append_history({"n": self.execution_count, "write": module,
                              "ok": True, "ts": time.time()})
        return out

    def load_harness_module(self, module: str, register: bool = True) -> dict:
        import importlib

        module = _safe_mod(module)
        if str(self.harness_dir) not in sys.path:
            sys.path.insert(0, str(self.harness_dir))
        try:
            if module in sys.modules:
                mod = importlib.reload(sys.modules[module])
            else:
                mod = importlib.import_module(module)
        except BaseException as e:  # noqa: BLE001
            return {"success": False, "error": _format_runtime_error(e)}

        exports, registered = [], []
        for attr, obj in vars(mod).items():
            if attr.startswith("_"):
                continue
            if getattr(obj, "__module__", None) != module:
                continue
            if callable(obj) or not inspect.ismodule(obj):
                exports.append(attr)
            self.user_ns[attr] = obj
            meta = getattr(obj, "_tb_tool", None)
            if register and meta and self.agent is not None:
                try:
                    self.agent.add_tool(
                        tool_func=obj,
                        name=meta["name"],
                        description=meta["description"],
                        category=meta["category"],
                        flags=meta["flags"],
                    )
                    registered.append(meta["name"])
                except Exception as e:
                    registered.append(f"{meta['name']} (FAILED: {e})")
        self.user_ns[module] = mod
        if registered:
            self.registered_tools[module] = [r for r in registered if "FAILED" not in r]
        return {"success": True, "exports": sorted(exports), "registered_tools": registered}

    def remove_harness(self, module: str) -> dict:
        module = _safe_mod(module)
        path = self.harness_dir / f"{module}.py"
        if not path.exists():
            return {"success": False, "error": f"No harness module '{module}'"}
        for name in self.registered_tools.pop(module, []):
            try:
                self.agent.remove_tool(name)
            except Exception:
                pass
        path.unlink()
        sys.modules.pop(module, None)
        self.user_ns.pop(module, None)
        return {"success": True, "removed": module}

    # -- introspection ------------------------------------------------------
    def state(self) -> dict:
        self.bootstrap()
        user_vars, user_funcs = [], []
        for k, v in self.user_ns.items():
            if k.startswith("__") or k in _NON_PERSIST:
                continue
            if inspect.isfunction(v) or inspect.isclass(v):
                user_funcs.append(f"{k}{_safe_sig(v)}" if inspect.isfunction(v) else f"class {k}")
            elif not inspect.ismodule(v):
                user_vars.append(f"{k}: {type(v).__name__}")
        tools_available = []
        if self.agent is not None and getattr(self.agent, "tool_manager", None):
            try:
                tools_available = sorted(self.agent.tool_manager.list_names())
            except Exception:
                pass
        return {
            "agent": self.agent_name,
            "session": self.session_id,
            "privileged": self.privileged,
            "execution_count": self.execution_count,
            "harness_modules": self.list_harness(),
            "harness_tools": {k: v for k, v in self.registered_tools.items()},
            "variables": sorted(user_vars)[:60],
            "functions": sorted(user_funcs)[:60],
            "tools_available": tools_available,
            "mods": "all (privileged)" if self.privileged else sorted(self.allowed_mods),
            "work_dir": str(self.work_dir),
            "harness_dir": str(self.harness_dir),
        }

    def reset(self, wipe_harness: bool = False) -> dict:
        for module in list(self.registered_tools):
            if wipe_harness:
                self.remove_harness(module)
        self._reset_ns()
        self.execution_count = 0
        self._bootstrapped = False
        try:
            (self.root / "state.pkl").unlink(missing_ok=True)
        except Exception:
            pass
        return {"success": True, "reset": True, "wiped_harness": wipe_harness}


# =============================================================================
# HELPERS
# =============================================================================

class _Tee:
    """Buffer writer that optionally mirrors to a real stream."""

    def __init__(self, buffer, mirror=None):
        self._buf = buffer
        self._mirror = mirror or None

    def write(self, data):
        self._buf.write(data)
        if self._mirror:
            try:
                self._mirror.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        try:
            self._buf.flush()
        except Exception:
            pass
        if self._mirror:
            try:
                self._mirror.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _clip(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated {len(s) - n} chars]"


def _safe_mod(name: str) -> str:
    name = (name or "module").strip().replace("-", "_").replace(" ", "_")
    name = "".join(c for c in name if c.isalnum() or c == "_")
    if not name or name[0].isdigit():
        name = f"h_{name}"
    return name


def _format_syntax_error(code: str, e: SyntaxError) -> str:
    lines = code.splitlines()
    if e.lineno and e.lineno <= len(lines):
        arrow = " " * ((e.offset or 1) - 1) + "^"
        return f"SyntaxError line {e.lineno}: {e.msg}\n{lines[e.lineno - 1]}\n{arrow}"
    return f"SyntaxError: {e.msg}"


def _format_runtime_error(e: BaseException) -> str:
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    # strip harness frames so the agent sees its own code first
    keep = [f for f in tb if "live_harness.py" not in f]
    return "".join(keep or tb).strip()


def _agent_name(agent) -> str:
    if agent is None:
        return "anonymous"
    amd = getattr(agent, "amd", None)
    return getattr(amd, "name", None) or getattr(agent, "name", None) or "anonymous"


def _agent_session(agent):
    if agent is None:
        return None
    try:
        sm = getattr(agent, "session_manager", None)
        sid = getattr(agent, "active_session", None)
        if sm and sid:
            return sm.get(sid)
    except Exception:
        pass
    return None


def _get_app():
    try:
        from toolboxv2 import get_app
        return get_app()
    except Exception:
        return None


def _default_root() -> Path:
    app = _get_app()
    base = getattr(app, "appdata", None) if app else None
    return Path(base or Path.home() / ".toolboxv2") / ".live_harness"


def _static_analyze(path: Path) -> str | None:
    try:
        from toolboxv2.utils.extras.code_analyzer.tb_analyze import static_analyze
        rep = static_analyze(str(path))
        return _clip(rep if isinstance(rep, str) else json.dumps(rep, default=str), 1500)
    except Exception:
        return None
