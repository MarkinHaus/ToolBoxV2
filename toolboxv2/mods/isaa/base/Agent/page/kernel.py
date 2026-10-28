"""In-process python kernel (trusted: the capsule is NOT enforced inside the namespace).

Tools sind im Kernel als normale Objekte verfügbar:
    await tools.name(**args)   – async (empfohlen)
    tools.name.sync(**args)    – blockierend, wenn kein await möglich
    dir(tools)                 – listet alle verfügbaren Tools
    tools.name                 – zeigt Signatur (repr mit echter inspect.signature)

Dafür wird AgentToolProxy aus live_harness.py genutzt: er baut einen Duck-Type-Agent
der tool_manager.get_function(name) erfüllt, und gibt damit echte Signaturen + __dir__.

Für die Digest-Ausgabe in P4 erscheinen Tools genauso wie Kernel-Variablen.

Ein Subprocess/Docker-Kernel implementiert dieselben drei Methoden: run, digest, close.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import keyword
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .core import Exec


# ================================================================== tool proxy
@dataclass
class _ToolEntry:
    """Minimales Duck-Type für ToolManager.get_function / _registry."""
    name: str
    function: Callable
    description: str = ""


class _ProxyToolManager:
    """Duck-Type für AgentToolProxy: braucht get_function() und _registry.

    Jeder Tool-Aufruf geht durch `gateway` (die echte Capsule-geprüfte Funktion
    aus PageEngine). Dadurch ist die Capsule auch für Kernel-Code wirksam.
    """

    def __init__(self, gateway: Callable, tool_names: Callable[[], list[str]]):
        self._gateway = gateway
        self._get_names = tool_names  # lambda -> aktuelle Liste

    @property
    def _registry(self) -> dict[str, _ToolEntry]:
        """Wird von AgentToolProxy.__dir__ genutzt."""
        return {
            name: _ToolEntry(name=name, function=self._make_fn(name))
            for name in self._get_names()
        }

    def get_function(self, name: str) -> Callable | None:
        """Wird von AgentToolProxy.__getattr__ genutzt."""
        if name not in self._get_names():
            return None
        return self._make_fn(name)

    def _make_fn(self, name: str) -> Callable:
        """Baut eine async Funktion für das Tool – mit **kwargs für echte Signatur."""
        gateway = self._gateway

        async def _tool_fn(**kwargs):
            return await gateway(name, kwargs)

        _tool_fn.__name__ = name
        _tool_fn.__qualname__ = f"tools.{name}"
        _tool_fn.__doc__ = f"Call tool '{name}' via the PageEngine gateway."
        return _tool_fn


class _ProxyAgent:
    """Minimaler Duck-Type-Agent für AgentToolProxy."""

    def __init__(self, tool_manager: _ProxyToolManager):
        self.tool_manager = tool_manager


# ================================================================== kernel
class InProcessKernel:
    trusted = True

    def __init__(self):
        self.ns: dict = {"__name__": "__kernel__"}
        self._media: list = []
        self._show_sink: Callable | None = None
        self._builtin: set = set()
        self._tool_manager: _ProxyToolManager | None = None
        self._actions: _KernelActions | None = None

    def bind(
        self,
        gateway: Callable,          # async (name, args) -> Any  — capsule already checked
        get_tool_names: Callable,   # () -> list[str]
        show: Callable,             # (bytes, mime, caption) -> media_ref dict
        cap_check: Callable | None = None,  # (cap) -> bool; ignored (trusted kernel)
        action_dispatch: Callable | None = None,   # async (name, args) -> str  — page actions
        actions_schema: list | None = None,        # [(name, desc, props, required), ...]
    ) -> None:
        """Bindet den Kernel an die Engine.  Muss vor dem ersten run() aufgerufen werden.

        ``cap_check`` wird nur vom isolierten ``SubprocessKernel`` genutzt (import/fs
        Gating); der In-Process-Kernel ist trusted und ignoriert es.

        ``action_dispatch`` + ``actions_schema`` machen die PAGE-ACTIONS (recall,
        page_edit, plan, ...) im Kernel als normale Funktionen verfügbar — mit
        echter Signatur + Docstring (sichtbar in ``digest``).  Beide Aufruf-Formen
        funktionieren::

            await recall(query="x")          # bare name (injiziert in den Namespace)
            await actions.recall(query="x")  # über das actions-Objekt
            await tools("recall", query="x") # Fallback über die tools-Shim
        """
        self._show_sink = show
        self._tool_manager = _ProxyToolManager(gateway, get_tool_names)
        self._actions = (
            _KernelActions(action_dispatch, actions_schema or [])
            if action_dispatch is not None
            else None
        )
        self.ns.update(
            tools=_KernelTools(self._tool_manager, self._actions),
            show=self._show,
        )
        if self._actions is not None:
            # Expose each page action as a bare callable in the namespace so the
            # model can write `await recall(...)` directly, and as `actions.x`.
            self.ns["actions"] = self._actions
            for name in self._actions.names():
                self.ns[name] = self._actions.fn(name)
        self._builtin = set(self.ns)

    def _show(
        self,
        data: bytes | str | Path,
        mime: str = "image/png",
        caption: str = "",
    ) -> str:
        """show(bytes_or_path, mime, caption) — Bild in den Tail einstellen."""
        if self._show_sink is None:
            raise RuntimeError("kernel not bound: call bind() first")
        part = self._show_sink(
            Path(data).read_bytes() if isinstance(data, (str, Path)) else data,
            mime,
            caption,
        )
        self._media.append(part)
        return part["ref"]

    async def _eval(self, node, mode: str, timeout: float) -> Any:
        value = eval(
            compile(node, "<cell>", mode, flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
            self.ns,
        )
        return (
            await asyncio.wait_for(value, timeout)
            if inspect.iscoroutine(value)
            else value
        )

    async def run(self, code: str, timeout: float) -> Exec:
        self._media = []
        out = io.StringIO()
        try:
            tree = ast.parse(code)
            last = (
                ast.Expression(tree.body.pop().value)
                if tree.body and isinstance(tree.body[-1], ast.Expr)
                else None
            )
            with contextlib.redirect_stdout(out):
                await self._eval(tree, "exec", timeout)
                value = await self._eval(last, "eval", timeout) if last else None
        except asyncio.TimeoutError:
            return Exec(
                False,
                f"error: cell exceeded {timeout}s\n{out.getvalue()}",
                None,
                list(self._media),
            )
        except Exception:
            return Exec(
                False,
                f"error\n{out.getvalue()}{traceback.format_exc(limit=4)}",
                None,
                list(self._media),
            )
        text = "ok" + (
            f"\nstdout:\n{out.getvalue().rstrip()}" if out.getvalue() else ""
        )
        if value is not None:
            text += f"\nresult: {repr(value)[:2000]}"
        if self._media:
            text += "\nmedia: " + ", ".join(p["ref"] for p in self._media)
        return Exec(True, text, bool(value) if last else None, list(self._media))

    def digest(self, max_chars: int) -> str:
        """Namen, Typen, Signaturen für P4 – keine Werte.

        Tools aus dem Proxy erscheinen genauso wie Kernel-Variablen, damit das
        Modell sieht welche Tools direkt im Code aufrufbar sind.
        """
        parts: list[str] = []

        # 1. Kernel-Namespace-Variablen (keine builtins, keine Module)
        for k, v in self.ns.items():
            if k.startswith("_") or k in self._builtin or inspect.ismodule(v):
                continue
            if k == "tools":
                # Tools-Proxy separat behandeln (s.u.)
                continue
            if callable(v):
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(f"{k}{inspect.signature(v)}")
                    continue
                parts.append(f"{k}(...)")
            else:
                size = ""
                with contextlib.suppress(Exception):
                    size = f"[{len(v)}]"
                parts.append(f"{k}: {type(v).__name__}{size}")

        # 2. Verfügbare Tools (zeigt echte Signaturen dank AgentToolProxy)
        if self._tool_manager is not None:
            tool_names = self._tool_manager._get_names()
            if tool_names:
                tool_sigs: list[str] = []
                for name in tool_names[:20]:  # cap bei 20 damit P4 nicht volläuft
                    fn = self._tool_manager.get_function(name)
                    if fn is not None:
                        with contextlib.suppress(Exception):
                            tool_sigs.append(f"tools.{name}{inspect.signature(fn)}")
                            continue
                    tool_sigs.append(f"tools.{name}(...)")
                parts.append("# tools\n" + "\n".join(tool_sigs))

        # 3. Page actions (yellow tools) callable directly in the kernel.
        if self._actions is not None and self._actions.names():
            act_sigs: list[str] = []
            for name in self._actions.names():
                fn = self._actions.fn(name)
                with contextlib.suppress(Exception):
                    act_sigs.append(f"{name}{inspect.signature(fn)}")
                    continue
                act_sigs.append(f"{name}(...)")
            parts.append("# actions (await directly)\n" + "\n".join(act_sigs))

        return "   ".join(parts)[:max_chars]

    async def close(self) -> None:
        self.ns.clear()
        self._tool_manager = None
        self._actions = None


# ================================================================== kernel actions
_JSON_PY = {"string": str, "integer": int, "number": float, "boolean": bool,
            "object": dict, "array": list}


def _make_action_fn(name: str, desc: str, props: dict, required: list,
                    dispatch: Callable) -> Callable:
    """Build an async function for a page action with a REAL signature + doc.

    The signature mirrors the action schema (required params first, optional
    ones default to ``None``) so ``inspect.signature`` / ``repr`` show the model
    exactly how to call it; ``digest`` renders the same.
    """
    params: list[inspect.Parameter] = []
    for key in list(required) + [k for k in props if k not in required]:
        # A JSON param name may be a Python keyword (e.g. cut_tail's "from").
        # Such names are illegal as inspect.Parameter names, so we omit them
        # from the displayed signature — they still arrive via **kwargs at call
        # time and are validated below. The docstring lists them so the model
        # still sees how to call it.
        if keyword.iskeyword(key) or not key.isidentifier():
            continue
        ann = _JSON_PY.get(props.get(key, {}).get("type", "string"), str)
        params.append(inspect.Parameter(
            key,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect._empty if key in required else None,
            annotation=ann,
        ))
    # Accept any extra (keyword-named) params too.
    params.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
    sig = inspect.Signature(params)
    _required = list(required)

    async def _action_fn(**kwargs):
        missing = [k for k in _required if k not in kwargs]
        if missing:
            raise TypeError(f"{name}() missing required: {', '.join(missing)}")
        # Drop unset optionals so action defaults apply.
        clean = {k: v for k, v in kwargs.items() if v is not None or k in _required}
        return await dispatch(name, clean)

    _action_fn.__name__ = name
    _action_fn.__qualname__ = f"actions.{name}"
    _action_fn.__doc__ = desc
    _action_fn.__signature__ = sig  # type: ignore[attr-defined]
    return _action_fn


class _KernelActions:
    """Page actions as callables inside the kernel (recall, page_edit, plan, ...).

    ``actions.recall(query="x")`` and the bare ``recall(query="x")`` both reach
    the engine via the bound ``dispatch`` callback.
    """

    def __init__(self, dispatch: Callable, schema: list):
        self._dispatch = dispatch
        self._fns: dict[str, Callable] = {
            name: _make_action_fn(name, desc, props, required, dispatch)
            for name, desc, props, required in schema
        }

    def names(self) -> list[str]:
        return list(self._fns)

    def fn(self, name: str) -> Callable:
        return self._fns[name]

    def has(self, name: str) -> bool:
        return name in self._fns

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        fns = object.__getattribute__(self, "_fns")
        if name not in fns:
            raise AttributeError(f"no page action {name!r}; have: {list(fns)}")
        return fns[name]

    def __dir__(self):
        return list(self._fns)

    def __repr__(self):
        return f"<actions: {', '.join(self._fns)}>"


# ================================================================== kernel tools
class _KernelTools:
    """Tools-Schnittstelle im Kernel.  Unterstützt beide Aufruf-Formen:

        await tools("name", **args)   – explizit (robust, immer korrekt)
        await tools.name(**args)      – attributiv (hübsch, mit echter Signatur)
        tools.search("query")         – Tool-Suche
        dir(tools)                    – Liste aller Tools
        repr(tools.name)              – Signatur via inspect.signature

    Für die attributive Form wird AgentToolProxy aus live_harness.py genutzt
    (gibt _CallableTool mit echtem repr/signature).  Fehlt live_harness, kommt
    ein schlanker Fallback mit gleicher Semantik.

    Fällt ein Name nicht auf ein Harness-Tool, wird eine Page-ACTION versucht
    (``actions``), damit ``await tools("recall", ...)`` nicht mit KeyError bricht.
    """

    def __init__(self, tm: _ProxyToolManager, actions: "_KernelActions | None" = None):
        self._tm = tm
        self._actions = actions
        self._proxy = None
        try:
            from toolboxv2.mods.isaa.base.Agent.live_harness import AgentToolProxy
            self._proxy = AgentToolProxy(_ProxyAgent(tm))
        except ImportError:
            self._proxy = None

    # ---- explizite Form: await tools("name", **args)
    async def __call__(self, name: str, **args):
        fn = self._tm.get_function(name)
        if fn is None:
            # Fall through to page actions before failing (fixes KeyError: no tool 'recall').
            if self._actions is not None and self._actions.has(name):
                return await self._actions.fn(name)(**args)
            raise KeyError(f"no tool {name!r}")
        return await fn(**args)

    # ---- attributive Form: tools.name  -> _CallableTool oder async fn
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "search":
            return self._search
        # Page actions win only if not a real harness tool.
        tm = object.__getattribute__(self, "_tm")
        actions = object.__getattribute__(self, "_actions")
        if tm.get_function(name) is None and actions is not None and actions.has(name):
            return actions.fn(name)
        if self._proxy is not None:
            return getattr(self._proxy, name)  # _CallableTool mit echtem repr
        fn = tm.get_function(name)
        if fn is None:
            available = list(tm._registry)[:40]
            raise AttributeError(f"Tool '{name}' not found. Available: {available}")
        return fn

    async def _search(self, query: str, k: int = 8):
        """Tool-Suche – gibt [(name, doc), ...] zurück."""
        # _ProxyToolManager hat keinen Suchindex; die Namen reichen für den Kernel.
        names = self._tm._get_names()
        return [(n, "") for n in names if query.lower() in n.lower()][:k]

    def __dir__(self):
        return list(self._tm._registry)

    def __repr__(self):
        n = len(list(self._tm._registry))
        return f"<tools: {n} available — dir(tools) to list>"
