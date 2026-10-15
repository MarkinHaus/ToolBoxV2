"""Child-process worker for :class:`SubprocessKernel` (Phase 2).

Self-contained, stdlib-only.  Launched as a plain script (``python -I
kernel_worker.py``) so it does not depend on package context.

It owns the real python namespace.  A background *reader thread* consumes stdin
and routes two kinds of lines:
  * ``reply_to`` lines  -> resolve a pending RPC (thread-safe queue per id);
  * ``op`` lines        -> queued for the main thread to execute.

RPC to the parent is *synchronous blocking* from the worker's point of view
(send a request line, wait on a per-id queue for the reply).  This lets the
synchronous ``import`` hook gate modules, while ``tools`` / ``fs`` expose async
wrappers that run the blocking RPC in a thread executor so ``await`` works.
"""
from __future__ import annotations

import ast
import asyncio
import base64
import builtins
import contextlib
import inspect
import io
import itertools
import json
import queue
import sys
import threading
import traceback

_OUT = sys.stdout
_LOCK = threading.Lock()
_rpc_ids = itertools.count(1)
_replies: dict[int, queue.Queue] = {}
_replies_lock = threading.Lock()
_ops: queue.Queue = queue.Queue()


def _write(obj: dict) -> None:
    with _LOCK:
        _OUT.write(json.dumps(obj) + "\n")
        _OUT.flush()


def _reader() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "reply_to" in msg:
            rid = msg["reply_to"]
            with _replies_lock:
                q = _replies.get(rid)
            if q is not None:
                q.put(msg)
        elif "op" in msg:
            _ops.put(msg)


def _rpc(payload: dict):
    """Send an RPC request and block until the parent replies.  Raises on error."""
    rid = next(_rpc_ids)
    q: queue.Queue = queue.Queue(maxsize=1)
    with _replies_lock:
        _replies[rid] = q
    try:
        _write({**payload, "id": rid})
        reply = q.get()  # block
    finally:
        with _replies_lock:
            _replies.pop(rid, None)
    if not reply.get("ok", False):
        raise PermissionError(reply.get("error") or "rpc denied")
    return reply.get("value")


# ================================================================== import hook
_real_import = builtins.__import__
# Imports performed by the worker's own machinery (and by granted cell imports
# while resolving submodules) must not be gated.  We only gate imports that
# originate from *cell* code, tracked by a thread-local "in cell" flag.
_import_guard = threading.local()


def _checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Cell-scoped ``__import__``: gate top-level module on an ``import:`` cap.

    Unlike a ``sys.meta_path`` hook, this checks the capability *every* time the
    cell imports a name, even if the module is already in ``sys.modules`` (the
    worker itself pulls in e.g. ``socket`` via asyncio, but the cell must still
    be denied without the capability).

    Only imports whose *calling frame* is the cell source (``<cell>``) are gated;
    imports done by the asyncio/runtime machinery (even while a cell ``await`` is
    in flight) are left alone.
    """
    if level == 0 and name and _from_cell_frame():
        top = name.split(".")[0]
        checking = getattr(_import_guard, "checking", None)
        if checking is None:
            checking = _import_guard.checking = set()
        if top not in checking:
            checking.add(top)
            try:
                _rpc({"rpc": "import.check", "name": name})
            finally:
                checking.discard(top)
    return _real_import(name, globals, locals, fromlist, level)


def _from_cell_frame() -> bool:
    """True if the *immediate* caller is the compiled cell (``<cell>``).

    An ``import`` statement written in the cell compiles to a ``__import__`` call
    whose direct calling frame is ``<cell>``.  Machinery imports triggered deep
    inside an awaited coroutine have a stdlib frame as their immediate caller
    (even though ``<cell>`` is still further up the await chain), so checking the
    direct caller only — not the whole stack — cleanly separates the two.
    """
    try:
        caller = sys._getframe(2)  # 0=_from_cell_frame, 1=_checked_import, 2=caller
    except ValueError:
        return False
    return caller is not None and caller.f_code.co_filename == "<cell>"


# ================================================================== kernel tools
class _Tools:
    async def __call__(self, name, **args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _rpc({"rpc": "tool.call", "name": name, "args": args}))

    async def search(self, query, k=8):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _rpc({"rpc": "tool.search", "query": query, "k": k}))


class _FS:
    async def read(self, path):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _rpc({"rpc": "fs.read", "path": path}))

    async def write(self, path, data):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: _rpc({"rpc": "fs.write", "path": path, "data": data}))


def _show(data, mime="image/png", caption=""):
    raw = data if isinstance(data, (bytes, bytearray)) else bytes(data)
    return _rpc({"rpc": "media.put",
                 "data_b64": base64.b64encode(raw).decode(),
                 "mime": mime, "caption": caption})["ref"]


# ================================================================== cell runner
_NS: dict = {}


def _init_ns() -> None:
    _NS.clear()
    _NS.update(__name__="__kernel__", tools=_Tools(), fs=_FS(), show=_show)
    _NS["__builtin_names__"] = set(_NS)


async def _eval(node, mode):
    value = eval(compile(node, "<cell>", mode,
                         flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), _NS)
    return await value if inspect.iscoroutine(value) else value


async def _run_cell(code: str) -> dict:
    out = io.StringIO()
    media: list = []
    truthy = None
    value = None
    try:
        tree = ast.parse(code)
        last = (ast.Expression(tree.body.pop().value)
                if tree.body and isinstance(tree.body[-1], ast.Expr) else None)
        with contextlib.redirect_stdout(out):
            await _eval(tree, "exec")
            value = await _eval(last, "eval") if last else None
    except Exception:
        return {"ok": False, "truthy": None, "media": media,
                "text": f"error\n{out.getvalue()}{traceback.format_exc(limit=4)}"}
    text = "ok" + (f"\nstdout:\n{out.getvalue().rstrip()}" if out.getvalue() else "")
    if value is not None:
        text += f"\nresult: {repr(value)[:2000]}"
    truthy = bool(value) if last else None
    return {"ok": True, "truthy": truthy, "media": media, "text": text}


def _digest(max_chars: int = 2000) -> str:
    builtin = _NS.get("__builtin_names__", set())
    parts: list[str] = []
    for k, v in list(_NS.items()):
        if k.startswith("_") or k in builtin or inspect.ismodule(v):
            continue
        if callable(v):
            try:
                parts.append(f"{k}{inspect.signature(v)}")
                continue
            except (TypeError, ValueError):
                parts.append(f"{k}(...)")
        else:
            size = ""
            try:
                size = f"[{len(v)}]"
            except Exception:
                pass
            parts.append(f"{k}: {type(v).__name__}{size}")
    return "   ".join(parts)[:max_chars]


def _main() -> None:
    builtins.__import__ = _checked_import
    _init_ns()
    threading.Thread(target=_reader, daemon=True).start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        msg = _ops.get()
        op = msg.get("op")
        if op == "shutdown":
            break
        if op == "run":
            try:
                res = loop.run_until_complete(_run_cell(msg.get("code", "")))
            except Exception as e:  # pragma: no cover - defensive
                res = {"ok": False, "truthy": None, "media": [],
                       "text": f"error: worker crash {type(e).__name__}: {e}"}
            res["result"] = msg.get("id")
            res["digest"] = _digest()
            _write(res)
        elif op == "digest":
            _write({"result": msg.get("id"), "ok": True, "text": _digest(
                msg.get("max_chars", 2000)), "truthy": None, "media": []})
    with contextlib.suppress(Exception):
        loop.close()


if __name__ == "__main__":
    try:
        _main()
    except KeyboardInterrupt:
        pass
