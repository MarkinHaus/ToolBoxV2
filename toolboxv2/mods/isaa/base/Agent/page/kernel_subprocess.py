"""Isolated python kernel in a child process (Phase 2).

Same interface as ``InProcessKernel`` — ``bind`` / ``run`` / ``digest`` /
``close`` — but the python namespace lives in a *separate* process reached over
JSON-Lines on stdin/stdout.  That buys three things the in-process kernel cannot
give:

  * a synchronous endless loop can be *killed* (timeout -> SIGKILL + restart);
  * ``import`` is gated: the child's import hook asks the parent whether an
    ``import:<name>`` capability is granted before the module loads;
  * host-filesystem access is only possible through ``fs.read`` / ``fs.write``
    RPC, which the parent gates on ``read:`` / ``write:`` capabilities.

``tools(...)`` round-trips through the parent gateway, so the capsule is
enforced for kernel code exactly like in the in-process kernel.

Protocol (one JSON object per line):

    parent -> child : {"op":"run","id":N,"code":"..."}
                      {"op":"digest","id":N,"max_chars":M}
    child  -> parent: {"rpc":"tool.call","id":R,"name":..,"args":..}   (request)
                      {"rpc":"tool.search","id":R,"query":..,"k":..}
                      {"rpc":"import.check","id":R,"name":..}
                      {"rpc":"fs.read","id":R,"path":..}
                      {"rpc":"fs.write","id":R,"path":..,"data":..}
                      {"rpc":"media.put","id":R,"data_b64":..,"mime":..,"caption":..}
    parent -> child : {"reply_to":R,"ok":bool,"value":..,"error":..}   (reply)
    child  -> parent: {"result":N,"ok":bool,"text":..,"truthy":..,"media":[..]}

``core.py`` never imports this module directly; the engine only knows the
``Kernel`` protocol.  This file is stdlib-only and imports nothing from
toolboxv2, so it is unit-testable isolated (iso_pkg).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from .core import Exec

_WORKER = str(Path(__file__).with_name("kernel_worker.py"))


class SubprocessKernel:
    trusted = False

    def __init__(self):
        self._gateway: Callable | None = None
        self._get_names: Callable | None = None
        self._show: Callable | None = None
        self._cap_check: Callable[[str], bool] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._rpc_id = 0
        self._op_id = 0
        self._closed = False
        self._pump_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._op_future: asyncio.Future | None = None
        self._op_target: int = -1
        self._lock = asyncio.Lock()
        self._digest_cache: str = ""

    # ------------------------------------------------------------ bind
    def bind(
        self,
        gateway: Callable,          # async (name, args) -> Any  (capsule-checked)
        get_tool_names: Callable,   # () -> list[str]
        show: Callable,             # (bytes, mime, caption) -> media_ref dict
        cap_check: Callable[[str], bool] | None = None,  # (cap) -> bool for import/fs
    ) -> None:
        self._gateway = gateway
        self._get_names = get_tool_names
        self._show = show
        self._cap_check = cap_check or (lambda cap: False)

    # ------------------------------------------------------------ process mgmt
    async def _ensure_proc(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", _WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
        )
        self._pending.clear()
        self._op_future = None
        self._op_target = -1
        self._pump_task = asyncio.ensure_future(self._pump())

    async def _pump(self) -> None:
        """Read child lines; dispatch RPC requests, resolve op results."""
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if "rpc" in msg:
                    await self._handle_rpc(msg)
                elif "result" in msg:
                    if self._op_future and not self._op_future.done() \
                            and msg["result"] == self._op_target:
                        self._op_future.set_result(msg)
        except (asyncio.CancelledError, Exception):
            pass

    async def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    # ------------------------------------------------------------ RPC handlers
    async def _handle_rpc(self, msg: dict) -> None:
        rpc, rid = msg.get("rpc"), msg.get("id")
        ok, value, error = True, None, None
        try:
            if rpc == "tool.call":
                value = await self._gateway(msg["name"], msg.get("args") or {})
            elif rpc == "tool.search":
                names = self._get_names() if self._get_names else []
                q = (msg.get("query") or "").lower()
                value = [n for n in names if q in n.lower()][: msg.get("k", 8)]
            elif rpc == "import.check":
                name = msg.get("name", "")
                top = name.split(".")[0]
                ok = self._cap_allows(f"import:{name}") or self._cap_allows(
                    f"import:{top}")
                if not ok:
                    error = f"import:{top} not granted - capability denied"
            elif rpc == "fs.read":
                path = msg.get("path", "")
                if not self._cap_allows(f"read:host:{path}"):
                    raise PermissionError(f"read:host:{path} not granted")
                value = Path(path).read_text("utf-8")
            elif rpc == "fs.write":
                path = msg.get("path", "")
                if not self._cap_allows(f"write:host:{path}"):
                    raise PermissionError(f"write:host:{path} not granted")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(msg.get("data", ""), "utf-8")
                value = len(msg.get("data", ""))
            elif rpc == "media.put":
                data = base64.b64decode(msg.get("data_b64", ""))
                part = self._show(data, msg.get("mime", "image/png"),
                                  msg.get("caption", ""))
                value = part
            else:
                ok, error = False, f"unknown rpc {rpc!r}"
        except Exception as e:  # surface as an RPC error -> raised in the child
            ok, error, value = False, f"{type(e).__name__}: {e}", None
        await self._send({"reply_to": rid, "ok": ok, "value": value, "error": error})

    def _cap_allows(self, cap: str) -> bool:
        try:
            return bool(self._cap_check(cap))
        except Exception:
            return False

    # ------------------------------------------------------------ run / digest
    async def run(self, code: str, timeout: float) -> Exec:
        async with self._lock:
            await self._ensure_proc()
            self._op_id += 1
            op_id = self._op_id
            self._op_target = op_id
            self._op_future = asyncio.get_event_loop().create_future()
            await self._send({"op": "run", "id": op_id, "code": code})
            try:
                msg = await asyncio.wait_for(
                    asyncio.shield(self._op_future), timeout)
            except asyncio.TimeoutError:
                await self._kill()
                return Exec(False, f"error: cell exceeded {timeout}s (killed)",
                            None, [])
            self._digest_cache = msg.get("digest", self._digest_cache)
            return Exec(
                bool(msg.get("ok")),
                msg.get("text", ""),
                msg.get("truthy"),
                msg.get("media") or [],
            )

    def digest(self, max_chars: int) -> str:
        """Namespace digest (names/types/sigs) + tool sigs.

        Synchronous to match the ``Kernel`` protocol (``InProcessKernel.digest``
        is sync, and ``core.py`` calls it synchronously).  The child ships its
        namespace digest with every ``run`` result; we cache it and append the
        tool signatures here.
        """
        vars_part = self._digest_cache
        tool_part = self._tool_sigs()
        joined = "   ".join(p for p in (vars_part, tool_part) if p)
        return joined[:max_chars]

    def _tool_sigs(self) -> str:
        names = self._get_names() if self._get_names else []
        if not names:
            return ""
        sigs = [f"tools.{n}(...)" for n in names[:20]]
        return "# tools\n" + "\n".join(sigs)

    # ------------------------------------------------------------ lifecycle
    async def _kill(self) -> None:
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
        self._digest_cache = ""
        p = self._proc
        self._proc = None
        if p is not None and p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(p.wait(), 5)
            except (asyncio.TimeoutError, Exception):
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        p = self._proc
        if p is not None and p.returncode is None:
            try:
                await self._send({"op": "shutdown"})
            except Exception:
                pass
            try:
                await asyncio.wait_for(p.wait(), 2)
            except (asyncio.TimeoutError, Exception):
                await self._kill()
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
        self._proc = None
