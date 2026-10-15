"""Page engine: bounded working memory (page + fixed-length tail) instead of a growing history.

    INPUT -> ACTION <-> CONTEXT(fold | validate) -> FINAL

Branches depend only on measured values (tokens, counts) or flags set by actions.
Meaning is decided by the model or the Decider port - never by text heuristics.
Commits are synchronous: cancelling a task can never leave a half-applied page.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import difflib
import hashlib
import json
import mimetypes
import time
import uuid
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from .store import DirStore, lock_file


# ================================================================== ports
@dataclass
class Call:
    id: str
    name: str
    args: dict


@dataclass
class Reply:
    content: str = ""
    calls: list[Call] = field(default_factory=list)
    reasoning: str = ""
    usage: dict = field(default_factory=dict)
    interrupted: bool = False         # P5a: stream stopped early by a mid-stream steer


@dataclass
class Exec:
    ok: bool
    text: str
    truthy: bool | None = None        # truthiness of the last expression, None = none
    media: list = field(default_factory=list)


class LLM(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict]) -> Reply: ...
    # Optional: implement stream() to enable live token streaming.
    # Yields str chunks; the engine accumulates them into a full Reply.
    # If absent, the engine falls back to complete().
    async def stream(self, messages: list[dict], tools: list[dict]) -> AsyncIterator[str]: ...  # type: ignore[empty-body]


class Tools(Protocol):
    async def call(self, name: str, args: dict) -> Any: ...
    def search(self, query: str, k: int) -> list[tuple[str, str]]: ...  # [(name, doc)]


class Store(Protocol):
    def put(self, path: str, data: str | bytes) -> str: ...             # text gets indexed
    def get(self, path: str) -> bytes: ...
    def search(self, query: str, k: int, prefix: str = "") -> list[tuple[str, str]]: ...  # [(path, snippet)]


class Kernel(Protocol):
    async def run(self, code: str, timeout: float) -> Exec: ...
    def digest(self, max_chars: int) -> str: ...
    async def close(self) -> None: ...


class Decider(Protocol):
    """Jev-style typed decisions. questions: [{key, kind: noul|choice|score, q, options}] -> {key: value}"""
    async def decide(self, state: str, questions: list[dict]) -> dict[str, Any]: ...


class _NoTools:
    async def call(self, name, args):
        raise KeyError(f"no tool {name!r}")

    def search(self, query, k):
        return []


# ================================================================== config + capsule
@dataclass
class Config:
    page_cap: int = 16_000            # prompt tokens; above it the overrun counter runs
    hard_ratio: float = 1.5           # prompt above page_cap * hard_ratio -> fold now
    target_ratio: float = 0.6         # need_more only repeats a fold above page_cap * target_ratio
    overrun_steps: int = 3
    slot_tokens: dict = field(default_factory=lambda: {"P1": 1200, "P2": 3000, "P3": 1500,
                                                       "P4": 600, "PS": 1200, "P5": 1200})
    tail_items: int = 10              # t1 (newest) always stays raw
    tail_tokens: int = 8_000
    tail_media: int = 2
    inline_chars: int = 4_000         # longer tool output -> store + preview
    max_steps: int = 60
    max_ctx: int = 2                  # context steps in a row
    error_hypotheses: int = 3         # P2.9: max probable-cause hypotheses in an error report
    auto_fold: bool = False           # True: fold after every action unless the agent calls chain()
    max_chain: int = 4
    done_gate: bool = True
    mode: str = "act"                 # P2.10: "act" (Done-Gate + boxes) | "talk" (free-form, no gate)
    kernel_timeout: float = 120
    decider_timeout: float = 0.15
    prep_timeout: float = 5
    vision: bool = True
    live_role: str = "system"         # some local chat templates need "user"
    chars_per_token: float = 3.7
    image_tokens: int = 765


@dataclass
class Capsule:
    """Capabilities are strings: tool:<name>  read:<vfs|sbox|host>:<path>  write:...  (fnmatch globs)."""
    allow: list[str] = field(default_factory=lambda: ["tool:*"])
    pool: list[str] = field(default_factory=list)       # granted automatically for the run
    path: str | None = None                             # "always" grants are saved here
    _run: set = field(default_factory=set, repr=False)
    _step: set = field(default_factory=set, repr=False)

    def allows(self, cap: str) -> bool:
        if cap in self._step:
            self._step.discard(cap)
            return True
        return cap in self._run or any(fnmatch(cap, p) for p in self.allow)

    def intersect(self, requested: list[str] | None) -> list[str]:
        """P5c: capability allow-list for a sub-agent = narrowing of this (parent)
        capsule.  ``None`` inherits the parent's allow-list verbatim.  A requested
        cap is kept only if the parent already allows it (fnmatch) — never widened.
        """
        if requested is None:
            return list(self.allow)
        return [cap for cap in requested if any(fnmatch(cap, p) for p in self.allow)]

    async def request(self, cap: str, reason: str, ask) -> str:
        if any(fnmatch(cap, p) for p in self.allow) or cap in self._run:
            return "already"
        if any(fnmatch(cap, p) for p in self.pool):
            self._run.add(cap)
            return "pool"
        how = await ask(cap, reason) if ask else "deny"  # deny | step | run | always
        {"step": self._step.add, "run": self._run.add, "always": self.allow.append}.get(
            how, lambda c: None
        )(cap)
        if how == "always" and self.path:
            Path(self.path).write_text(
                json.dumps({"allow": self.allow, "pool": self.pool}, indent=1)
            )
        return how

    @classmethod
    def load(cls, path: str) -> "Capsule":
        p = Path(path)
        return cls(**(json.loads(p.read_text()) if p.exists() else {}), path=str(p))


# ================================================================== prompt + actions
MODES = {"A": "action", "F": "context_fold", "V": "context_validate"}
SLOTS = {"P1": ("Task plan", "plan"), "P2": ("Context", "agent"), "P3": ("Helpers", "harness"),
         "P4": ("Code space", "harness"), "PS": ("Offloaded", "harness"), "P5": ("Chat", "agent")}

RULES = """\
You work on a PAGE (bounded working memory, above) and a TAIL (your last raw steps, below).
Old tail items get folded: their content moves to storage and leaves a stub line in PS; `recall` brings it back.
Your slots: P2 context and P5 chat (page_edit), P1 task plan (plan). P3/P4/PS are written by the harness.
Page edits become visible after the next commit; the tool result confirms them.
- Multi-step task: first plan(op="add") one box per deliverable with a `check` (python expression, truthy when
  done, run in your kernel) or check="soft". When done: propose_done. final_answer needs every box done or
  plan(op="reason").
- Put facts you need later into P2 at once. Reasoning alone is lost after a fold.
- code_run: persistent python. Inside: await tools("name", **args), await tools.search("q"), show(path_or_bytes).
  Bundle several operations in one cell.
Examples:
  page_edit(slot="P2", op="append", text="- config: vfs:/project/app.yaml")
  page_edit(slot="P2", op="replace", old="port 8000", text="port 8080")
  plan(op="add", title="tests pass", check="__import__('subprocess').run(['pytest','-q']).returncode == 0")
"""
FOLD = """\
CONTEXT STEP (fold). No world actions. Tail items {first}..t2 get folded, t1 stays.
Copy facts you still need into P2 / P1 notes, then fold_mark(labels, decision): keep = fact is on the page now,
stub = keep a pointer line, drop = irrelevant. Unmarked items become stubs. Everything in ONE response.
To compress a run of related items into ONE line, use summarize(labels=[...], text="what happened", title="x")
instead of many stubs; the originals stay recall-able and a prior summary can be summarized again.
ITEMS:
{items}"""
VALIDATE = """\
CONTEXT STEP (validate) box {id}: {title}
check: {check}
evidence: {evidence}
Judge the evidence, then run_check(box_id="{id}"). Record facts in P2 if needed."""

_S = {"type": "string"}
_ENUM = lambda *v: {"type": "string", "enum": list(v)}  # noqa: E731
# name: (description, properties, required, modes it is allowed in)
ACTIONS: dict[str, tuple[str, dict, list, str]] = {
    "code_run": ("Run python in your persistent kernel (top-level await ok).", {"code": _S}, ["code"], "A"),
    "tool": ("Call a harness tool by name.", {"name": _S, "args": {"type": "object"}}, ["name"], "A"),
    "tool_search": ("Find harness tools.", {"query": _S}, ["query"], "A"),
    "page_edit": ("Edit P2 or P5. replace needs `old` matching exactly once.",
                  {"slot": _ENUM("P2", "P5"), "op": _ENUM("append", "replace", "set"), "text": _S, "old": _S},
                  ["slot", "op", "text"], "AFV"),
    "plan": ("Task plan P1: add(title, check) | note(box_id, text) | reason(box_id, text).",
             {"op": _ENUM("add", "note", "reason"), "box_id": _S, "title": _S, "check": _S, "text": _S},
             ["op"], "AFV"),
    "propose_done": ("Propose a box as done; the harness validates it.", {"box_id": _S, "evidence": _S},
                     ["box_id"], "A"),
    "run_check": ("Run the check of the proposed box (once).", {"box_id": _S}, ["box_id"], "V"),
    "recall": ("Search offloaded content (query) or read one entry (ref, offset).",
               {"query": _S, "ref": _S, "offset": {"type": "integer"}}, [], "AF"),
    "look": ("Bring an offloaded image back into the tail.", {"ref": _S}, ["ref"], "A"),
    "request_capability": ("Ask for a capability, e.g. tool:web_fetch or write:host:C:/out/*.",
                           {"cap": _S, "reason": _S}, ["cap", "reason"], "A"),
    "activate_skill": ("Load a matched skill: its instruction lands on the page (P3) and its "
                       "tools become usable. Use after a skill is suggested or found via tool_search.",
                       {"name": _S}, ["name"], "A"),
    "chain": ("Skip the automatic context step after this action.", {}, [], "A"),
    "fold_mark": ("Decide tail items by label (t2, t3, ...).",
                  {"labels": {"type": "array", "items": _S}, "decision": _ENUM("keep", "stub", "drop")},
                  ["labels", "decision"], "F"),
    "summarize": ("Collapse a run of related tail items (labels t2,t3,...) into ONE summary note. "
                  "`text` is the human summary, `title` labels it. The originals are archived and "
                  "stay recall-able; a prior summary note can be summarized again (summary of summaries).",
                  {"labels": {"type": "array", "items": _S}, "text": _S, "title": _S},
                  ["labels", "text"], "F"),
    "need_more": ("Ask for one more context step.", {}, [], "F"),
    "final_answer": ("Deliver the answer to the user.", {"answer": _S, "success": {"type": "boolean"}},
                     ["answer"], "A"),
}
TOOL_SCHEMA = [{"type": "function", "function": {"name": n, "description": d, "parameters": {
    "type": "object", "properties": p, "required": r}}} for n, (d, p, r, _) in ACTIONS.items()]

# P5b: the sidecar exposes exactly ONE tool — a live steering channel into a
# running session.  It is intentionally NOT part of ACTIONS (not reachable from
# the main loop or the kernel); it only exists for sidecar() calls.
STEER_TOOL = [{"type": "function", "function": {
    "name": "steer",
    "description": "Redirect the running session: inject a steering instruction it will "
                   "see at its next step (interrupts a live stream).",
    "parameters": {"type": "object", "properties": {"text": _S}, "required": ["text"]},
}}]

# Page actions the kernel exposes as callable functions (yellow tools).  Every
# action whose mode set includes "A" is reachable from code_run EXCEPT code_run
# itself (no re-entrancy).  Each entry: (name, description, properties, required).
KERNEL_ACTIONS: list[tuple[str, str, dict, list]] = [
    (n, d, p, r)
    for n, (d, p, r, modes) in ACTIONS.items()
    if "A" in modes and n != "code_run"
]


@dataclass
class Result:
    text: str
    ok: bool = True
    media: list = field(default_factory=list)


@dataclass
class Step:
    mode: str
    chain: bool = False
    need_more: bool = False
    proposed: str | None = None
    checked: bool = False
    final: tuple[str, bool] | None = None
    marks: dict = field(default_factory=dict)
    summaries: list = field(default_factory=list)  # [{labels, text, title}] from summarize()


def _empty_page() -> dict:
    return {"v": 0, "slots": {k: "" for k in SLOTS}, "boxes": []}


@dataclass
class Session:
    id: str
    dir: Path
    page: dict                   # committed: what the model sees
    work: dict                   # working copy: edits are staged here
    tail: list[dict]             # {id, msgs, tok, media, meta}
    next_id: int = 1
    kernel: Kernel | None = None
    lock_handle: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    bg: asyncio.Task | None = None
    prep: asyncio.Task | None = None
    helpers: dict = field(default_factory=dict)  # prep hook name -> text, rendered into P3
    evidence: dict = field(default_factory=dict)
    notices: list = field(default_factory=list)
    steer_pending: list = field(default_factory=list)  # P5a: mid-stream steering, drained per step
    overrun: int = 0
    gate_fired: bool = False
    run: str = ""
    rendered: tuple = (-1, "", 0)


def _parse_streamed(text: str) -> Reply:
    """Build a Reply from accumulated streaming text.

    The engine expects tool calls embedded as a JSON block (the same format most
    OpenAI-compatible streaming endpoints emit when the full text is collected):

        ```json
        [{"id": "c1", "name": "final_answer", "args": {"answer": "42"}}]
        ```

    Plain text with no such block = answer with no tool calls.
    """
    import re as _re
    block = _re.search(r"```json\s*(\[.*?\])\s*```", text, _re.DOTALL)
    if not block:
        # Try a bare JSON array at the end of the text
        block = _re.search(r"(\[\s*\{.*?\}\s*\])\s*$", text, _re.DOTALL)
    if block:
        try:
            raw_calls = json.loads(block.group(1))
            calls = [
                Call(
                    id=c.get("id", f"s{i}"),
                    name=c.get("name", ""),
                    args=c.get("args", {}),
                )
                for i, c in enumerate(raw_calls)
                if c.get("name")
            ]
            # Remove the JSON block from the visible content
            content = text[: block.start()].strip()
            return Reply(content=content, calls=calls)
        except (json.JSONDecodeError, TypeError):
            pass
    return Reply(content=text)


def _clean_chunks(chunks: list[str], visible: str) -> list[str]:
    """Beschneidet die rohen Stream-Chunks auf den sichtbaren Text (visible).

    _parse_streamed() entfernt den Tool-Call-JSON-Block aus dem content.  Die rohen
    Chunks enthalten ihn aber noch.  Hier werden Chunks so gekürzt, dass ihre Summe
    exakt `visible` ergibt – der JSON-Block-Teil wird verworfen.
    """
    if not visible:
        return []
    out: list[str] = []
    remaining = len(visible)
    for ch in chunks:
        if remaining <= 0:
            break
        if len(ch) <= remaining:
            out.append(ch)
            remaining -= len(ch)
        else:
            out.append(ch[:remaining])
            remaining = 0
    return out


# ================================================================== engine
class PageEngine:
    def __init__(self, root: str | Path, llm: LLM, *, tools: Tools | None = None,
                 kernel: Callable[[], Kernel] | None = None, store: Store | None = None,
                 decider: Decider | None = None, capsule: Capsule | None = None,
                 ask: Callable[[str, str], Awaitable[str]] | None = None, cfg: Config | None = None,
                 prep: dict[str, Callable] | None = None, static_prompt: str = "",
                 on_event: Callable[..., None] | None = None,
                 skill_loader: Callable[[str], Any] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.llm, self.tools, self.kernel_factory = llm, tools or _NoTools(), kernel
        self.store = store or DirStore(self.root / "store")
        self.decider, self.capsule, self.ask = decider, capsule or Capsule(), ask
        self.cfg, self.prep = cfg or Config(), prep or {}
        self.skill_loader = skill_loader  # (name) -> {instruction, tools, title} | None
        self.static = f"{static_prompt.strip()}\n\n{RULES}" if static_prompt else RULES
        self.static_hash = hashlib.sha256(self.static.encode()).hexdigest()[:12]
        self.emit = on_event or (lambda type, **data: None)
        self.sessions: dict[str, Session] = {}
        self._b64 = lru_cache(maxsize=32)(self._encode)  # each image is encoded once
        self._active: tuple[Session, Step] | None = None  # current (session, step) for kernel actions

    # ------------------------------------------------------------ public API
    async def run(self, query: str, session: str = "default", media: list = ()) -> str:
        answer = ""
        async for chunk in self.stream(query, session, media):
            answer = chunk.get("answer", answer)
        return answer

    def steer(self, session: str, text: str) -> None:
        s = self.session(session)
        s.notices.append(f"[steering] {text}")
        s.steer_pending.append(text)  # P5a: interrupt a running stream at the next chunk

    def context_view(self, session: str = "default") -> list[dict]:
        """What V3 called working_history (for sidecar / context overview)."""
        s = self.session(session)
        return [{"role": "system", "content": self.static}, {"role": "system", "content": self.render(s)},
                *[m for it in s.tail for m in it["msgs"]]]

    async def sidecar(self, session: str, question: str) -> str:
        """P5b: a single steering-only LLM call on the live ``context_view``.

        Lets an operator query / redirect a running (or paused) session without
        entering the main action loop.  Exactly ONE ``llm.complete`` call, the
        ONLY tool is ``steer``; a ``steer`` call is routed to ``self.steer`` (its
        only side effect is a pending steer + ``[steering]`` notice).  The sidecar
        never edits the page, boxes or tail.  Returns the model's answer text.
        """
        s = self.session(session)
        msgs = [*self.context_view(session), {"role": "user", "content": question}]
        reply = await self.llm.complete(msgs, STEER_TOOL)
        steered = 0
        for c in reply.calls:
            if c.name == "steer":  # the only honoured tool
                self.steer(session, str(c.args.get("text", "")))
                steered += 1
        self.emit("sidecar", run=s.run, steered=steered)
        content = reply.content
        return content if isinstance(content, str) else str(content or "")

    async def spawn_sub(self, parent: str, query: str, *,
                        allow: list[str] | None = None, media: list = ()) -> dict:
        """P5c: run ``query`` in a child ``sub_<id>`` session whose capsule is the
        INTERSECTION of this (parent) capsule with ``allow`` (never widened).  The
        result is stored, and a P3 helper is written into the PARENT page citing
        the result ref, the sub page version and the sub ``versions.jsonl`` path.
        Returns ``{session, answer, ref, version, versions, sub_allow}``.
        """
        p = self.session(parent)
        sub_allow = self.capsule.intersect(allow)
        sub_id = f"sub_{uuid.uuid4().hex[:8]}"
        # The sub runs on its own engine instance so its capsule is isolated,
        # while it shares the llm / tools / store / decider / kernel of the parent.
        sub = PageEngine(
            self.root, self.llm, tools=self.tools, kernel=self.kernel_factory,
            store=self.store, decider=self.decider,
            capsule=Capsule(allow=sub_allow, pool=list(self.capsule.pool)),
            ask=self.ask, cfg=self.cfg, prep=self.prep,
            static_prompt="", on_event=self.emit, skill_loader=self.skill_loader,
        )
        try:
            answer = await sub.run(query, session=sub_id, media=media)
            ss = sub.session(sub_id)
            version = ss.page["v"]
            versions = str(ss.dir / "versions.jsonl")
            ref = self.store.put(
                f"{sub_id}/result.json",
                json.dumps({"query": query, "answer": answer, "version": version,
                            "versions": versions}, ensure_ascii=False),
            )
        finally:
            # Tear down ONLY the sub's own sessions/kernels/locks — never the
            # SHARED store or event sink (those belong to the parent engine).
            for ss2 in sub.sessions.values():
                await sub._cancel_bg(ss2)
                if ss2.prep:
                    ss2.prep.cancel()
                if ss2.kernel:
                    await ss2.kernel.close()
                if ss2.lock_handle:
                    ss2.lock_handle.close()
            sub.sessions.clear()

        # Point the PARENT page (P3) at the sub result; commit so it is rendered.
        p.helpers[f"sub:{sub_id}"] = (
            f"sub-agent {sub_id} -> result {ref} (page v{version}, versions {versions}): "
            f"{answer[:300]}"
        )
        self._commit(p, "sub_agent", sub=sub_id, ref=ref, version=version)
        self.emit("sub_agent", parent=parent, sub=sub_id, ref=ref, version=version)
        return {"session": sub_id, "answer": answer, "ref": ref,
                "version": version, "versions": versions, "sub_allow": sub_allow}

    def session(self, sid: str) -> Session:
        if sid not in self.sessions:
            d = self.root / sid
            d.mkdir(parents=True, exist_ok=True)
            handle = lock_file(d / "session.lock")
            state = json.loads((d / "state.json").read_text("utf-8")) if (d / "state.json").exists() else {}
            page = state.get("page") or _empty_page()
            s = Session(sid, d, page, copy.deepcopy(page), state.get("tail", []),
                        state.get("next_id", 1),
                        kernel=self.kernel_factory() if self.kernel_factory else None,
                        lock_handle=handle)
            if s.kernel is not None and hasattr(s.kernel, "bind"):
                # gateway:        async (name, args) -> Any  – capsule-geprüft
                # get_tool_names: () -> list[str]            – für dir(tools) + digest
                # show:           (bytes, mime, caption) -> media_ref
                # cap_check:      (cap) -> bool              – import/fs Gating (Subprocess)
                s.kernel.bind(
                    self.gateway,
                    lambda: [n for n, _ in self.tools.search("", 200)],
                    lambda b, m, c, s=s: self._put_media(s, b, m, c),
                    cap_check=self.capsule.allows,
                    action_dispatch=self.action_dispatch,
                    actions_schema=KERNEL_ACTIONS,
                )
            self.sessions[sid] = s
        return self.sessions[sid]

    async def close(self) -> None:
        for s in self.sessions.values():
            await self._cancel_bg(s)
            if s.prep:
                s.prep.cancel()
            if s.kernel:
                await s.kernel.close()
            if s.lock_handle:
                s.lock_handle.close()
        self.sessions.clear()
        # Release SQLite / file handles so the store dir can be removed (Windows).
        close = getattr(self.store, "close", None)
        if callable(close):
            close()
        close_tools = getattr(self.tools, "close", None)  # P2.12: in-memory Fts index
        if callable(close_tools):
            close_tools()
        close_emit = getattr(self.emit, "close", None)
        if callable(close_emit):
            close_emit()

    async def gateway(self, name: str, args: dict) -> Any:
        """The single place harness tools run - from actions and from kernel code alike."""
        if not self.capsule.allows(f"tool:{name}"):
            raise PermissionError(f"tool:{name} not granted - use request_capability")
        return await self.tools.call(name, args)

    async def action_dispatch(self, name: str, args: dict) -> str:
        """Run a PAGE ACTION requested from inside the kernel (yellow tools).

        Dispatches against the step currently executing ``code_run``
        (``self._active``).  Returns the action's result text.  Raises
        ``KeyError`` for a name that is not a kernel-exposed action so the
        kernel's ``tools(...)`` shim can fall through to a real harness tool.
        """
        if not any(n == name for n, *_ in KERNEL_ACTIONS):
            raise KeyError(name)
        if self._active is None:
            raise RuntimeError(f"action {name!r} can only run inside code_run")
        s, st = self._active
        result = await self._dispatch(s, st, Call(id="k", name=name, args=args))
        return result.text

    async def search_tools(self, query: str) -> str:
        hits = self.tools.search(query, 20)  # bm25 decides membership
        names = [n for n, _ in hits]
        pick = (await self._decide(
            f"query: {query}",
            [{"key": "tool", "kind": "choice", "q": "best tool for the query", "options": names}]
        ) if names else {}).get("tool")
        hits.sort(key=lambda h: h[0] != pick)  # the decider only reorders
        return "\n".join(f"{n}: {d[:160]}" for n, d in hits[:8]) or "no tools found"

    # ------------------------------------------------------------ the loop
    async def stream(self, query: str, session: str = "default",
                     media: list = ()) -> AsyncIterator[dict]:
        s = self.session(session)
        await self._cancel_bg(s)  # new input never waits for a fold
        async with s.lock:
            s.run, s.gate_fired = uuid.uuid4().hex[:8], False
            self.capsule._run.clear()
            self.capsule._step.clear()
            self.emit("run_begin", run=s.run)
            yield {"type": "status", "status_msg": "start", "run_id": s.run}
            if s.page["v"] == 0:
                await self._prep(s)
                self._commit(s, "init")
            self._push_user(s, query, media)

            mode, last, ctx_n, chain_n, final = "A", "A", 0, 0, None
            for step in range(1, self.cfg.max_steps + 1):
                if mode == "V" and not any(b["state"] == "proposed" for b in s.work["boxes"]) \
                        or mode == "F" and len(s.tail) < 2:
                    mode = "A"
                yield {"type": "iteration_start", "iteration": step, "mode": MODES[mode],
                       "tokens_used": self.prompt_tokens(s), "tokens_max": self.cfg.page_cap}

                if mode != "A":  # ---- CONTEXT
                    st = await self._context(s, mode)
                    yield {"type": "context_step", "mode": MODES[mode], "page_version": s.page["v"]}
                    last, ctx_n = mode, ctx_n + 1
                    again = self._needs_fold(s) or st.need_more and \
                        self.prompt_tokens(s) > self.cfg.page_cap * self.cfg.target_ratio
                    if again and ctx_n < self.cfg.max_ctx:
                        mode = "F"
                        continue
                    if self._needs_fold(s):
                        self._fold(s, {})  # hard fold, no LLM
                    mode, ctx_n = "A", 0
                    continue

                st = Step("A")
                reply, chunks = await self._llm(s, "A")  # ---- ACTION
                if reply.reasoning:
                    yield {"type": "reasoning", "chunk": reply.reasoning}
                # chunks = live stream tokens already seen; emit individually so the client
                # can display them progressively. Without streaming, emit content at once.
                if chunks:
                    for ch in chunks:
                        yield {"type": "content_chunk", "chunk": ch}
                elif reply.content:
                    yield {"type": "content", "chunk": reply.content}
                if reply.interrupted:  # P5a: a steer arrived mid-stream
                    # Close the partial work as a tail item (lossless, recall-able),
                    # drain the steering into the tail as a user turn, then loop to a
                    # fresh action step that sees both.  Never a final answer.
                    if reply.content:
                        self._push(s, [{"role": "assistant", "content": reply.content}],
                                   status="interrupted")
                    steers, s.steer_pending = s.steer_pending, []
                    self._push(s, [{"role": "user",
                                    "content": "[steering] " + " | ".join(steers)}],
                               status="steering")
                    self.emit("steer_midstream", run=s.run, count=len(steers))
                    last = "A"
                    continue
                if not reply.calls:  # plain text == answer
                    final = (reply.content, True)
                    break
                for c in reply.calls:
                    yield {"type": "tool_start", "name": c.name, "args": c.args, "id": c.id}
                results = [await self._dispatch(s, st, c) for c in reply.calls]
                if st.final and (blocked := self._gate(s)):
                    results[[c.name for c in reply.calls].index("final_answer")] = Result(
                        blocked, False
                    )
                    st.final = None
                for c, r in zip(reply.calls, results):
                    yield {"type": "tool_result", "name": c.name, "result": r.text[:2000], "id": c.id}
                if st.final and len(reply.calls) == 1:
                    final = st.final
                    break
                self._push_turn(s, reply, results)
                if any(not r.ok for r in results):
                    await self._error_recall(s, results)  # hint only, never a decision
                if st.final:
                    final = st.final
                    break

                s.overrun = s.overrun + 1 if self.prompt_tokens(s) > self.cfg.page_cap else 0
                last, chain_n = "A", chain_n + 1 if st.chain else 0
                if st.proposed:
                    mode = "V"
                elif self._needs_fold(s) or (
                    self.cfg.auto_fold and not (st.chain and chain_n <= self.cfg.max_chain)
                ):
                    mode, chain_n = "F", 0

            final = final or (
                f"Stopped after {self.cfg.max_steps} steps; the state is on the page.", False
            )
            self._push(s, [{"role": "assistant", "content": final[0]}], status="final")
            yield {"type": "final_answer", "answer": final[0], "success": final[1]}
            # the user has the answer; finish the context afterwards
            if s.work["boxes"] and last == "A" and len(s.tail) > 1:
                s.bg = asyncio.create_task(self._bg_fold(s))
            elif s.work != s.page:
                self._commit(s, "run_end")
            self.emit("run_end", run=s.run)
            yield {"type": "done", "success": final[1], "final_answer": final[0]}

    # ------------------------------------------------------------ measures
    def tokens(self, text: str) -> int:
        return int(len(text) / self.cfg.chars_per_token)

    def prompt_tokens(self, s: Session) -> int:
        return self.render_tokens(s) + sum(it["tok"] for it in s.tail)

    def context_breakdown(self, session: str = "default") -> dict:
        """Native token breakdown for a pure page run (CLI /context parity).

        Returns per-slot token counts, the tail (item count + token sum), the
        static system prompt size, the grand total, and the cap — everything the
        CLI needs to render a CONTEXT X-RAY without the V3 context_overview.
        """
        s = self.session(session)
        slots = {}
        for k in SLOTS:
            body = (
                "\n".join(self._box_line(b) for b in s.page["boxes"])
                if k == "P1"
                else s.page["slots"].get(k, "")
            )
            slots[k] = self.tokens(body or "")
        static_tok = self.tokens(self.static)
        tail_tok = sum(it["tok"] for it in s.tail)
        # ---- V3-compatible X-Ray mapping (so the CLI shows the SAME categories
        #      as context_overview, with real page+tail content instead of 0) ----
        tool_defs = TOOL_SCHEMA
        tools_tok = self.tokens(json.dumps(tool_defs, default=str))
        media_count = sum(len(it["media"]) for it in s.tail)
        media_tok = media_count * self.cfg.image_tokens
        last_input = s.tail[-1]["tok"] if s.tail else 0
        # P1 plan + P3 helpers are stable guidance the agent always sees -> system.
        sys_total = static_tok + slots["P1"] + slots["P3"]
        # P2 context + P5 chat + the raw tail are the live working history.
        work_hist = slots["P2"] + slots["P5"] + slots["P4"] + slots["PS"] + tail_tok
        xray = {
            "System Prompt Total": sys_total,
            "Base System Prompt": static_tok,
            "Page Plan (P1)": slots["P1"],
            "Page Helpers (P3)": slots["P3"],
            "Active Tools": tools_tok,
            "tool_count": len(tool_defs),
            "History (Permanent)": 0,          # page engine keeps no separate RAG history
            "History (Working)": work_hist,
            "w_msg_count": len(s.tail),
            "Last Input": last_input,
            "Media": media_tok,
            "media_count": media_count,
            "Total": sys_total + tools_tok + work_hist + media_tok,
            "cap": self.cfg.page_cap,
        }
        return {
            "session": session,
            "page_version": s.page["v"],
            "static": static_tok,
            "slots": slots,
            "page_rendered": self.render_tokens(s),
            "tail": {"items": len(s.tail), "tokens": tail_tok},
            "total": self.render_tokens(s) + static_tok + tail_tok,
            "cap": self.cfg.page_cap,
            "xray": xray,
        }

    def _needs_fold(self, s: Session) -> bool:
        c = self.cfg
        return len(s.tail) > 1 and (
            len(s.tail) > c.tail_items
            or sum(it["tok"] for it in s.tail) > c.tail_tokens
            or sum(len(it["media"]) for it in s.tail) > c.tail_media
            or self.prompt_tokens(s) > c.page_cap * c.hard_ratio
            or s.overrun > c.overrun_steps
        )

    def _gate(self, s: Session) -> str | None:
        if self.cfg.mode == "talk":  # P2.10: TALK has no Done-Gate (free-form)
            return None
        open_ = [b["id"] for b in s.work["boxes"] if b["state"] in ("open", "proposed")]
        if not (self.cfg.done_gate and open_) or s.gate_fired:
            return None
        s.gate_fired = True
        return (f"final_answer blocked once: open boxes {', '.join(open_)}. propose_done them or "
                f"plan(op='reason'). The next final_answer goes through.")

    # ------------------------------------------------------------ prompt
    def render(self, s: Session) -> str:
        if s.rendered[0] != s.page["v"]:  # rendered once per commit
            p, parts = s.page, [f"# PAGE v{s.page['v']}"]
            for k, (title, owner) in SLOTS.items():
                body = (
                    "\n".join(self._box_line(b) for b in p["boxes"])
                    if k == "P1"
                    else p["slots"][k]
                )
                parts.append(f"## {k} {title} [{owner}]\n{body or '(empty)'}")
            text = "\n\n".join(parts)
            s.rendered = (p["v"], text, self.tokens(text))
        return s.rendered[1]

    def render_tokens(self, s: Session) -> int:
        self.render(s)
        return s.rendered[2]

    @staticmethod
    def _box_line(b: dict) -> str:
        mark = {"open": "[ ]", "proposed": "[~]", "done": "[x]", "open_reason": "[!]"}[b["state"]]
        return (
            f"- {mark} {b['id']} {b['title']}  (check: `{b['check']}`)"
            + (f"  - {b['note']}" if b["note"] else "")
        )

    def _live(self, s: Session, mode: str) -> str:
        lines = [
            f"[LIVE] {time.strftime('%Y-%m-%d %H:%M')} | {MODES[mode]} | {self.cfg.mode} mode "
            f"| page v{s.page['v']} "
            f"{self.render_tokens(s)} tok | tail {len(s.tail)} items "
            f"{sum(i['tok'] for i in s.tail)} tok | cap {self.cfg.page_cap}"
        ]
        if s.work != s.page:
            lines.append("staged page edits pending (visible after the next commit)")
        return "\n".join(lines + s.notices[-5:])

    def _encode(self, ref: str, mime: str) -> str:
        return f"data:{mime};base64," + base64.b64encode(self.store.get(ref)).decode()

    def _materialize(self, msg: dict) -> dict:
        """media_ref parts become image_url (base64, cached) only here, at send time."""
        content = msg.get("content")
        if not isinstance(content, list) or all(
            p.get("type") != "media_ref" for p in content
        ):
            return msg

        def part(p):
            if p.get("type") != "media_ref":
                return p
            if self.cfg.vision:
                return {"type": "image_url",
                        "image_url": {"url": self._b64(p["ref"], p["mime"])}}
            return {"type": "text",
                    "text": f"[image {p['ref']}: {p['caption'] or 'no caption'}]"}

        return {**msg, "content": [part(p) for p in content]}

    def _messages(self, s: Session, mode: str, instruction: str | None) -> list[dict]:
        # Native universal shape: exactly ONE system message at the very top
        # (static prompt + page render joined), then the raw tail, then the
        # volatile live block as a user turn (never a mid/trailing system msg),
        # then the optional instruction.  _sanitize_messages is the final,
        # information-preserving guarantee on top of this.
        msgs = [
            {"role": "system", "content": f"{self.static}\n\n{self.render(s)}"},
            *[m for it in s.tail for m in it["msgs"]],
            {"role": self.cfg.live_role, "content": self._live(s, mode)},
        ]
        if instruction:  # appended, never in the system prompt
            msgs.append({"role": "user", "content": instruction})
        return self._sanitize_messages([self._materialize(m) for m in msgs])

    @staticmethod
    def _sanitize_messages(msgs: list[dict]) -> list[dict]:
        """Guarantee a universally valid chat array WITHOUT losing information.

        Strict OpenAI-compatible providers (e.g. zglm/glm via the openai_compat
        adapter) reject arrays with a mid/trailing ``system`` message, an
        assistant ``content: None`` + ``tool_calls``, or a broken
        ``assistant(tool_calls) -> tool`` adjacency (HTTP 400, code 1214).

        This function makes the array valid everywhere while PRESERVING every
        piece of content — nothing is dropped:

          * leading consecutive ``system`` messages are merged into one block;
          * any later ``system`` message is re-roled to ``user`` (content kept);
          * assistant ``content: None`` becomes ``""``;
          * a broken tool sequence is NOT discarded: an incomplete assistant
            ``tool_calls`` is rewritten so the attempted calls appear in the
            message *text*, and an orphan ``tool`` message is converted to a
            ``user`` text message that keeps the tool name + full result.
        """
        # ---- 1. system handling (merge leading, re-role later ones) ----------
        out: list[dict] = []
        leading = True
        sys_buf: list[str] = []
        for m in msgs:
            role = m.get("role")
            if role == "system" and leading:
                c = m.get("content")
                sys_buf.append(c if isinstance(c, str) else json.dumps(c, default=str))
                continue
            if leading and sys_buf:
                out.append({"role": "system", "content": "\n\n".join(sys_buf)})
                sys_buf = []
            leading = False
            if role == "system":
                out.append({**m, "role": "user"})  # re-role, keep content
            else:
                out.append(dict(m))
        if sys_buf:
            out.append({"role": "system", "content": "\n\n".join(sys_buf)})

        # ---- 2. tool-call adjacency + content normalization (lossless) -------
        repaired: list[dict] = []
        i, n = 0, len(out)
        while i < n:
            m = out[i]
            role = m.get("role")

            if role == "assistant" and m.get("tool_calls"):
                call_ids = [tc.get("id") for tc in m["tool_calls"]]
                j = i + 1
                tool_msgs: list[dict] = []
                while j < n and out[j].get("role") == "tool":
                    tool_msgs.append(out[j])
                    j += 1
                have = {t.get("tool_call_id") for t in tool_msgs}
                if call_ids and all(cid in have for cid in call_ids):
                    # complete, valid turn — keep as-is (content None -> "")
                    repaired.append({**m, "content": m.get("content") or ""})
                    repaired.extend(tool_msgs)
                    i = j
                    continue
                # incomplete turn: keep the assistant message but encode the
                # attempted calls into text so NO information is lost, and drop
                # only the now-invalid structural tool_calls field.
                attempted = "; ".join(
                    f"{tc.get('function', {}).get('name', '?')}("
                    f"{tc.get('function', {}).get('arguments', '')})"
                    for tc in m["tool_calls"]
                )
                base = m.get("content") or ""
                text = (base + ("\n" if base else "")
                        + f"[tool calls attempted: {attempted}]")
                demoted = {k: v for k, v in m.items() if k != "tool_calls"}
                demoted["content"] = text
                repaired.append(demoted)
                # convert any partial tool results to user text (keep content)
                for t in tool_msgs:
                    repaired.append({
                        "role": "user",
                        "content": f"[tool result {t.get('name', '')}]: "
                                   f"{t.get('content', '')}",
                    })
                i = j
                continue

            if role == "tool":
                # orphan tool message (no preceding assistant tool_calls):
                # keep its content, just re-role to user text.
                repaired.append({
                    "role": "user",
                    "content": f"[tool result {m.get('name', '')}]: "
                               f"{m.get('content', '')}",
                })
                i += 1
                continue

            if role == "assistant" and m.get("content") is None:
                repaired.append({**m, "content": ""})
                i += 1
                continue
            repaired.append(m)
            i += 1
        return repaired


    async def _llm(self, s: Session, mode: str,
                   instruction: str | None = None) -> tuple[Reply, list[str]]:
        """Returns (reply, stream_chunks).

        stream_chunks is a list of str chunks emitted during streaming (empty when the LLM
        does not support streaming or in context steps). The caller yields them to the client
        so the user sees tokens live.
        """
        msgs = self._messages(s, mode, instruction)
        t0 = time.perf_counter()
        chunks: list[str] = []

        if mode == "A" and hasattr(self.llm, "stream") and callable(
            getattr(self.llm, "stream", None)
        ):
            # ---- streaming path: accumulate chunks, then hand the full text to the caller
            accumulated = ""
            interrupted = False
            try:
                # stream() is an async generator -> iterate directly (no await on the call)
                async for chunk in self.llm.stream(msgs, TOOL_SCHEMA):
                    if isinstance(chunk, str) and chunk:
                        accumulated += chunk
                        chunks.append(chunk)
                    if s.steer_pending:  # P5a: a steer arrived mid-stream -> stop here
                        interrupted = True
                        break
            except (AttributeError, NotImplementedError):
                # LLM declares stream() but does not actually support it -> fall back.
                chunks = []
                reply = await self.llm.complete(msgs, TOOL_SCHEMA)
            else:
                if interrupted:
                    # Keep the partial VISIBLE text verbatim; do NOT parse it as a
                    # tool-call/final — it is unfinished work to be closed as a tail item.
                    reply = Reply(content=accumulated, interrupted=True)
                else:
                    # Parse the accumulated text into a Reply (tool-call JSON block or plain text).
                    reply = _parse_streamed(accumulated)
                    # Only expose the VISIBLE text as chunks (reply.content). The tool-call
                    # JSON block that _parse_streamed stripped must not reach the client as
                    # content. Rebuild chunks so the sum equals reply.content exactly.
                    chunks = _clean_chunks(chunks, reply.content)
        else:
            # ---- non-streaming path (context steps, or LLM without stream())
            reply = await self.llm.complete(msgs, TOOL_SCHEMA)

        self.emit(
            "llm_call",
            mode=MODES[mode],
            static=self.static_hash,
            page_v=s.page["v"],
            tail=[it["id"] for it in s.tail],
            prompt_tok=self.prompt_tokens(s),
            ms=int((time.perf_counter() - t0) * 1000),
            usage=reply.usage,
            streamed=bool(chunks),
        )
        return reply, chunks

    async def _decide(self, state: str, questions: list[dict]) -> dict:
        if not self.decider:
            return {}
        try:
            answers = await asyncio.wait_for(
                self.decider.decide(state, questions), self.cfg.decider_timeout
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # timeout or backend error: no hint at all
            return {}
        self.emit("decider", keys=[q["key"] for q in questions], answers=answers)
        return answers

    async def _error_recall(self, s: Session, results: list[Result]) -> None:
        """After a tool/kernel error: surface the error and 1..N probable-cause
        hypotheses as NOTICES only — never a page edit, box, decision or action.

        P1.6 (recall hints): for each store hit the decider marks relevant, append
        a ``[recall?] <ref>: <snippet>`` line.

        P2.9 (error report): always append ONE ``[error-report]`` notice naming the
        error line, followed by up to ``Config.error_hypotheses`` decider-ranked
        hypotheses as ``* <ref>: <snippet>`` bullets.  No decider / no hits ⇒ the
        report still names the error line (so the model is not blind) but has no
        bullets.  Reuses the single ``store.search`` — no extra search, no LLM call.
        """
        line = next(
            (r.text.strip().splitlines()[0] for r in results if not r.ok and r.text.strip()),
            "",
        )
        if not line:
            return

        relevant: list[tuple[str, str]] = []
        if self.decider:
            hits = self.store.search(line, self.cfg.error_hypotheses, prefix=f"{s.id}/")
            if hits:
                answers = await self._decide(
                    f"error: {line}",
                    [
                        {"key": ref, "kind": "noul", "q": "is this relevant to the error"}
                        for ref, _ in hits
                    ],
                )
                relevant = [
                    (ref, snippet) for ref, snippet in hits if answers.get(ref)
                ][: self.cfg.error_hypotheses]

        # P1.6 recall hints (unchanged behaviour).
        for ref, snippet in relevant:
            s.notices.append(f"[recall?] {ref}: {snippet[:160]}")
        if relevant:
            self.emit("error_recall", error=line[:160], refs=[r for r, _ in relevant])

        # P2.9 structured error report (always, hypotheses optional).
        bullets = [f"* {ref}: {snippet[:160]}" for ref, snippet in relevant]
        report = f"[error-report] {line[:200]}"
        if bullets:
            report += " | probable causes:\n" + "\n".join(bullets)
        s.notices.append(report)
        self.emit("error_report", error=line[:200], hypotheses=[r for r, _ in relevant])

    # ------------------------------------------------------------ tail
    def _save(self, s: Session) -> None:
        """Page and tail live in one file, so a fold (commit + tail cut) is atomic on disk too."""
        tmp = s.dir / "state.json.tmp"
        tmp.write_text(
            json.dumps(
                {"page": s.page, "tail": s.tail, "next_id": s.next_id},
                ensure_ascii=False,
                default=str,
            ),
            "utf-8",
        )
        tmp.replace(s.dir / "state.json")

    def _push(self, s: Session, msgs: list[dict], media: list = (), **meta) -> None:
        tok = (
            self.tokens(json.dumps(msgs, ensure_ascii=False, default=str))
            + len(media) * self.cfg.image_tokens
        )
        s.tail.append(
            {"id": s.next_id, "msgs": msgs, "tok": tok, "media": list(media), "meta": meta}
        )
        s.next_id += 1
        self._save(s)

    def _put_media(self, s: Session, data: bytes, mime: str, caption: str) -> dict:
        ref = (
            f"{s.id}/media/{hashlib.sha256(data).hexdigest()[:16]}"
            f"{mimetypes.guess_extension(mime) or ''}"
        )
        return {"type": "media_ref", "ref": self.store.put(ref, data), "mime": mime,
                "caption": caption}

    def _push_user(self, s: Session, query: str, media: list) -> None:
        parts = [
            self._put_media(s, m, "image/png", "")
            if isinstance(m, bytes)
            else self._put_media(
                s, Path(m).read_bytes(), mimetypes.guess_type(str(m))[0] or "image/png", ""
            )
            for m in media
        ]
        text = f"[{time.strftime('%Y-%m-%d %H:%M')}] {query}"
        self._push(
            s,
            [{"role": "user",
              "content": [{"type": "text", "text": text}, *parts] if parts else text}],
            parts,
            status="user",
        )

    def _inline(self, s: Session, text: str, limit: int | None = None) -> str:
        """Long output goes to the store; the tail keeps head, end and a recall reference."""
        n = limit or self.cfg.inline_chars
        if len(text) <= n:
            return text
        ref = self.store.put(
            f"{s.id}/out/{hashlib.sha256(text.encode()).hexdigest()[:16]}.txt", text
        )
        return (
            f"{text[:n * 3 // 8]}\n"
            f"...[{len(text)} chars -> recall(ref={ref!r})]...\n"
            f"{text[-n // 8:]}"
        )

    def _push_turn(self, s: Session, reply: Reply, results: list[Result]) -> None:
        msgs = [{"role": "assistant", "content": reply.content or None, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.args)}}
            for c in reply.calls
        ]}]
        msgs += [
            {"role": "tool", "tool_call_id": c.id, "name": c.name,
             "content": self._inline(s, r.text)}
            for c, r in zip(reply.calls, results)
        ]
        media = [p for r in results for p in r.media]
        if media:  # after ALL tool results: API-safe
            msgs.append(
                {"role": "user",
                 "content": [{"type": "text", "text": "[tool media]"}, *media]}
            )
        self._push(s, msgs, media, tools=[c.name for c in reply.calls],
                   reasoning=reply.reasoning,
                   status="ok" if all(r.ok for r in results) else "error")

    # ------------------------------------------------------------ context steps
    def _item_line(self, s: Session, i: int) -> str:
        it = s.tail[i]
        return (
            f"t{len(s.tail) - i}: "
            f"{','.join(it['meta'].get('tools', [])) or it['meta'].get('status')} "
            f"status={it['meta'].get('status')} tok={it['tok']} media={len(it['media'])}"
        )

    async def _context(self, s: Session, mode: str) -> Step:
        st, snapshot = Step(mode), copy.deepcopy(s.work)
        try:
            if mode == "V":
                box = next(b for b in s.work["boxes"] if b["state"] == "proposed")
                reply, _ = await self._llm(
                    s, "V",
                    VALIDATE.format(
                        evidence=s.evidence.get(box["id"], "")[:1500], **box
                    ),
                )
                [await self._dispatch(s, st, c) for c in reply.calls]
                if not st.checked:  # the harness checks if the model did not
                    await self._check(s, box["id"])
                self._commit(s, "validate", box=box["id"], state=box["state"])
                return st
            items = s.tail[:-1]
            lines = [self._item_line(s, i) for i in range(len(items))]
            hints = await self._decide(
                self.render(s)[:2000] + "\n" + "\n".join(lines),
                [
                    {"key": str(it["id"]), "kind": "choice",
                     "q": "keep, stub or drop this tail item for the open task",
                     "options": ["keep", "stub", "drop"]}
                    for it in items
                ],
            )
            listing = "\n".join(
                ln + (f"  hint={hints[str(it['id'])]}" if str(it["id"]) in hints else "")
                for ln, it in zip(lines, items)
            )
            reply, _ = await self._llm(
                s, "F", FOLD.format(first=f"t{len(s.tail)}", items=listing)
            )
            [await self._dispatch(s, st, c) for c in reply.calls]
        except asyncio.CancelledError:
            s.work = snapshot  # a cancelled step leaves no trace
            self.emit("fold_cancelled", page_v=s.page["v"])
            raise
        by_label = {f"t{len(s.tail) - i}": it["id"] for i, it in enumerate(items)}
        summaries = [
            {
                "ids": [by_label[lbl] for lbl in sm["labels"] if lbl in by_label],
                "text": sm["text"],
                "title": sm["title"],
            }
            for sm in st.summaries
        ]
        summaries = [sm for sm in summaries if sm["ids"]]  # drop empty label sets
        self._fold(
            s,
            {
                **{int(k): v for k, v in hints.items() if k.isdigit()},
                **{by_label[k]: v for k, v in st.marks.items() if k in by_label},
            },
            summaries,
        )
        return st

    def _fold(self, s: Session, decisions: dict[int, str],
              summaries: list[dict] | None = None) -> None:
        """Synchronous on purpose. Every item t2.. goes to the store; stubs go to PS.

        ``summaries`` collapses a run of items into ONE note: the covered items
        are archived (still recall-able), a single summary line goes to PS with a
        store ref to the combined note, and those items are NOT stubbed per-item.
        A summary note stores ``{title, text, refs:[...]}`` — a later fold can
        summarize a previous summary line again (summary of summaries).
        """
        summaries = summaries or []
        items, s.tail = s.tail[:-1], s.tail[-1:]
        # Archive every folded item first so refs exist for stubs AND summaries.
        refs: dict[int, str] = {}
        for it in items:
            refs[it["id"]] = self.store.put(
                f"{s.id}/tail/{it['id']}.json",
                json.dumps(it["msgs"], ensure_ascii=False),
            )
        covered: set[int] = set()
        for sm in summaries:
            ids = [i for i in sm["ids"] if i in refs]
            if not ids:
                continue
            covered.update(ids)
            note = {
                "title": sm["title"],
                "text": sm["text"],
                "refs": [refs[i] for i in ids],
                "ids": ids,
            }
            payload = json.dumps(note, ensure_ascii=False)
            sref = self.store.put(
                f"{s.id}/summary/{hashlib.sha256(payload.encode()).hexdigest()[:16]}.json",
                payload,
            )
            tok = sum(it["tok"] for it in items if it["id"] in ids)
            title = sm["title"] or "summary"
            self._stub(
                s,
                f"[summary \"{title}\" #{min(ids)}..#{max(ids)} {tok}tok -> {sref} ] "
                f"{sm['text'][:200]}",
            )
        for it in items:
            if it["id"] in covered:
                continue  # already represented by a summary line
            if decisions.get(it["id"], "stub") == "stub":
                images = "; ".join(m["caption"] or m["ref"] for m in it["media"])
                self._stub(
                    s,
                    f"[#{it['id']} "
                    f"{','.join(it['meta'].get('tools', [])) or it['meta'].get('status')} "
                    f"{it['meta'].get('status')} {it['tok']}tok -> {refs[it['id']]}"
                    + (f" | images: {images}]" if images else "]"),
                )
        s.overrun = 0
        self._commit(
            s, "fold", folded=[it["id"] for it in items], decisions=decisions,
            summaries=[sm["title"] or "summary" for sm in summaries],
        )

    def _stub(self, s: Session, line: str) -> None:
        lines = (s.work["slots"]["PS"] + "\n" + line).strip().splitlines()
        while len(lines) > 1 and self.tokens("\n".join(lines)) > self.cfg.slot_tokens["PS"]:
            lines.pop(0)  # oldest roll off, still searchable
        s.work["slots"]["PS"] = "\n".join(lines)

    async def _bg_fold(self, s: Session) -> None:
        async with s.lock:
            with contextlib.suppress(asyncio.CancelledError):
                await self._context(s, "F")

    @staticmethod
    async def _cancel_bg(s: Session) -> None:
        if s.bg and not s.bg.done():
            s.bg.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await s.bg
        s.bg = None

    # ------------------------------------------------------------ commit + prep
    def _render_helpers(self, s: Session) -> None:
        """Render prep-hook / skill helpers into the P3 slot (work copy)."""
        if s.helpers:
            text = "\n\n".join(f"### {k}\n{v}" for k, v in s.helpers.items())
            s.work["slots"]["P3"] = text[
                : int(self.cfg.slot_tokens["P3"] * self.cfg.chars_per_token)
            ]

    def _commit(self, s: Session, reason: str, **data) -> None:
        self._render_helpers(s)
        before = self.render(s)
        s.work["v"] = s.page["v"] + 1
        s.page = copy.deepcopy(s.work)
        diff = "".join(
            difflib.unified_diff(before.splitlines(True), self.render(s).splitlines(True))
        )
        self._save(s)
        with open(s.dir / "versions.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"v": s.page["v"], "ts": time.time(), "reason": reason, "diff": diff}
                )
                + "\n"
            )
        self.emit("page_commit", v=s.page["v"], reason=reason, **data)
        if reason != "init" and self.prep and (s.prep is None or s.prep.done()):
            s.prep = asyncio.create_task(self._prep(s))  # lands in P3 at the next commit

    async def _prep(self, s: Session) -> None:
        async def one(name, hook):
            try:
                return name, await asyncio.wait_for(hook(self, s), self.cfg.prep_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.emit("prep_error", hook=name, error=repr(e))
                return name, None

        for name, text in await asyncio.gather(
            *(one(n, h) for n, h in self.prep.items())
        ):
            if text:
                s.helpers[name] = text

    # ------------------------------------------------------------ actions
    async def _dispatch(self, s: Session, st: Step, call: Call) -> Result:
        spec = ACTIONS.get(call.name)
        if spec is None:
            return Result(f"error: unknown action {call.name!r}", False)
        if st.mode not in spec[3]:
            return Result(f"error: {call.name} is not allowed in {MODES[st.mode]}", False)
        try:
            return await getattr(self, f"_do_{call.name}")(s, st, **call.args)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # an error is a result, not a crash
            return Result(f"error: {type(e).__name__}: {e}", False)

    async def _do_code_run(self, s, st, code):
        if s.kernel is None:
            return Result("error: no kernel configured", False)
        self._active = (s, st)  # kernel-invoked page actions target this step
        try:
            r = await s.kernel.run(code, self.cfg.kernel_timeout)
        finally:
            self._active = None
        s.work["slots"]["P4"] = s.kernel.digest(
            int(self.cfg.slot_tokens["P4"] * self.cfg.chars_per_token)
        )
        return Result(r.text, r.ok, r.media)

    async def _do_tool(self, s, st, name, args=None):
        out = await self.gateway(name, args or {})
        return Result(
            out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)
        )

    async def _do_tool_search(self, s, st, query):
        return Result(await self.search_tools(query))

    async def _do_request_capability(self, s, st, cap, reason):
        how = await self.capsule.request(cap, reason, self.ask)
        self.emit("cap_decision", cap=cap, how=how)
        return Result(
            f"{'denied' if how == 'deny' else 'granted'} ({how}): {cap}", how != "deny"
        )

    async def _do_activate_skill(self, s, st, name):
        if self.skill_loader is None:
            return Result("error: no skill loader configured", False)
        loaded = self.skill_loader(name)
        if asyncio.iscoroutine(loaded):
            loaded = await loaded
        if not loaded:
            return Result(f"error: skill {name!r} not found", False)
        instruction = (loaded.get("instruction") or "").strip()
        title = loaded.get("title") or name
        tools = loaded.get("tools") or []
        if instruction:
            s.helpers[f"skill:{title}"] = instruction  # rendered into P3 at commit
            self._render_helpers(s)  # make it visible in work now (so run_end commits it)
        for t in tools:  # make the skill's tools usable
            self.capsule._run.add(f"tool:{t}")
        self.emit("skill_activated", skill=title, tools=list(tools))
        return Result(
            f"ok: skill '{title}' activated"
            + (f" (+{len(tools)} tools: {', '.join(tools)})" if tools else "")
            + "; instruction added to P3 (visible after the next commit)"
        )

    async def _do_page_edit(self, s, st, slot, op, text="", old=None):
        if SLOTS.get(slot, ("", ""))[1] != "agent":
            return Result(
                f"error: {slot} is not your slot (P2, P5; P1 via plan)", False
            )
        cur = s.work["slots"][slot]
        if op == "replace" and (not old or cur.count(old) != 1):
            return Result(
                f"error: old must match exactly once in {slot}, "
                f"found {cur.count(old) if old else 0}",
                False,
            )
        new = {
            "append": f"{cur}\n{text}" if cur else text,
            "set": text,
            "replace": cur.replace(old or "", text, 1),
        }.get(op)
        if new is None:
            return Result("error: op is append | replace | set", False)
        if (n := self.tokens(new)) > (limit := self.cfg.slot_tokens[slot]):
            return Result(
                f"error: {slot} would be {n} tokens (limit {limit}); shorten first", False
            )
        s.work["slots"][slot] = new
        return Result(f"ok: {slot} {n}/{limit} tok, visible after the next commit")

    def _box(self, s, box_id) -> dict:
        box = next((b for b in s.work["boxes"] if b["id"] == box_id), None)
        if box is None:
            raise KeyError(f"unknown box {box_id!r}")
        return box

    async def _do_plan(self, s, st, op, box_id=None, title=None, check="soft", text=""):
        boxes = s.work["boxes"]
        if op == "add":
            if not title:
                return Result("error: add needs a title", False)
            box = {"id": f"B{len(boxes) + 1}", "title": title,
                   "check": check or "soft", "state": "open", "note": ""}
            if self.tokens(
                "\n".join(map(self._box_line, boxes + [box]))
            ) > self.cfg.slot_tokens["P1"]:
                return Result("error: P1 is full; finish or merge boxes first", False)
            boxes.append(box)
            return Result(f"ok: {box['id']} added (check: {box['check']})")
        box = self._box(s, box_id)
        if op == "note" or op == "reason" and text:
            box["note"] = text
            box["state"] = "open_reason" if op == "reason" else box["state"]
            return Result(f"ok: {box['id']} {box['state']}")
        return Result("error: op is add | note | reason (reason needs text)", False)

    async def _do_propose_done(self, s, st, box_id, evidence=""):
        box = self._box(s, box_id)
        box["state"], s.evidence[box_id], st.proposed = "proposed", evidence, box_id
        return Result(f"ok: {box_id} proposed, validation follows")

    async def _do_run_check(self, s, st, box_id):
        if st.checked:
            return Result("error: run_check runs once per validation step", False)
        st.checked = True
        ok, detail = await self._check(s, box_id)
        return Result(f"check {'passed' if ok else 'failed'}: {detail}", ok)

    async def _check(self, s: Session, box_id: str) -> tuple[bool, str]:
        box = self._box(s, box_id)
        if box["check"].strip() == "soft":
            v = (
                await self._decide(
                    f"box: {box['title']}\nevidence: {s.evidence.get(box_id, '')}",
                    [{"key": "soft", "kind": "noul",
                      "q": "is the box done according to the evidence"}],
                )
            ).get("soft")
            ok, detail = (
                (True, "soft check: model judgement")
                if v is None
                else (bool(v), "soft check: decider")
            )
        elif s.kernel is None:
            ok, detail = False, "no kernel to run the check"
        else:
            r = await s.kernel.run(box["check"], self.cfg.kernel_timeout)
            ok, detail = r.ok and r.truthy is not False, self._inline(s, r.text, 300)
        box["state"], box["note"] = (
            ("done", "check ok") if ok else ("open", f"check failed: {detail}")
        )
        self.emit("check", box=box_id, ok=ok, detail=detail)
        return ok, detail

    async def _do_recall(self, s, st, query="", ref="", offset=0):
        if ref:
            if not ref.startswith(f"{s.id}/"):
                return Result("error: ref belongs to another session", False)
            text = self.store.get(ref).decode("utf-8", "replace")
            n = self.cfg.inline_chars
            more = (
                f"\n[more: recall(ref={ref!r}, offset={offset + n})]"
                if offset + n < len(text)
                else ""
            )
            return Result(text[offset: offset + n] + more)
        hits = self.store.search(query, 6, prefix=f"{s.id}/")
        return Result(
            "\n".join(f"{ref}  {snippet}" for ref, snippet in hits) or "no matches"
        )

    async def _do_look(self, s, st, ref):
        self.store.get(ref)  # raises if missing
        part = {"type": "media_ref", "ref": ref,
                "mime": mimetypes.guess_type(ref)[0] or "image/png", "caption": ""}
        return Result(f"image {ref} loaded", media=[part])

    async def _do_chain(self, s, st):
        st.chain = True
        return Result("ok")

    async def _do_need_more(self, s, st):
        st.need_more = True
        return Result("ok")

    async def _do_fold_mark(self, s, st, labels, decision):
        if decision not in ("keep", "stub", "drop"):
            return Result("error: decision is keep | stub | drop", False)
        st.marks.update(dict.fromkeys(labels, decision))
        return Result("ok")

    async def _do_summarize(self, s, st, labels, text, title=""):
        if not labels or not isinstance(labels, list):
            return Result("error: summarize needs a non-empty labels list", False)
        st.summaries.append({"labels": list(labels), "text": text, "title": title or ""})
        return Result(
            f"ok: summarizing {len(labels)} item(s) into one note '{title or 'summary'}'"
        )

    async def _do_final_answer(self, s, st, answer, success=True):
        st.final = (answer, bool(success))
        return Result("delivered")
