"""
omni.py
=======

Omni (speech-to-speech) layer for the ISAA audio stack.

A single drop-in module that sits *beside* the classic STT->LLM->TTS pipeline
without touching it. The classic pipeline stays as fallback; this module adds
end-to-end Omni backends (one model, audio-in -> audio-out, tool-calls as
mid-stream events) plus the tool bridge the FlowAgent already has the pieces
for.

Design constraints honoured here:
  - minimal invasive: the tested core imports NOTHING from toolboxv2. It
    couples to the existing AudioRecorder / AudioPlayer / ToolManager /
    FlowAgent purely by duck-typing, so it is unit-testable in isolation.
  - reuse, don't duplicate: OmniSession drives the *existing* recorder/player
    ABCs and routes tool-calls into the *existing* ToolManager.execute().
    Agent delegation reuses FlowAgent.a_run / list_executions.
  - the only untested surface is the network/model I/O (LocalOmniBackend,
    CloudOmniBackend); everything else is driven by StubOmniBackend + fakes.

Public, tested surface
----------------------
    OmniEventType, OmniEvent          event protocol between backend & session
    OmniBackend (ABC)                 backend contract
    StubOmniBackend                   deterministic scripted backend (tests + dev)
    JobManager                        background jobs: agent delegations + nonblocking tools
    OmniSession                       the loop: recorder -> backend -> player + tool bridge
    make_agent_tools                  delegate / agent_result / agent_status tools
    make_vfs_peek_tools               vfs_peek / vfs_tree_peek (read-only into delegated VFS)
    pcm16_to_wav                      PCM int16 -> WAV bytes
    VoiceModeConfig                   config + backend factory (omni_local|omni_cloud|pipeline|stub)

Network surface (lazy, guarded, not unit-tested)
------------------------------------------------
    LocalOmniBackend                  Qwen2.5-Omni-7B-AWQ via transformers
    CloudOmniBackend                  Qwen-Omni-Flash via DashScope realtime WS
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re as _re
import time
import uuid
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
)

try:
    from toolboxv2 import get_logger, Style

    logger = get_logger()
except ImportError:
    logger = logging.getLogger(__name__)

TARGET_SR = 16000


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------

def pcm16_to_wav(pcm: bytes, sample_rate: int = TARGET_SR, channels: int = 1) -> bytes:
    """Wrap raw PCM int16 bytes in a minimal WAV container.

    AudioPlayer.queue_audio expects WAV bytes; Omni backends emit raw PCM.
    Uses stdlib `wave` only — no numpy dependency in the tested core.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # int16
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Event protocol
# ---------------------------------------------------------------------------

class OmniEventType(Enum):
    AUDIO = "audio"          # raw PCM int16 chunk out (ev.audio)
    TEXT = "text"            # partial / final text out (ev.text)
    TOOL_CALL = "tool_call"  # model wants a tool (ev.tool_call: {id,name,arguments})
    TURN_END = "turn_end"    # model finished a turn
    INTERRUPTED = "interrupted"  # user barge-in: model output should be flushed/stopped
    ERROR = "error"          # backend-level failure (ev.text = message)


@dataclass
class OmniEvent:
    type: OmniEventType
    audio: Optional[bytes] = None
    text: Optional[str] = None
    tool_call: Optional[dict] = None  # {"id": str, "name": str, "arguments": dict}
    meta: dict = field(default_factory=dict)

    # convenience constructors -------------------------------------------------
    @classmethod
    def audio_chunk(cls, pcm: bytes, **meta) -> "OmniEvent":
        return cls(OmniEventType.AUDIO, audio=pcm, meta=meta)

    @classmethod
    def text_chunk(cls, text: str, **meta) -> "OmniEvent":
        return cls(OmniEventType.TEXT, text=text, meta=meta)

    @classmethod
    def call(cls, name: str, arguments: dict, call_id: Optional[str] = None) -> "OmniEvent":
        return cls(
            OmniEventType.TOOL_CALL,
            tool_call={
                "id": call_id or uuid.uuid4().hex[:8],
                "name": name,
                "arguments": arguments or {},
            },
        )

    @classmethod
    def turn_end(cls, **meta) -> "OmniEvent":
        return cls(OmniEventType.TURN_END, meta=meta)

    @classmethod
    def interrupted(cls, **meta) -> "OmniEvent":
        return cls(OmniEventType.INTERRUPTED, meta=meta)

    @classmethod
    def error(cls, message: str) -> "OmniEvent":
        return cls(OmniEventType.ERROR, text=message)


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class OmniBackend(ABC):
    """Contract every Omni backend implements.

    Lifecycle:
        await backend.start(tools=[...])      # open model/session, advertise tools
        await backend.send_audio(pcm)         # push input frames (PCM int16 16k mono)
        await backend.send_tool_result(id, s) # answer a TOOL_CALL event
        async for ev in backend.events(): ... # consume AUDIO/TEXT/TOOL_CALL/TURN_END/ERROR
        await backend.stop()                  # teardown
    """

    @abstractmethod
    async def start(self, tools: Optional[list[dict]] = None) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...

    @abstractmethod
    async def send_tool_result(self, call_id: str, result: str) -> None: ...

    async def send_text(self, text: str) -> None:
        """Inject a text turn into the live session (e.g. a system notice so the
        model speaks). Optional — backends that can't do this leave it a no-op."""
        return None

    @abstractmethod
    def events(self) -> AsyncIterator[OmniEvent]: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @property
    def backend_name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# StubOmniBackend — deterministic, scriptable (tests + offline dev)
# ---------------------------------------------------------------------------

_SENTINEL = object()


class StubOmniBackend(OmniBackend):
    """Deterministic backend driven entirely by a pre/inline-queued script.

    No timing, no threads — events come out in the exact order they were
    enqueued. Ordering against tool results is guaranteed because OmniSession
    awaits handle_tool_call (which calls send_tool_result) before pulling the
    next event, and send_tool_result enqueues its follow-ups synchronously.
    """

    def __init__(self, close_on_turn_end: bool = False):
        self._q: "asyncio.Queue[Any]" = asyncio.Queue()
        self.close_on_turn_end = close_on_turn_end
        self.started = False
        self.stopped = False
        self.advertised_tools: list[dict] = []
        self.received_audio: list[bytes] = []
        self.tool_results: list[tuple[str, str]] = []
        self.sent_texts: list[str] = []
        self.on_tool_result: Optional[Callable[[str, str], None]] = None

    def queue(self, *events: OmniEvent) -> "StubOmniBackend":
        for ev in events:
            self._q.put_nowait(ev)
        return self

    async def start(self, tools: Optional[list[dict]] = None) -> None:
        self.started = True
        self.advertised_tools = list(tools or [])

    async def send_audio(self, pcm: bytes) -> None:
        self.received_audio.append(pcm)

    async def send_tool_result(self, call_id: str, result: str) -> None:
        self.tool_results.append((call_id, result))
        if self.on_tool_result is not None:
            self.on_tool_result(call_id, result)

    async def send_text(self, text: str) -> None:
        self.sent_texts.append(text)

    async def events(self) -> AsyncIterator[OmniEvent]:
        while True:
            ev = await self._q.get()
            if ev is _SENTINEL:
                return
            yield ev
            if ev.type == OmniEventType.TURN_END and self.close_on_turn_end:
                return

    async def stop(self) -> None:
        self.stopped = True
        self._q.put_nowait(_SENTINEL)


# ---------------------------------------------------------------------------
# JobManager — background jobs (agent delegations + nonblocking custom tools)
# ---------------------------------------------------------------------------

@dataclass
class _Job:
    job_id: str
    kind: str           # "agent" | "tool"
    label: str          # query or tool name
    status: str = "running"   # running | completed | failed
    result: Optional[str] = None
    started: float = field(default_factory=time.monotonic)
    ended: Optional[float] = None
    task: Optional[asyncio.Task] = None
    session_id: Optional[str] = None   # for agent jobs: the delegated session id


class JobManager:
    """Fire-and-forget background runner with later result retrieval and a
    mini live-state view across multiple concurrent jobs.

    Backs both:
      - delegate-to-agent nonblocking (kind="agent")
      - nonblocking custom tools (kind="tool")
    One mechanism, no duplication.
    """

    def __init__(self, max_finished: int = 50):
        self._jobs: dict[str, _Job] = {}
        self._max_finished = max_finished
        self._name_counts: dict[str, int] = {}

    def _make_job_id(self, kind: str, label: str) -> str:
        """Human-readable, unique id: a slug of the label (or kind), plus a
        per-slug counter. So '29da9373' becomes e.g. 'fuehre-eine-1'. Readable
        for the user AND for the model when it calls agent_result(job_id)."""
        import unicodedata
        text = label or kind
        # German umlauts / ß to ASCII, then strip remaining accents
        text = (text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
                    .replace("Ä", "ae").replace("Ö", "oe").replace("Ü", "ue")
                    .replace("ß", "ss"))
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        slug = _re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        slug = "-".join(slug.split("-")[:3]) or kind  # max ~3 words, keep short
        slug = slug[:24] or kind
        n = self._name_counts.get(slug, 0) + 1
        self._name_counts[slug] = n
        candidate = f"{slug}-{n}"
        while candidate in self._jobs:  # paranoia: never collide
            n += 1
            self._name_counts[slug] = n
            candidate = f"{slug}-{n}"
        return candidate

    def spawn(self, kind: str, label: str, coro_factory: Callable[[], Awaitable[Any]],
              session_id: Optional[str] = None) -> str:
        job_id = self._make_job_id(kind, label)
        job = _Job(job_id=job_id, kind=kind, label=label, session_id=session_id)
        self._jobs[job_id] = job
        job.task = asyncio.ensure_future(self._run(job, coro_factory))
        self._evict()
        return job_id

    async def _run(self, job: _Job, coro_factory: Callable[[], Awaitable[Any]]) -> None:
        try:
            res = await coro_factory()
            job.result = res if isinstance(res, str) else str(res)
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "failed"
            job.result = "Error: cancelled"
            raise
        except Exception as e:  # noqa: BLE001 - jobs must never crash the session
            job.status = "failed"
            job.result = f"Error: {e}"
        finally:
            job.ended = time.monotonic()
            self._evict()

    def result(self, job_id: str) -> Optional[str]:
        job = self._jobs.get(job_id)
        if job is None or job.status == "running":
            return None
        return job.result

    def status(self, job_id: str) -> Optional[str]:
        job = self._jobs.get(job_id)
        return job.status if job else None

    def session_id(self, job_id: str) -> Optional[str]:
        job = self._jobs.get(job_id)
        return job.session_id if job else None

    def live_state(self) -> list[dict]:
        out = []
        for job in self._jobs.values():
            dur = (job.ended or time.monotonic()) - job.started
            out.append(
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "label": job.label[:60],
                    "status": job.status,
                    "duration_s": round(dur, 2),
                    "has_result": job.result is not None,
                    "session_id": job.session_id,
                }
            )
        out.sort(key=lambda d: d["job_id"])
        return out

    async def join(self, job_id: str, timeout: Optional[float] = None) -> Optional[str]:
        job = self._jobs.get(job_id)
        if job is None or job.task is None:
            return None
        try:
            await asyncio.wait_for(asyncio.shield(job.task), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        except Exception:  # noqa: BLE001
            pass
        return job.result

    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.task is None or job.task.done():
            return False
        job.task.cancel()
        try:
            await job.task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        if job.status == "running":
            job.status = "failed"
            job.result = "Error: cancelled"
            job.ended = time.monotonic()
        return True

    def _evict(self) -> None:
        finished = [jid for jid, j in self._jobs.items() if j.status != "running"]
        overflow = len(finished) - self._max_finished
        if overflow <= 0:
            return
        finished.sort(key=lambda jid: self._jobs[jid].ended or 0.0)
        for jid in finished[:overflow]:
            self._jobs.pop(jid, None)


# ---------------------------------------------------------------------------
# Agent delegation tools (delegate / agent_result / agent_status)
# ---------------------------------------------------------------------------

def _deleg_session_id(query: str) -> str:
    """Stable session id for a delegated query. Single source of truth so the
    delegate tool and the VFS-peek tools agree on the session name."""
    return f"deleg-{query[:12]}"


def make_agent_tools(
    jobs: JobManager,
    agent_provider: Callable[[str], Any],
) -> list[dict]:
    """Build the agent-delegation tools, bound to a JobManager.

    Tools:
        delegate(query, agent="default")  -> job_id  (returns immediately)
        agent_result(job_id)              -> result string or "<running>"
        agent_status()                    -> JSON live-state of all jobs
    """

    async def delegate(query: str, agent: str = "default") -> str:
        """Delegate a task to a background agent run. Returns a job_id immediately
        (nonblocking). Poll with agent_result(job_id) / agent_status()."""
        target = agent_provider(agent)
        if target is None:
            return f"Error: unknown agent {agent!r}"
        session_id = _deleg_session_id(query)

        def _factory() -> Awaitable[Any]:
            return target.a_run(query, session_id=session_id)

        job_id = jobs.spawn("agent", query, _factory, session_id=session_id)
        logger.info("omni.delegate: job=%s session=%s query=%r", job_id, session_id, query[:60])
        return job_id

    async def agent_result(job_id: str) -> str:
        """Fetch the result of a delegated job. Returns '<running>' if not done."""
        res = jobs.result(job_id)
        if res is None:
            st = jobs.status(job_id)
            return "<running>" if st == "running" else f"Error: unknown job {job_id!r}"
        return res

    async def agent_status() -> str:
        """Mini live-state of all running/finished delegated jobs as JSON."""
        return json.dumps(jobs.live_state(), ensure_ascii=False)

    return [
        {
            "tool_func": delegate,
            "name": "delegate",
            "description": (
                "Delegate a task to a background agent (nonblocking). "
                "Returns a job_id immediately; poll agent_result(job_id)."
            ),
            "category": ["agent", "delegate"],
        },
        {
            "tool_func": agent_result,
            "name": "agent_result",
            "description": "Fetch a delegated job's result by job_id, or '<running>'.",
            "category": ["agent", "read"],
        },
        {
            "tool_func": agent_status,
            "name": "agent_status",
            "description": "Live-state (JSON) of all delegated agent jobs.",
            "category": ["agent", "read"],
        },
    ]


# ---------------------------------------------------------------------------
# VFS peek tools — read-only window into a delegated agent's VFS
# ---------------------------------------------------------------------------

def _resolve_delegated_vfs(agent: Any, session_id: str) -> Any:
    """Return the VFS of a delegated agent session, or None. Uses the
    synchronous SessionManager.get (no create — peeking must never spin up a
    session)."""
    sm = getattr(agent, "session_manager", None)
    if sm is None:
        return None
    session = sm.get(session_id)
    if session is None:
        return None
    return getattr(session, "vfs", None)


def _slice_read_only(vfs: Any, path: str, line_start: int, line_end: int,
                     scroll_to: Optional[str], context_lines: int) -> dict:
    """Read a file slice WITHOUT mutating VFS state.

    Critically does NOT touch f.state / f.view_start / f.view_end — so the
    delegated agent's build_context_string() is unchanged. Goes through
    vfs.read() which lazy-loads shadow content but leaves state closed.
    """
    np = vfs._normalize_path(path)
    if not vfs._is_file(np):
        return {"success": False, "error": f"file not found: {path}"}

    rr = vfs.read(np, max_chars=10_000_000)
    if not rr.get("success"):
        return rr
    content = rr.get("content", "")
    all_lines = content.split("\n")
    total = len(all_lines)
    match_info: Optional[dict] = None

    if scroll_to:
        try:
            pat = _re.compile(scroll_to, _re.IGNORECASE)
        except _re.error:
            pat = _re.compile(_re.escape(scroll_to), _re.IGNORECASE)
        found = None
        for idx, ln in enumerate(all_lines, 1):
            if pat.search(ln):
                found = idx
                break
        if found is None:
            return {
                "success": False,
                "error": f"pattern '{scroll_to}' not found in {path}",
                "hint": f"file has {total} lines; try a different pattern or explicit line range",
            }
        half = max(1, context_lines // 2)
        line_start = max(1, found - half)
        line_end = min(total, found + half)
        match_info = {"matched_line": found, "pattern": scroll_to}

    start_idx = max(0, line_start - 1)
    end_idx = line_end if line_end > 0 else total
    end_idx = min(end_idx, total)
    visible = all_lines[start_idx:end_idx]

    out = {
        "success": True,
        "path": np,
        "content": "\n".join(visible),
        "showing": f"lines {start_idx + 1}-{end_idx} of {total}",
        "total_lines": total,
    }
    if match_info:
        out["match"] = match_info
    return out


def make_vfs_peek_tools(
    agent: Any,
    jobs: JobManager,
    *,
    default_max_chars: int = 6000,
) -> list[dict]:
    """Build read-only VFS inspection tools bound to the Omni agent + JobManager.

    Lets the Omni model look INTO a delegated agent's VFS directly, read-only,
    in tight slices, WITHOUT delegating again and WITHOUT mutating the delegated
    agent's context (no file opened, no state flipped). The agent's own
    build_context_string() is too bloated for the Omni model; these return only
    the exact slice asked for.

        vfs_peek(job_id, path, scroll_to=None, line_start=1, line_end=-1,
                 context_lines=40)        -> one focused file slice
        vfs_tree_peek(job_id, path="/", max_depth=2)   -> partial tree, on request only
    """

    def _vfs_for_job(job_id: str) -> tuple[Any, Optional[str]]:
        session_id = jobs.session_id(job_id)
        if session_id is None:
            return None, f"unknown job {job_id!r}"
        vfs = _resolve_delegated_vfs(agent, session_id)
        if vfs is None:
            return None, f"no VFS for session {session_id!r}"
        return vfs, None

    async def vfs_peek(
        job_id: str,
        path: str,
        scroll_to: Optional[str] = None,
        line_start: int = 1,
        line_end: int = -1,
        context_lines: int = 40,
    ) -> str:
        """Look at a tight slice of one file in a delegated agent's VFS, read-only.
        Use scroll_to to center on a pattern; otherwise give a line range. Does
        not open the file or change the other agent's context."""
        vfs, err = _vfs_for_job(job_id)
        if vfs is None:
            return json.dumps({"success": False, "error": err})
        res = _slice_read_only(vfs, path, int(line_start), int(line_end),
                               scroll_to, int(context_lines))
        if res.get("success") and len(res.get("content", "")) > default_max_chars:
            res["content"] = res["content"][:default_max_chars] + "\n... [slice truncated]"
            res["truncated"] = True
        return json.dumps(res, ensure_ascii=False)

    async def vfs_tree_peek(job_id: str, path: str = "/", max_depth: int = 2) -> str:
        """Show a PARTIAL directory tree of a delegated agent's VFS (on request only,
        shallow by default to stay cheap). Read-only; does not alter context."""
        vfs, err = _vfs_for_job(job_id)
        if vfs is None:
            return json.dumps({"success": False, "error": err})
        try:
            np = vfs._normalize_path(path)
            tree = vfs._build_tree_string(np, max_depth=int(max_depth))
            return json.dumps(
                {"success": True, "path": np, "tree": tree or "(empty)"},
                ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"success": False, "error": str(e)})

    return [
        {
            "tool_func": vfs_peek,
            "name": "vfs_peek",
            "description": (
                "Read-only peek into a delegated agent's VFS file (by job_id). "
                "Returns a tight slice; use scroll_to to center on a pattern. "
                "Does NOT open the file or change the other agent's context."
            ),
            "category": ["vfs", "read"],
        },
        {
            "tool_func": vfs_tree_peek,
            "name": "vfs_tree_peek",
            "description": (
                "Show a partial, shallow directory tree of a delegated agent's VFS "
                "(by job_id). On request only. Read-only."
            ),
            "category": ["vfs", "read"],
        },
    ]


def _build_fallback(agent, recorder, player):
    """Wire the classic STT->LLM->TTS LiveModeEngine as an OmniSession fallback.

    Returns an object exposing async start()/stop(). Best-effort: the live mode
    is decoupled and may be unstable, so failures here must not crash the session
    (OmniSession._switch_to_fallback already guards start()).
    """
    try:
        from toolboxv2.mods.isaa.base.audio_io.audio_live import (
            LiveModeEngine, LiveModeConfig,
        )
    except Exception as e:
        logging.getLogger("isaa_voice").warning("fallback unavailable: %s", e)
        return None

    class _PipelineFallback:
        def __init__(self):
            self._engine = None

        async def start(self):
            #if verbose:
            print(Style.YELLOW("[fallback] starting classic STT->LLM->TTS pipeline"))

            async def _on_utterance(audio_bytes, speaker=None):
                #if verbose:
                print(Style.YELLOW("[fallback] STT->LLM"))
                audio_output, text_output, tool_calls, metadata = await agent.a_audio(audio_bytes)
                #if verbose:
                print(Style.YELLOW("[fallback] LLM->TTS"))
                await player.queue_audio(audio_output, metadata)

            self._engine = LiveModeEngine(
                config=LiveModeConfig(),
                on_utterance=_on_utterance,
                recorder=recorder,
            )
            await self._engine.start()

        async def stop(self):
            if self._engine is not None:
                await self._engine.stop()

    return _PipelineFallback()
# ---------------------------------------------------------------------------
# OmniSession — the loop
# ---------------------------------------------------------------------------

class OmniSession:
    """Wires recorder -> Omni backend -> player, and bridges tool-calls into
    the existing ToolManager. Decoupled from FlowAgent's text ReAct loop on
    purpose (FlowAgent stays the heavy thinker behind the `delegate` tool).

    Audio output buffering
    ----------------------
    Omni backends (Gemini etc.) stream audio as MANY small PCM chunks. Playing
    each chunk with its own sd.play()/sd.wait() makes the next play() cut off
    the previous one — you hear only the first ~half-second. So this session
    BUFFERS chunks per turn and flushes them as ONE contiguous WAV when the turn
    ends (or on interruption / explicit flush). Set buffer_audio=False to forward
    each chunk immediately (only sensible for players that concatenate).

    All collaborators are injected and duck-typed:
        backend          : OmniBackend
        recorder         : AudioRecorder-like (start/stop/frames())  [optional]
        player           : AudioPlayer-like (start/stop/queue_audio())  [optional]
        tools            : ToolManager-like (execute/get_function)  [optional]
        jobs             : JobManager (created if None)
        background_tools : set[str] tool names to run fire-and-forget
        fallback         : object with start()/stop() — classic pipeline, on backend ERROR
        enhancer         : object with enhance(wav_bytes)->wav_bytes  [optional]
        output_sample_rate : default SR for AUDIO events lacking meta['sample_rate_out']
    """

    def __init__(
        self,
        backend: OmniBackend,
        recorder: Any = None,
        player: Any = None,
        tools: Any = None,
        *,
        jobs: Optional[JobManager] = None,
        background_tools: Optional[set[str]] = None,
        fallback: bool = None,
        sample_rate: int = TARGET_SR,
        output_sample_rate: Optional[int] = None,
        on_text: Optional[Callable[[str], Any]] = None,
        buffer_audio: bool = True,
        enhancer: Any = None,
        vad: Any = None,
        vad_threshold: float = 0.5,
        vad_hangover_frames: int = 8,
        on_job_done: Optional[Callable[[dict], Any]] = None,
    ):
        self.backend = backend
        self.recorder = recorder
        self.player = player
        self.tools = tools
        self.jobs = jobs or JobManager()
        self.background_tools = set(background_tools or set())

        self.fallback = _build_fallback(agent=fallback, recorder=recorder, player=player) if fallback is not None else None
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate
        self.on_text = on_text
        self.buffer_audio = buffer_audio
        self.enhancer = enhancer

        # VAD gate: only stream audio to the backend while someone is speaking.
        # Saves cloud cost (no silence streamed). vad must expose is_speech(pcm)->float.
        self.vad = vad
        self.vad_threshold = vad_threshold
        self.vad_hangover_frames = vad_hangover_frames
        self._vad_hangover = 0  # frames still sent after speech stops (avoid clipping)

        # Notify when a delegated agent job finishes (so the system can speak).
        self.on_job_done = on_job_done
        self._announced_jobs: set[str] = set()
        self._notify_task: Optional[asyncio.Task] = None

        self._pump_task: Optional[asyncio.Task] = None
        self._event_task: Optional[asyncio.Task] = None
        self._running = False
        self._fell_back = False

        # per-turn audio buffer: list of (pcm_bytes, sample_rate)
        self._audio_buf: list[tuple[bytes, int]] = []

        # introspection
        self.audio_chunks_out = 0
        self.audio_flushes = 0
        self.tool_calls_handled = 0
        self.turns = 0
        self.frames_sent = 0
        self.frames_gated = 0

    # lifecycle --------------------------------------------------------------
    async def start(self, tool_specs: Optional[list[dict]] = None) -> None:
        logger.info("OmniSession.start backend=%s tools=%d", self.backend.backend_name,
                    len(tool_specs or []))
        try:
            await self.backend.start(tools=tool_specs)
        except Exception as e:  # noqa: BLE001
            logger.warning("Omni backend start failed (%s) — falling back.", e)
            import traceback
            logger.debug(traceback.format_exc())
            await self._switch_to_fallback()
            return

        if self.recorder is not None:
            await self.recorder.start()
            logger.info("OmniSession: recorder started (%s)",
                        getattr(self.recorder, "recorder_type", type(self.recorder).__name__))
        if self.player is not None:
            await self.player.start()
            logger.info("OmniSession: player started (%s)", type(self.player).__name__)

        self._running = True
        self._event_task = asyncio.ensure_future(self._consume_events())
        if self.recorder is not None:
            self._pump_task = asyncio.ensure_future(self._pump_audio())
        self._notify_task = asyncio.ensure_future(self._notify_loop())
        logger.info("OmniSession: live loop running")

    async def stop(self) -> None:
        logger.info("OmniSession.stop — %s", self.status_line())
        self._running = False
        for t in (self._pump_task, self._event_task, self._notify_task):
            if t is not None and not t.done():
                t.cancel()
        await self.backend.stop()
        if self.recorder is not None:
            await self.recorder.stop()
        if self.player is not None:
            await self.player.stop()
        if self.fallback is not None:
            await self.fallback.stop()

    async def wait(self, timeout: Optional[float] = None) -> None:
        if self._event_task is None:
            return
        await asyncio.wait_for(asyncio.shield(self._event_task), timeout=timeout)

    # internal loops ---------------------------------------------------------
    def _should_send_frame(self, frame: bytes) -> bool:
        """VAD gate. True if this frame should be streamed to the backend.

        No VAD configured -> always send (open mic). With VAD: send while speech
        prob >= threshold, plus a short hangover of N frames after speech stops
        so word endings aren't clipped. Frames where the VAD hasn't accumulated
        enough samples yet (is_speech returns -1.0) inherit the current state.
        """
        if self.vad is None:
            return True
        try:
            prob = self.vad.is_speech(frame)
        except Exception as e:  # noqa: BLE001 - VAD must never break the stream
            logger.debug("VAD error, sending frame: %s", e)
            return True
        if prob < 0:  # not enough samples buffered yet -> keep current behaviour
            return self._vad_hangover > 0
        if prob >= self.vad_threshold:
            self._vad_hangover = self.vad_hangover_frames
            return True
        if self._vad_hangover > 0:
            self._vad_hangover -= 1
            return True
        return False

    async def _pump_audio(self) -> None:
        try:
            async for frame in self.recorder.frames():
                if not self._running:
                    break
                if not self._should_send_frame(frame):
                    self.frames_gated += 1
                    continue
                await self.backend.send_audio(frame)
                self.frames_sent += 1
                if self.frames_sent % 50 == 0:
                    logger.debug("OmniSession: %d frames sent, %d gated (silence)",
                                 self.frames_sent, self.frames_gated)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession audio pump stopping: %s", e)
            self._running = False

    async def _consume_events(self) -> None:
        try:
            async for ev in self.backend.events():
                await self.dispatch(ev)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession event loop error: %s", e)

    async def _notify_loop(self, poll_s: float = 0.5) -> None:
        """Watch the JobManager; when a delegated agent job finishes, announce it
        ONCE so the model speaks the result. Replaces the caller polling
        live_state() in a print loop (that was the repeating-log noise)."""
        try:
            while self._running:
                await asyncio.sleep(poll_s)
                for st in self.jobs.live_state():
                    jid = st["job_id"]
                    if st["kind"] != "agent" or st["status"] == "running":
                        continue
                    if jid in self._announced_jobs:
                        continue
                    self._announced_jobs.add(jid)
                    await self._announce_job(jid, st)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession notify loop error: %s", e)

    async def _announce_job(self, job_id: str, state: dict) -> None:
        """Tell the model a delegated job finished, so it speaks. Custom handler
        (on_job_done) wins; otherwise inject a system text turn into the backend."""
        result = self.jobs.result(job_id) or ""
        logger.info("OmniSession: job %s (%s) done -> announcing", job_id, state.get("label"))
        if self.on_job_done is not None:
            res = self.on_job_done({**state, "result": result})
            if asyncio.iscoroutine(res):
                await res
            return
        label = state.get("label", job_id)
        note = (
            f"[system] The delegated task '{label}' (job {job_id}) just finished. "
            f"Briefly tell the user it's done and summarize the result:\n{result[:1500]}"
        )
        try:
            await self.backend.send_text(note)
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession: announce send_text failed: %s", e)

    # audio buffering --------------------------------------------------------
    def _buffer_or_play(self, ev: OmniEvent) -> Optional[Awaitable]:
        sr = int(ev.meta.get("sample_rate_out") or self.output_sample_rate or self.sample_rate)
        if self.buffer_audio:
            self._audio_buf.append((ev.audio, sr))
            return None
        return self._emit_wav(ev.audio, sr, dict(ev.meta))

    async def _flush_audio(self) -> None:
        """Concatenate buffered same-rate PCM and hand the player ONE wav."""
        if not self._audio_buf or self.player is None:
            self._audio_buf.clear()
            return
        # group consecutive chunks by sample rate (normally all identical)
        groups: list[tuple[int, bytearray]] = []
        for pcm, sr in self._audio_buf:
            if groups and groups[-1][0] == sr:
                groups[-1][1].extend(pcm)
            else:
                groups.append((sr, bytearray(pcm)))
        self._audio_buf.clear()
        for sr, buf in groups:
            await self._emit_wav(bytes(buf), sr, {})
        self.audio_flushes += 1
        logger.debug("OmniSession: flushed audio (%d group(s))", len(groups))

    async def _emit_wav(self, pcm: bytes, sr: int, meta: dict) -> None:
        if self.player is None or not pcm:
            return
        wav = pcm16_to_wav(pcm, sr)
        if self.enhancer is not None:
            try:
                logger.debug("OmniSession: enhancing wav")
                wav = self.enhancer.enhance(wav)
                logger.debug("OmniSession: done enhancing wav")
            except Exception as e:  # noqa: BLE001
                logger.warning("OmniSession: enhancer failed, using raw audio: %s", e)
        await self.player.queue_audio(wav, meta)

    # event dispatch ---------------------------------------------------------
    async def dispatch(self, ev: OmniEvent) -> None:
        if ev.type == OmniEventType.AUDIO:
            self.audio_chunks_out += 1
            if ev.audio:
                awaitable = self._buffer_or_play(ev)
                if awaitable is not None:
                    await awaitable
        elif ev.type == OmniEventType.TEXT:
            if ev.text:
                logger.debug("OmniSession TEXT[%s]: %s", ev.meta.get("source", "?"), ev.text[:80])
            if self.on_text is not None and ev.text is not None:
                res = self.on_text(ev.text)
                if asyncio.iscoroutine(res):
                    await res
        elif ev.type == OmniEventType.TOOL_CALL:
            await self.handle_tool_call(ev.tool_call or {})
        elif ev.type == OmniEventType.TURN_END:
            self.turns += 1
            logger.info("OmniSession: turn %d complete (%d audio chunks buffered)",
                        self.turns, len(self._audio_buf))
            await self._flush_audio()
        elif ev.type == OmniEventType.INTERRUPTED:
            logger.info("OmniSession: user barge-in — dropping buffered audio")
            self._audio_buf.clear()
        elif ev.type == OmniEventType.ERROR:
            logger.warning("Omni backend error: %s", ev.text)
            await self._switch_to_fallback()

    # tool bridge ------------------------------------------------------------
    async def handle_tool_call(self, call: dict) -> str:
        call_id = call.get("id") or uuid.uuid4().hex[:8]
        name = call.get("name", "")
        args = call.get("arguments") or {}
        self.tool_calls_handled += 1
        logger.info("OmniSession tool-call: %s(%s)", name, ", ".join(args.keys()))

        if self.tools is None:
            result = f"Error: no tool provider for {name!r}"
            await self.backend.send_tool_result(call_id, result)
            return result

        if self.tools.get_function(name) is None:
            result = f"Error: unknown tool {name!r}"
            logger.warning("OmniSession: %s", result)
            await self.backend.send_tool_result(call_id, result)
            return result

        if name in self.background_tools:
            def _factory() -> Awaitable[Any]:
                return self.tools.execute(name, **args)
            job_id = self.jobs.spawn("tool", name, _factory)
            result = json.dumps({"job_id": job_id, "status": "started"})
            await self.backend.send_tool_result(call_id, result)
            return result

        try:
            raw = await self.tools.execute(name, **args)
            result = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            result = f"Error: {e}"
            logger.warning("OmniSession: tool %s raised: %s", name, e)
        await self.backend.send_tool_result(call_id, result)
        return result

    async def _switch_to_fallback(self) -> None:
        if self._fell_back or self.fallback is None:
            return
        self._fell_back = True
        logger.info("OmniSession: switching to classic STT->LLM->TTS fallback.")
        try:
            await self.fallback.start()
        except Exception as e:  # noqa: BLE001
            logger.error("Fallback start failed: %s", e)

    @property
    def fell_back(self) -> bool:
        return self._fell_back

    def status_line(self) -> str:
        return (
            f"backend={self.backend.backend_name} running={self._running} "
            f"audio_out={self.audio_chunks_out} flushes={self.audio_flushes} "
            f"tool_calls={self.tool_calls_handled} turns={self.turns} "
            f"fallback={self._fell_back}"
        )


# ---------------------------------------------------------------------------
# VoiceModeConfig — config + backend factory
# ---------------------------------------------------------------------------

@dataclass
class VoiceModeConfig:
    """Selects the voice path. `build_backend()` returns the matching backend.

    mode:
        "omni_local"  -> LocalOmniBackend (Qwen2.5-Omni-7B-AWQ)
        "omni_cloud"  -> GeminiLiveBackend (Gemini Live)
        "stub"        -> StubOmniBackend (tests / offline)
        "pipeline"    -> None (caller uses the classic STT/LLM/TTS path)
    Pass `custom` to inject any OmniBackend directly (overrides mode).
    """

    mode: str = "stub"
    local_model_id: str = "Qwen/Qwen2.5-Omni-7B-AWQ"
    cloud_model_id: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    device: str = "cuda"
    sample_rate: int = TARGET_SR
    api_key_env: str = "GEMINI_API_KEY"
    custom: Optional[OmniBackend] = None

    def build_backend(self) -> Optional[OmniBackend]:
        if self.custom is not None:
            return self.custom
        if self.mode == "stub":
            return StubOmniBackend()
        if self.mode == "omni_local":
            return LocalOmniBackend(self.local_model_id, device=self.device,
                                    sample_rate=self.sample_rate)
        if self.mode == "omni_cloud":
            from toolboxv2.mods.isaa.base.audio_io.native.omni_gemini import GeminiLiveBackend
            return GeminiLiveBackend(self.cloud_model_id, api_key_env=self.api_key_env)
        if self.mode == "pipeline":
            return None
        raise ValueError(f"Unknown voice mode: {self.mode!r}")


# ===========================================================================
# NETWORK / MODEL BACKENDS — lazy, guarded, NOT unit-tested
# ===========================================================================

class LocalOmniBackend(OmniBackend):
    """Qwen2.5-Omni-7B-AWQ end-to-end on local GPU (~8-10 GB VRAM at 4-bit AWQ).

    Integration seam: the Thinker-Talker streaming + mid-stream tool-call
    extraction is model-version specific.
    """

    def __init__(self, model_id: str, device: str = "cuda", sample_rate: int = TARGET_SR):
        self.model_id = model_id
        self.device = device
        self.sample_rate = sample_rate
        self._model = None
        self._q: "asyncio.Queue[Any]" = asyncio.Queue()
        self._in_buf = bytearray()
        self._tools: list[dict] = []

    async def start(self, tools: Optional[list[dict]] = None) -> None:
        self._tools = list(tools or [])
        try:
            from toolboxv2.mods.isaa.base.audio_io import model_registry  # type: ignore
            self._model = model_registry.get_qwen3(self.model_id, device=self.device)
        except Exception as e:  # noqa: BLE001
            await self._q.put(OmniEvent.error(f"local omni load failed: {e}"))
            raise

    async def send_audio(self, pcm: bytes) -> None:
        self._in_buf += pcm

    async def send_tool_result(self, call_id: str, result: str) -> None:
        raise NotImplementedError("wire to Qwen2.5-Omni tool-result re-injection")

    async def events(self) -> AsyncIterator[OmniEvent]:
        while True:
            ev = await self._q.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def stop(self) -> None:
        await self._q.put(_SENTINEL)


class CloudOmniBackend(OmniBackend):
    """Legacy DashScope (Qwen-Omni-Flash) seam. Prefer GeminiLiveBackend via
    omni_gemini.py for the working cloud path; kept for parity/future use."""

    def __init__(self, model_id: str, api_key_env: str = "DASHSCOPE_API_KEY",
                 sample_rate: int = TARGET_SR):
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.sample_rate = sample_rate
        self._ws = None
        self._q: "asyncio.Queue[Any]" = asyncio.Queue()
        self._tools: list[dict] = []

    async def start(self, tools: Optional[list[dict]] = None) -> None:
        import os
        self._tools = list(tools or [])
        key = os.getenv(self.api_key_env)
        if not key:
            await self._q.put(OmniEvent.error(f"{self.api_key_env} not set"))
            raise RuntimeError(f"{self.api_key_env} not set")
        raise NotImplementedError("open DashScope realtime websocket session")

    async def send_audio(self, pcm: bytes) -> None:
        raise NotImplementedError("ws.send append-audio-buffer event")

    async def send_tool_result(self, call_id: str, result: str) -> None:
        raise NotImplementedError("ws.send function_call_output event")

    async def events(self) -> AsyncIterator[OmniEvent]:
        while True:
            ev = await self._q.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def stop(self) -> None:
        await self._q.put(_SENTINEL)
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
