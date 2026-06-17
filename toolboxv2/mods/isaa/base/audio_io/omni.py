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
    from toolboxv2 import get_logger
    from toolboxv2.utils.extras import Style
    logger = get_logger()
except ImportError as e:
    print("Failed to import toolboxv2 logger", e)
    logger = logging.getLogger(__name__)

TARGET_SR = 16000


# ---------------------------------------------------------------------------
# PCM helpers
# ---------------------------------------------------------------------------

def _wav_to_pcm(wav_bytes: bytes) -> "tuple[bytes, int]":
    """Reverse of pcm16_to_wav: raw PCM + sample rate from a WAV container.
    Stdlib only. Assumes mono (TTS output). Returns (b"", TARGET_SR) on failure."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.readframes(w.getnframes()), w.getframerate()
    except Exception:  # noqa: BLE001
        return b"", TARGET_SR

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
# Lifecycle phases (update 1) — observable session state machine
# ---------------------------------------------------------------------------

class OmniPhase(Enum):
    WAITING = "waiting"                  # waiting for user speech
    SPEECH_DETECTED = "speech_detected"  # VAD rising edge
    SPEAKER_DETECTED = "speaker_detected"  # update 2 (optional)
    SPEAKING = "speaking"                # user speaking
    SPEECH_END = "speech_end"            # VAD falling edge (hangover spent)
    THINKING = "thinking"                # model internal processing
    TOOL_START = "tool_start"            # meta: name, arguments
    TOOL_END = "tool_end"                # meta: name, result
    AUDIO_PROCESSING = "audio_processing"  # decode/enhance/buffer
    AUDIO_PLAYING = "audio_playing"      # handed to player


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
    # capability flags OmniSession honours (overridden per backend) -----------
    needs_silence = False  # True -> OmniSession streams ALL frames (no VAD gate)
    supports_restart = True  # False -> OmniSession never restart/reseeds this backend

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

    async def send_media(self, data: bytes, mime_type: str, *, as_turn: bool = False) -> None:
        """Send an image / video frame|blob (update 3). as_turn=True -> discrete
        content turn (one-shot image/file); else realtime video frame. Optional —
        no-op by default."""
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
        self.received_media: list[tuple[bytes, str, bool]] = []
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

    async def send_media(self, data: bytes, mime_type: str, *, as_turn: bool = False) -> None:
        self.received_media.append((data, mime_type, as_turn))

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
    last_thought: str = ""             # streamed reasoning/narrator (agent_progress)
    last_tool: str = ""                # current/last tool name
    last_tool_result: str = ""         # last tool result (truncated)
    iteration: int = 0                 # last reported ReAct iteration


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

    def spawn_stream(self, kind: str, label: str, agen_factory: Callable[[], Any],
                     session_id: Optional[str] = None) -> str:
        """Like spawn(), but agen_factory returns an async generator (a_stream).
        Streams progress (thought/tool/iteration) into the _Job so agent_progress
        can show the delegate's live thinking. Final result from done/final_answer."""
        job_id = self._make_job_id(kind, label)
        job = _Job(job_id=job_id, kind=kind, label=label, session_id=session_id)
        self._jobs[job_id] = job
        job.task = asyncio.ensure_future(self._run_stream(job, agen_factory))
        self._evict()
        return job_id

    async def _run_stream(self, job: _Job, agen_factory: Callable[[], Any]) -> None:
        final, acc = "", ""
        try:
            async for ch in agen_factory():
                t = ch.get("type", "")
                if t == "reasoning":
                    job.last_thought = (ch.get("chunk") or "")[:500]
                elif t == "narrator":
                    job.last_thought = (ch.get("narrator_msg") or "")[:500]
                elif t == "tool_start":
                    job.last_tool, job.last_tool_result = ch.get("name", ""), ""
                elif t == "tool_result":
                    job.last_tool = ch.get("name", job.last_tool)
                    job.last_tool_result = str(ch.get("result", ""))[:500]
                elif t == "iteration_start":
                    job.iteration = ch.get("iteration", job.iteration)
                elif t == "content":
                    acc += ch.get("chunk", "")
                elif t == "final_answer":
                    final = ch.get("answer", "") or final
                elif t == "done":
                    final = ch.get("final_answer", "") or final
                elif t == "error":
                    job.status, job.result = "failed", f"Error: {ch.get('error', '')}"
                    return
            job.result = final or acc
            job.status = "completed"
        except asyncio.CancelledError:
            job.status, job.result = "failed", "Error: cancelled"
            raise
        except Exception as e:  # noqa: BLE001
            job.status, job.result = "failed", f"Error: {e}"
        finally:
            job.ended = time.monotonic()
            self._evict()

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

    def progress(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job.job_id, "status": job.status, "iteration": job.iteration,
            "last_thought": job.last_thought, "last_tool": job.last_tool,
            "last_tool_result": job.last_tool_result,
            "has_result": job.result is not None,
        }

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
# Persistent state — World Model + summaries + resumable history
# ---------------------------------------------------------------------------

@dataclass
class WorldModel:
    """Tiny, flat, always-on model of the user + agent role.

    Re-injected on every session restart, so it MUST stay small. Deep/volatile
    data (people graph, detailed attributes, workflows) does NOT live here — it
    is referenced by short recall-keys in `routines` and fetched on demand via
    the agent's memory_recall.
    """
    user: str = ""        # who the user is, one line
    agent_role: str = ""  # what the user expects the agent to be, one line
    routines: list[str] = field(default_factory=list)  # recall-keys for recurring tasks

    def to_dict(self) -> dict:
        return {"user": self.user, "agent_role": self.agent_role,
                "routines": list(self.routines)}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WorldModel":
        d = d or {}
        return cls(
            user=str(d.get("user", "")),
            agent_role=str(d.get("agent_role", "")),
            routines=list(d.get("routines", []) or []),
        )

    def render(self) -> str:
        """Compact text block for the seed/system prompt."""
        lines = []
        if self.user:
            lines.append(f"User: {self.user}")
        if self.agent_role:
            lines.append(f"Your role: {self.agent_role}")
        if self.routines:
            lines.append("Known routines (recall-keys): " + ", ".join(self.routines))
        return "\n".join(lines)


@dataclass
class OmniState:
    """Everything that survives a session restart / process restart."""
    world_model: WorldModel = field(default_factory=WorldModel)
    full_summary: str = ""                                       # rolling summary of the live session
    session_summaries: list[str] = field(default_factory=list)  # one ultra-short line per past session
    active_history: list[dict] = field(default_factory=list)    # recent {role, text} turns to resume

    def to_dict(self) -> dict:
        return {
            "world_model": self.world_model.to_dict(),
            "full_summary": self.full_summary,
            "session_summaries": list(self.session_summaries),
            "active_history": list(self.active_history),
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "OmniState":
        d = d or {}
        return cls(
            world_model=WorldModel.from_dict(d.get("world_model")),
            full_summary=str(d.get("full_summary", "")),
            session_summaries=list(d.get("session_summaries", []) or []),
            active_history=list(d.get("active_history", []) or []),
        )


class BlobStateStore:
    """Auto-persisting JSON store for OmniState, backed by an INJECTED BlobFile.

    BlobFile is injected (not imported) so this core stays toolboxv2-free and
    unit-testable with a fake. Reading a missing/empty blob returns {} (safe),
    so a fresh store starts from a default OmniState. Encryption: key=None lets
    the BlobFile use the storage's own (device-bound) crypto; pass a key to
    override. Save uses mode "w" (clean overwrite — avoids write_json's append).
    """

    def __init__(self, path: str, blob_file_cls: Any, key: Optional[bytes] = None):
        self._path = path
        self._BlobFile = blob_file_cls
        self._key = key
        self.state: OmniState = self.load()

    def load(self) -> OmniState:
        with self._BlobFile(self._path, "r", key=self._key) as f:
            raw = f.read_json()
        self.state = OmniState.from_dict(raw if isinstance(raw, dict) else {})
        return self.state

    def save(self) -> None:
        with self._BlobFile(self._path, "w", key=self._key) as f:
            f.write_json(self.state.to_dict())


async def _summarize_omni(
    agent: Any,
    transcript: str,
    prior_summary: str = "",
    max_tokens: int = 400,
    one_line: bool = False,
) -> str:
    """Single fast LLM completion that (re)builds a rolling summary. Uses
    a_run_llm_completion (NOT a_run) — no ReAct, no tools, no session context."""
    if one_line:
        instr = (
            "Compress the following summary into ONE short line (max 20 words). "
            "Keep only the most important durable facts and the outcome."
        )
        body = transcript
    else:
        instr = (
            "You maintain a rolling summary of an ongoing voice conversation. "
            "Produce an updated, compact summary preserving the user's goals, "
            "decisions, open threads and key facts. No preamble, just the summary."
        )
        body = (f"Previous summary:\n{prior_summary}\n\n" if prior_summary else "") \
               + f"Recent transcript:\n{transcript}"
    out = await agent.a_run_llm_completion(
        messages=[{"role": "system", "content": instr},
                  {"role": "user", "content": body}],
        model_preference="fast",
        with_context=False,
        stream=False,
        max_tokens=max_tokens,
        temperature=0.3,
        task_id="omni_summary",
    )
    return (out or "").strip()


def make_world_model_tools(store: BlobStateStore) -> list[dict]:
    """The atomic, token-cheap World Model editor (one tool)."""

    _FIELDS = ("user", "agent_role", "routines")

    async def world_model_edit(field_name: str, op: str = "set", value: str = "") -> str:
        """Atomically edit the always-on World Model.

        field_name: 'user' | 'agent_role' | 'routines'
        op: 'set' (user/agent_role), 'append'/'remove' (routines only)
        value: one short line of text, or a single recall-key.

        Call ONLY when you learn a STABLE fact: who the user is, what role they
        expect of you, or a recurring task (stored as a short recall-key). Do NOT
        use it for one-off facts or detailed/volatile data — those belong in
        memory, not the always-on model.
        """
        wm = store.state.world_model
        if field_name not in _FIELDS:
            return f"Error: unknown field {field_name!r}; valid: {', '.join(_FIELDS)}"
        if field_name in ("user", "agent_role"):
            if op != "set":
                return f"Error: field {field_name!r} only supports op='set'"
            setattr(wm, field_name, value.strip())
        else:  # routines
            v = value.strip()
            if op == "append":
                if v and v not in wm.routines:
                    wm.routines.append(v)
            elif op == "remove":
                if v in wm.routines:
                    wm.routines.remove(v)
            else:
                return "Error: 'routines' supports op='append' or 'remove'"
        store.save()
        return json.dumps({"ok": True, "world_model": wm.to_dict()}, ensure_ascii=False)

    return [
        {
            "tool_func": world_model_edit,
            "name": "world_model_edit",
            "description": (
                "Atomically edit the always-on World Model. field_name: 'user' "
                "(who the user is), 'agent_role' (what role they expect of you), "
                "'routines' (recurring tasks as short recall-keys). op='set' for "
                "user/agent_role; op='append'/'remove' for routines. Call ONLY for "
                "STABLE facts; one short line per value. Not for one-off or detailed data."
            ),
            "category": ["world_model", "memory"],
        }
    ]


def make_session_tools(session: "OmniSession") -> list[dict]:
    """The compress_session tool (one tool). Non-blocking: schedules the work
    and returns immediately so the live audio loop is never stalled by the LLM
    summary call."""

    async def compress_session(merge_old: bool = True) -> str:
        """Compress the conversation so far into the rolling summary and clear the
        replayed history, keeping latency/quality high.

        WHEN to call: at every topic change, and once the exchange has more than a
        few turns — at the latest before the session gets long. merge_old=True folds
        the existing summary into the new one (default); False starts the summary
        fresh from recent turns only. Returns immediately; compression runs in the
        background.
        """
        session.request_compress(merge_old=merge_old)
        return "compression scheduled"

    return [
        {
            "tool_func": compress_session,
            "name": "compress_session",
            "description": (
                "Compress the conversation so far into a rolling summary and drop "
                "the replayed history (keeps the voice model fast and accurate). "
                "Call on EVERY topic change, once past a few turns, and before the "
                "session gets long. merge_old=True (default) folds the old summary in."
            ),
            "category": ["session", "memory"],
        }
    ]

# ---------------------------------------------------------------------------
# Agent delegation tools (delegate / agent_result / agent_status)
# ---------------------------------------------------------------------------

def _deleg_session_id(query: str) -> str:
    """Stable, FILESYSTEM-SAFE session id for a delegated query. The id becomes a
    memory filename, so strip to [A-Za-z0-9]; quotes/spaces/etc. raise OSError
    [Errno 22] on Windows. Single source of truth for delegate + VFS-peek tools."""
    slug = _re.sub(r"[^A-Za-z0-9]+", "-", (query or "").strip()).strip("-")[:12] or "task"
    return f"deleg-{slug}"


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

        def _factory():
            return target.a_stream(query, session_id=session_id)

        job_id = jobs.spawn_stream("agent", query, _factory, session_id=session_id)
        logger.info("omni.delegate: job=%s session=%s query=%r", job_id, session_id, query[:60])
        return job_id

    async def agent_result(job_id: str) -> str:
        """Fetch the result of a delegated job. Returns '<running>' if not done."""
        res = jobs.result(job_id)
        if res is None:
            st = jobs.status(job_id)
            return "<running>" if st == "running" else f"Error: unknown job {job_id!r}"
        return res

    # ponytail: track which finished jobs agent_status already surfaced, so a
    # newly-finished delegation is reported ONCE — not re-listed as "just done"
    # on every later completion.
    reported: set[str] = set()

    async def agent_status() -> str:
        """Live-state of delegated jobs as JSON. Running jobs always shown;
        each finished job is flagged 'finished_new' exactly once, then
        'finished' on later calls — so the model announces it only once."""
        out = []
        for st in jobs.live_state():
            jid = st["job_id"]
            if st["status"] == "running":
                out.append(st)
                continue
            # finished/failed
            if jid in reported:
                st = {**st, "status": f"{st['status']} (already reported)"}
            else:
                reported.add(jid)
                st = {**st, "status": f"{st['status']} (new)"}
            out.append(st)
        return json.dumps(out, ensure_ascii=False)

    async def agent_progress(job_id: str) -> str:
        """Show a delegate's CURRENT thinking: last reasoning/narrator, current
        tool + last tool result, iteration, status. JSON."""
        p = jobs.progress(job_id)
        if p is None:
            return f"Error: unknown job {job_id!r}"
        return json.dumps(p, ensure_ascii=False)

    async def agent_stop(job_id: str) -> str:
        """Stop (cancel) a running delegated job by job_id. Idempotent."""
        ok = await jobs.cancel(job_id)
        return "stopped" if ok else f"job {job_id!r} not running or unknown"

    async def agent_resume(job_id: str, query: str, agent: str = "default") -> str:
        """Resume a delegation's SESSION with new context. Cancels the old run if
        active, then re-streams on the SAME session (a_stream auto-resumes it)."""
        session_id = jobs.session_id(job_id)
        if session_id is None:
            return f"Error: unknown job {job_id!r}"
        target = agent_provider(agent)
        if target is None:
            return f"Error: unknown agent {agent!r}"
        if jobs.status(job_id) == "running":
            await jobs.cancel(job_id)

        def _factory():
            return target.a_stream(query, session_id=session_id)

        new_id = jobs.spawn_stream("agent", query, _factory, session_id=session_id)
        logger.info("omni.resume: job=%s -> new=%s session=%s", job_id, new_id, session_id)
        return new_id

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
        {
            "tool_func": agent_progress,
            "name": "agent_progress",
            "description": (
                "Current thinking of a delegate by job_id: last reasoning/tool + "
                "iteration + status. Use to narrate what the background agent is doing."
            ),
            "category": ["agent", "read"],
        },
        {
            "tool_func": agent_stop,
            "name": "agent_stop",
            "description": "Stop (cancel) a running delegated job by job_id.",
            "category": ["agent", "control"],
        },
        {
            "tool_func": agent_resume,
            "name": "agent_resume",
            "description": (
                "Resume a delegation's session with new context: agent_resume("
                "job_id, query). Cancels the old run if active; returns a new job_id."
            ),
            "category": ["agent", "control"],
        },
    ]


# ---------------------------------------------------------------------------
# Chat-view tools — read-only window into the ICLI chat the Omni session rides on
# ---------------------------------------------------------------------------

def make_chat_view_tools(
    executions_provider: Callable[[], list[dict]],
    history_provider: Callable[[int], list[dict]],
) -> list[dict]:
    """Read-only tools so the Omni model can see the CHAT it is part of:
    currently-running chat executions and recent chat history (which already
    contains the last finished results as assistant turns).

    ponytail: read, don't share. No shared JobManager/registry refactor — the
    Omni model just *reads* the chat's existing state through two thin providers
    injected by the host (duck-typed; this module imports nothing from toolboxv2).
    """

    async def chat_executions() -> str:
        """List the chat's currently running/active executions (JSON). These are
        tasks started in the text chat, separate from your own delegate() jobs."""
        try:
            return json.dumps(executions_provider() or [], ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            return f"Error: chat_executions failed: {e}"

    async def chat_history(last_n: int = 10) -> str:
        """Last N messages of the chat conversation (JSON), most recent last.
        Includes the latest finished results (assistant answers)."""
        try:
            return json.dumps(history_provider(last_n) or [], ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            return f"Error: chat_history failed: {e}"

    return [
        {
            "tool_func": chat_executions,
            "name": "chat_executions",
            "description": (
                "Read-only: tasks currently running in the text chat (JSON). "
                "Separate from your own delegate() jobs."
            ),
            "category": ["chat", "read"],
        },
        {
            "tool_func": chat_history,
            "name": "chat_history",
            "description": (
                "Read-only: last N chat messages (JSON), incl. the latest "
                "finished results. Call before answering questions about what "
                "the user did in the text chat."
            ),
            "category": ["chat", "read"],
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

# ---------------------------------------------------------------------------
# Speaker recognition (update 2) — optional, blob-persisted embeddings
# ---------------------------------------------------------------------------

class SpeakerRegistry:
    """Optional speaker recognition with blob-persisted voice embeddings.

    Couples to an embedder by DUCK-TYPING only (embedder.embed(pcm)->list[float]
    |None), so the registry is toolboxv2-free + testable with a fake. Embeddings
    persist via an INJECTED BlobFile (auto-load on construct, auto-save on
    enroll). Similarity is pure-Python cosine — no numpy in the tested core.
    The label_hook names unknown voices: (embedding, best_score)->label|None.
    """

    def __init__(self, path: str, blob_file_cls: Any, embedder: Any = None, *,
                 threshold: float = 0.75, key: Optional[bytes] = None,
                 label_hook: Optional[Callable[[list[float], float], Optional[str]]] = None):
        self._path = path
        self._BlobFile = blob_file_cls
        self._embedder = embedder
        self._key = key
        self.threshold = threshold
        self.label_hook = label_hook
        self._embeddings: dict[str, list[float]] = self._load()

    def _load(self) -> dict[str, list[float]]:
        try:
            with self._BlobFile(self._path, "r", key=self._key) as f:
                raw = f.read_json()
        except Exception as e:  # noqa: BLE001
            logger.debug("SpeakerRegistry load failed: %s", e)
            return {}
        if not isinstance(raw, dict):
            return {}
        return {k: [float(x) for x in v] for k, v in raw.items() if isinstance(v, list)}

    def save(self) -> None:
        try:
            with self._BlobFile(self._path, "w", key=self._key) as f:
                f.write_json(self._embeddings)
        except Exception as e:  # noqa: BLE001
            logger.debug("SpeakerRegistry save failed: %s", e)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dot = sum(a[i] * b[i] for i in range(n))
        na = sum(x * x for x in a[:n]) ** 0.5
        nb = sum(x * x for x in b[:n]) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def embed(self, pcm: bytes) -> Optional[list[float]]:
        if self._embedder is None:
            return None
        vec = self._embedder.embed(pcm)
        return list(vec) if vec is not None else None

    def identify(self, embedding: list[float]) -> tuple[str, float]:
        """Best match (label, score), or assign via label_hook, else ('unknown',
        best_score). Does NOT adapt on match -> no blob write storms."""
        best_label, best_score = "unknown", 0.0
        for label, ref in self._embeddings.items():
            s = self._cosine(embedding, ref)
            if s > best_score:
                best_label, best_score = label, s
        if best_score >= self.threshold:
            return best_label, best_score
        if self.label_hook is not None:
            assigned = self.label_hook(embedding, best_score)
            if assigned:
                self.enroll(assigned, embedding)
                return assigned, best_score
        return "unknown", best_score

    def enroll(self, label: str, embedding: list[float], alpha: float = 0.3) -> None:
        """Add or update a speaker centroid (running mean) and persist."""
        cur = self._embeddings.get(label)
        if cur is None:
            self._embeddings[label] = list(embedding)
        else:
            n = min(len(cur), len(embedding))
            self._embeddings[label] = [(1 - alpha) * cur[i] + alpha * embedding[i]
                                       for i in range(n)]
        self.save()


class StubSpeakerEmbedder:
    """Deterministic, dependency-free embedder for tests/offline dev. A real
    ECAPA/Resemblyzer embedder plugs in via the same .embed(pcm)->list[float]."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, pcm: bytes) -> list[float]:
        if not pcm:
            return [0.0] * self.dim
        buckets = [0.0] * self.dim
        for i, b in enumerate(pcm):
            buckets[i % self.dim] += b
        total = sum(buckets) or 1.0
        return [v / total for v in buckets]

class FallbackOmniBackend(OmniBackend):
    """The classic STT->LLM->TTS pipeline wrapped as a native OmniBackend.

    OmniSession drives it like any Omni backend: it pumps mic frames via
    send_audio() into an internal WebRecorder that a LiveModeEngine consumes;
    each completed utterance runs FlowAgent.a_audio and is emitted back as the
    SAME OmniEvents (TEXT + AUDIO + TURN_END). So phase hook, player, on_text and
    persistence all fire normally — indistinguishable from a real Omni backend.

    Integration surface (not unit-tested). agent is duck-typed (a_audio / a_run /
    tts); LiveModeEngine/WebRecorder/LiveModeConfig imported lazily in start().
    Tools run INSIDE the agent's own ReAct loop, so send_tool_result is a no-op.
    """

    needs_silence = True      # own VAD ends turns -> OmniSession must stream silence too
    supports_restart = False  # no live model session -> never reseed-speak

    def __init__(self, agent: Any, *, session_id: str = "fallback",
                 config: Any = None, require_wake_word: bool = False,
                 engine_factory: Optional[Callable[[Callable], Any]] = None):
        self._agent = agent
        self._session_id = session_id
        self._config = config
        self._require_wake_word = require_wake_word
        self._engine_factory = engine_factory
        self._engine = None
        self._recorder = None
        self._q: "asyncio.Queue[Any]" = asyncio.Queue()

    async def start(self, tools: Optional[list[dict]] = None) -> None:
        if self._engine_factory is not None:  # test seam
            self._engine = self._engine_factory(self._on_utterance)
            self._recorder = getattr(self._engine, "recorder", None)
            await self._engine.start()
            logger.info("FallbackOmniBackend: injected engine started")
            return
        from toolboxv2.mods.isaa.base.audio_io.audio_live import (
            LiveModeEngine, LiveModeConfig,
        )
        from toolboxv2.mods.isaa.base.audio_io.audio_recorder import WebRecorder
        self._recorder = WebRecorder(src_sample_rate=TARGET_SR, src_channels=1)
        self._engine = LiveModeEngine(
            config=self._config or LiveModeConfig(),
            on_utterance=self._on_utterance,
            recorder=self._recorder,
            require_wake_word=self._require_wake_word,
        )
        await self._engine.start()
        logger.info("FallbackOmniBackend: classic pipeline started")

    async def send_audio(self, pcm: bytes) -> None:
        if self._recorder is not None:
            await self._recorder.feed(pcm)
        elif self._engine is not None and hasattr(self._engine, "feed"):
            await self._engine.feed(pcm)

    async def send_tool_result(self, call_id: str, result: str) -> None:
        return None  # agent runs its own tools internally

    async def send_text(self, text: str) -> None:
        asyncio.ensure_future(self._run_text(text))

    async def events(self) -> AsyncIterator[OmniEvent]:
        while True:
            ev = await self._q.get()
            if ev is _SENTINEL:
                return
            yield ev

    async def stop(self) -> None:
        if self._engine is not None:
            try:
                await self._engine.stop()
            except Exception:  # noqa: BLE001
                pass
        await self._q.put(_SENTINEL)

    # -- pipeline -> OmniEvents ----------------------------------------------
    async def _on_utterance(self, wav: bytes, speaker: Optional[str]) -> None:
        try:
            audio_out, text_out, _tc, _meta = await self._agent.a_audio(
                wav, session_id=self._session_id
            )
        except Exception as e:  # noqa: BLE001
            self._q.put_nowait(OmniEvent.error(f"fallback a_audio: {e}"))
            return
        self._emit_response(text_out, audio_out)

    async def _run_text(self, text: str) -> None:
        try:
            out = await self._agent.a_run(text, session_id=self._session_id)
            res = await self._agent.tts(out)
            self._emit_response(out, getattr(res, "audio", None))
        except Exception as e:  # noqa: BLE001
            self._q.put_nowait(OmniEvent.error(f"fallback send_text: {e}"))

    def _emit_response(self, text: Optional[str], wav: Optional[bytes]) -> None:
        if text:
            self._q.put_nowait(OmniEvent.text_chunk(text, source="output"))
        if wav:
            pcm, sr = _wav_to_pcm(wav)
            if pcm:
                self._q.put_nowait(OmniEvent.audio_chunk(pcm, sample_rate_out=sr))
        self._q.put_nowait(OmniEvent.turn_end())


class PhaseCallback:
    def __init__(self):
        self.active_spinner = None

    def stop_spinner(self):
        """Stoppt den aktuellen Spinner und räumt die Zeile auf."""
        if self.active_spinner:
            self.active_spinner.__exit__(None, None, None)
            self.active_spinner = None

    def start_spinner(self, message, symbol):
        """Stoppt den alten Spinner und startet einen neuen."""
        from toolboxv2 import Spinner
        self.stop_spinner()
        self.active_spinner = Spinner(message=message, symbols=symbol)
        self.active_spinner.__enter__()

    def __call__(self, phase, meta=None):
        meta = meta or {}
        # Erlaubt Enum-Objekte oder direkte Strings
        phase_val = phase.value if hasattr(phase, 'value') else phase

        if phase_val == "waiting":
            self.start_spinner(Style.GREY("Waiting for speech..."), symbol="i")

        elif phase_val == "speech_detected":
            self.stop_spinner()
            print(Style.GREEN2("🎤 Speech detected"))

        elif phase_val == "speaker_detected":
            self.stop_spinner()
            speaker = meta.get("speaker", "Unknown")
            score = meta.get("score", "N/A")
            print(Style.CYAN(f"🗣️  Speaker: {speaker} (Score: {score})"))

        elif phase_val == "speaking":
            self.start_spinner(Style.CYAN("User speaking..."), symbol="b")

        elif phase_val == "speech_end":
            self.stop_spinner()
            print(Style.GREEN2("🛑 Speech ended"))

        elif phase_val == "thinking":
            self.start_spinner(Style.VIOLET2("Thinking..."), symbol="d")

        elif phase_val == "tool_start":
            self.stop_spinner()
            name = meta.get("name", "Unknown")
            args = meta.get("arguments", "{}")
            print(Style.YELLOW(f"🔧 Tool Start: {name} | Args: {args}"))
            self.start_spinner(Style.YELLOW(f"Executing {name}..."), symbol="t")

        elif phase_val == "tool_end":
            self.stop_spinner()
            name = meta.get("name", "Unknown")
            result = str(meta.get("result", ""))
            # Resultat kürzen, falls es das Terminal sprengt
            if len(result) > 150:
                result = result[:150] + "..."
            print(Style.YELLOW(f"✅ Tool End: {name} | Result: {result}"))

        elif phase_val == "audio_processing":
            self.start_spinner(Style.BLUE("Processing audio..."), symbol="e")

        elif phase_val == "audio_playing":
            self.start_spinner(Style.GREEN("Playing audio..."), symbol="s")

        else:
            self.stop_spinner()
            print(Style.GREY(f"Phase: {phase_val}"))
# ---------------------------------------------------------------------------
# OmniSession — the loop
# ---------------------------------------------------------------------------
OMNI_SYSTEM_INSTRUCTION = """
You are ISAA's voice layer: a spoken, real-time assistant. You talk; the actual work is done by a stronger background agent you delegate to.

VOICE
- Speak in short, natural spoken sentences. No markdown, lists, code, or symbols read aloud.
- Say only what is in this context or came from a tool result. Never invent facts, names, paths, or numbers.

DELEGATE — don't do real work yourself
- For any task, lookup, file, code, or research: call delegate(query). It returns a job_id at once and runs in the background.
- Briefly tell the user you're on it. You are notified when a job finishes — then speak the result. You may also check agent_status() or fetch agent_result(job_id).
- To look into a delegation's files: vfs_peek(job_id, path, scroll_to=...) for a tight slice, or vfs_tree_peek(job_id) for a shallow tree. Read-only.

WORLD MODEL — remember the user
- On a STABLE fact, save it with world_model_edit: field_name 'user' (who they are) or 'agent_role' (what they want you to be) with op='set'; or 'routines' (a recurring task as a short recall-key) with op='append'/'remove'.
- One short line per value. Not for one-off or detailed facts — those go to the agent's memory.

STAY FAST — compress
- Call compress_session() at every topic change, once past a few turns, and before the conversation gets long. It folds the talk into a summary and keeps your latency low. Keep the default merge_old=True.
"""

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
        sample_rate: int = TARGET_SR,
        output_sample_rate: Optional[int] = None,
        on_text: Optional[Callable[[str], Any]] = None,
        buffer_audio: bool = True,
        enhancer: Any = None,
        vad: Any = None,
        vad_threshold: float = 0.5,
        vad_hangover_frames: int = 8,
        on_job_done: Optional[Callable[[dict], Any]] = None,
        backend_factory: Optional[Callable[[], OmniBackend]] = None,
        state_store: Optional[BlobStateStore] = None,
        compress_min_turns: int = 3,
        restart_at_turns: int = 10,
        restart_compress: bool = True,
        resume_tail_turns: int = 6,
        summary_max_tokens: int = 400,
        stream_flush_bytes: int = 12000,  # ~250ms @24k: flush buffered audio early
        summarizer_agent: Any = None,
        on_phase: Optional[Callable[["OmniPhase", dict], Any]] = None,
        speakers: Any = None,
        video_source: Any = None,
        idle_reconnect_s: float = 10.0,
    ):
        self.backend = backend
        self.recorder = recorder
        self.player = player
        self.tools = tools
        self.jobs = jobs or JobManager()
        self.background_tools = set(background_tools or set())

        # summarizer MUST be independent of the optional fallback agent — see
        # analysis below. Falls back to `fallback` only for backward compat.
        self.summarizer = summarizer_agent

        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate
        self.on_text = on_text
        self.buffer_audio = buffer_audio
        self.stream_flush_bytes = stream_flush_bytes
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

        # ponytail: idle watchdog. If no backend event arrives for idle_reconnect_s
        # while WAITING (WS half-dead -> "waiting for speaker" hang), pause + reopen
        # the backend. 0 disables. Reuses backend_factory; keeps _announced_jobs.
        self.idle_reconnect_s = idle_reconnect_s
        self._last_event_mono = 0.0
        self._watchdog_task: Optional[asyncio.Task] = None

        # per-turn audio buffer: list of (pcm_bytes, sample_rate)
        self._audio_buf: list[tuple[bytes, int]] = []

        # introspection
        self.audio_chunks_out = 0
        self.audio_flushes = 0
        self.tool_calls_handled = 0
        self.turns = 0
        self.frames_sent = 0
        self.frames_gated = 0

        # persistent state / infinite session
        self.backend_factory = backend_factory
        self.state_store = state_store
        self.compress_min_turns = compress_min_turns
        self.restart_at_turns = restart_at_turns
        self.restart_compress = restart_compress
        self.resume_tail_turns = resume_tail_turns
        self.summary_max_tokens = summary_max_tokens
        self._tool_specs: list[dict] = []
        self._turn_buf: list[tuple[str, str]] = []  # (source, text) within current turn
        self._restarting = False
        # ponytail: serialize text inputs. While the agent is speaking, queue
        # incoming text (user + system announcements) and flush as ONE combined
        # message on TURN_END — otherwise two send_text calls overlap mid-speech.
        self._pending_text: list[str] = []

        # phase hook
        self.on_phase = on_phase or PhaseCallback()
        # update 1: lifecycle phase hook — single source of truth via flags
        self._phase: OmniPhase = OmniPhase.WAITING
        self._speaking = False  # user speech in progress (VAD start->end)
        self._agent_speaking = False  # agent audio in progress (chunk->drain)
        self._thinking = False  # speech ended, agent audio not yet started
        self._playback_task: Optional[asyncio.Task] = None
        self._user_end_task: Optional[asyncio.Task] = None  # debounces transcription speech-end
        self._mic_muted_until = 0.0  # half-duplex: suppress mic until this monotonic ts
        # update 2: optional speaker recognition
        self.speakers = speakers
        self._speech_frames = bytearray()
        self._last_speaker: Optional[str] = None
        # update 3: optional video source pump
        self.video_source = video_source
        self._video_task: Optional[asyncio.Task] = None

    # lifecycle --------------------------------------------------------------
    async def _warmup_vad(self) -> None:
        """Load/JIT the VAD model OFF the event loop BEFORE the mic starts. The
        first vad.is_speech() lazily loads the model (torch.hub) which blocks the
        loop ~10s on first run — the recorder queue then floods and the mic stream
        self-stops. Warming up in an executor keeps the loop responsive. Generic:
        duck-types is_speech/reset, no SileroVAD import. Never blocks start."""
        if self.vad is None:
            return
        probe = getattr(self.vad, "is_speech", None)
        if probe is None:
            return
        try:
            loop = asyncio.get_event_loop()
            silence = b"\x00\x00" * self.sample_rate  # ~1s @ sample_rate, enough to JIT
            await loop.run_in_executor(None, probe, silence)
            reset = getattr(self.vad, "reset", None)
            if reset is not None:
                reset()  # drop the warmup frame so real speech state starts clean
            logger.info("OmniSession: VAD warmed up")
        except Exception as e:  # noqa: BLE001 - warmup must never break start
            logger.debug("OmniSession: VAD warmup skipped: %s", e)

    async def start(self, tool_specs: Optional[list[dict]] = None) -> None:
        self._tool_specs = tool_specs or []
        logger.info("OmniSession.start backend=%s tools=%d", self.backend.backend_name,
                    len(self._tool_specs))
        try:
            await self.backend.start(tools=tool_specs)
        except Exception as e:  # noqa: BLE001
            logger.error("Omni backend start failed: %s", e)
            import traceback
            logger.debug(traceback.format_exc())
            raise

        # ponytail: warm the VAD BEFORE the mic produces frames, so the first
        # is_speech() doesn't load the model on the loop while frames pile up.
        await self._warmup_vad()

        if self.recorder is not None:
            await self.recorder.start()
            logger.info("OmniSession: recorder started (%s)",
                        getattr(self.recorder, "recorder_type", type(self.recorder).__name__))
        if self.player is not None:
            await self.player.start()
            logger.info("OmniSession: player started (%s)", type(self.player).__name__)

        self._running = True
        self._last_event_mono = time.monotonic()
        self._event_task = asyncio.ensure_future(self._consume_events())
        if self.recorder is not None:
            self._pump_task = asyncio.ensure_future(self._pump_audio())
        self._notify_task = asyncio.ensure_future(self._notify_loop())
        if self.video_source is not None:
            self._video_task = asyncio.ensure_future(self._pump_video())
        if self.idle_reconnect_s > 0 and self.backend_factory is not None:
            self._watchdog_task = asyncio.ensure_future(self._watchdog_loop())
        logger.info("OmniSession: live loop running")

    async def stop(self) -> None:
        logger.info("OmniSession.stop — %s", self.status_line())
        self._running = False
        # exit guard: flush the in-flight turn, fold + archive into persistent
        # state, then save — BEFORE tearing down backend/summarizer. Prevents the
        # raw active_history from leaking into the next session's seed.
        try:
            self._flush_turn()
            if self.state_store is not None:
                # persist current turns FIRST — survives selbst wenn der LLM-Compress
                # beim Shutdown (Ctrl-C) abbricht. _compress speichert bei Erfolg erneut.
                self.state_store.save()
                if self.summarizer is not None:
                    await self._compress(merge_old=True)
                    await self._archive_session()
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession exit-compress failed: %s", e)
            # stop running delegations cleanly before tearing down
        for st in self.jobs.live_state():
            if st["status"] == "running":
                try:
                    await self.jobs.cancel(st["job_id"])
                except Exception:  # noqa: BLE001
                    pass
        for t in (self._pump_task, self._event_task, self._notify_task,
                  self._video_task, self._playback_task, self._user_end_task,
                  self._watchdog_task):
            if t is not None and not t.done():
                t.cancel()
        await self.backend.stop()
        if self.recorder is not None:
            await self.recorder.stop()
        if self.player is not None:
            await self.player.stop()

    async def wait(self, timeout: Optional[float] = None) -> None:
        if self._event_task is None:
            return
        await asyncio.wait_for(asyncio.shield(self._event_task), timeout=timeout)

    # update 1/2/3 hooks ------------------------------------------------------
    def _set_phase(self, phase: "OmniPhase", **meta) -> None:
        """Fire the lifecycle hook on transition. Sync-safe: a coroutine result is
        scheduled fire-and-forget so the audio loop never blocks. Deduped: same
        phase without meta won't re-fire."""
        if phase is self._phase and not meta:
            return
        self._phase = phase
        if self.on_phase is None:
            return
        try:
            res = self.on_phase(phase, meta)
            if asyncio.iscoroutine(res):
                asyncio.ensure_future(res)
        except Exception as e:  # noqa: BLE001 - hook must never break the loop
            logger.debug("on_phase hook error: %s", e)

    def _update_phase(self) -> None:
        """Single source of truth: derive the held phase from intent flags so
        independent writers (mic VAD vs event loop) can't clobber each other."""
        if self._speaking:
            self._set_phase(OmniPhase.SPEAKING)
        elif self._agent_speaking:
            self._set_phase(OmniPhase.AUDIO_PLAYING)
        elif self._thinking:
            self._set_phase(OmniPhase.THINKING)
        else:
            self._set_phase(OmniPhase.WAITING)

    def _begin_user_speech(self) -> None:
        """Idempotent. Driven by BOTH the VAD rising edge (fallback) and the first
        inputTranscription chunk (cloud)."""
        if self._speaking:
            return
        self._speaking = True
        self._speech_frames = bytearray()
        self._set_phase(OmniPhase.SPEECH_DETECTED)  # momentary signal
        self._update_phase()  # -> SPEAKING (held)


    def _end_user_speech(self) -> None:
        """Idempotent. Driven by the VAD falling edge, the input-transcription
        debounce, or the first agent output."""
        if not self._speaking:
            return
        self._speaking = False
        self._set_phase(OmniPhase.SPEECH_END)  # momentary signal
        if self.speakers is not None and self._speech_frames:
            asyncio.ensure_future(self._identify_speaker(bytes(self._speech_frames)))
        self._thinking = True
        self._update_phase()  # -> THINKING (or AUDIO_PLAYING)


    def _begin_agent_speech(self) -> None:
        """Idempotent. Driven by the first outputTranscription chunk OR first AUDIO."""
        self._thinking = False
        if self._agent_speaking:
            return
        self._agent_speaking = True
        self._update_phase()  # -> AUDIO_PLAYING (unless user speaking)


    def _bump_user_speech(self) -> None:
        """Transcription path: (re)open the user-speech window and debounce its end
        (no explicit 'input ended' event from Gemini)."""
        self._begin_user_speech()
        if self._user_end_task is not None and not self._user_end_task.done():
            self._user_end_task.cancel()
        self._user_end_task = asyncio.ensure_future(self._end_user_speech_after(0.8))


    async def _end_user_speech_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        self._end_user_speech()

    async def _await_playback_done(self) -> None:
        """Hold AUDIO_PLAYING until the player actually drains, THEN -> WAITING.
        Playback runs async in the player worker, so TURN_END is NOT the end of
        audio. Players without is_active (NullPlayer) fall through immediately."""
        try:
            await asyncio.sleep(0.05)
            while (self._running and self.player is not None
                   and getattr(self.player, "is_active", False)):
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return
        self._agent_speaking = False
        self._mic_muted_until = time.monotonic() + 0.3  # swallow echo tail
        self._update_phase()

    async def _identify_speaker(self, pcm: bytes) -> None:
        """Embed the finished utterance, match against the persisted registry, fire
        SPEAKER_DETECTED, and inject [speaker: X] into the live context on change."""
        reg = self.speakers
        if reg is None:
            return
        try:
            emb = reg.embed(pcm)
            if emb is None:
                return
            label, score = reg.identify(emb)
            self._set_phase(OmniPhase.SPEAKER_DETECTED, speaker=label, score=score)
            if label and label != self._last_speaker:
                self._last_speaker = label
                await self.backend.send_text(f"[speaker: {label}]")
        except Exception as e:  # noqa: BLE001
            logger.debug("speaker identify error: %s", e)

    async def _pump_video(self) -> None:
        """Stream JPEG frames from an injected video_source (duck-typed .frames()
        async iterator of bytes) to the backend as realtime video frames."""
        try:
            async for frame in self.video_source.frames():
                if not self._running:
                    break
                await self.backend.send_media(frame, "image/jpeg", as_turn=False)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession video pump stopping: %s", e)

    async def send_media(self, data: bytes, mime_type: str, *, as_turn: bool = False) -> None:
        """Forward an image/video blob to the backend (one-shot turn or realtime)."""
        await self.backend.send_media(data, mime_type, as_turn=as_turn)

    async def send_image(self, source: Any, mime_type: Optional[str] = None) -> None:
        """Send an image from a path or raw bytes as a one-shot turn."""
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            mime_type = mime_type or "image/jpeg"
        else:
            import mimetypes
            with open(source, "rb") as f:
                data = f.read()
            mime_type = mime_type or (mimetypes.guess_type(str(source))[0] or "image/jpeg")
        await self.send_media(data, mime_type, as_turn=True)

    # internal loops ---------------------------------------------------------
    def _should_send_frame(self, frame: bytes) -> bool:
        """VAD gate. True if this frame should be streamed to the backend.

        No VAD configured -> always send (open mic). With VAD: send while speech
        prob >= threshold, plus a short hangover of N frames after speech stops
        so word endings aren't clipped. Frames where the VAD hasn't accumulated
        enough samples yet (is_speech returns -1.0) inherit the current state.
        """
        # half-duplex: never feed the mic while the agent is (or just was) speaking
        # — without echo cancellation the model hears itself and replies to itself.
        if self._agent_speaking or time.monotonic() < self._mic_muted_until:
            return False
        if self.vad is None:
            return True
        try:
            prob = self.vad.is_speech(frame)
        except Exception as e:  # noqa: BLE001 - VAD must never break the stream
            logger.debug("VAD error, sending frame: %s", e)
            return True
        in_speech = self._update_speech_state(prob)
        # Fallback ends turns via its OWN VAD -> it must receive the silence too.
        if getattr(self.backend, "needs_silence", False):
            return True
        return in_speech

    def _update_speech_state(self, prob: float) -> bool:
        """VAD gate + (fallback) speech edges via the idempotent helpers.
        Returns True while inside the speech window (speech or hangover)."""
        if prob < 0:  # not enough samples yet -> keep current state
            return self._vad_hangover > 0
        if prob >= self.vad_threshold:
            self._begin_user_speech()
            self._vad_hangover = self.vad_hangover_frames
            return True
        if self._vad_hangover > 0:
            self._vad_hangover -= 1
            return True
        self._end_user_speech()
        return False

    async def _pump_audio(self) -> None:
        try:
            async for frame in self.recorder.frames():
                if not self._running:
                    break
                res = self._should_send_frame(frame)
                logger.debug("VAD info, frame: %s", res)
                if not res:
                    self.frames_gated += 1
                    continue
                await self.backend.send_audio(frame)
                if self.speakers is not None and self._speaking:
                    self._speech_frames += frame
                self.frames_sent += 1
                if self.frames_sent % 50 == 0:
                    logger.debug("OmniSession: %d frames sent, %d gated (silence)",
                                 self.frames_sent, self.frames_gated)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession audio pump stopping: %s", e)
            self._running = False

    async def send_user_text(self, text: str) -> None:
        """Single serialized entry for ALL injected text (user input + system
        announcements). While the agent is speaking, queue it; the queued text
        is flushed as one combined message on TURN_END. Routes everything through
        one place so two inputs can never overlap the model mid-speech."""
        if not text:
            return
        if self._agent_speaking:
            self._pending_text.append(text)
            logger.debug("OmniSession: queued text while speaking (%d pending)",
                         len(self._pending_text))
            return
        await self.backend.send_text(text)

    async def _flush_pending_text(self) -> None:
        """Send everything queued during the last turn as ONE message."""
        if not self._pending_text:
            return
        combined = "\n".join(self._pending_text)
        self._pending_text.clear()
        logger.debug("OmniSession: flushing %d queued text input(s)", combined.count("\n") + 1)
        await self.backend.send_text(combined)

    async def _flush_player(self) -> None:
        """Drop in-flight player audio if the player supports it (barge-in /
        restart / reconnect). No-op for players without flush()."""
        if self.player is None:
            return
        f = getattr(self.player, "flush", None)
        if f is None:
            return
        try:
            res = f()
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:  # noqa: BLE001
            logger.debug("OmniSession: player flush failed: %s", e)

    async def interrupt(self) -> None:
        """User barge-in: stop the agent talking NOW. Drops buffered + in-flight
        audio and returns to WAITING. Reuses the INTERRUPTED dispatch semantics,
        just user-triggered (e.g. an icli keybind)."""
        logger.info("OmniSession: user interrupt")
        self._audio_buf.clear()
        await self._flush_player()
        self._agent_speaking = False
        bi = getattr(self.backend, "interrupt", None)
        if bi is not None:
            try:
                res = bi()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:  # noqa: BLE001
                logger.debug("OmniSession: backend interrupt failed: %s", e)
        self._update_phase()

    async def _watchdog_loop(self, poll_s: float = 1.0) -> None:
        """Pause+reopen the backend if it goes silent for idle_reconnect_s while
        WAITING — the WS can half-die and just hang on 'waiting for speaker'."""
        try:
            while self._running:
                await asyncio.sleep(poll_s)
                if self._restarting or self._agent_speaking or self.idle_reconnect_s <= 0:
                    continue
                if self._phase is not OmniPhase.WAITING:
                    continue
                if self.backend_factory is None or not getattr(self.backend, "supports_restart", True):
                    continue
                idle = time.monotonic() - self._last_event_mono
                if idle >= self.idle_reconnect_s:
                    logger.info("OmniSession: idle %.1fs while waiting -> reconnecting", idle)
                    self._restarting = True
                    await self._reconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession watchdog error: %s", e)

    async def _reconnect(self) -> None:
        """Pause + reopen the backend after an idle stall. Subset of _do_restart:
        NO compress (nothing new to fold) and KEEP _announced_jobs (so finished
        delegations are not re-announced). Reseeds from persistent state."""
        try:
            seed = self._build_seed_text()
            if self._event_task is not None and not self._event_task.done():
                self._event_task.cancel()
            try:
                await self.backend.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("OmniSession: reconnect old-backend stop failed: %s", e)
            self.backend = self.backend_factory()
            seed_via_text = False
            if seed:
                if hasattr(self.backend, "system_instruction"):
                    base = getattr(self.backend, "system_instruction", "") or ""
                    self.backend.system_instruction = (base + "\n\n" + seed).strip()
                else:
                    seed_via_text = True
            await self.backend.start(tools=self._tool_specs)
            self._audio_buf.clear()
            await self._flush_player()
            self._last_event_mono = time.monotonic()
            self._event_task = asyncio.ensure_future(self._consume_events())
            if seed_via_text:
                await self.backend.send_text(seed)
            logger.info("OmniSession: reconnect complete (backend=%s)", self.backend.backend_name)
        except Exception as e:  # noqa: BLE001
            logger.error("OmniSession: reconnect failed: %s", e)
        finally:
            self._restarting = False

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
            await self.send_user_text(note)
        except Exception as e:  # noqa: BLE001
            logger.warning("OmniSession: announce send_text failed: %s", e)

    # audio buffering --------------------------------------------------------
    def _buffer_or_play(self, ev: OmniEvent) -> Optional[Awaitable]:
        sr = int(ev.meta.get("sample_rate_out") or self.output_sample_rate or self.sample_rate)
        if self.buffer_audio:
            self._audio_buf.append((ev.audio, sr))
            return None
        return self._emit_wav(ev.audio, sr, dict(ev.meta))

    def _buffered_bytes(self) -> int:
        return sum(len(pcm) for pcm, _ in self._audio_buf)

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
        # self._set_phase(OmniPhase.AUDIO_PLAYING)
        await self.player.queue_audio(wav, meta)

    # event dispatch ---------------------------------------------------------
    async def dispatch(self, ev: OmniEvent) -> None:
        self._last_event_mono = time.monotonic()  # ponytail: watchdog liveness stamp
        if ev.type == OmniEventType.AUDIO:
            self.audio_chunks_out += 1
            if ev.audio:
                self._begin_agent_speech()
                awaitable = self._buffer_or_play(ev)
                if awaitable is not None:
                    await awaitable
                elif self._buffered_bytes() >= self.stream_flush_bytes:
                    await self._flush_audio()  # stream early -> low latency
        elif ev.type == OmniEventType.TEXT:
            if ev.text:
                src = ev.meta.get("source")
                if src == "input":  # user is speaking (Gemini inputTranscription)
                    self._bump_user_speech()
                elif src == "output":  # agent is answering (outputTranscription)
                    if self._user_end_task is not None and not self._user_end_task.done():
                        self._user_end_task.cancel()
                    self._end_user_speech()
                    self._begin_agent_speech()
                logger.debug("OmniSession TEXT[%s]: %s", src or "?", ev.text[:80])
                if self.state_store is not None:
                    self._turn_buf.append((src or "model", ev.text))
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
            self._flush_turn()
            if self.state_store is not None:
                self.state_store.save()
            if self._should_restart() and not self._restarting:
                self._restarting = True
                # never restart synchronously from inside the event task (self-cancel)
                asyncio.ensure_future(self._do_restart(self.restart_compress))
            await self._flush_audio()
            self._thinking = False
            # ponytail: turn done -> deliver any text queued while speaking as one
            await self._flush_pending_text()
            # playback runs async in the player; hold AUDIO_PLAYING until it drains
            if self._agent_speaking:
                if self._playback_task is None or self._playback_task.done():
                    self._playback_task = asyncio.ensure_future(self._await_playback_done())
            else:
                self._update_phase()  # text-only turn -> WAITING
        elif ev.type == OmniEventType.INTERRUPTED:
            logger.info("OmniSession: user barge-in — dropping buffered audio")
            self._audio_buf.clear()
            self._agent_speaking = False
            self._update_phase()
        elif ev.type == OmniEventType.ERROR:
            logger.warning("Omni backend error: %s", ev.text)

    # tool bridge ------------------------------------------------------------
    async def handle_tool_call(self, call: dict) -> str:
        call_id = call.get("id") or uuid.uuid4().hex[:8]
        name = call.get("name", "")
        args = call.get("arguments") or {}
        self.tool_calls_handled += 1
        logger.info("OmniSession tool-call: %s(%s)", name, ", ".join(args.keys()))
        self._set_phase(OmniPhase.TOOL_START, name=name, arguments=args)

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
            self._set_phase(OmniPhase.TOOL_END, name=name, result=result)
            await self.backend.send_tool_result(call_id, result)
            return result

        try:
            raw = await self.tools.execute(name, **args)
            result = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001
            result = f"Error: {e}"
            logger.warning("OmniSession: tool %s raised: %s", name, e)
        self._set_phase(OmniPhase.TOOL_END, name=name, result=result)
        await self.backend.send_tool_result(call_id, result)
        return result

    @property
    def compress_tool(self) -> list[dict]:
        return make_session_tools(self)

    def status_line(self) -> str:
        return (
            f"backend={self.backend.backend_name} running={self._running} "
            f"audio_out={self.audio_chunks_out} flushes={self.audio_flushes} "
            f"tool_calls={self.tool_calls_handled} turns={self.turns}"
        )

    # persistent state / infinite session ------------------------------------
    @staticmethod
    def _render_history(hist: list[dict]) -> str:
        return "\n".join(f"{h.get('role', '?')}: {h.get('text', '')}"
                         for h in hist if h.get("text"))

    def _flush_turn(self) -> None:
        """Consolidate this turn's TEXT fragments into active_history entries."""
        if self.state_store is None or not self._turn_buf:
            self._turn_buf = []
            return
        entries: list[dict] = []
        for src, txt in self._turn_buf:
            role = "user" if src in ("user", "input") else "assistant"
            if entries and entries[-1]["role"] == role:
                entries[-1]["text"] += txt
            else:
                entries.append({"role": role, "text": txt})
        self.state_store.state.active_history.extend(entries)
        self._turn_buf = []

    def _should_restart(self) -> bool:
        return (
            self.state_store is not None
            and self.backend_factory is not None
            and getattr(self.backend, "supports_restart", True)
            and self.turns >= self.restart_at_turns
        )

    def request_restart(self, compress: bool = True) -> None:
        """Manually schedule a session restart (dev / CLI). Non-blocking."""
        if self._restarting:
            return
        self._restarting = True
        asyncio.ensure_future(self._do_restart(compress))

    def request_compress(self, merge_old: bool = True) -> None:
        """Schedule a background compression (tool entrypoint). Non-blocking."""
        asyncio.ensure_future(self._compress(merge_old=merge_old))

    async def _compress(self, merge_old: bool = True) -> str:
        if getattr(self, "_compressing", False):
            return "compression already running"
        self._compressing = True
        try:
            return await self._compress_inner(merge_old)
        finally:
            self._compressing = False

    async def _compress_inner(self, merge_old: bool = True) -> str:
        if self.summarizer is None or self.state_store is None:
            return "Error: no summarizer/state_store for compression"
        s = self.state_store.state
        n = len(s.active_history)  # snapshot — keep turns that arrive during the LLM call
        transcript = self._render_history(s.active_history[:n])
        if not transcript and not s.full_summary:
            return "nothing to compress"
        prior = s.full_summary if merge_old else ""
        new_summary = await _summarize_omni(
            self.summarizer, transcript, prior, max_tokens=self.summary_max_tokens
        )
        if new_summary:
            s.full_summary = new_summary
            s.active_history = s.active_history[n:]
            self.state_store.save()
            logger.info("OmniSession: compressed (%d turns folded)", n)
        return new_summary or "Error: empty summary"

    async def _archive_session(self) -> None:
        """Append one ultra-short collective line for the just-ended session."""
        if self.summarizer is None or self.state_store is None:
            return
        s = self.state_store.state
        if not s.full_summary:
            return
        line = await _summarize_omni(self.summarizer, s.full_summary, max_tokens=80, one_line=True)
        if line:
            s.session_summaries.append(line)
            self.state_store.save()

    def _build_seed_text(self) -> str:
        if self.state_store is None:
            return ""
        s = self.state_store.state
        parts = ["[session resume — you are continuing an ongoing voice conversation]"]
        wm = s.world_model.render()
        if wm:
            parts.append("Known context:\n" + wm)
        if s.full_summary:
            parts.append("Conversation so far:\n" + s.full_summary)
        tail = s.active_history[-self.resume_tail_turns:]
        rendered = self._render_history(tail)
        if rendered:
            parts.append("Recent turns:\n" + rendered)
        return "\n\n".join(parts)

    async def _do_restart(self, compress: bool) -> None:
        """Tear down the degraded backend and bring up a fresh one, reseeded from
        persistent state. Runs as its own task (never inside the event loop)."""
        try:
            logger.info("OmniSession: restarting session (turns=%d, compress=%s)",
                        self.turns, compress)
            # ponytail: compress on the LIVE backend, swap only when seed is ready.
            # Cancelling events / stopping the backend before _compress made the
            # session go silent for the whole summary LLM call (the freeze).
            if compress:
                await self._compress(merge_old=True)
            await self._archive_session()
            seed = self._build_seed_text()
            # seed ready -> now tear down the old session and bring up the fresh one
            if self._event_task is not None and not self._event_task.done():
                self._event_task.cancel()
            try:
                await self.backend.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("OmniSession: old backend stop failed: %s", e)
            self.backend = self.backend_factory()
            seed_via_text = False
            if seed:
                if hasattr(self.backend, "system_instruction"):
                    base = getattr(self.backend, "system_instruction", "") or ""
                    self.backend.system_instruction = (base + "\n\n" + seed).strip()
                else:
                    seed_via_text = True
            await self.backend.start(tools=self._tool_specs)
            self.turns = 0
            self._audio_buf.clear()
            await self._flush_player()  # ponytail: clear stale playback backlog on restart
            self._turn_buf = []
            self._announced_jobs.clear()
            self._event_task = asyncio.ensure_future(self._consume_events())
            if seed_via_text:
                await self.backend.send_text(seed)
            logger.info("OmniSession: restart complete (backend=%s)", self.backend.backend_name)
        except Exception as e:  # noqa: BLE001
            logger.error("OmniSession: restart failed: %s", e)
        finally:
            self._restarting = False

    def export_state(self) -> dict:
        """Dev hook: flush the current turn, persist, and return the full state."""
        self._flush_turn()
        if self.state_store is not None:
            self.state_store.save()
            return self.state_store.state.to_dict()
        return {}
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
    agent: Any = None
    local_model_id: str = "Qwen/Qwen2.5-Omni-7B-AWQ"
    cloud_model_id: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    device: str = "cuda"
    sample_rate: int = TARGET_SR
    api_key_env: str = "GEMINI_API_KEY"
    custom: Optional[OmniBackend] = None
    kwargs: dict[str, Any] = field(default_factory=dict)

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
            return GeminiLiveBackend(self.cloud_model_id, api_key_env=self.api_key_env, **self.kwargs)
        if self.mode in ("pipeline", "fallback"):
            if self.agent is None:
                return None  # no agent -> caller handles the classic path itself
            return FallbackOmniBackend(self.agent)
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
