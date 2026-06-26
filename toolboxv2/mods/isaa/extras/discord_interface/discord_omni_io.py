"""
discord_omni_io.py
==================

Bridge between Discord voice and the existing OmniSession (S2S) layer.

Design: reuse, don't reimplement.
  - The whole Omni machinery (OmniSession, VoiceModeConfig, delegate/vfs/chat
    tools, world-model, compress, restart, persistence) is imported from
    ``omni.py`` and wired EXACTLY like the icli ``_handle_omni_command`` — the
    ONLY swap points are the recorder and the player.
  - Output reuses the proven Discord playback path (``FFmpegPCMAudio`` + a
    sequential worker, like the classic ``_tts_worker``); FFmpeg handles any
    sample-rate -> 48k/stereo, so no manual DSP on the way out.
  - Input is the only genuinely new piece: Discord delivers 20 ms 48k-stereo
    PCM *per user*, OmniSession wants one 16k-mono stream -> a tick-driven
    mixer sums the active speakers and decimates to 16k mono. Speaker identity
    is taken straight from Discord (``user.display_name``) and injected as
    ``[speaker: name]`` — no embedding guessing needed.

Public surface
--------------
    DiscordOmniRecorder   recorder-contract adapter (start/stop/frames) + mixer
    OmniMixSink           voice_recv AudioSink feeding the recorder per user
    DiscordOmniPlayer     player-contract adapter (queue_audio/is_active/flush)
    DiscordOmniController  builds + owns OmniSession(s) per guild; one-shot a_audio
"""
from __future__ import annotations

import array
import asyncio
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

try:
    import discord
except ImportError:
    raise ImportError("pip install discord.py[voice]")

# --- reuse the whole Omni core (single source of truth) ---------------------
from toolboxv2.mods.isaa.base.audio_io.omni import (  # type: ignore
    OmniSession,
    VoiceModeConfig,
    JobManager,
    BlobStateStore,
    OMNI_SYSTEM_INSTRUCTION,
    make_agent_tools,
    make_vfs_peek_tools,
    make_chat_view_tools,
    TARGET_SR,  # 16000
)

try:
    from toolboxv2 import get_logger
    logger = get_logger()
except Exception:  # noqa: BLE001
    import logging
    logger = logging.getLogger(__name__)

# voice receive is optional (same guard as voice_mode.py)
try:
    from discord.ext import voice_recv
    VOICE_RECV_AVAILABLE = True
except ImportError:
    VOICE_RECV_AVAILABLE = False
    voice_recv = None  # type: ignore

try:
    import numpy as _np
    _HAVE_NP = True
except ImportError:  # numpy missing -> degraded single-speaker fallback
    _HAVE_NP = False

# Discord voice frame geometry (discord.py native): 48 kHz, 16-bit, stereo.
DISCORD_SR = 48000
DISCORD_CH = 2
FRAME_MS = 20
DECIMATE = DISCORD_SR // TARGET_SR  # 48k -> 16k => 3


# ---------------------------------------------------------------------------
# PCM mix + downsample (48k stereo -> 16k mono int16)
# ---------------------------------------------------------------------------

def _mix_downsample(frames: list[bytes]) -> bytes:
    """Mix N equal-length 48k-stereo PCM frames into one 16k-mono PCM frame.

    int32 accumulate -> clip -> stereo mean -> decimate by 3. Reuses the same
    stereo->mono + decimate recipe the classic sink already used, just for the
    summed signal. Falls back to first-speaker passthrough without numpy.
    """
    if not frames:
        return b""

    if not _HAVE_NP:
        a = array.array("h")
        a.frombytes(frames[0])
        # stereo -> mono (average L/R), then decimate by 3
        mono = array.array(
            "h",
            [(a[i] + a[i + 1]) // 2 for i in range(0, len(a) - 1, 2)],
        )
        return mono[::DECIMATE].tobytes()

    arrs = [_np.frombuffer(f, dtype=_np.int16).astype(_np.int32) for f in frames]
    n = min(a.shape[0] for a in arrs)
    mixed = _np.zeros(n, dtype=_np.int32)
    for a in arrs:
        mixed += a[:n]
    mixed = _np.clip(mixed, -32768, 32767).astype(_np.int16)
    if n % DISCORD_CH == 0:
        mono = mixed.reshape(-1, DISCORD_CH).mean(axis=1).astype(_np.int16)
    else:
        mono = mixed
    return mono[::DECIMATE].astype(_np.int16).tobytes()


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    if _HAVE_NP:
        a = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32)
        return float(_np.sqrt(_np.mean(a * a))) if a.size else 0.0
    a = array.array("h")
    a.frombytes(pcm)
    return (sum(s * s for s in a) / len(a)) ** 0.5 if a else 0.0


# ---------------------------------------------------------------------------
# Recorder adapter — Discord per-user audio -> one 16k-mono frame stream
# ---------------------------------------------------------------------------

class DiscordOmniRecorder:
    """recorder-contract for OmniSession: start/stop/frames().

    Fed (from the voice_recv reader thread) via ``feed()``; a tick loop on the
    bot loop mixes one 20 ms frame per active speaker into a single 16k-mono
    stream. Emits frames ONLY when someone is speaking — Discord already
    silence-gates, so gaps become natural turn boundaries (no silence streamed).
    """

    recorder_type = "discord_omni"

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_speaker: Optional[Callable[[str], Awaitable[None]]] = None,
        *,
        frame_ms: int = FRAME_MS,
        max_user_frames: int = 100,  # ~2 s jitter buffer per user
        emit_silence: bool = False,  # True for needs_silence backends (pipeline)
    ):
        self._loop = loop
        self._on_speaker = on_speaker
        self._buffers: dict[int, deque] = {}
        self._names: dict[int, str] = {}
        self._lock = threading.Lock()
        self._out: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._frame_dt = frame_ms / 1000.0
        self._max = max_user_frames
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_speaker: Optional[str] = None
        # One 16k-mono silence frame; emitted in gaps so a backend whose own VAD
        # ends turns (needs_silence=True) actually sees the silence. Cloud backends
        # (needs_silence=False) leave this off and stay gap-gated.
        self._emit_silence = emit_silence
        self._silence = b"\x00" * (int(TARGET_SR * frame_ms / 1000) * 2)

    # -- thread side (called from voice_recv reader thread) ------------------
    def feed(self, user_id: int, name: str, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            dq = self._buffers.get(user_id)
            if dq is None:
                dq = deque(maxlen=self._max)
                self._buffers[user_id] = dq
            self._names[user_id] = name
            dq.append(pcm)

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._names.clear()

    # -- recorder contract ---------------------------------------------------
    async def start(self) -> None:
        self._running = True
        self._task = asyncio.ensure_future(self._mix_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._out.put_nowait(None)
        self.clear()

    async def frames(self):
        while self._running:
            frame = await self._out.get()
            if frame is None:
                return
            yield frame

    # -- mixer ---------------------------------------------------------------
    async def _mix_loop(self) -> None:
        try:
            next_t = self._loop.time()
            while self._running:
                next_t += self._frame_dt
                present: list[tuple[int, str, bytes]] = []
                with self._lock:
                    for uid, dq in self._buffers.items():
                        if dq:
                            present.append((uid, self._names.get(uid, f"user_{uid}"),
                                            dq.popleft()))
                if present:
                    if self._on_speaker is not None:
                        name = max(present, key=lambda p: _rms(p[2]))[1]
                        if name and name != self._last_speaker:
                            self._last_speaker = name
                            asyncio.ensure_future(self._safe_speaker(name))
                    mixed = _mix_downsample([p[2] for p in present])
                    if mixed:
                        self._out.put_nowait(mixed)
                elif self._emit_silence:
                    # gap: feed silence so needs_silence backends can end the turn
                    self._last_speaker = None
                    self._out.put_nowait(self._silence)
                delay = next_t - self._loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_t = self._loop.time()  # fell behind, resync clock
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001 - mixer must never kill the loop
            logger.warning("DiscordOmniRecorder mix loop stopped: %s", e)
            self._running = False

    async def _safe_speaker(self, name: str) -> None:
        try:
            await self._on_speaker(name)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001
            logger.debug("on_speaker inject failed: %s", e)


# ---------------------------------------------------------------------------
# voice_recv sink — routes each user's PCM into the recorder
# ---------------------------------------------------------------------------

if VOICE_RECV_AVAILABLE:

    class OmniMixSink(voice_recv.AudioSink):
        """Per-user PCM -> DiscordOmniRecorder.feed(). PCM (not opus).

        `is_allowed(user_id)` gates whose audio is ingested — only admins +
        the voice allowlist are heard; everyone else in the channel is ignored.
        """

        def __init__(self, recorder: DiscordOmniRecorder,
                     is_allowed: Optional[Callable[[int], bool]] = None):
            super().__init__()
            self.recorder = recorder
            self._is_allowed = is_allowed

        def wants_opus(self) -> bool:
            return False

        def write(self, user, data: "voice_recv.VoiceData") -> None:
            if user is None or not getattr(data, "pcm", None):
                return
            if self._is_allowed is not None and not self._is_allowed(user.id):
                return
            self.recorder.feed(
                user.id,
                getattr(user, "display_name", None) or f"user_{user.id}",
                data.pcm,
            )

        def cleanup(self) -> None:
            self.recorder.clear()

else:

    class OmniMixSink:  # type: ignore[no-redef]
        def __init__(self, recorder, is_allowed=None):
            raise RuntimeError("discord-ext-voice-recv not installed")


# ---------------------------------------------------------------------------
# Player adapter — OmniSession WAV out -> Discord voice (FFmpeg path reused)
# ---------------------------------------------------------------------------

class DiscordOmniPlayer:
    """player-contract for OmniSession: queue_audio/is_active/flush + start/stop.

    OmniSession (buffer_audio=True) hands ONE WAV per turn; a sequential worker
    plays it via ``FFmpegPCMAudio`` — the same mechanism the classic TTS path
    already uses — so FFmpeg resamples 24k/16k -> 48k stereo for free. ``flush``
    is the barge-in / INTERRUPTED hook.
    """

    def __init__(self, vc_getter: Callable[[], Optional[discord.VoiceClient]],
                 loop: asyncio.AbstractEventLoop):
        self._vc_getter = vc_getter
        self._loop = loop
        self._q: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._playing = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.ensure_future(self._worker())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
        await self.flush()

    async def queue_audio(self, wav: bytes, meta: Optional[dict] = None) -> None:
        if wav:
            self._q.put_nowait(wav)

    @property
    def is_active(self) -> bool:
        vc = self._vc_getter()
        return self._playing or not self._q.empty() or bool(vc and vc.is_playing())

    async def flush(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except Exception:  # noqa: BLE001
                break
        vc = self._vc_getter()
        if vc and vc.is_playing():
            vc.stop()
        self._playing = False

    async def _worker(self) -> None:
        try:
            while self._running:
                wav = await self._q.get()
                vc = self._vc_getter()
                if not (vc and vc.is_connected()):
                    continue
                while vc.is_playing():
                    await asyncio.sleep(0.05)
                self._playing = True
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
                    _tf.write(wav)
                    tmp = Path(_tf.name)
                done = asyncio.Event()

                def _after(err, _ev=done):
                    if err:
                        logger.debug("DiscordOmniPlayer playback error: %s", err)
                    self._loop.call_soon_threadsafe(_ev.set)

                try:
                    vc.play(discord.FFmpegPCMAudio(str(tmp)), after=_after)
                    await done.wait()
                finally:
                    self._playing = False
                    try:
                        tmp.unlink()
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning("DiscordOmniPlayer worker stopped: %s", e)


# ---------------------------------------------------------------------------
# Controller — builds + owns OmniSession(s); mirrors icli _handle_omni_command
# ---------------------------------------------------------------------------

class DiscordOmniController:
    """Per-guild OmniSession lifecycle + one-shot audio-file I/O.

    ``attach(vc, guild_id)`` is the Discord twin of the icli ``/omni start``:
    same VoiceModeConfig + system_instruction + tool stack, recorder/player
    swapped for the Discord adapters. ``handle_audio_file`` is the one-shot
    "audio in -> audio out" path, reusing ``agent.a_audio`` directly.
    """

    def __init__(
        self,
        interface: Any,
        *,
        mode: str = "omni_cloud",
        state_path: str = "isaa/omni/discord_omni_state.json",
    ):
        self.interface = interface
        self.agent = interface.agent
        self.mode = mode
        self.state_path = state_path
        self._sessions: dict[int, OmniSession] = {}
        self._recorders: dict[int, DiscordOmniRecorder] = {}
        self._vcs: dict[int, discord.VoiceClient] = {}
        self._jobs: Optional[JobManager] = None
        self._state_store: Optional[BlobStateStore] = None
        self._tools_added = False

    # -- lazy shared infra ---------------------------------------------------
    def _ensure_jobs(self) -> JobManager:
        if self._jobs is None:
            self._jobs = JobManager()
        return self._jobs

    def _ensure_state_store(self) -> Optional[BlobStateStore]:
        if self._state_store is None:
            try:
                from toolboxv2 import BlobFile  # type: ignore
                self._state_store = BlobStateStore(self.state_path, BlobFile)
            except Exception as e:  # noqa: BLE001
                logger.warning("Omni state store unavailable: %s", e)
                self._state_store = None
        return self._state_store

    def _add_tools_once(self, session: OmniSession) -> None:
        """delegate/vfs/chat + world-model + compress — added once to the agent.
        Uses the GENERIC omni.py tool factories (no icli all_executions needed)."""
        agent = self.agent
        jobs = self._ensure_jobs()
        try:
            if not self._tools_added:
                agent.add_tools(make_agent_tools(jobs, lambda name="default": agent))
                agent.add_tools(make_vfs_peek_tools(agent, jobs))

                def _history(n: int) -> list:
                    sm = getattr(agent, "session_manager", None)
                    sid = getattr(agent, "active_session", None)
                    if sm is None or sid is None:
                        return []
                    sess = sm.get(sid)
                    return sess.get_history_for_llm(n) if sess else []

                if hasattr(agent, "list_executions"):
                    agent.add_tools(make_chat_view_tools(agent.list_executions, _history))
                self._tools_added = True
            # compress_tool is session-bound -> add per session (name-deduped)
            agent.add_tools(session.compress_tool)
        except Exception as e:  # noqa: BLE001
            logger.warning("Omni tool registration partial: %s", e)

    def _make_backend_factory(self):
        cfg = VoiceModeConfig(
            mode=self.mode,
            agent=self.agent,
            kwargs=(
                {"voice": "Algenib", "input_transcription": True,
                 "output_transcription": True, "thinking_level": None}
                if self.mode == "omni_cloud" else {}
            ),
        )

        def _make_backend():
            b = cfg.build_backend()
            if b is not None and hasattr(b, "system_instruction"):
                b.system_instruction = OMNI_SYSTEM_INSTRUCTION
            return b

        return _make_backend

    def _tool_specs(self):
        tm = getattr(self.agent, "tool_manager", None)
        if tm is None:
            return []
        try:
            return tm.get_all_litellm()
        except Exception:  # noqa: BLE001
            try:
                return tm.get_all()
            except Exception:  # noqa: BLE001
                return []

    # -- live voice ----------------------------------------------------------
    def is_attached(self, guild_id: int) -> bool:
        return guild_id in self._sessions

    async def attach(self, vc: discord.VoiceClient, guild_id: int) -> dict:
        """Start an OmniSession for this guild, fed by ``vc``. Must run on the
        bot loop. Returns a small status dict."""
        if guild_id in self._sessions:
            return {"success": True, "already": True}
        if not VOICE_RECV_AVAILABLE or not hasattr(vc, "listen"):
            return {"success": False, "error": "discord-ext-voice-recv required for Omni"}

        loop = asyncio.get_running_loop()
        self._vcs[guild_id] = vc

        make_backend = self._make_backend_factory()
        backend = make_backend()
        if backend is None:
            return {"success": False, "error": f"mode={self.mode!r} produced no backend"}

        recorder = DiscordOmniRecorder(
            loop,
            on_speaker=lambda name, g=guild_id: self._inject_speaker(g, name),
            emit_silence=getattr(backend, "needs_silence", False),
        )
        player = DiscordOmniPlayer(lambda g=guild_id: self._vcs.get(g), loop)

        session = OmniSession(
            backend,
            backend_factory=make_backend,
            recorder=recorder,
            player=player,
            tools=getattr(self.agent, "tool_manager", None),
            jobs=self._ensure_jobs(),
            buffer_audio=True,            # Discord plays one source/turn via FFmpeg
            output_sample_rate=None,      # honour meta['sample_rate_out'] (24k Gemini)
            state_store=self._ensure_state_store(),
            summarizer_agent=self.agent,  # compression-fix: ALWAYS set
        )
        self._sessions[guild_id] = session
        self._recorders[guild_id] = recorder
        self._add_tools_once(session)

        try:
            await session.start(tool_specs=self._tool_specs())
        except Exception as e:  # noqa: BLE001 - roll back partial registration
            self._sessions.pop(guild_id, None)
            self._recorders.pop(guild_id, None)
            self._vcs.pop(guild_id, None)
            logger.warning("Omni attach failed, rolled back guild %s: %s", guild_id, e)
            return {"success": False, "error": str(e)}

        is_allowed = getattr(self.interface, "is_voice_allowed", None)
        vc.listen(OmniMixSink(recorder, is_allowed))  # feeding starts after loop is up
        logger.info("Omni attached to guild %s (backend=%s)", guild_id, backend.backend_name)
        return {"success": True, "backend": backend.backend_name}

    async def detach(self, guild_id: int) -> dict:
        session = self._sessions.pop(guild_id, None)
        self._recorders.pop(guild_id, None)
        self._vcs.pop(guild_id, None)
        if session is None:
            return {"success": False, "error": "no omni session"}
        try:
            await session.stop()
        except Exception as e:  # noqa: BLE001
            logger.warning("Omni detach: session stop failed: %s", e)
        return {"success": True}

    async def _inject_speaker(self, guild_id: int, name: str) -> None:
        session = self._sessions.get(guild_id)
        if session is None:
            return
        try:
            await session.send_user_text(f"[speaker: {name}]")
        except Exception as e:  # noqa: BLE001
            logger.debug("speaker inject failed: %s", e)

    # -- one-shot audio file in -> audio out --------------------------------
    async def handle_audio_file(
        self, wav_bytes: bytes, *, session_id: str = "discord_audio"
    ) -> tuple[Optional[bytes], str]:
        """S2S for a single audio attachment. Returns (audio_out_wav, text_out).
        Reuses the agent's existing ``a_audio`` end-to-end path — no session."""
        try:
            audio_out, text_out, _tc, _meta = await self.agent.a_audio(
                wav_bytes, session_id=session_id
            )
            return audio_out, text_out or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("Omni one-shot a_audio failed: %s", e)
            return None, f"[omni audio error: {e}]"
