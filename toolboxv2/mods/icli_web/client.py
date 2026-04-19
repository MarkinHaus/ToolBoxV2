"""
icli-side WebSocket client.

Connects to the FastAPI server at ws://<host>:<port>/ws/icli?key=...
with robust reconnection, heartbeat, and per-cid session cleanup.

Wire-up (in ICli.__init__):
    from toolboxv2.mods.icli_web.client import IcliWebClient
    IcliWebClient.get().attach(self)

The client is transport-agnostic — all agent lifecycle goes through
ICli.run_agent_for_web() which wraps _create_execution + _drain_agent_stream.
"""
from __future__ import annotations

import asyncio
import contextvars
import io
import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Optional

from toolboxv2 import get_logger

log = get_logger()

# Tracks which cid's agent task is currently running. Set by _run_agent
# before calling into the agent; read by interactive_panel tool so the
# tool knows which orb to send the panel to.
_current_cid: contextvars.ContextVar[Optional[str]] = \
    contextvars.ContextVar("_current_cid", default=None)


# ─── Sentence splitter for low-latency TTS ───────────────────────────────────

_SENTENCE_ENDS = ".!?。！？\n"

def _first_sentence_end(s: str) -> Optional[int]:
    """Return index of first sentence-ending char at >=30 chars, or the
    earliest ending >=12 chars. None if none qualify."""
    best = None
    for i, ch in enumerate(s):
        if ch in _SENTENCE_ENDS:
            if i >= 30: return i
            best = i
    return best if best and best >= 12 else None


def _normalize_for_dedup(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    import re as _re
    return _re.sub(r"\s+", " ", s).lower().strip(" \t\n.,!?;:—-\"'")


def _dedupe_remainder(already_spoken: str, full_answer: str) -> str:
    """If most of `full_answer` has already been spoken via content-stream,
    return only the remainder (tail). If nothing was spoken yet, return
    the full answer. If we can't tell, play it safe and return the full
    answer — a tiny duplication is better than silence.

    Heuristic based on normalized strings:
      1. Full answer is already covered (substring of spoken) → return ""
      2. Spoken is a prefix of answer (even lossy, allowing some drift) →
         slice full_answer at roughly the same character ratio and return
         the tail. We match the FIRST 20 normalized chars to decide
         "prefix-like", then slice by length ratio on the ORIGINAL string.
      3. Otherwise, return the full answer — better to repeat a short
         answer than to lose it to a misdetected prefix.
    """
    spoken_norm = _normalize_for_dedup(already_spoken)
    answer_norm = _normalize_for_dedup(full_answer)
    if not answer_norm: return ""
    if not spoken_norm: return full_answer.strip()

    # Case 1: full answer already covered
    if answer_norm in spoken_norm: return ""

    # Case 2: spoken is prefix-like. We declare prefix-like iff:
    #   - spoken normalized is a prefix of answer normalized, OR
    #   - the first 20 normalized chars match AND spoken is at least
    #     ~30 chars (long enough that prefix match is not accidental).
    prefix_match = answer_norm.startswith(spoken_norm)
    fuzzy_prefix = (
        len(spoken_norm) >= 30
        and answer_norm[:20] == spoken_norm[:20]
    )
    if prefix_match or fuzzy_prefix:
        # Find where spoken ends in the answer, using normalized lengths
        # proportionally. Then walk forward in the ORIGINAL string until
        # we hit a word boundary.
        ratio = min(1.0, len(spoken_norm) / max(1, len(answer_norm)))
        cut = int(len(full_answer) * ratio)
        while cut < len(full_answer) and full_answer[cut] not in " \n":
            cut += 1
        tail = full_answer[cut:].strip()
        # If we sliced but ended up with essentially nothing, return empty
        if len(_normalize_for_dedup(tail)) < 3: return ""
        return tail

    # Case 3: default — speak the full answer
    return full_answer.strip()


# ─── Per-session state ───────────────────────────────────────────────────────

class Session:
    """One per active cid. Holds TTS player + stream buffer + cancel handle."""

    def __init__(self, cid: str, send_queue: asyncio.Queue):
        self.cid = cid
        self.send_queue = send_queue
        self.player = None
        self.web_player = None
        self.drain_task: Optional[asyncio.Task] = None
        self.agent_task: Optional[asyncio.Task] = None   # the run_agent_for_web task
        self.audio_buf = io.BytesIO()
        self.sentence_buf = ""
        self.emotion = "neutral"
        self.closed = False
        # Monotonic counter of audio chunks sent to the orb for this cid.
        # We don't trust WebPlayer's meta['chunk_index'] because it may
        # restart per synth call; we need one sequence across the whole cid.
        self.chunk_index = 0
        # Orb-side state echoed into agent context (for interactive panel)
        self.context: dict = {}
        # Concatenation of everything we've sent to TTS for this cid.
        # Used to dedupe: if final_answer matches what content-stream
        # already spoke, don't speak it a second time.
        self.spoken_text = ""
        # Top-level task identity, so narrator events + sub-agent events
        # can be attached to the right monitor task without the chunk
        # having to carry it every time. Populated in _run_agent as soon
        # as the first chunk with task metadata arrives.
        self.task_id: Optional[str] = None
        self.agent_name: Optional[str] = None

    async def start_tts(self, tts_cfg: dict):
        from toolboxv2.mods.isaa.base.audio_io.audioIo import (
            AudioStreamPlayer, WebPlayer,
        )
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSConfig

        kw = _coerce_tts_kwargs(tts_cfg)
        try: cfg = TTSConfig(**kw)
        except Exception as e:
            log.warning("bad TTSConfig, using defaults: %s", e)
            cfg = TTSConfig()

        log.info("[%s] starting TTS backend=%s voice=%s lang=%s",
                 self.cid, cfg.backend.value, cfg.voice, cfg.language)

        self.emotion = (tts_cfg.get("emotion") or "neutral")
        self.web_player = WebPlayer(max_queue=50)
        self.player = AudioStreamPlayer(
            player_backend=self.web_player, tts_config=cfg,
            session_id=f"web-{self.cid}",
        )
        await self.player.start()
        self.drain_task = asyncio.create_task(self._drain_audio())
        log.info("[%s] TTS ready, drain task started", self.cid)

    async def _drain_audio(self):
        log.info("[%s] drain loop entered", self.cid)
        try:
            async for wav_bytes, meta in self.web_player.iter_chunks():
                if self.closed: break
                size = len(wav_bytes) if wav_bytes else 0
                # Header-only WAVs are 44 bytes. Some malformed outputs are
                # slightly larger (e.g. 46, 78) but contain no real samples.
                # We use a heuristic: reject anything small enough that it
                # cannot realistically be audible (< 1024 bytes = < ~23ms at
                # 22050 Hz / 16-bit mono).
                if size < 1024:
                    log.warning(
                        "[%s] DROPPING chunk: %d bytes is too small to be "
                        "audible (text=%r). TTS backend produced no usable "
                        "samples — check Tts.py:_synthesize_piper output.",
                        self.cid, size, (meta.get("text", "") or "")[:80],
                    )
                    continue
                idx = self.chunk_index
                self.chunk_index += 1
                log.info("[%s] drain got chunk idx=%d %d bytes text='%s'",
                         self.cid, idx, size,
                         (meta.get("text", "") or "")[:40])
                await self.send_queue.put(("text", {
                    "type": "audio", "cid": self.cid,
                    "text": meta.get("text", ""),
                    "emotion": meta.get("emotion", ""),
                    "duration_s": meta.get("duration_s"),
                    "chunk_index": idx,
                }))
                await self.send_queue.put(("bytes", wav_bytes))
        except asyncio.CancelledError:
            log.info("[%s] drain cancelled", self.cid)
        except Exception as e:
            log.warning("[%s] audio drain crashed: %s", self.cid, e)

    async def speak(self, text: str, emotion: Optional[str] = None):
        if not self.player or self.closed:
            log.warning("[%s] speak() but player=%s closed=%s",
                        self.cid, self.player, self.closed)
            return
        from toolboxv2.mods.isaa.base.audio_io.Tts import TTSEmotion
        try: em = TTSEmotion((emotion or self.emotion).lower())
        except Exception: em = TTSEmotion.NEUTRAL
        log.info("[%s] queue_text len=%d emotion=%s text='%s'",
                 self.cid, len(text), em.value, text[:60])
        await self.player.queue_text(text, emotion=em)
        # Track concat so final_answer can be deduped against streamed content
        self.spoken_text = (self.spoken_text + " " + text).strip()

    async def wait_for_tts_done(self, timeout: float = 60.0) -> None:
        """Wait until ALL queued text has been synthesized, delivered to
        the WebPlayer queue, forwarded through the drain_task, AND written
        out over the websocket.

        Four-stage wait:
          1. AudioStreamPlayer.is_busy == False
             (Piper done + WebPlayer queue empty + text_queue empty)
          2. Our drain_task has a moment to pick up the last chunk
          3. Our _send_queue is empty
             (the ("text", meta) + ("bytes", payload) pair has been sent)
          4. One tiny final tick for OS-level socket send to actually flush.

        Polls at 50ms. Exits early on session close. Cancellation re-raises.
        """
        if self.closed or not self.player:
            return
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        # Stage 1: synth + webplayer fully quiet
        while loop.time() < deadline:
            if self.closed: return
            if not getattr(self.player, "is_busy", False):
                break
            await asyncio.sleep(0.05)

        # Stage 2: give drain_task a chance to consume the last webplayer chunk
        await asyncio.sleep(0.05)

        # Stage 3: our outgoing send_queue is empty (both meta + bytes pushed)
        while loop.time() < deadline:
            if self.closed: return
            if self.send_queue.empty():
                break
            await asyncio.sleep(0.05)

        # Stage 4: tiny final tick so the websocket has flushed the bytes
        await asyncio.sleep(0.05)

    async def stop_tts(self):
        """Barge-in: discard all pending/current TTS. Agent task stays alive.
        Also clears the sentence buffer so the just-interrupted sentence
        won't be re-spoken when more content_chunks arrive."""
        self.sentence_buf = ""
        if self.player:
            try: await self.player.stop()
            except Exception: pass
        # Drain anything still queued in WebPlayer
        if self.web_player:
            try:
                clear = getattr(self.web_player, "clear", None)
                if clear: clear()
            except Exception: pass

    async def close(self):
        if self.closed: return
        self.closed = True
        if self.agent_task and not self.agent_task.done():
            self.agent_task.cancel()
        if self.drain_task:
            self.drain_task.cancel()
        if self.player:
            try: await self.player.stop()
            except Exception: pass


def _coerce_tts_kwargs(d: dict) -> dict:
    from toolboxv2.mods.isaa.base.audio_io.Tts import (
        TTSBackend, TTSEmotion, TTSQuality,
    )
    out = {}
    for key, cast in (("backend", TTSBackend), ("emotion", TTSEmotion),
                      ("quality", TTSQuality)):
        if key in d:
            try: out[key] = cast(d[key])
            except Exception: pass
    for key in ("voice", "language", "piper_model_path", "groq_model",
                "elevenlabs_model", "index_tts_reference_audio",
                "index_tts_reference_text", "index_tts_model_dir",
                "index_tts_device"):
        if key in d and d[key]: out[key] = str(d[key])
    for key in ("speed", "elevenlabs_stability", "elevenlabs_similarity_boost",
                "elevenlabs_style", "index_tts_cfg_scale"):
        if key in d:
            try: out[key] = float(d[key])
            except Exception: pass
    if "vibevoice_speaker_id" in d:
        try: out["vibevoice_speaker_id"] = int(d["vibevoice_speaker_id"])
        except Exception: pass
    if "style_prompt" in d and d["style_prompt"]:
        out["style_prompt"] = str(d["style_prompt"])
    return out


def _coerce_stt_kwargs(d: dict) -> dict:
    from toolboxv2.mods.isaa.base.audio_io.Stt import STTBackend
    out = {}
    if "backend" in d:
        try: out["backend"] = STTBackend(d["backend"])
        except Exception: pass
    for key in ("model", "language", "device", "compute_type"):
        if key in d and d[key]: out[key] = str(d[key])
    return out


# ─── The client ──────────────────────────────────────────────────────────────

class IcliWebClient:
    _inst: Optional["IcliWebClient"] = None

    # Heartbeat / reconnect timing
    PING_INTERVAL = 20.0       # seconds between pings
    PING_TIMEOUT = 10.0        # wait for pong
    BACKOFF_INITIAL = 1.0
    BACKOFF_MAX = 30.0
    BACKOFF_FACTOR = 1.5

    def __init__(self):
        self._icli = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._sessions: dict[str, Session] = {}
        # Persistent across reconnects — items queued while disconnected
        # are preserved and drained once a new WS comes up. Bounded so a
        # long outage can't OOM the icli process.
        self._send_queue: Optional[asyncio.Queue] = None
        self._ws = None
        self._url = ""

    @classmethod
    def get(cls) -> "IcliWebClient":
        if cls._inst is None: cls._inst = IcliWebClient()
        return cls._inst

    def attach(self, icli, url: Optional[str] = None,
               api_key: Optional[str] = None) -> None:
        """Call once from ICli.__init__."""
        # Force visible logging for this module — icli_web.client messages
        # would otherwise be swallowed if no root handler is configured,
        # or hidden if icli reroutes stdout.
        if not log.handlers:
            handler = logging.StreamHandler()  # writes to stderr by default
            handler.setFormatter(logging.Formatter(
                "[icli_web] %(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            ))
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            log.propagate = False

        self._icli = icli
        self._install_task_hook()

        if url is None:
            host = os.environ.get("ICLI_WEB_HOST", "127.0.0.1")
            port = os.environ.get("ICLI_WEB_PORT", "5055")
            url = f"ws://{host}:{port}/ws/icli"
        if api_key is None:
            api_key = os.environ.get("ICLI_WEB_API_KEY", "")
            if not api_key:
                try:
                    from pathlib import Path
                    kf = Path.home() / ".toolbox" / "icli_web.key"
                    api_key = kf.read_text().strip() if kf.exists() else ""
                except Exception: api_key = ""

        self._url = f"{url}?key={api_key}" if api_key else url
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="icli_web_client",
        )
        self._thread.start()
        log.info("icli_web client thread started → %s", url)

    def stop(self):
        """Graceful shutdown — call from ICli.on_exit."""
        self._running = False

    # ── Task broadcast hook ──────────────────────────────────────────────
    def _install_task_hook(self):
        try:
            from toolboxv2.utils.workers import get_registry
            reg = get_registry()
            original = reg.publish_sync

            def wrapped(id: str, data: Any):
                original(id=id, data=data)
                if id.startswith("icli.task.") and self._running:
                    self._enqueue_text({"type": "task", "data": data})
            reg.publish_sync = wrapped
        except Exception as e:
            log.warning("registry hook unavailable: %s", e)

    def _enqueue_text(self, msg: dict):
        if self._loop and self._send_queue:
            try:
                self._loop.call_soon_threadsafe(
                    self._send_queue.put_nowait, ("text", msg))
            except Exception: pass

    # ── Main loop ────────────────────────────────────────────────────────
    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_loop())
        except Exception as e:
            log.exception("client loop died: %s", e)

    async def _connect_loop(self):
        """Reconnect forever with exponential backoff + jitter."""
        try:
            import websockets
        except ImportError:
            log.error("pip install websockets"); return

        backoff = self.BACKOFF_INITIAL
        while self._running:
            connected_at = 0.0
            try:
                log.info("connecting to %s …",
                         self._url.split("?")[0])
                async with websockets.connect(
                    self._url,
                    max_size=None,
                    ping_interval=self.PING_INTERVAL,
                    ping_timeout=self.PING_TIMEOUT,
                    close_timeout=5.0,
                ) as ws:
                    self._ws = ws
                    connected_at = time.time()
                    log.info("connected to icli_web server")
                    backoff = self.BACKOFF_INITIAL
                    # Queue is persistent — only create on first connect.
                    # Subsequent reconnects re-use any items buffered
                    # during the outage, so chunks don't get dropped.
                    if self._send_queue is None:
                        self._send_queue = asyncio.Queue(maxsize=2048)

                    await self._send_hello()

                    try:
                        await asyncio.gather(
                            self._recv_loop(ws),
                            self._send_loop(ws),
                            self._keepalive_loop(),
                        )
                    except Exception as e:
                        log.info("session ended: %s", e)

            except (ConnectionRefusedError, OSError) as e:
                log.warning("cannot reach server: %s", e)
            except Exception as e:
                # websockets.exceptions.InvalidStatus, timeouts, etc.
                log.warning("ws error: %s", type(e).__name__)

            finally:
                self._ws = None
                # If we were connected for a while, reset backoff so a
                # brief server restart doesn't push us to 30s delay.
                if connected_at and time.time() - connected_at > 30:
                    backoff = self.BACKOFF_INITIAL
                # DON'T blindly close sessions here — a transient reconnect
                # shouldn't kill an in-flight agent task. The send_queue
                # is persistent; chunks produced during the outage will
                # drain over the new WS. If the disconnect turns out to
                # be long (>60s stuck), sessions will self-terminate via
                # their own timeouts. Only clean up on explicit stop().
                if not self._running:
                    await self._cleanup_sessions("client stopping")

            if not self._running: break

            # Backoff with ±20% jitter
            jitter = backoff * 0.2 * (2 * (uuid.uuid4().int & 0xFF) / 0xFF - 1)
            delay = max(0.1, backoff + jitter)
            log.info("reconnect in %.1fs", delay)
            await asyncio.sleep(delay)
            backoff = min(backoff * self.BACKOFF_FACTOR, self.BACKOFF_MAX)

    async def _cleanup_sessions(self, reason: str):
        """Cancel all in-flight sessions. Called on disconnect."""
        if not self._sessions: return
        log.info("cleaning up %d session(s): %s", len(self._sessions), reason)
        for sess in list(self._sessions.values()):
            try: await sess.close()
            except Exception: pass
        self._sessions.clear()

    # ── Hello ────────────────────────────────────────────────────────────
    async def _send_hello(self):
        try:
            from toolboxv2.mods.isaa.base.audio_io.Tts import (
                TTSBackend, TTSEmotion, TTSQuality,
            )
            from toolboxv2.mods.isaa.base.audio_io.Stt import STTBackend
            caps = {
                "tts": {
                    "backends": [b.value for b in TTSBackend],
                    "emotions": [e.value for e in TTSEmotion],
                    "qualities": [q.value for q in TTSQuality],
                    "voices": {
                        "piper": ["en_US-amy-medium", "de_DE-thorsten-high",
                                  "de_DE-thorsten-medium", "en_GB-alan-medium"],
                        "vibevoice": ["Carter", "Maya"],
                        "groq_tts": ["Fritz-PlayAI", "Ava-PlayAI",
                                     "Zola-PlayAI", "Celeste-PlayAI"],
                        "elevenlabs": ["21m00Tcm4TlvDq8ikWAM",
                                       "EXAVITQu4vr4xnSDxMaL"],
                        "index_tts": [],
                    },
                    "supports_style_prompt": ["qwen3_tts"],
                },
                "stt": {
                    "backends": [b.value for b in STTBackend],
                    "models": {
                        "faster_whisper": ["tiny","base","small","medium","large-v3"],
                        "groq_whisper": ["whisper-large-v3-turbo",
                                         "whisper-large-v3"],
                    },
                    "devices": ["cpu","cuda","auto"],
                    "compute_types": ["int8","float16","float32"],
                },
                "defaults": {
                    "tts_backend": "piper",
                    "stt_backend": "faster_whisper",
                    "language": os.environ.get("ICLI_WEB_LANG", "de"),
                },
                "agents": self._discover_agents(),
            }
        except ImportError:
            caps = {"tts": {"backends": []}, "stt": {"backends": []},
                    "agents": self._discover_agents()}

        # List live cids so the server can decide whether to tell orbs
        # "your cid is still alive — send resume" vs. ignore stale cids.
        active_cids = [
            {"cid": s.cid, "chunk_index": s.chunk_index}
            for s in self._sessions.values() if not s.closed
        ]

        await self._send_queue.put(("text", {
            "type": "hello", "capabilities": caps,
            "supports": {
                "interactive_panel": True,
                "chunk_index": True,
                "resume": True,
            },
            "active_cids": active_cids,
        }))

    def _discover_agents(self) -> list[str]:
        """Best-effort enumeration of agent names currently registered
        on the ICli / isaa_tools. Several attribute paths exist across
        ToolBoxV2 versions, so we probe in order and fall back to the
        orb's default set if nothing works — the selector always has
        something to show.

        Returns a sorted, deduplicated list of names. Never raises.
        """
        if self._icli is None:
            return ["self", "isaa", "coder"]
        names: set[str] = set()
        icli = self._icli
        isaa = getattr(icli, "isaa_tools", None) or getattr(icli, "isaa", None)
        # Probe 1: explicit registry attribute
        for attr in ("agents", "registered_agents", "_agents", "agent_names"):
            try:
                reg = getattr(isaa, attr, None) if isaa else None
                if reg is None: continue
                if isinstance(reg, dict):
                    names.update(str(k) for k in reg.keys())
                elif isinstance(reg, (list, tuple, set)):
                    names.update(str(x) for x in reg)
                if names: break
            except Exception: pass
        # Probe 2: list_agents() / get_agents() method
        if not names and isaa is not None:
            for method in ("list_agents", "get_agents",
                           "list_registered_agents"):
                try:
                    fn = getattr(isaa, method, None)
                    if fn is None: continue
                    result = fn()
                    if isinstance(result, dict):
                        names.update(str(k) for k in result.keys())
                    elif isinstance(result, (list, tuple, set)):
                        names.update(str(x) for x in result)
                    if names: break
                except Exception: pass
        # Probe 3: ICli itself might track them
        if not names:
            for attr in ("agents", "_agents", "active_agents"):
                try:
                    reg = getattr(icli, attr, None)
                    if reg is None: continue
                    if isinstance(reg, dict):
                        names.update(str(k) for k in reg.keys())
                    elif isinstance(reg, (list, tuple, set)):
                        names.update(str(x) for x in reg)
                    if names: break
                except Exception: pass
        if not names:
            return ["self", "isaa", "coder"]
        # Always include 'self' since that's the default entry point
        names.add("self")
        return sorted(names)

    async def _keepalive_loop(self):
        """Push a JSON {type:"ping"} onto the send queue every 15s.

        The server uses `ws.receive()` with a 45s recv timeout to detect
        dead clients. websockets-library-level pings (binary control
        frames) don't satisfy that check because they're handled below
        the FastAPI layer. Sending a real JSON ping every 15s keeps the
        recv timer fresh. Server already ignores {type:"ping"}.
        """
        try:
            while True:
                await asyncio.sleep(15.0)
                if not self._send_queue: return
                try:
                    self._send_queue.put_nowait(("text", {"type": "ping"}))
                except asyncio.QueueFull:
                    pass
        except asyncio.CancelledError:
            return

    # ── IO loops ─────────────────────────────────────────────────────────
    async def _send_loop(self, ws):
        while True:
            kind, payload = await self._send_queue.get()
            try:
                if kind == "text":
                    await ws.send(json.dumps(payload, default=str))
                else:
                    await ws.send(payload)
            except Exception as e:
                log.warning("send failed: %s", e)
                return

    async def _recv_loop(self, ws):
        pending_audio_cid: Optional[str] = None
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                if pending_audio_cid:
                    sess = self._sessions.get(pending_audio_cid)
                    if sess and not sess.closed:
                        sess.audio_buf.write(raw)
                continue
            try: msg = json.loads(raw)
            except json.JSONDecodeError: continue
            t = msg.get("type")

            if t == "audio_chunk_in":
                pending_audio_cid = msg.get("cid")
                continue

            pending_audio_cid = None
            cid = msg.get("cid")

            if t == "query":
                asyncio.create_task(self._handle_query(msg))
            elif t == "tts_preview":
                asyncio.create_task(self._handle_tts_preview(msg))
            elif t == "audio_start":
                await self._start_session_for_audio(msg)
            elif t == "audio_end":
                if cid: asyncio.create_task(self._handle_audio_end(cid, msg))
            elif t == "stop_tts":
                # TTS-only stop: keep agent task alive. Barge-in where user
                # wants the current speech to shut up but still get the
                # same answer streamed / visible in transcript.
                sess = self._sessions.get(cid) if cid else None
                if sess: await sess.stop_tts()
            elif t == "cancel":
                # Full cancel: agent task + TTS + buffer. This is the
                # "agent redet mit sich selbst, unterbrechen" path.
                if cid:
                    sess = self._sessions.pop(cid, None)
                    if sess: await sess.close()

    # ── Session helpers ──────────────────────────────────────────────────
    async def _start_session_for_audio(self, msg: dict):
        cid = msg["cid"]
        if cid in self._sessions: return
        sess = Session(cid, self._send_queue)
        await sess.start_tts(msg.get("tts") or {})
        self._sessions[cid] = sess

    async def _get_or_start_session(self, cid: str, tts_cfg: dict) -> Session:
        s = self._sessions.get(cid)
        if s: return s
        s = Session(cid, self._send_queue)
        await s.start_tts(tts_cfg)
        self._sessions[cid] = s
        return s

    # ── Query (text path) ────────────────────────────────────────────────
    async def _handle_query(self, msg: dict):
        cid = msg["cid"]
        agent = msg.get("agent", "self")
        query = (msg.get("query") or "").strip()
        if not query:
            await self._send_done(cid, "empty query"); return

        sess = await self._get_or_start_session(cid, msg.get("tts") or {})
        # Store orb-side state (selected/entered panel fields) on the
        # session so the interactive_panel tool can read it and the agent
        # prompt can inject it as context.
        sess.context = msg.get("context") or {}
        try:
            await self._run_agent(cid, sess, agent, query)
        except asyncio.CancelledError:
            log.info("query %s cancelled", cid)
        except Exception as e:
            log.exception("query failed: %s", e)
            await self._send_done(cid, str(e))

    # ── TTS preview (no agent — direct synth for "test voice") ───────────
    async def _handle_tts_preview(self, msg: dict):
        cid = msg["cid"]
        text = (msg.get("text") or "").strip()
        if not text:
            await self._send_done(cid, "empty preview text"); return

        sess = await self._get_or_start_session(cid, msg.get("tts") or {})
        try:
            await sess.speak(text)
            await sess.wait_for_tts_done(timeout=30.0)
        except asyncio.CancelledError:
            log.info("tts_preview %s cancelled", cid)
        except Exception as e:
            log.exception("tts_preview failed: %s", e)
            await self._send_done(cid, str(e))
            return
        await self._send_done(cid)

    # ── Audio path (server STT) ──────────────────────────────────────────
    async def _handle_audio_end(self, cid: str, msg: dict):
        sess = self._sessions.get(cid)
        if not sess:
            await self._send_done(cid, "no active session"); return
        # Refresh orb-side state (may have been updated while recording)
        if msg.get("context"):
            sess.context = msg["context"]
        data = sess.audio_buf.getvalue()
        sess.audio_buf = io.BytesIO()
        if not data:
            await self._send_done(cid, "no audio data"); return

        try:
            from toolboxv2.mods.isaa.base.audio_io.Stt import (
                transcribe, STTConfig,
            )
            stt_kw = _coerce_stt_kwargs(msg.get("stt") or {})
            cfg = STTConfig(**stt_kw) if stt_kw else STTConfig()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: transcribe(data, config=cfg))
            text = result.text if result else ""
        except Exception as e:
            log.exception("transcribe: %s", e)
            await self._send_done(cid, f"stt: {e}"); return

        await self._send_queue.put(("text", {
            "type": "transcription", "cid": cid, "text": text,
        }))
        if not text.strip():
            await self._send_done(cid); return

        agent = msg.get("agent", "self")
        try:
            await self._run_agent(cid, sess, agent, text)
        except asyncio.CancelledError:
            log.info("audio query %s cancelled", cid)
        except Exception as e:
            await self._send_done(cid, str(e))

    # ── Agent execution via ICli.run_agent_for_web ────────────────────────
    async def _run_agent(self, cid: str, sess: Session, agent: str, query: str):
        """
        Drives the ICli.run_agent_for_web async generator and forwards all
        chunk types to the web. Tracks sess.agent_task so disconnect can
        cancel mid-flight.
        """
        run = getattr(self._icli, "run_agent_for_web", None)
        if run is None:
            await self._send_done(cid, "ICli missing run_agent_for_web"); return

        async def _drive():
            """The actual consumer. Runs as a task so we can cancel it."""
            _current_cid.set(cid)
            # Top-level agent for this cid. Used by narrator-event routing
            # when icli's enrich() didn't supply an explicit 'agent' field.
            sess.agent_name = agent
            chunk_count = 0
            async for chunk in run(agent, query):
                chunk_count += 1
                if sess.closed: break
                # Latch task_id from the first chunk that carries one
                # (enrich() sets it on most chunk types except narrator).
                # Sub-agent chunks carry the SAME task_id — they belong
                # to the parent task in the monitor.
                if sess.task_id is None:
                    tid = chunk.get("task_id")
                    if tid: sess.task_id = tid
                await self._forward_chunk(cid, sess, chunk)

            log.info("[%s] stream ended, %d chunks, buf_len=%d",
                     cid, chunk_count, len(sess.sentence_buf))
            # Flush remaining sentence buffer as TTS
            if sess.sentence_buf.strip() and not sess.closed:
                log.info("[%s] flushing buf: '%s'",
                         cid, sess.sentence_buf[:80])
                await sess.speak(sess.sentence_buf.strip())
                sess.sentence_buf = ""

        sess.agent_task = asyncio.create_task(_drive())
        error_msg: Optional[str] = None
        try:
            await sess.agent_task
        except asyncio.CancelledError:
            log.info("agent task %s cancelled", cid)
            raise
        except Exception as e:
            log.exception("agent task %s failed: %s", cid, e)
            error_msg = str(e)
        finally:
            # Critical: wait for the ENTIRE TTS pipeline to flush before
            # tearing down. The race we were losing:
            #   agent stream ends (fast) → finally runs → _send_done →
            #   sess.close() → drain_task.cancel() — all while Piper is
            #   still synthesizing on a thread and the queue is empty.
            # Result: WAV lands in WebPlayer after iter_chunks() is dead.
            try:
                log.info("[%s] waiting for TTS pipeline to drain…", cid)
                await sess.wait_for_tts_done(timeout=60.0)
                log.info("[%s] TTS pipeline drained", cid)
            except asyncio.CancelledError:
                # Caller (browser disconnect, user cancel) trumps drain wait
                log.info("[%s] drain wait cancelled", cid)
                raise
            except Exception as e:
                log.warning("[%s] drain wait failed: %s", cid, e)
            await self._send_done(cid, error=error_msg)

    async def _forward_chunk(self, cid: str, sess: Session, chunk: dict):
        """Translate an agent stream chunk into orb messages + TTS."""
        t = chunk.get("type", "")
        log.debug("[%s] chunk type=%s keys=%s", cid, t, list(chunk.keys()))

        if t == "content" and chunk.get("chunk"):
            piece = chunk["chunk"]
            await self._send_queue.put(("text", {
                "type": "text_chunk", "cid": cid, "text": piece,
            }))
            sess.sentence_buf += piece
            # Split at sentence boundaries — speak each complete sentence
            while True:
                idx = _first_sentence_end(sess.sentence_buf)
                if idx is None: break
                sentence = sess.sentence_buf[:idx+1]
                sess.sentence_buf = sess.sentence_buf[idx+1:]
                if sentence.strip():
                    await sess.speak(sentence.strip())

        elif t == "reasoning" and chunk.get("chunk"):
            # Show the agent's thinking in the transcript (sys-style)
            await self._send_queue.put(("text", {
                "type": "reasoning", "cid": cid,
                "text": chunk.get("chunk", ""),
                "iter": chunk.get("iter"),
            }))

        elif t == "tool_start":
            await self._send_queue.put(("text", {
                "type": "tool_start", "cid": cid,
                "name": chunk.get("name", "?"),
                "iter": chunk.get("iter"),
            }))

        elif t == "tool_result":
            result_raw = chunk.get("result", "")
            if isinstance(result_raw, (dict, list)):
                try: result_str = json.dumps(result_raw)[:200]
                except Exception: result_str = str(result_raw)[:200]
            else:
                result_str = str(result_raw)[:200]
            await self._send_queue.put(("text", {
                "type": "tool_result", "cid": cid,
                "name": chunk.get("name", "?"),
                "success": chunk.get("success", True),
                "info": result_str,
                "iter": chunk.get("iter"),
            }))

        elif t == "narrator":
            # icli enrich() guarantees: agent, iter, is_sub, narrator_msg.
            # We use those directly instead of guessing.
            narrator_msg = chunk.get("narrator_msg") or chunk.get("message", "")
            if narrator_msg:
                # Orb TTS path: speak / show in transcript
                await self._send_queue.put(("text", {
                    "type": "narrator", "cid": cid, "text": narrator_msg,
                }))
                # Monitor path: attach to the active task under the right
                # agent. enrich() sets 'agent' on every chunk; is_sub=True
                # marks sub-agent activity so the monitor can nest it
                # below the parent. task_id isn't on narrator chunks —
                # we latch it from earlier enriched chunks in _drive().
                if sess.task_id:
                    await self._send_queue.put(("text", {
                        "type": "narrator_event",
                        "task_id": sess.task_id,
                        "agent": chunk.get("agent") or sess.agent_name,
                        "is_sub": bool(chunk.get("is_sub")),
                        "iter": chunk.get("iter"),
                        "persona": chunk.get("persona"),
                        "text": narrator_msg,
                        "timestamp": time.time(),
                    }))

        elif t == "final_answer":
            answer = chunk.get("answer") or chunk.get("content", "") \
                     or chunk.get("chunk", "")
            log.info("[%s] final_answer received, len=%d, buf_len=%d, spoken=%d",
                     cid, len(answer), len(sess.sentence_buf),
                     len(sess.spoken_text))
            if answer:
                # Send the canonical answer as a 'response' for UI display
                # (replaces the streamed preview in the transcript)
                await self._send_queue.put(("text", {
                    "type": "response", "cid": cid, "text": answer,
                }))
                # Speak only if content-stream didn't already cover it.
                # Dedup heuristic: strip whitespace+punct, compare. If the
                # spoken text already covers ≥80% of the answer length and
                # the same leading words are there, skip — otherwise speak
                # the remainder (or the whole thing for small answers).
                leftover = _dedupe_remainder(sess.spoken_text, answer)
                # Cancel any mid-sentence buffer — final_answer supersedes.
                sess.sentence_buf = ""
                if leftover:
                    log.info("[%s] speaking final_answer leftover: %r",
                             cid, leftover[:80])
                    await sess.speak(leftover)
                else:
                    log.info("[%s] final_answer already spoken by stream, "
                             "skipping TTS", cid)

        elif t == "error":
            await self._send_queue.put(("text", {
                "type": "error", "cid": cid,
                "error": chunk.get("message", chunk.get("error", "?")),
            }))

        elif t == "done":
            # Server-side run_agent_for_web emits its own 'done' via
            # _send_done when generator exits; ignore inner 'done' chunks
            # from sub-agents etc.
            pass

    async def _send_done(self, cid: str, error: Optional[str] = None):
        if error:
            await self._send_queue.put(("text", {
                "type": "error", "cid": cid, "error": error,
            }))
        await self._send_queue.put(("text", {"type": "done", "cid": cid}))
        sess = self._sessions.pop(cid, None)
        if sess: await sess.close()

    # ── Public API for the interactive_panel agent tool ──────────────────
    # The agent calls these via a registered tool function. The tool reads
    # the current cid from the _current_cid contextvar and routes the panel
    # event back to the correct orb.

    def send_interactive_panel(self, template: str,
                               state: Optional[dict] = None,
                               content: Optional[dict] = None,
                               panel_id: Optional[str] = None) -> dict:
        """Thread-safe entry point for agent tool code. Schedules the
        panel event onto the client loop. Returns what the agent sees
        (incl. the panel_id so the agent can refer to it later)."""
        cid = _current_cid.get()
        if not cid:
            return {"ok": False, "error": "no active cid in this context"}
        pid = panel_id or uuid.uuid4().hex[:10]
        payload = {
            "type": "interactive_panel",
            "cid": cid,
            "panel_id": pid,
            "template": template,
            "state": state or {},
            "content": content or {},
        }
        if self._loop and self._send_queue:
            try:
                self._loop.call_soon_threadsafe(
                    self._send_queue.put_nowait, ("text", payload))
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "panel_id": pid}

    def get_panel_state(self) -> dict:
        """Tool-side accessor: what does the orb currently have selected
        in its interactive panel? Returns the stored sess.context dict."""
        cid = _current_cid.get()
        if not cid: return {}
        sess = self._sessions.get(cid)
        return dict(sess.context) if sess else {}
