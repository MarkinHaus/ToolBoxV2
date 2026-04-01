"""
OmniCore Audio I/O Integration
==============================

High-level audio processing for the ISAA Agent.

Pipeline:
    STT → ISAA (streaming, tool-augmented) → TTS → [LavaSR enhance] → Player

Player Backends:
    - LocalPlayer:  sounddevice, direct hardware output (RYZEN local)
    - WebPlayer:    async queue relay — streams WAV chunks to a websocket
                    consumer. Caller owns the consumer (FastAPI WS endpoint,
                    browser AudioWorklet, etc.). No local hardware needed.
    - NullPlayer:   silent, for testing / server-only deployments

speak() Tool Contract:
    The ISAA agent receives this tool in its tool registry. Any response
    intended for audio output MUST pass through speak(). The tool:
      1. Accepts text + optional emotion
      2. Queues it to the configured player (non-blocking)
      3. Returns immediately so the agent can continue

Version: 2.1.0
"""

import asyncio
import io
import os
import re
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Optional,
    Union,
)

from .Stt import STTBackend, STTConfig, STTResult, transcribe, transcribe_stream
from .Tts import (
    TTSBackend,
    TTSConfig,
    TTSEmotion,
    TTSResult,
    synthesize,
    synthesize_stream,
)

# Enhancer is optional — import gracefully
try:
    from .audio_enhancer import AudioEnhancer, EnhancerConfig
    _ENHANCER_AVAILABLE = True
except ImportError:
    AudioEnhancer = None  # type: ignore
    EnhancerConfig = None  # type: ignore
    _ENHANCER_AVAILABLE = False

# Type aliases
AudioData = Union[bytes, Path, str]
AsyncTextProcessor = Callable[[str], "AsyncGenerator[str, None]"]


class ProcessingMode(Enum):
    PIPELINE = "pipeline"
    NATIVE_AUDIO = "native"


class AudioQuality(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"


# =============================================================================
# SENTENCE SPLITTER
# =============================================================================

_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


def split_sentences(text: str, min_chars: int = 20) -> list:
    """
    Split text into TTS-ready sentence chunks.

    Short fragments (< min_chars) are merged with the next sentence
    to avoid excessive TTS calls with very short audio.

    Args:
        text: Input text to split
        min_chars: Minimum character count per chunk

    Returns:
        List of sentence strings, each ready for TTS synthesis
    """
    parts = _SENTENCE_RE.split(text.strip())
    result = []
    buffer = ""
    for part in parts:
        buffer = (buffer + " " + part).strip() if buffer else part
        if len(buffer) >= min_chars:
            result.append(buffer)
            buffer = ""
    if buffer:
        if result:
            result[-1] = result[-1] + " " + buffer
        else:
            result.append(buffer)
    return result or [text]


# =============================================================================
# AUDIO PLAYER ABSTRACTION
# =============================================================================


class AudioPlayer(ABC):
    """
    Abstract base for audio output backends.

    All players share the same async interface:
        await player.start()                         # initialize
        await player.queue_audio(bytes, metadata)    # enqueue WAV chunk
        await player.stop()                          # flush + teardown

    Subclasses handle actual delivery:
        - LocalPlayer:  writes to sounddevice hardware
        - WebPlayer:    pushes to asyncio.Queue for WS relay
        - NullPlayer:   discards audio, tracks metadata (testing)
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize the player and start any background tasks."""

    @abstractmethod
    async def stop(self) -> None:
        """Flush queued audio and release resources."""

    @abstractmethod
    async def queue_audio(self, wav_bytes: bytes, metadata: dict) -> None:
        """
        Enqueue a WAV audio chunk for output. Non-blocking.

        Args:
            wav_bytes: Complete WAV file bytes (including RIFF header)
            metadata:  Dict with keys: text, emotion, duration_s, chunk_index, session_id
        """

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """True if player has queued or in-progress audio."""

    @property
    def player_type(self) -> str:
        return self.__class__.__name__


# =============================================================================
# LOCAL PLAYER (sounddevice hardware output)
# =============================================================================


class LocalPlayer(AudioPlayer):
    """
    Plays audio through local hardware via sounddevice.

    Designed for RYZEN server with audio output or developer workstations.
    Audio is played sequentially — one chunk finishes before the next starts.

    Requirements:
        pip install sounddevice numpy
    """

    def __init__(self, device=0):
        self._queue = asyncio.Queue()
        self._task = None
        self._stop_event = asyncio.Event()
        self._playing = False
        self.device = device

    async def start(self) -> None:
        self._stop_event.clear()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    async def queue_audio(self, wav_bytes: bytes, metadata: dict) -> None:
        await self._queue.put((wav_bytes, metadata))

    @property
    def is_active(self) -> bool:
        return self._playing or not self._queue.empty()

    async def _worker(self):
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "LocalPlayer requires sounddevice + numpy:\n"
                "  pip install sounddevice numpy\n"
                f"Original error: {e}"
            )

        while not self._stop_event.is_set():
            try:
                try:
                    wav_bytes, meta = await asyncio.wait_for(
                        self._queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                self._playing = True
                try:
                    audio_np, sample_rate = _wav_to_numpy(wav_bytes)
                    sd.play(audio_np, sample_rate, device=self.device)
                    sd.wait()
                except Exception as e:
                    print(f"[LocalPlayer] Playback error: {e}")
                finally:
                    self._playing = False
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[LocalPlayer] Worker error: {e}")
                self._playing = False


# =============================================================================
# WEB PLAYER (async queue relay for WebSocket / HTTP streaming)
# =============================================================================


class WebPlayer(AudioPlayer):
    """
    Relays audio chunks to an async consumer (WebSocket endpoint, etc.).

    The WebPlayer does NOT handle the WebSocket itself. It maintains an
    asyncio.Queue that the caller drains. This decouples the player from
    any web framework (FastAPI, aiohttp, raw asyncio, etc.).

    Typical live usage:
        web_player = WebPlayer(max_queue=50)
        stream_player = setup_isaa_audio(agent, player_backend=web_player)
        await stream_player.start()

        # In FastAPI WebSocket handler:
        @app.websocket("/audio")
        async def ws_audio(ws: WebSocket):
            await ws.accept()
            async for chunk, meta in web_player.iter_chunks():
                await ws.send_bytes(chunk)  # raw WAV bytes per sentence

    Mock/test usage:
        player = WebPlayer(mock_mode=True)
        await player.start()
        await player.queue_audio(wav_bytes, meta)
        # Check player.received_chunks

    Wire format:
        Each chunk is a complete WAV file (with RIFF header).
        Clients can play chunks sequentially, e.g.:
            const src = new AudioBufferSourceNode(ctx);
            ctx.decodeAudioData(e.data).then(buf => { src.buffer = buf; src.start(); });

    Metadata per chunk (available to consumer via iter_chunks):
        text:        original text string
        emotion:     emotion name string
        duration_s:  float, audio duration in seconds
        chunk_index: int, monotonically increasing per session
        session_id:  str
    """

    def __init__(self, max_queue: int = 100, mock_mode: bool = False):
        self._queue = asyncio.Queue(maxsize=max_queue)
        self._running = False
        self._chunk_index = 0
        self.mock_mode = mock_mode
        # In mock_mode, chunks are stored here for test assertions
        self.received_chunks = []  # list[tuple[bytes, dict]]

    async def start(self) -> None:
        self._running = True
        self._chunk_index = 0

    async def stop(self) -> None:
        self._running = False
        if self.mock_mode:
            while not self._queue.empty():
                try:
                    chunk, meta = self._queue.get_nowait()
                    self.received_chunks.append((chunk, meta))
                except Exception:
                    break

    async def queue_audio(self, wav_bytes: bytes, metadata: dict) -> None:
        """
        Enqueue a WAV chunk for relay.

        In mock_mode: stored directly in received_chunks.
        In live mode: put into asyncio.Queue for consumer to drain.
        """
        meta = {**metadata, "chunk_index": self._chunk_index}
        self._chunk_index += 1

        if self.mock_mode:
            self.received_chunks.append((wav_bytes, meta))
        else:
            try:
                self._queue.put_nowait((wav_bytes, meta))
            except asyncio.QueueFull:
                print(f"[WebPlayer] Queue full ({self._queue.maxsize}), dropping chunk")

    async def iter_chunks(self) -> AsyncGenerator:
        """
        Async generator yielding (wav_bytes, metadata) as chunks arrive.

        Exits when stop() is called and queue is drained.

        Usage in WebSocket handler:
            async for chunk, meta in player.iter_chunks():
                await ws.send_bytes(chunk)
        """
        while self._running or not self._queue.empty():
            try:
                chunk, meta = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
                yield chunk, meta
                self._queue.task_done()
            except asyncio.TimeoutError:
                if not self._running:
                    break
            except Exception as e:
                print(f"[WebPlayer] iter_chunks error: {e}")
                break

    async def get_next_chunk(self, timeout: float = 5.0):
        """Get the next chunk with timeout. Returns None on timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    @property
    def is_active(self) -> bool:
        return not self._queue.empty()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def total_chunks_sent(self) -> int:
        return self._chunk_index

    def get_received_audio_frames(self) -> bytes:
        """
        In mock_mode: concatenate raw PCM frames from all received chunks.
        Useful for asserting audio content and duration.
        """
        all_frames = b""
        for wav_bytes, _ in self.received_chunks:
            try:
                with io.BytesIO(wav_bytes) as buf:
                    with wave.open(buf, "rb") as wav:
                        all_frames += wav.readframes(wav.getnframes())
            except Exception:
                all_frames += wav_bytes
        return all_frames

    def get_received_texts(self) -> list:
        return [meta.get("text", "") for _, meta in self.received_chunks]

    def get_received_emotions(self) -> list:
        return [meta.get("emotion", "neutral") for _, meta in self.received_chunks]

    def get_total_duration(self) -> float:
        """Total audio duration in seconds across all received chunks."""
        total = 0.0
        for wav_bytes, meta in self.received_chunks:
            d = meta.get("duration_s")
            if d is not None:
                total += d
            else:
                try:
                    total += wav_duration(wav_bytes)
                except Exception:
                    pass
        return total


# =============================================================================
# NULL PLAYER (silent, for headless / test deployments)
# =============================================================================


class NullPlayer(AudioPlayer):
    """
    Discards all audio. Tracks chunks in received_chunks for assertions.

    Use in unit tests where TTS output format matters but playback doesn't.
    No external dependencies required.
    """

    def __init__(self):
        self.received_chunks = []  # list[tuple[bytes, dict]]
        self._running = False
        self._chunk_index = 0

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def queue_audio(self, wav_bytes: bytes, metadata: dict) -> None:
        meta = {**metadata, "chunk_index": self._chunk_index}
        self._chunk_index += 1
        self.received_chunks.append((wav_bytes, meta))

    @property
    def is_active(self) -> bool:
        return False

    @property
    def total_chunks(self) -> int:
        return len(self.received_chunks)

    def total_audio_bytes(self) -> int:
        return sum(len(wav) for wav, _ in self.received_chunks)

    def get_received_texts(self) -> list:
        return [meta.get("text", "") for _, meta in self.received_chunks]

    def get_received_emotions(self) -> list:
        return [meta.get("emotion", "neutral") for _, meta in self.received_chunks]


# =============================================================================
# AUDIO STREAM PLAYER (TTS queue with injected player backend)
# =============================================================================


class AudioStreamPlayer:
    """
    Manages TTS synthesis and delegates audio delivery to an AudioPlayer.

    Accepts text + emotion via queue_text(), synthesizes with the configured
    TTS backend, optionally enhances with LavaSR, then calls player.queue_audio().

    The player backend (local / web / null) is injected at construction time,
    making this class framework-agnostic.

    Example (local):
        player = AudioStreamPlayer(
            player_backend=LocalPlayer(),
            tts_config=TTSConfig(backend=TTSBackend.GROQ_TTS, voice="autumn"),
        )
        await player.start()
        await player.queue_text("Hello!", emotion=TTSEmotion.FRIENDLY)

    Example (web relay):
        web = WebPlayer(max_queue=50)
        player = AudioStreamPlayer(player_backend=web, tts_config=...)
        await player.start()
        # Consumer drains web.iter_chunks() in a separate coroutine
    """

    def __init__(
        self,
        player_backend=None,
        tts_config=None,
        enhancer=None,
        session_id: str = "default",
    ):
        self._text_queue = asyncio.Queue()
        self._task = None
        self._stop_event = asyncio.Event()
        self._synthesizing = False
        self._chunk_index = 0

        self.player = player_backend if player_backend is not None else NullPlayer()
        self.tts_config = tts_config or TTSConfig(
            backend=TTSBackend.GROQ_TTS,
            voice="autumn",
            language="de",
        )
        self.enhancer = enhancer
        self.session_id = session_id

    async def start(self) -> None:
        await self.player.start()
        self._stop_event.clear()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._tts_worker())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.player.stop()

    async def queue_text(self, text: str, emotion=None) -> None:
        """Enqueue text for TTS synthesis + delivery. Non-blocking."""
        if emotion is None:
            emotion = TTSEmotion.NEUTRAL
        if text.strip():
            await self._text_queue.put((text.strip(), emotion))

    async def _tts_worker(self):
        """
        Background: dequeue text → TTS synthesize → optional enhance → player.queue_audio().
        TTS runs in executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()

        while not self._stop_event.is_set():
            try:
                try:
                    text, emotion = await asyncio.wait_for(
                        self._text_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                self._synthesizing = True
                try:
                    import dataclasses
                    cfg = dataclasses.replace(self.tts_config, emotion=emotion)

                    result = await loop.run_in_executor(
                        None, lambda t=text, c=cfg: synthesize(t, config=c)
                    )
                    audio_bytes = result.audio

                    if self.enhancer is not None:
                        audio_bytes = await loop.run_in_executor(
                            None, lambda b=audio_bytes: self.enhancer.enhance(b)
                        )

                    duration_s = None
                    try:
                        duration_s = wav_duration(audio_bytes)
                    except Exception:
                        pass

                    metadata = {
                        "text": text,
                        "emotion": emotion.value if hasattr(emotion, "value") else str(emotion),
                        "duration_s": duration_s,
                        "session_id": self.session_id,
                        "chunk_index": self._chunk_index,
                    }
                    self._chunk_index += 1

                    await self.player.queue_audio(audio_bytes, metadata)

                except Exception as e:
                    print(f"[AudioStreamPlayer] TTS error for '{text[:40]}': {e}")
                finally:
                    self._synthesizing = False
                    self._text_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[AudioStreamPlayer] Worker error: {e}")
                self._synthesizing = False

    @property
    def is_busy(self) -> bool:
        return self._synthesizing or not self._text_queue.empty() or self.player.is_active

    @property
    def pending_texts(self) -> int:
        return self._text_queue.qsize()


# =============================================================================
# AUDIO UTILITIES
# =============================================================================


def _wav_to_numpy(wav_bytes: bytes):
    """Convert WAV bytes to (numpy int16 array, sample_rate)."""
    import numpy as np
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            sr = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16), sr


def wav_duration(wav_bytes: bytes) -> float:
    """Return duration in seconds of a WAV byte buffer."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            return wav.getnframes() / wav.getframerate()


def wav_sample_rate(wav_bytes: bytes) -> int:
    """Return sample rate of a WAV byte buffer."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            return wav.getframerate()


def wav_channels(wav_bytes: bytes) -> int:
    """Return channel count of a WAV byte buffer."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            return wav.getnchannels()


def wav_is_valid(wav_bytes: bytes) -> bool:
    """Check if bytes are a valid WAV file."""
    if len(wav_bytes) < 12:
        return False
    return wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"


def make_silent_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """
    Generate a silent WAV buffer. Useful for testing pipeline without real TTS.
    """
    import struct
    n_samples = int(sample_rate * duration_s)
    pcm_data = b"\x00\x00" * n_samples  # 16-bit silence
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buf.getvalue()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class AudioIOConfig:
    """
    Configuration for the full audio I/O pipeline.

    Player backend selection:
        player_backend_type: "local" | "web" | "null"
            - "local":  LocalPlayer (sounddevice hardware output)
            - "web":    WebPlayer (async queue relay, caller consumes)
            - "null":   NullPlayer (silent, tracks chunks for tests)
        web_player_max_queue: Queue depth for WebPlayer

    Pipeline components:
        stt_config: STT backend config
        tts_config: TTS backend config (supports INDEX_TTS + emotion)
        enable_enhancement: Toggle LavaSR post-processing
    """

    mode: ProcessingMode = ProcessingMode.PIPELINE
    quality: AudioQuality = AudioQuality.BALANCED
    language: str = "en"

    # Player
    player_backend_type: str = "null"        # "local" | "web" | "null"
    web_player_max_queue: int = 100

    # Pipeline components
    stt_config: Optional[STTConfig] = None
    tts_config: Optional[TTSConfig] = None
    enable_enhancement: bool = False
    enhancer_config: Optional[Any] = None

    # Audio settings
    sample_rate: int = 16000
    channels: int = 1

    # Streaming behavior
    speak_min_chars: int = 40
    speak_flush_timeout: float = 1.5

    # Native mode (legacy)
    native_model: str = "LiquidAI/LFM2.5-Audio-1.5B"
    native_backend: str = "transformers"
    native_device: str = "cpu"
    native_quantization: Optional[str] = None

    def __post_init__(self):
        if self.stt_config is None:
            self.stt_config = STTConfig(
                backend=STTBackend.FASTER_WHISPER,
                language=self.language,
                device="cpu",
                compute_type="int8",
            )
        if self.tts_config is None:
            self.tts_config = TTSConfig(
                backend=TTSBackend.GROQ_TTS,
                language=self.language,
            )

    def build_player(self) -> AudioPlayer:
        """Instantiate the configured player backend."""
        if self.player_backend_type == "local":
            return LocalPlayer()
        elif self.player_backend_type == "web":
            return WebPlayer(max_queue=self.web_player_max_queue)
        elif self.player_backend_type == "null":
            return NullPlayer()
        raise ValueError(
            f"Unknown player_backend_type: '{self.player_backend_type}'. "
            "Valid: 'local', 'web', 'null'"
        )

    def build_stream_player(self, session_id: str = "default") -> "AudioStreamPlayer":
        """Instantiate a fully configured AudioStreamPlayer."""
        enhancer = None
        if self.enable_enhancement and _ENHANCER_AVAILABLE and self.enhancer_config:
            enhancer = AudioEnhancer(self.enhancer_config)
        return AudioStreamPlayer(
            player_backend=self.build_player(),
            tts_config=self.tts_config,
            enhancer=enhancer,
            session_id=session_id,
        )


@dataclass
class AudioIOResult:
    text_input: str
    text_output: str
    audio_output: Optional[bytes] = None
    tool_calls: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        return self.metadata.get("duration")


# =============================================================================
# SPEAK TOOL
# =============================================================================

SPEAK_TOOL_SYSTEM_PROMPT = """
## Audio Output Contract

You are operating in AUDIO MODE. The user receives your responses as speech.

**MANDATORY RULES:**
1. You MUST call the `speak` tool for EVERY response the user should hear.
2. Call `speak` EARLY — as soon as you have the first meaningful sentence.
3. Call `speak` multiple times for long responses (one call per logical paragraph).
4. The `final_answer` you return is a text transcript only; the user hears what you spoke.
5. Set `emotion` to match the content:
   - neutral: default, informational
   - friendly: greetings, casual conversation
   - excited: good news, discoveries
   - calm: explanations, instructions
   - serious: warnings, critical information
   - empathetic: apologies, user errors
   - urgent: time-critical information

**Pattern:**
speak(text="I found 3 results.", emotion="neutral")
[do more work if needed]
speak(text="Here is the most relevant one: ...", emotion="calm")
final_answer("I found 3 results. The most relevant one is ...")

Never return a final_answer without having called speak() at least once.
""".strip()


def create_speak_tool(player: AudioStreamPlayer):
    """
    Create a speak() tool function bound to an AudioStreamPlayer.

    Register on the ISAA agent:
        agent.add_tool(create_speak_tool(player), name="speak")

    Non-blocking: schedules TTS work and returns immediately.
    The player's backend handles delivery (local / web / null).
    """

    async def speak(text: str, emotion: str = "neutral") -> str:
        """
        Speak text aloud through the audio output system.

        MUST be called for any response the user should hear.
        Returns immediately — audio plays/streams in background.

        Args:
            text: Text to speak. One or two sentences per call works best.
            emotion: Tone preset — neutral / calm / excited / serious /
                     friendly / empathetic / urgent

        Returns:
            "[queued for speech]"
        """
        try:
            emo = TTSEmotion(emotion.lower())
        except ValueError:
            emo = TTSEmotion.NEUTRAL

        for sentence in split_sentences(text):
            await player.queue_text(sentence, emotion=emo)

        return "[queued for speech]"

    return speak


# =============================================================================
# PIPELINE PROCESSING
# =============================================================================


async def _process_pipeline_stream(
    audio_chunks,
    st_processor,
    config,
    stream_player=None,
):
    """
    STT → ISAA stream → sentence-split TTS → player delivery.

    If stream_player is provided, audio is delivered through it.
    If not, raw WAV bytes are yielded for the caller to handle.
    """
    full_text_parts = []
    for segment in transcribe_stream(audio_chunks, config=config.stt_config):
        full_text_parts.append(segment.text)

    text_input = " ".join(full_text_parts).strip()
    if not text_input:
        return

    enhancer = None
    if config.enable_enhancement and _ENHANCER_AVAILABLE and config.enhancer_config:
        enhancer = AudioEnhancer(config.enhancer_config)

    sentence_buffer = ""

    async for text_chunk in st_processor(text_input):
        sentence_buffer += text_chunk
        match = re.search(r"[.!?;:]\s", sentence_buffer)
        if match and len(sentence_buffer[:match.end()].strip()) >= config.speak_min_chars:
            to_speak = sentence_buffer[:match.end()].strip()
            sentence_buffer = sentence_buffer[match.end():]
            async for chunk in _synthesize_and_deliver(to_speak, config, enhancer, stream_player):
                yield chunk

    if sentence_buffer.strip():
        async for chunk in _synthesize_and_deliver(
            sentence_buffer.strip(), config, enhancer, stream_player
        ):
            yield chunk


async def _synthesize_and_deliver(text, config, enhancer, stream_player):
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(
        None, lambda: synthesize(text, config=config.tts_config)
    )
    audio_bytes = result.audio

    if enhancer is not None:
        audio_bytes = await loop.run_in_executor(
            None, lambda: enhancer.enhance(audio_bytes)
        )

    if stream_player is not None:
        await stream_player.player.queue_audio(audio_bytes, {"text": text})
    else:
        yield audio_bytes


async def _process_pipeline_raw(audio, processor, config):
    stt_result = transcribe(audio, config=config.stt_config)
    text_input = stt_result.text

    full_output = []
    async for chunk in processor(text_input):
        full_output.append(chunk)
    text_output = "".join(full_output)

    enhancer = None
    if config.enable_enhancement and _ENHANCER_AVAILABLE and config.enhancer_config:
        enhancer = AudioEnhancer(config.enhancer_config)

    tts_result = synthesize(text_output, config=config.tts_config)
    audio_bytes = tts_result.audio
    if enhancer:
        audio_bytes = enhancer.enhance(audio_bytes)

    return AudioIOResult(
        text_input=text_input,
        text_output=text_output,
        audio_output=audio_bytes,
        metadata={
            "mode": "pipeline",
            "stt_language": stt_result.language,
            "stt_duration": stt_result.duration,
            "enhanced": enhancer is not None,
            "duration": wav_duration(audio_bytes) if wav_is_valid(audio_bytes) else None,
        },
    )


# =============================================================================
# PUBLIC API
# =============================================================================


async def process_audio_raw(audio, processor, config=None, **kwargs):
    if config is None:
        config = AudioIOConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = AudioIOConfig(**config_dict)

    if config.mode == ProcessingMode.PIPELINE:
        return await _process_pipeline_raw(audio, processor, config)
    raise ValueError(f"Mode {config.mode} not supported here")


async def process_audio_stream(audio_chunks, processor, config=None, stream_player=None, **kwargs):
    if config is None:
        config = AudioIOConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = AudioIOConfig(**config_dict)

    if config.mode == ProcessingMode.PIPELINE:
        async for chunk in _process_pipeline_stream(
            audio_chunks, processor, config, stream_player
        ):
            yield chunk
    else:
        raise ValueError(f"Mode {config.mode} not supported here")


# =============================================================================
# SETUP HELPER
# =============================================================================


def setup_isaa_audio(
    agent,
    tts_config=None,
    player_backend=None,
    enable_enhancement: bool = False,
    enhancer_config=None,
    session_id: str = "default",
) -> AudioStreamPlayer:
    """
    One-call setup for ISAA audio.

    Creates AudioStreamPlayer, registers speak() tool, appends audio system prompt.

    Args:
        agent:             ISAAAgent / FlowAgent instance
        tts_config:        TTSConfig (backend, voice, emotion defaults)
        player_backend:    AudioPlayer instance. Defaults to NullPlayer.
                           Pass LocalPlayer() for hardware output.
                           Pass WebPlayer() for WS streaming.
        enable_enhancement: Enable LavaSR (requires installation)
        enhancer_config:   EnhancerConfig for LavaSR
        session_id:        Session identifier embedded in audio metadata

    Returns:
        AudioStreamPlayer — call `await player.start()` before first use.

    Examples:
        # Local hardware output
        player = setup_isaa_audio(agent, player_backend=LocalPlayer())

        # WebSocket relay
        web = WebPlayer(max_queue=50)
        player = setup_isaa_audio(agent, player_backend=web)
        await player.start()
        # async for chunk, meta in web.iter_chunks(): await ws.send_bytes(chunk)

        # Headless / testing
        null = NullPlayer()
        player = setup_isaa_audio(agent, player_backend=null)
        await player.start()
        # null.received_chunks for assertions
    """
    enhancer = None
    if enable_enhancement and _ENHANCER_AVAILABLE and enhancer_config is not None:
        enhancer = AudioEnhancer(enhancer_config)

    stream_player = AudioStreamPlayer(
        player_backend=player_backend if player_backend is not None else NullPlayer(),
        tts_config=tts_config,
        enhancer=enhancer,
        session_id=session_id,
    )

    speak_tool = create_speak_tool(stream_player)
    agent.add_tool(
        speak_tool,
        name="speak",
        description=(
            "Speak text aloud to the user. MUST be called for every response. "
            "Call early and multiple times for long answers. "
            "Set emotion to match content tone."
        ),
        category=["audio", "output"],
    )

    if hasattr(agent, "amd"):
        amd = agent.amd
        attr = "system_message" if hasattr(amd, "system_message") else "system_prompt"
        existing = getattr(amd, attr, "") or ""
        if "AUDIO MODE" not in existing:
            setattr(amd, attr, existing + "\n\n" + SPEAK_TOOL_SYSTEM_PROMPT)

    return stream_player
