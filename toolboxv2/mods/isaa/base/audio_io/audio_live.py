# ─── audio_live.py ────────────────────────────────────────────────────────────
# toolboxv2/mods/isaa/base/audio_io/audio_live.py
"""
Hands-free live audio mode for icli.

Stack:
  VAD:          Silero VAD  (snakers4/silero-vad) — 1MB, ~0.4% CPU
  Wake word:    openWakeWord (dscripka/openWakeWord) — ONNX, custom models
  End detect:   Silero VAD silence + keyword/intent heuristics
  Mic capture:  sounddevice InputStream (non-blocking)
  Speaker ID:   pyannote speaker embeddings (optional, lazy-loaded)

Install:
  pip install silero-vad openwakeword sounddevice numpy
  pip install pyannote.audio          # optional, for speaker profiles

Flow:
  sounddevice → 80ms frames → openWakeWord
                                   │ wake detected
                                   ▼
                            Silero VAD streaming
                                   │ speech frames buffered
                                   ▼
                            End detector
                            (silence / keyword / intent)
                                   │ utterance complete
                                   ▼
                            callback(audio_bytes, speaker_name)
"""

import asyncio
import io
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

class EndMode(Enum):
    SILENCE  = "silence"   # Send after N ms of silence (most reliable)
    KEYWORD  = "keyword"   # Send when stop-word spoken ("fertig", "send it")
    INTENT   = "intent"    # Heuristic: falling energy at sentence end
    AUTO     = "auto"      # SILENCE + INTENT combined (recommended)


@dataclass
class LiveModeConfig:
    # Wake word
    wake_word_model:   str   = "hey_mycroft"  # built-in oww model or path to .tflite
    wake_sensitivity:  float = 0.5            # 0.0–1.0
    vad_threshold:     float = 0.5            # Silero VAD inside openWakeWord

    # Recording
    sample_rate:       int   = 16000
    frame_ms:          int   = 80             # 80ms = optimal for openWakeWord
    device:            Optional[int] = None   # None = system default mic

    # End detection
    end_mode:          EndMode = EndMode.AUTO
    silence_ms:        int   = 800            # ms of silence before send
    partial_interval_s: float = 1.5  # Live transcript re-run cadence
    max_partial_duration_s: float = 8.0  # After this, finalize-current + start new segment
    max_utterance_s:   float = 30.0           # hard cap

    # STT preset (resolved at live-mode start via resolve_stt_preset)
    stt_preset: str = "default"  # STTPreset.value

    # Stop keywords (any language — checked post-VAD on the raw text buffer)
    stop_keywords: list = field(default_factory=lambda: [
        "fertig", "done", "send", "send it", "ok send",
        "stop", "ende", "abschicken", "go",
    ])

    # Speaker profiles
    enable_speaker_id: bool  = False
    speaker_embed_model: str = "pyannote/embedding"  # HF model id


# =============================================================================
# SPEAKER PROFILE STORE
# =============================================================================

@dataclass
class SpeakerProfile:
    name:      str
    embedding: np.ndarray   # shape [D], L2-normalized


class SpeakerProfileStore:
    """
    Lightweight speaker profile registry.
    Stores L2-normalized embeddings, matches via cosine similarity.

    Persistence: JSON + numpy .npy files in profile_dir.

    Speaker profiles (the dream):
      1. User says /audio speaker add Markin
      2. System records 5s → extracts pyannote embedding → saves
      3. On each utterance, embedding extracted → cosine match → tag
      4. Agent receives [Markin]: <transcript> in query
    """

    def __init__(self, profile_dir: str = "~/.config/isaa/speaker_profiles"):
        self.dir = Path(profile_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.profiles: dict[str, SpeakerProfile] = {}
        self._load_all()

    def _load_all(self):
        for npy in self.dir.glob("*.npy"):
            name = npy.stem
            emb = np.load(str(npy))
            self.profiles[name] = SpeakerProfile(name=name, embedding=emb)

    def add(self, name: str, embedding: np.ndarray) -> None:
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.profiles[name] = SpeakerProfile(name=name, embedding=emb)
        np.save(str(self.dir / f"{name}.npy"), emb)

    def remove(self, name: str) -> bool:
        if name in self.profiles:
            del self.profiles[name]
            p = self.dir / f"{name}.npy"
            if p.exists():
                p.unlink()
            return True
        return False

    def identify(self, embedding: np.ndarray, threshold: float = 0.75) -> Optional[str]:
        """Return best matching speaker name or None if below threshold."""
        if not self.profiles:
            return None
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_name, best_sim = None, -1.0
        for prof in self.profiles.values():
            sim = float(np.dot(emb, prof.embedding))
            if sim > best_sim:
                best_sim, best_name = sim, prof.name
        return best_name if best_sim >= threshold else None

    def list_names(self) -> list:
        return list(self.profiles.keys())


# =============================================================================
# SILERO VAD WRAPPER
# =============================================================================

class SileroVAD:
    """
    Minimal Silero VAD wrapper for streaming frame-level detection.

    Install: pip install silero-vad
    Model:   ~1MB JIT, ~0.4% CPU on AMD desktop

    Silero operates on 512-sample chunks at 16kHz (32ms).
    We accumulate frames internally to match any input frame size.
    """

    CHUNK = 512  # samples @ 16kHz = 32ms

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._model = None
        self._buffer = np.array([], dtype=np.float32)

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                force_reload=False, trust_repo=True,
            )
            model.eval()
            self._model = model
            self._torch = torch
        except Exception as e:
            raise ImportError(
                "Silero VAD not available.\n"
                "  pip install silero-vad\n"
                f"  (also needs torch)  — {e}"
            )

    def is_speech(self, pcm_int16: bytes) -> float:
        """
        Returns speech probability [0.0, 1.0] for the given PCM frame.
        Accumulates internally, returns -1.0 when not enough data yet.
        """
        self._load()
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, samples])

        probs = []
        while len(self._buffer) >= self.CHUNK:
            chunk = self._buffer[:self.CHUNK]
            self._buffer = self._buffer[self.CHUNK:]
            tensor = self._torch.tensor(chunk).unsqueeze(0)
            with self._torch.no_grad():
                prob = self._model(tensor, 16000).item()
            probs.append(prob)

        return float(np.mean(probs)) if probs else -1.0

    def reset(self):
        self._buffer = np.array([], dtype=np.float32)
        if self._model is not None:
            self._model.reset_states()


# =============================================================================
# END DETECTOR
# =============================================================================

class EndDetector:
    """
    Decides when an utterance is complete.

    Modes:
      SILENCE:  N ms of consecutive VAD-silence after speech started
      KEYWORD:  stop-word detected in rolling whisper-tiny transcript
      INTENT:   falling RMS energy heuristic (sentence completion)
      AUTO:     SILENCE + INTENT OR'd
    """

    def __init__(self, config: LiveModeConfig):
        self.config   = config
        self._speech_started = False
        self._silence_start  = None
        self._rms_history:   list = []
        self._start_time:    Optional[float] = None

    def reset(self):
        self._speech_started = False
        self._silence_start  = None
        self._rms_history    = []
        self._start_time     = None

    def update(self, vad_prob: float, pcm_int16: bytes) -> bool:
        """
        Feed a frame. Returns True when utterance should end.
        """
        now = time.monotonic()
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))

        is_speech = vad_prob >= self.config.vad_threshold if vad_prob >= 0 else True

        if is_speech and not self._speech_started:
            self._speech_started = True
            self._start_time = now

        if not self._speech_started:
            return False

        # Hard cap
        if self._start_time and (now - self._start_time) > self.config.max_utterance_s:
            return True

        mode = self.config.end_mode

        # --- SILENCE ---
        if mode in (EndMode.SILENCE, EndMode.AUTO):
            if not is_speech:
                if self._silence_start is None:
                    self._silence_start = now
                elif (now - self._silence_start) * 1000 >= self.config.silence_ms:
                    return True
            else:
                self._silence_start = None

        # --- INTENT (falling energy) ---
        if mode in (EndMode.INTENT, EndMode.AUTO):
            self._rms_history.append(rms)
            if len(self._rms_history) > 10:
                self._rms_history.pop(0)
            if len(self._rms_history) >= 8:
                first_half = float(np.mean(self._rms_history[:4]))
                second_half = float(np.mean(self._rms_history[4:]))
                # Utterance falling energy + VAD silence = likely end
                if first_half > 0 and second_half / first_half < 0.35 and not is_speech:
                    return True

        return False

    def check_keyword(self, text: str) -> bool:
        """Return True if text contains a stop keyword."""
        t = text.lower().strip()
        return any(kw in t for kw in self.config.stop_keywords)


# =============================================================================
# LIVE MODE ENGINE
# =============================================================================

class LiveModeEngine:
    """
    Hands-free VAD + wake word engine.

    Usage:
        engine = LiveModeEngine(config, on_utterance=my_callback)
        await engine.start()
        # ... runs in background ...
        await engine.stop()

    Callback signature:
        async def on_utterance(audio_bytes: bytes, speaker: Optional[str]) -> None

    Speaker ID (optional):
        If config.enable_speaker_id and pyannote.audio is installed,
        each utterance is tagged with the closest registered speaker profile
        or None if unknown.
    """

    def __init__(
        self,
        config: LiveModeConfig,
        on_utterance: Callable,
        speaker_store: Optional[SpeakerProfileStore] = None,
        recorder: Optional[Any] = None,  # AudioRecorder; None = local mic
        on_partial: Optional[Callable] = None,  # async (text: str) -> None
        partial_transcriber: Optional[Callable[[bytes], str]] = None,
        require_wake_word: bool = True,
    ):
        self.config = config
        self.on_utterance = on_utterance
        self.on_partial = on_partial
        self._partial_transcriber = partial_transcriber
        self.speaker_store = speaker_store
        self._vad = SileroVAD(threshold=config.vad_threshold)
        self._end_detector = EndDetector(config)
        self._running = False
        self._task = None
        self._state = "idle"  # idle | listening | recording
        self._buffer: list = []  # PCM int16 bytes per frame
        self._oww_model = None
        self._embed_model = None  # pyannote embedding model
        self._recorder = recorder
        self._require_wake_word = require_wake_word
        self._last_partial_at: float = 0.0
        self._segment_start_idx: int = 0  # where the "current" partial starts
        self._last_partial_text: str = ""

        # Stats
        self.utterances_captured = 0
        self.wake_activations = 0
        self.current_speaker: Optional[str] = None

    # ── Wake word model ────────────────────────────────────────────────────────

    def _load_oww(self):
        if self._oww_model is not None:
            return
        try:
            import openwakeword
            from openwakeword.model import Model as OWWModel
            openwakeword.utils.download_models()
            self._oww_model = OWWModel(
                wakeword_models=[self.config.wake_word_model]
                    if Path(self.config.wake_word_model).exists()
                    else [],
                vad_threshold=0.0,  # We use our own Silero VAD
            )
        except ImportError:
            raise ImportError(
                "openWakeWord not installed.\n"
                "  pip install openwakeword\n"
                "  (automatically downloads ONNX models on first run)"
            )

    def _oww_predict(self, pcm_int16: bytes) -> bool:
        """Return True if wake word detected in frame."""
        if self._oww_model is None:
            return False
        samples = np.frombuffer(pcm_int16, dtype=np.int16)
        preds = self._oww_model.predict(samples)
        return any(v >= self.config.wake_sensitivity for v in preds.values())

    # ── Speaker embedding ──────────────────────────────────────────────────────

    def _extract_embedding(self, wav_bytes: bytes) -> Optional[np.ndarray]:
        """Extract pyannote speaker embedding from WAV bytes."""
        if not self.config.enable_speaker_id:
            return None
        try:
            import torch
            import torchaudio
            from pyannote.audio import Model
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

            if self._embed_model is None:
                self._embed_model = PretrainedSpeakerEmbedding(
                    self.config.speaker_embed_model, device=torch.device("cpu")
                )

            waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            with torch.no_grad():
                emb = self._embed_model({"waveform": waveform, "sample_rate": 16000})

            return emb.squeeze().numpy()
        except Exception:
            return None

    # ── PCM → WAV ──────────────────────────────────────────────────────────────

    def _pcm_to_wav(self, pcm_frames: list) -> bytes:
        raw = b"".join(pcm_frames)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.config.sample_rate)
            w.writeframes(raw)
        return buf.getvalue()

    # ── Main loop ──────────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._state = "idle"

    async def _run(self):
        """
        Main capture loop. Reads frames from self._recorder, routes via
        wake-word + VAD. If no recorder was injected, we spin up a default
        LocalMicRecorder for backwards compat.
        """
        if self._require_wake_word:
            self._load_oww()

        if self._recorder is None:
            from toolboxv2.mods.isaa.base.audio_io.audio_recorder import (
                LocalMicRecorder,
            )
            self._recorder = LocalMicRecorder(
                device=self.config.device,
                sample_rate=self.config.sample_rate,
                frame_ms=self.config.frame_ms,
            )

        await self._recorder.start()
        try:
            async for pcm in self._recorder.frames():
                if not self._running:
                    break
                await self._process_frame(pcm)
        finally:
            await self._recorder.stop()

    async def _process_frame(self, pcm: bytes):
        vad_prob = self._vad.is_speech(pcm)

        if self._state == "idle":
            if self._require_wake_word:
                if self._oww_predict(pcm):
                    self._enter_listening()
            else:
                # Push-to-talk / manual start: VAD-edge triggers listening
                if vad_prob >= self.config.vad_threshold:
                    self._enter_listening()
                    self._buffer.append(pcm)

        elif self._state == "listening":
            self._buffer.append(pcm)
            should_end = self._end_detector.update(vad_prob, pcm)

            if should_end and len(self._buffer) >= 3:
                await self._dispatch_utterance()
                return

            if len(self._buffer) == 0 and vad_prob < 0.1:
                self._state = "idle"
                return

            # Partial transcribe cadence — rolling re-transcribe up to
            # max_partial_duration_s; after that, cap the segment and
            # continue accumulating from a new anchor.
            await self._maybe_emit_partial()

    def _enter_listening(self) -> None:
        self._state = "listening"
        self.wake_activations += 1
        self._vad.reset()
        self._end_detector.reset()
        self._buffer = []
        self._last_partial_at = time.monotonic()
        self._segment_start_idx = 0
        self._last_partial_text = ""

    async def _maybe_emit_partial(self) -> None:
        if self.on_partial is None or self._partial_transcriber is None:
            return

        now = time.monotonic()
        if (now - self._last_partial_at) < self.config.partial_interval_s:
            return

        # Window = frames since current segment anchor
        window_frames = self._buffer[self._segment_start_idx:]
        if not window_frames:
            return

        # Duration of current partial window (frames × frame_ms)
        window_s = len(window_frames) * self.config.frame_ms / 1000.0
        pcm_window = b"".join(window_frames)

        # Run transcribe in executor — never block the capture loop
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None, lambda b=pcm_window: self._partial_transcriber(b)
            )
        except Exception as e:
            # Partial is best-effort; don't tear the session down
            print(f"[LiveMode] partial transcribe error: {e}")
            self._last_partial_at = now
            return

        text = (text or "").strip()
        if text and text != self._last_partial_text:
            self._last_partial_text = text
            try:
                await self.on_partial(text)
            except Exception as e:
                print(f"[LiveMode] on_partial error: {e}")

        self._last_partial_at = now

        # Intelligent segment cap: if we hit max duration AND the current
        # partial ends in sentence-terminator punctuation, close the segment
        # and continue with a fresh anchor (rolling window).
        if window_s >= self.config.max_partial_duration_s:
            try:
                from toolboxv2.mods.isaa.base.audio_io.Stt import is_sentence_end
                at_boundary = is_sentence_end(text)
            except Exception:
                at_boundary = False
            if at_boundary:
                self._segment_start_idx = len(self._buffer)
                self._last_partial_text = ""

    async def _dispatch_utterance(self):
        """Convert buffer to WAV, identify speaker, fire callback."""
        self._state = "idle"
        wav = self._pcm_to_wav(self._buffer)
        self._buffer = []
        self._vad.reset()
        self._end_detector.reset()

        # Speaker identification
        speaker = None
        if self.config.enable_speaker_id and self.speaker_store:
            emb = self._extract_embedding(wav)
            if emb is not None:
                speaker = self.speaker_store.identify(emb)
        self.current_speaker = speaker
        self.utterances_captured += 1

        try:
            await self.on_utterance(wav, speaker)
        except Exception as e:
            print(f"[LiveMode] on_utterance error: {e}")

    @property
    def state(self) -> str:
        return self._state

    def status_line(self) -> str:
        s = self._state
        spk = f" | speaker: {self.current_speaker}" if self.current_speaker else ""
        return (
            f"state={s}  wake_hits={self.wake_activations}"
            f"  utterances={self.utterances_captured}{spk}"
        )
