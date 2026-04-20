"""
OmniCore STT Module - Speech-to-Text with Multiple Backends
============================================================

Supported Backends:
- faster_whisper: Local CPU/GPU inference (recommended for privacy)
- groq_whisper: Groq Cloud API (fast, reliable)

All functions are "dumb" - they receive all config directly and return text.
No state, no side effects, pure transformations.

Author: OmniCore Team
Version: 1.0.0
"""

import io
import os
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Generator, Optional, Union

import numpy as np

# Type aliases
AudioData = Union[bytes, BinaryIO, Path, str]


class STTBackend(Enum):
    """Available STT backends."""

    FASTER_WHISPER = "faster_whisper"
    GROQ_WHISPER = "groq_whisper"
    PARAKEET = "parakeet"


@dataclass(frozen=True)
class STTConfig:
    """
    Configuration for STT operations.

    Attributes:
        backend: Which STT engine to use
        model: Model identifier (backend-specific)
        language: ISO 639-1 language code (e.g., "en", "de")
        temperature: Sampling temperature (0.0 = deterministic)
        prompt: Context hint for better recognition
        word_timestamps: Include word-level timestamps

    Backend-specific defaults:
        faster_whisper: model="small" (good CPU balance)
        groq_whisper: model="whisper-large-v3-turbo" (fastest)
    """

    backend: STTBackend = STTBackend.FASTER_WHISPER
    model: str = ""  # Empty = use backend default
    language: Optional[str] = None
    temperature: float = 0.0
    prompt: Optional[str] = None
    word_timestamps: bool = False

    # Groq-specific
    groq_api_key: Optional[str] = None

    # faster-whisper specific
    device: str = "cpu"  # "cpu", "cuda", "auto"
    compute_type: str = "int8"  # "int8", "float16", "float32"

    parakeet_preset: str = "balanced"

    def __post_init__(self):
        # Set default models if not specified
        if not self.model:
            defaults = {
                STTBackend.FASTER_WHISPER: "small",
                STTBackend.GROQ_WHISPER: "whisper-large-v3-turbo",
                STTBackend.PARAKEET: "nvidia/parakeet-tdt-0.6b-v3",
            }
            object.__setattr__(self, "model", defaults.get(self.backend, "small"))


@dataclass
class STTResult:
    """
    Result from STT transcription.

    Attributes:
        text: Transcribed text
        language: Detected language code
        duration: Audio duration in seconds
        segments: List of text segments with timestamps
        confidence: Average confidence score (if available)
    """

    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: list = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class STTSegment:
    """A segment of transcribed text with timing."""

    text: str
    start: float
    end: float
    words: list = field(default_factory=list)


# =============================================================================
# AUDIO UTILITIES
# =============================================================================


def _normalize_audio_input(audio: AudioData) -> bytes:
    """
    Convert various audio input types to bytes.

    Accepts:
        - bytes: Raw audio data
        - BinaryIO: File-like object
        - Path/str: Path to audio file

    Returns:
        Audio data as bytes
    """
    if isinstance(audio, bytes):
        return audio

    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return path.read_bytes()

    if hasattr(audio, "read"):
        return audio.read()

    if isinstance(audio, np.ndarray):
        return audio.tobytes()

    raise TypeError(f"Unsupported audio input type: {type(audio)}")


def _save_temp_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Saves raw PCM audio bytes into a valid WAV file with a header.
    """
    # 1. Temporäre Datei erstellen
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = temp_file.name
    temp_file.close()  # Wir schließen sie kurz, damit 'wave' sie öffnen kann

    try:
        # 2. Mit dem wave-Modul öffnen, um den Header zu schreiben
        with wave.open(temp_path, "wb") as wav_file:
            # WICHTIG: Diese Werte müssen exakt zu deinem Recorder passen!
            # Basierend auf deinem Code: samplerate=16000, channels=1, dtype="int16"
            n_channels = 1
            sampwidth = 2  # 2 Bytes = 16-bit (int16)
            framerate = 16000

            wav_file.setnchannels(n_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(framerate)

            # 3. Die rohen Bytes schreiben
            wav_file.writeframes(audio_bytes)

    except Exception as e:
        print(f"Error writing WAV file: {e}")
        raise

    return temp_path

def _ensure_wav_format(audio_bytes: bytes) -> bytes:
    """
    Ensure audio is in WAV format.
    If already WAV, return as-is. Otherwise, attempt conversion.
    """
    # Check WAV header
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes

    # For non-WAV, we'd need ffmpeg or similar
    # For now, assume input is valid audio
    return audio_bytes


# =============================================================================
# FASTER-WHISPER BACKEND (LOCAL)
# =============================================================================


def _transcribe_faster_whisper(audio: AudioData, config: STTConfig) -> STTResult:
    """
    Transcribe audio using faster-whisper (local inference).

    Requirements:
        pip install faster-whisper

    Note: First run downloads the model (~500MB for 'small')
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        )

    audio_bytes = _normalize_audio_input(audio)

    temp_path = _save_temp_audio(audio_bytes, suffix=".wav")

    try:
        # Cached load via registry
        from toolboxv2.mods.isaa.base.audio_io.model_registry import get_faster_whisper
        model = get_faster_whisper(
            config.model, device=config.device, compute_type=config.compute_type,
        )

        # Transcribe
        segments_gen, info = model.transcribe(
            temp_path,
            language=config.language,
            temperature=config.temperature,
            initial_prompt=config.prompt,
            word_timestamps=config.word_timestamps,
            vad_filter=True,  # Voice activity detection
        )

        # Collect segments
        segments = []
        full_text_parts = []

        for segment in segments_gen:
            seg = STTSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=[
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (segment.words or [])
                ]
                if config.word_timestamps
                else [],
            )
            segments.append(seg)
            full_text_parts.append(segment.text)

        return STTResult(
            text=" ".join(full_text_parts).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments,
            confidence=None,  # faster-whisper doesn't provide overall confidence
        )
    finally:
        # Clean up
        # if os.path.exists(temp_path):
        #    os.unlink(temp_path)
        print(temp_path)


def _stream_faster_whisper(
    audio_chunks: Generator[bytes, None, None], config: STTConfig
) -> Generator[STTSegment, None, None]:
    """
    Stream transcription using faster-whisper.

    Note: faster-whisper doesn't natively support streaming,
    so we batch chunks and transcribe incrementally.

    Yields:
        STTSegment objects as they become available
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed")

    from toolboxv2.mods.isaa.base.audio_io.model_registry import get_faster_whisper
    model = get_faster_whisper(
        config.model, device=config.device, compute_type=config.compute_type,
    )

    # Accumulate audio chunks
    accumulated = b""
    chunk_size_seconds = 5.0  # Process every 5 seconds of audio
    sample_rate = 16000
    bytes_per_second = sample_rate * 2  # 16-bit audio
    chunk_bytes = int(chunk_size_seconds * bytes_per_second)

    for chunk in audio_chunks:
        accumulated += chunk

        if len(accumulated) >= chunk_bytes:
            # Save and transcribe
            temp_path = _save_temp_audio(accumulated)
            try:
                segments_gen, _ = model.transcribe(
                    temp_path,
                    language=config.language,
                    temperature=config.temperature,
                    vad_filter=True,
                )

                for segment in segments_gen:
                    yield STTSegment(
                        text=segment.text.strip(), start=segment.start, end=segment.end
                    )
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            accumulated = b""

    # Process remaining audio
    if accumulated:
        temp_path = _save_temp_audio(accumulated)
        try:
            segments_gen, _ = model.transcribe(
                temp_path,
                language=config.language,
                temperature=config.temperature,
                vad_filter=True,
            )
            for segment in segments_gen:
                yield STTSegment(
                    text=segment.text.strip(), start=segment.start, end=segment.end
                )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# =============================================================================
# GROQ WHISPER BACKEND (API)
# =============================================================================


def _transcribe_groq_whisper(audio: AudioData, config: STTConfig) -> STTResult:
    """
    Transcribe audio using Groq's Whisper API.

    Requirements:
        pip install groq
        Set GROQ_API_KEY environment variable or pass in config

    Models available:
        - whisper-large-v3-turbo (fastest, multilingual)
        - whisper-large-v3 (highest quality)
        - distil-whisper-large-v3-en (English only, fast)
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed. Install with: pip install groq")

    api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key required. Set GROQ_API_KEY environment variable "
            "or pass groq_api_key in config."
        )

    client = Groq(api_key=api_key)
    audio_bytes = _normalize_audio_input(audio)

    # Groq requires a file-like object with a name
    temp_path = _save_temp_audio(audio_bytes, suffix=".wav")

    try:
        with open(temp_path, "rb") as audio_file:
            # Prepare transcription parameters
            params = {
                "file": audio_file,
                "model": config.model,
                "response_format": "verbose_json",  # Get timestamps
                "temperature": config.temperature,
            }

            if config.language:
                params["language"] = config.language

            if config.prompt:
                params["prompt"] = config.prompt

            if config.word_timestamps:
                params["timestamp_granularities"] = ["word", "segment"]

            # Call API
            transcription = client.audio.transcriptions.create(**params)

        # Parse response
        segments = []
        if hasattr(transcription, "segments") and transcription.segments:
            for seg in transcription.segments:
                segments.append(
                    STTSegment(
                        text=seg.get("text", "").strip(),
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        words=seg.get("words", []) if config.word_timestamps else [],
                    )
                )

        return STTResult(
            text=transcription.text.strip() if hasattr(transcription, "text") else "",
            language=getattr(transcription, "language", config.language),
            duration=getattr(transcription, "duration", None),
            segments=segments,
            confidence=None,
        )

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# =============================================================================
# PARAKEET BACKEND (LOCAL CPU, streaming-native)
# =============================================================================

def _transcribe_parakeet(audio: AudioData, config: STTConfig) -> STTResult:
    """
    Transcribe audio using NVIDIA Parakeet-TDT via parakeet_stream.

    CPU-optimized (ONNX-Runtime INT8). Typically 20-30x realtime on
    modern CPUs. Supports 25 languages with auto-detection.

    Requirements:
        pip install parakeet-stream
    """
    from toolboxv2.mods.isaa.base.audio_io.model_registry import get_parakeet
    pk = get_parakeet(
        model_name=config.model,
        device=config.device,
        config_preset=config.parakeet_preset,
    )

    audio_bytes = _normalize_audio_input(audio)
    temp_path = _save_temp_audio(audio_bytes, suffix=".wav")
    try:
        result = pk.transcribe(temp_path)
        # parakeet_stream's Result exposes .text and .language
        text = getattr(result, "text", "") or ""
        lang = getattr(result, "language", config.language)
        duration = getattr(result, "duration", None)

        segments_raw = getattr(result, "segments", None) or []
        segments: list[STTSegment] = []
        for s in segments_raw:
            segments.append(STTSegment(
                text=getattr(s, "text", "").strip(),
                start=float(getattr(s, "start", 0.0)),
                end=float(getattr(s, "end", 0.0)),
                words=[],
            ))

        return STTResult(
            text=text.strip(),
            language=lang,
            duration=duration,
            segments=segments,
            confidence=None,
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _stream_parakeet(
    audio_chunks: Generator[bytes, None, None], config: STTConfig
) -> Generator[STTSegment, None, None]:
    """
    Parakeet streaming path. Batches chunks into ~2s windows and transcribes.

    Note: parakeet_stream has its own streaming API but the public pipeline
    here operates on bytes-chunks — we batch for consistency with the
    other _stream_* handlers.
    """
    from toolboxv2.mods.isaa.base.audio_io.model_registry import get_parakeet
    pk = get_parakeet(
        model_name=config.model,
        device=config.device,
        config_preset=config.parakeet_preset,
    )
    accumulated = b""
    window_seconds = 2.0
    window_bytes = int(window_seconds * 16000 * 2)

    for chunk in audio_chunks:
        accumulated += chunk
        if len(accumulated) >= window_bytes:
            temp_path = _save_temp_audio(accumulated)
            try:
                result = pk.transcribe(temp_path)
                text = (getattr(result, "text", "") or "").strip()
                if text:
                    yield STTSegment(text=text, start=0.0, end=window_seconds)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            accumulated = b""

    if accumulated:
        temp_path = _save_temp_audio(accumulated)
        try:
            result = pk.transcribe(temp_path)
            text = (getattr(result, "text", "") or "").strip()
            if text:
                yield STTSegment(text=text, start=0.0, end=0.0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# =============================================================================
# STT PRESETS (for /audio live and other auto-routed surfaces)
# =============================================================================

class STTPreset(Enum):
    """
    High-level presets picking a (partial, final) STT pair.

    Exact model/backend choices are resolved at runtime by
    resolve_stt_preset() based on what is installed + which API
    keys are in the environment.
    """
    DEFAULT        = "default"         # auto: best available
    SPEED_LOCAL    = "speed_local"     # parakeet cpu everywhere
    SPEED_API      = "speed_api"       # groq whisper-turbo everywhere
    QUALITY_LOCAL  = "quality_local"   # parakeet partial, whisper medium/large final
    QUALITY_API    = "quality_api"     # groq turbo partial, groq large-v3 final


def _available_backends() -> dict[str, bool]:
    """Snapshot of what's actually usable right now."""
    avail: dict[str, bool] = {}
    try:
        import parakeet_stream  # noqa: F401
        avail["parakeet"] = True
    except ImportError:
        avail["parakeet"] = False
    try:
        import faster_whisper  # noqa: F401
        avail["faster_whisper"] = True
    except ImportError:
        avail["faster_whisper"] = False
    try:
        import groq  # noqa: F401
        avail["groq_pkg"] = True
    except ImportError:
        avail["groq_pkg"] = False
    avail["groq_key"] = bool(os.environ.get("GROQ_API_KEY"))
    avail["groq"] = avail["groq_pkg"] and avail["groq_key"]
    try:
        import torch
        avail["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        avail["cuda"] = False
    return avail


def resolve_stt_preset(
    preset: STTPreset,
    language: Optional[str] = None,
) -> tuple[STTConfig, STTConfig, str]:
    """
    Return (partial_config, final_config, human_description).

    Raises RuntimeError with an actionable message if the preset cannot
    be satisfied by the current environment.

    DEFAULT picks the best available option in priority order:
        groq_api > parakeet_local > faster_whisper_local

    Language is propagated into both configs when given.
    """
    a = _available_backends()

    def _parakeet_fast() -> STTConfig:
        return STTConfig(
            backend=STTBackend.PARAKEET,
            model="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
            parakeet_preset="fast",
            language=language,
        )

    def _parakeet_balanced() -> STTConfig:
        return STTConfig(
            backend=STTBackend.PARAKEET,
            model="nvidia/parakeet-tdt-0.6b-v3",
            device="cpu",
            parakeet_preset="balanced",
            language=language,
        )

    def _whisper_small_cpu() -> STTConfig:
        return STTConfig(
            backend=STTBackend.FASTER_WHISPER,
            model="small", device="cpu", compute_type="int8",
            language=language,
        )

    def _whisper_big_gpu() -> STTConfig:
        return STTConfig(
            backend=STTBackend.FASTER_WHISPER,
            model="large-v3",
            device="cuda" if a["cuda"] else "cpu",
            compute_type="float16" if a["cuda"] else "int8",
            language=language,
        )

    def _groq_turbo() -> STTConfig:
        return STTConfig(
            backend=STTBackend.GROQ_WHISPER,
            model="whisper-large-v3-turbo",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            language=language,
        )

    def _groq_large() -> STTConfig:
        return STTConfig(
            backend=STTBackend.GROQ_WHISPER,
            model="whisper-large-v3",
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            language=language,
        )

    if preset == STTPreset.SPEED_API:
        if not a["groq"]:
            raise RuntimeError(
                "speed_api preset needs: pip install groq + export GROQ_API_KEY=..."
            )
        cfg = _groq_turbo()
        return cfg, cfg, "groq whisper-turbo (API) partial+final"

    if preset == STTPreset.QUALITY_API:
        if not a["groq"]:
            raise RuntimeError(
                "quality_api preset needs: pip install groq + export GROQ_API_KEY=..."
            )
        return _groq_turbo(), _groq_large(), "groq turbo → large-v3 (API)"

    if preset == STTPreset.SPEED_LOCAL:
        if not a["parakeet"]:
            raise RuntimeError(
                "speed_local preset needs: pip install parakeet-stream"
            )
        cfg = _parakeet_fast()
        return cfg, cfg, "parakeet fast CPU partial+final"

    if preset == STTPreset.QUALITY_LOCAL:
        if not a["parakeet"] or not a["faster_whisper"]:
            raise RuntimeError(
                "quality_local preset needs: pip install parakeet-stream faster-whisper"
            )
        return (
            _parakeet_fast(),
            _whisper_big_gpu(),
            f"parakeet fast partial → faster-whisper large-v3 "
            f"{'GPU' if a['cuda'] else 'CPU'} final",
        )

    # DEFAULT — auto-pick
    if a["groq"]:
        cfg = _groq_turbo()
        return cfg, cfg, "auto: groq whisper-turbo (API)"
    if a["parakeet"]:
        cfg = _parakeet_fast()
        return cfg, cfg, "auto: parakeet fast CPU"
    if a["faster_whisper"]:
        cfg = _whisper_small_cpu()
        return cfg, cfg, "auto: faster-whisper small CPU"

    raise RuntimeError(
        "No STT backend available.\n"
        "Install at least one of:\n"
        "  pip install parakeet-stream       (fastest local)\n"
        "  pip install faster-whisper        (local fallback)\n"
        "  pip install groq  +  export GROQ_API_KEY=...   (cloud)"
    )


def describe_stt_presets() -> str:
    """Human-readable status of all presets + which are currently usable."""
    a = _available_backends()
    lines = ["STT availability:"]
    lines.append(f"  parakeet-stream : {'✓' if a['parakeet'] else '✗'}")
    lines.append(f"  faster-whisper  : {'✓' if a['faster_whisper'] else '✗'}")
    lines.append(f"  groq package    : {'✓' if a['groq_pkg'] else '✗'}")
    lines.append(f"  GROQ_API_KEY    : {'✓' if a['groq_key'] else '✗'}")
    lines.append(f"  CUDA            : {'✓' if a['cuda'] else '✗'}")
    lines.append("")
    lines.append("Presets:")
    for p in STTPreset:
        try:
            _, _, desc = resolve_stt_preset(p)
            lines.append(f"  {p.value:<15} OK  — {desc}")
        except RuntimeError as e:
            lines.append(f"  {p.value:<15} --  {str(e).splitlines()[0]}")
    return "\n".join(lines)

# =============================================================================
# PUBLIC API
# =============================================================================


def transcribe(
    audio: AudioData, config: Optional[STTConfig] = None, **kwargs
) -> STTResult:
    """
    Transcribe audio to text.

    This is the main entry point for STT operations.

    Args:
        audio: Audio data (bytes, file path, or file-like object)
        config: STTConfig object with all settings
        **kwargs: Override config settings

    Returns:
        STTResult with transcribed text and metadata

    Examples:
        # Simple usage with defaults (local faster-whisper)
        result = transcribe("audio.wav")
        print(result.text)

        # Using Groq API
        result = transcribe(
            audio_bytes,
            config=STTConfig(
                backend=STTBackend.GROQ_WHISPER,
                groq_api_key="your-key"
            )
        )

        # German audio with context hint
        result = transcribe(
            "interview.mp3",
            config=STTConfig(
                language="de",
                prompt="Interview über KI und Technologie"
            )
        )
    """
    if config is None:
        config = STTConfig(**kwargs)
    elif kwargs:
        # Merge kwargs into config
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = STTConfig(**config_dict)

    # Route to appropriate backend
    backends = {
        STTBackend.FASTER_WHISPER: _transcribe_faster_whisper,
        STTBackend.GROQ_WHISPER: _transcribe_groq_whisper,
        STTBackend.PARAKEET: _transcribe_parakeet,
    }

    handler = backends.get(config.backend)
    if handler is None:
        raise ValueError(f"Unknown backend: {config.backend}")

    return handler(audio, config)


def transcribe_stream(
    audio_chunks: Generator[bytes, None, None],
    config: Optional[STTConfig] = None,
    **kwargs,
) -> Generator[STTSegment, None, None]:
    """
    Stream transcription from audio chunks.

    Note: Not all backends support true streaming.
    For non-streaming backends, chunks are batched internally.

    Args:
        audio_chunks: Generator yielding audio bytes
        config: STTConfig object
        **kwargs: Override config settings

    Yields:
        STTSegment objects as transcription progresses

    Example:
        def mic_stream():
            # Your microphone capture logic
            while recording:
                yield audio_chunk

        for segment in transcribe_stream(mic_stream()):
            print(f"[{segment.start:.1f}s] {segment.text}")
    """
    if config is None:
        config = STTConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = STTConfig(**config_dict)

    if config.backend == STTBackend.FASTER_WHISPER:
        yield from _stream_faster_whisper(audio_chunks, config)
    elif config.backend == STTBackend.PARAKEET:
        yield from _stream_parakeet(audio_chunks, config)
    else:
        all_audio = b"".join(audio_chunks)
        result = transcribe(all_audio, config)
        yield from result.segments


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def transcribe_local(
    audio: AudioData, model: str = "small", language: Optional[str] = None, **kwargs
) -> STTResult:
    """
    Transcribe using local faster-whisper.

    Convenience function for quick local transcription.
    """
    return transcribe(
        audio,
        config=STTConfig(
            backend=STTBackend.FASTER_WHISPER, model=model, language=language, **kwargs
        ),
    )


def transcribe_groq(
    audio: AudioData,
    api_key: Optional[str] = None,
    model: str = "whisper-large-v3-turbo",
    language: Optional[str] = None,
    **kwargs,
) -> STTResult:
    """
    Transcribe using Groq Whisper API.

    Convenience function for quick API transcription.
    """
    return transcribe(
        audio,
        config=STTConfig(
            backend=STTBackend.GROQ_WHISPER,
            groq_api_key=api_key,
            model=model,
            language=language,
            **kwargs,
        ),
    )

def transcribe_parakeet(
    audio: AudioData,
    model: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: str = "cpu",
    preset: str = "balanced",
    language: Optional[str] = None,
    **kwargs,
) -> STTResult:
    """Transcribe using Parakeet-TDT (CPU-optimized streaming STT)."""
    return transcribe(
        audio,
        config=STTConfig(
            backend=STTBackend.PARAKEET,
            model=model,
            device=device,
            parakeet_preset=preset,
            language=language,
            **kwargs,
        ),
    )


# =============================================================================
# PARTIAL TRANSCRIBE — for live streaming with sentence-end detection
# =============================================================================

_SENTENCE_END_CHARS = (".", "!", "?", "…")


def is_sentence_end(text: str) -> bool:
    """Heuristic: does `text` end in terminating punctuation?"""
    if not text:
        return False
    t = text.rstrip()
    return any(t.endswith(c) for c in _SENTENCE_END_CHARS)


def transcribe_partial(
    pcm_bytes: bytes,
    config: Optional[STTConfig] = None,
    sample_rate: int = 16000,
) -> str:
    """
    Cheap one-shot transcribe used for live partial updates.

    Wraps raw PCM int16 into a WAV, calls transcribe(), returns only the
    text string. No metadata — caller typically just wants something to
    show in a spinner.

    `config` is mutated-replaced so the backend's CPU preset is used
    (int8 / fast / balanced) regardless of what was configured for the
    final pass. Caller can override via config arg.
    """
    if config is None:
        config = STTConfig(
            backend=STTBackend.PARAKEET,
            device="cpu",
            parakeet_preset="fast",
        )
    wav = _save_temp_audio(pcm_bytes, suffix=".wav")
    try:
        result = transcribe(wav, config=config)
        return result.text.strip()
    finally:
        if os.path.exists(wav):
            os.unlink(wav)

# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("OmniCore STT Module")
    print("=" * 50)
    print("\nAvailable backends:")
    for backend in STTBackend:
        print(f"  - {backend.value}")

    print("\nUsage example:")
    print("""
    from stt import transcribe, STTConfig, STTBackend

    # Local transcription
    result = transcribe("audio.wav")
    print(result.text)

    # Groq API
    result = transcribe(
        "audio.wav",
        config=STTConfig(
            backend=STTBackend.GROQ_WHISPER,
            groq_api_key="your-key"
        )
    )
    """)
