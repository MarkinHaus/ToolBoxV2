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

# Type aliases
AudioData = Union[bytes, BinaryIO, Path, str]


class STTBackend(Enum):
    """Available STT backends."""

    FASTER_WHISPER = "faster_whisper"
    GROQ_WHISPER = "groq_whisper"


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

    def __post_init__(self):
        # Set default models if not specified
        if not self.model:
            defaults = {
                STTBackend.FASTER_WHISPER: "small",
                STTBackend.GROQ_WHISPER: "whisper-large-v3-turbo",
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

    raise TypeError(f"Unsupported audio input type: {type(audio)}")


def _save_temp_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Save audio bytes to a temporary file and return the path."""
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


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
    temp_path = _save_temp_audio(audio_bytes)

    try:
        # Initialize model
        model = WhisperModel(
            config.model, device=config.device, compute_type=config.compute_type
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
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


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

    model = WhisperModel(
        config.model, device=config.device, compute_type=config.compute_type
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
    else:
        # For non-streaming backends, collect all audio first
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
