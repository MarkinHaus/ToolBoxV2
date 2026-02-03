"""
OmniCore TTS Module - Text-to-Speech with Multiple Backends
============================================================

Supported Backends:
- piper: Local CPU inference (fast, lightweight)
- vibevoice: Local GPU inference (high quality, requires GPU)
- groq_tts: Groq Cloud API (Orpheus model, fast)
- elevenlabs: ElevenLabs API (highest quality)

All functions are "dumb" - they receive all config directly and return audio.
No state, no side effects, pure transformations.

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
from typing import BinaryIO, Generator, Literal, Optional, Union

from groq._response import BinaryAPIResponse

# Type aliases
TextInput = Union[str, list[str]]
AudioOutput = bytes


class TTSBackend(Enum):
    """Available TTS backends."""

    PIPER = "piper"
    VIBEVOICE = "vibevoice"
    GROQ_TTS = "groq_tts"
    ELEVENLABS = "elevenlabs"


class TTSQuality(Enum):
    """Audio quality presets."""

    LOW = "low"  # Fast, acceptable quality
    MEDIUM = "medium"  # Balanced
    HIGH = "high"  # Best quality, slower


@dataclass(frozen=True)
class TTSConfig:
    """
    Configuration for TTS operations.

    Attributes:
        backend: Which TTS engine to use
        voice: Voice identifier (backend-specific)
        language: ISO 639-1 language code
        speed: Speech speed multiplier (0.5 - 2.0)
        quality: Quality preset affecting output
        sample_rate: Output sample rate in Hz
        output_format: Audio format ("wav", "mp3", "opus")

    Backend-specific:
        piper: model_path for custom models
        vibevoice: speaker_id for multi-speaker
        groq_tts: voice from Orpheus voices
        elevenlabs: voice_id, model_id, stability, similarity_boost
    """

    backend: TTSBackend = TTSBackend.PIPER
    voice: str = ""  # Backend-specific voice ID
    language: str = "en"
    speed: float = 1.0
    quality: TTSQuality = TTSQuality.MEDIUM
    sample_rate: int = 22050
    output_format: str = "wav"

    # Piper-specific
    piper_model_path: Optional[str] = None

    # VibeVoice-specific
    vibevoice_speaker_id: int = 0
    vibevoice_reference_audio: Optional[str] = None  # For voice cloning

    # Groq TTS specific
    groq_api_key: Optional[str] = None
    groq_model: str = "canopylabs/orpheus-v1-english"

    # ElevenLabs specific
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0

    def __post_init__(self):
        # Validate speed
        if not 0.25 <= self.speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        # Set default voices if not specified
        if not self.voice:
            defaults = {
                TTSBackend.PIPER: "en_US-amy-medium",
                TTSBackend.VIBEVOICE: "Carter",
                TTSBackend.GROQ_TTS: "Fritz-PlayAI",
                TTSBackend.ELEVENLABS: "21m00Tcm4TlvDq8ikWAM",  # Rachel
            }
            object.__setattr__(self, "voice", defaults.get(self.backend, "default"))


@dataclass
class TTSResult:
    """
    Result from TTS synthesis.

    Attributes:
        audio: Raw audio bytes
        format: Audio format (wav, mp3, etc.)
        sample_rate: Audio sample rate in Hz
        duration: Audio duration in seconds
        channels: Number of audio channels
    """

    audio: bytes
    format: str = "wav"
    sample_rate: int = 22050
    duration: Optional[float] = None
    channels: int = 1

    def save(self, path: Union[str, Path]) -> None:
        """Save audio to file."""
        Path(path).write_bytes(self.audio)

    def to_numpy(self):
        """Convert to numpy array (requires numpy)."""
        import numpy as np

        if self.format == "wav":
            with io.BytesIO(self.audio) as buf:
                with wave.open(buf, "rb") as wav:
                    frames = wav.readframes(wav.getnframes())
                    return np.frombuffer(frames, dtype=np.int16)

        raise NotImplementedError(f"to_numpy not supported for format: {self.format}")


# =============================================================================
# AUDIO UTILITIES
# =============================================================================


def _create_wav_header(
    sample_rate: int, bits_per_sample: int = 16, channels: int = 1, data_size: int = 0
) -> bytes:
    """Create a WAV file header."""
    import struct

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # File size - 8
        b"WAVE",
        b"fmt ",
        16,  # Subchunk size
        1,  # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def _estimate_duration(text: str, speed: float = 1.0) -> float:
    """Estimate audio duration from text length."""
    # Average: ~150 words per minute, ~5 characters per word
    chars_per_second = (150 * 5) / 60  # ~12.5 chars/sec
    return len(text) / (chars_per_second * speed)


# =============================================================================
# PIPER BACKEND (LOCAL CPU)
# =============================================================================


def _synthesize_piper(text: str, config: TTSConfig) -> TTSResult:
    """
    Synthesize speech using Piper TTS (local CPU).

    Requirements:
        pip install piper-tts

    Models:
        Downloaded automatically from Hugging Face
        Format: {lang}_{country}-{name}-{quality}.onnx
        Example: en_US-amy-medium, de_DE-thorsten-high

    Note: First run downloads model (~50-100MB)
    """
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise ImportError("piper-tts not installed. Install with: pip install piper-tts")

    # Determine model path
    if config.piper_model_path:
        model_path = config.piper_model_path
    else:
        # Use voice name as model identifier
        # Piper will download from Hugging Face if not present
        model_path = config.voice

    # Load voice model
    voice = PiperVoice.load(model_path)

    # Synthesize to WAV bytes
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(voice.config.sample_rate)

        # Synthesize
        voice.synthesize(text, wav_file)

    audio_bytes = audio_buffer.getvalue()

    # Calculate duration
    with io.BytesIO(audio_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()

    return TTSResult(
        audio=audio_bytes,
        format="wav",
        sample_rate=voice.config.sample_rate,
        duration=duration,
        channels=1,
    )


def _stream_piper(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    """
    Stream audio synthesis using Piper TTS.

    Yields raw PCM audio chunks (no WAV header).
    """
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise ImportError("piper-tts not installed")

    model_path = config.piper_model_path or config.voice
    voice = PiperVoice.load(model_path)

    # Use streaming synthesis
    for audio_bytes in voice.synthesize_stream_raw(text):
        yield audio_bytes


# =============================================================================
# VIBEVOICE BACKEND (LOCAL GPU)
# =============================================================================


def _synthesize_vibevoice(text: str, config: TTSConfig) -> TTSResult:
    """
    Synthesize speech using VibeVoice (local GPU).

    Requirements:
        pip install vibevoice
        Requires NVIDIA GPU with 8GB+ VRAM

    Models:
        - VibeVoice-Streaming-0.5B: Fast, single speaker
        - VibeVoice-1.5B: Multi-speaker, high quality
        - VibeVoice-7B: Best quality, highest VRAM

    Note: Not suitable for CPU-only systems!
    """
    try:
        import torch
        from vibevoice import VibeVoice
    except ImportError:
        raise ImportError(
            "vibevoice not installed. "
            "Install with: pip install vibevoice\n"
            "Note: Requires NVIDIA GPU with 8GB+ VRAM"
        )

    # Check for GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VibeVoice requires NVIDIA GPU with CUDA. "
            "Use Piper or API backends for CPU-only systems."
        )

    # Load model
    model = VibeVoice.from_pretrained("microsoft/VibeVoice-Streaming-0.5B")

    # Prepare speaker
    if config.vibevoice_reference_audio:
        # Voice cloning from reference audio
        speaker = model.load_speaker(config.vibevoice_reference_audio)
    else:
        # Use built-in speaker
        speaker = config.voice

    # Synthesize
    audio = model.synthesize(text=text, speaker=speaker, speed=config.speed)

    # Convert to WAV bytes
    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)  # VibeVoice default
        wav_file.writeframes(audio.cpu().numpy().tobytes())

    audio_bytes = audio_buffer.getvalue()

    return TTSResult(
        audio=audio_bytes,
        format="wav",
        sample_rate=24000,
        duration=len(audio) / 24000,
        channels=1,
    )


def _stream_vibevoice(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    """
    Stream audio synthesis using VibeVoice.

    Uses the real-time streaming model for low-latency output.
    First chunk in ~300ms.
    """
    try:
        import torch
        from vibevoice import VibeVoice
    except ImportError:
        raise ImportError("vibevoice not installed")

    if not torch.cuda.is_available():
        raise RuntimeError("VibeVoice requires NVIDIA GPU")

    model = VibeVoice.from_pretrained("microsoft/VibeVoice-Streaming-0.5B")
    speaker = config.voice

    # Use streaming API
    for chunk in model.synthesize_stream(text=text, speaker=speaker):
        yield chunk.cpu().numpy().tobytes()


# =============================================================================
# GROQ TTS BACKEND (API)
# =============================================================================


def _synthesize_groq_tts(text: str, config: TTSConfig) -> TTSResult:
    """
    Synthesize speech using Groq TTS API (Orpheus model).

    Requirements:
        pip install groq
        Set GROQ_API_KEY environment variable

    Models:
        - canopylabs/orpheus-v1-english: English voices

    Voices (English):
        [autumn diana hannah austin daniel troy]
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

    # Call TTS API
    response = client.audio.speech.create(
        model=config.groq_model,
        voice=config.voice,
        input=text,
        response_format="wav",
        speed=config.speed,
    )

    # Get audio bytes - BinaryAPIResponse needs .read() or iterate
    audio_bytes = b"".join(response.iter_bytes())

    # Parse WAV to get metadata
    with io.BytesIO(audio_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            sample_rate = wav.getframerate()
            duration = wav.getnframes() / sample_rate
            channels = wav.getnchannels()

    return TTSResult(
        audio=audio_bytes,
        format="wav",
        sample_rate=sample_rate,
        duration=duration,
        channels=channels,
    )

def _stream_groq_tts(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    """
    Stream audio synthesis using Groq TTS API.

    Note: Groq TTS doesn't support true streaming yet,
    so we return the full audio in one chunk.
    """
    result = _synthesize_groq_tts(text, config)
    yield result.audio


# =============================================================================
# ELEVENLABS BACKEND (API)
# =============================================================================


def _synthesize_elevenlabs(text: str, config: TTSConfig) -> TTSResult:
    """
    Synthesize speech using ElevenLabs API.

    Requirements:
        pip install elevenlabs
        Set ELEVENLABS_API_KEY environment variable

    Models:
        - eleven_multilingual_v2: Best quality, multilingual
        - eleven_turbo_v2_5: Faster, good quality
        - eleven_flash_v2_5: Fastest, acceptable quality

    Popular Voices:
        - 21m00Tcm4TlvDq8ikWAM: Rachel (narrative)
        - EXAVITQu4vr4xnSDxMaL: Bella (young)
        - ErXwobaYiN019PkySvjV: Antoni (professional)
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        raise ImportError(
            "elevenlabs not installed. Install with: pip install elevenlabs"
        )

    api_key = config.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ElevenLabs API key required. Set ELEVENLABS_API_KEY environment variable "
            "or pass elevenlabs_api_key in config."
        )

    client = ElevenLabs(api_key=api_key)

    # Map quality to output format
    format_map = {
        TTSQuality.LOW: "mp3_22050_32",
        TTSQuality.MEDIUM: "mp3_44100_128",
        TTSQuality.HIGH: "pcm_44100",
    }
    output_format = format_map.get(config.quality, "mp3_44100_128")

    # Synthesize
    audio_generator = client.text_to_speech.convert(
        text=text,
        voice_id=config.voice,
        model_id=config.elevenlabs_model,
        output_format=output_format,
        voice_settings={
            "stability": config.elevenlabs_stability,
            "similarity_boost": config.elevenlabs_similarity_boost,
            "style": config.elevenlabs_style,
        },
    )

    # Collect all chunks
    audio_bytes = b"".join(audio_generator)

    # Determine format and sample rate from output_format string
    if "pcm" in output_format:
        fmt = "pcm"
        sample_rate = int(output_format.split("_")[1])
    else:
        fmt = "mp3"
        sample_rate = int(output_format.split("_")[1])

    return TTSResult(
        audio=audio_bytes,
        format=fmt,
        sample_rate=sample_rate,
        duration=_estimate_duration(text, config.speed),
        channels=1,
    )


def _stream_elevenlabs(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    """
    Stream audio synthesis using ElevenLabs API.

    True streaming support - yields chunks as generated.
    """
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        raise ImportError("elevenlabs not installed")

    api_key = config.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key required")

    client = ElevenLabs(api_key=api_key)

    # Use streaming endpoint
    audio_stream = client.text_to_speech.stream(
        text=text,
        voice_id=config.voice,
        model_id=config.elevenlabs_model,
    )

    for chunk in audio_stream:
        if isinstance(chunk, bytes):
            yield chunk


# =============================================================================
# PUBLIC API
# =============================================================================


def synthesize(text: str, config: Optional[TTSConfig] = None, **kwargs) -> TTSResult:
    """
    Synthesize speech from text.

    This is the main entry point for TTS operations.

    Args:
        text: Text to convert to speech
        config: TTSConfig object with all settings
        **kwargs: Override config settings

    Returns:
        TTSResult with audio data and metadata

    Examples:
        # Simple usage with defaults (local Piper)
        result = synthesize("Hello, world!")
        result.save("output.wav")

        # Using ElevenLabs API
        result = synthesize(
            "Professional narration text.",
            config=TTSConfig(
                backend=TTSBackend.ELEVENLABS,
                elevenlabs_api_key="your-key",
                voice="21m00Tcm4TlvDq8ikWAM"
            )
        )

        # German with Piper
        result = synthesize(
            "Guten Tag, wie geht es Ihnen?",
            config=TTSConfig(
                voice="de_DE-thorsten-medium",
                language="de"
            )
        )
    """
    if config is None:
        config = TTSConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = TTSConfig(**config_dict)

    # Route to appropriate backend
    backends = {
        TTSBackend.PIPER: _synthesize_piper,
        TTSBackend.VIBEVOICE: _synthesize_vibevoice,
        TTSBackend.GROQ_TTS: _synthesize_groq_tts,
        TTSBackend.ELEVENLABS: _synthesize_elevenlabs,
    }

    handler = backends.get(config.backend)
    if handler is None:
        raise ValueError(f"Unknown backend: {config.backend}")

    return handler(text, config)


def synthesize_stream(
    text: str, config: Optional[TTSConfig] = None, **kwargs
) -> Generator[bytes, None, None]:
    """
    Stream audio synthesis from text.

    Yields audio chunks as they become available.
    Useful for real-time playback or progressive download.

    Args:
        text: Text to convert to speech
        config: TTSConfig object
        **kwargs: Override config settings

    Yields:
        Audio bytes chunks

    Example:
        for chunk in synthesize_stream("Long text to speak..."):
            audio_player.write(chunk)
    """
    if config is None:
        config = TTSConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = TTSConfig(**config_dict)

    stream_handlers = {
        TTSBackend.PIPER: _stream_piper,
        TTSBackend.VIBEVOICE: _stream_vibevoice,
        TTSBackend.GROQ_TTS: _stream_groq_tts,
        TTSBackend.ELEVENLABS: _stream_elevenlabs,
    }

    handler = stream_handlers.get(config.backend)
    if handler is None:
        raise ValueError(f"Unknown backend: {config.backend}")

    yield from handler(text, config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def synthesize_piper(text: str, voice: str = "en_US-amy-medium", **kwargs) -> TTSResult:
    """
    Synthesize using local Piper TTS.

    Convenience function for quick local synthesis.
    """
    return synthesize(
        text, config=TTSConfig(backend=TTSBackend.PIPER, voice=voice, **kwargs)
    )


def synthesize_groq(
    text: str, api_key: Optional[str] = None, voice: str = "Fritz-PlayAI", **kwargs
) -> TTSResult:
    """
    Synthesize using Groq TTS API.

    Convenience function for quick API synthesis.
    """
    return synthesize(
        text,
        config=TTSConfig(
            backend=TTSBackend.GROQ_TTS, groq_api_key=api_key, voice=voice, **kwargs
        ),
    )


def synthesize_elevenlabs(
    text: str,
    api_key: Optional[str] = None,
    voice: str = "21m00Tcm4TlvDq8ikWAM",
    **kwargs,
) -> TTSResult:
    """
    Synthesize using ElevenLabs API.

    Convenience function for premium quality synthesis.
    """
    return synthesize(
        text,
        config=TTSConfig(
            backend=TTSBackend.ELEVENLABS,
            elevenlabs_api_key=api_key,
            voice=voice,
            **kwargs,
        ),
    )


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("OmniCore TTS Module")
    print("=" * 50)
    print("\nAvailable backends:")
    for backend in TTSBackend:
        print(f"  - {backend.value}")

    print("\nQuality presets:")
    for quality in TTSQuality:
        print(f"  - {quality.value}")
    result = synthesize("Hello, world!")
    result.save("output.wav")

    # ElevenLabs API
    result = synthesize(
        "Professional narration.",
        config=TTSConfig(
            backend=TTSBackend.ELEVENLABS,
            elevenlabs_api_key="your-key"
        )
    )

    # Streaming
    for chunk in synthesize_stream("Long text..."):
        play(chunk)
    print("\nUsage example:")
    print("""
    from tts import synthesize, TTSConfig, TTSBackend

    # Local synthesis with Piper
    result = synthesize("Hello, world!")
    result.save("output.wav")

    # ElevenLabs API
    result = synthesize(
        "Professional narration.",
        config=TTSConfig(
            backend=TTSBackend.ELEVENLABS,
            elevenlabs_api_key="your-key"
        )
    )

    # Streaming
    for chunk in synthesize_stream("Long text..."):
        play(chunk)
    """)
