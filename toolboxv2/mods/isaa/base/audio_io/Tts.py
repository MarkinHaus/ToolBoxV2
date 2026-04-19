"""
OmniCore TTS Module - Text-to-Speech with Multiple Backends
============================================================

Supported Backends:
- piper: Local CPU inference (fast, lightweight)
- vibevoice: Local GPU inference (high quality, requires GPU)
- groq_tts: Groq Cloud API (Orpheus model, fast)
- elevenlabs: ElevenLabs API (highest quality)
- index_tts: IndexTTS local GPU (zero-shot voice cloning, emotion-aware)

All functions are "dumb" - they receive all config directly and return audio.
No state, no side effects, pure transformations.

Version: 2.0.0
"""

import io
import os
import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Generator, Literal, Optional, Union

# Type aliases
TextInput = Union[str, list[str]]
AudioOutput = bytes


class TTSBackend(Enum):
    """Available TTS backends."""

    PIPER = "piper"
    VIBEVOICE = "vibevoice"
    GROQ_TTS = "groq_tts"
    ELEVENLABS = "elevenlabs"
    INDEX_TTS = "index_tts"  # Zero-shot voice cloning, emotion-aware


class TTSQuality(Enum):
    """Audio quality presets."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TTSEmotion(Enum):
    """
    Emotion/tone presets for supported backends (IndexTTS).

    These are injected as SSML-style hints or prompt prefixes
    depending on the backend.
    """

    NEUTRAL = "neutral"
    CALM = "calm"          # Slow, measured, reassuring
    EXCITED = "excited"    # Fast, energetic, enthusiastic
    SERIOUS = "serious"    # Deep, authoritative, formal
    FRIENDLY = "friendly"  # Warm, conversational
    EMPATHETIC = "empathetic"  # Soft, understanding
    URGENT = "urgent"      # Fast, tense


# Prompt prefixes injected before text for emotion-aware models
_EMOTION_PREFIXES: dict[TTSEmotion, str] = {
    TTSEmotion.NEUTRAL:    "",
    TTSEmotion.CALM:       "[calm, slow] ",
    TTSEmotion.EXCITED:    "[excited, energetic] ",
    TTSEmotion.SERIOUS:    "[serious, formal] ",
    TTSEmotion.FRIENDLY:   "[warm, friendly] ",
    TTSEmotion.EMPATHETIC: "[soft, empathetic] ",
    TTSEmotion.URGENT:     "[urgent, fast] ",
}


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
        emotion: Tone/mood for supported backends (IndexTTS)

    Backend-specific:
        piper: model_path for custom models
        vibevoice: speaker_id for multi-speaker
        groq_tts: voice from Orpheus voices
        elevenlabs: voice_id, model_id, stability, similarity_boost
        index_tts: reference_audio, reference_text, model_dir, device
    """

    backend: TTSBackend = TTSBackend.PIPER
    voice: str = ""
    language: str = "en"
    speed: float = 1.0
    quality: TTSQuality = TTSQuality.MEDIUM
    sample_rate: int = 22050
    output_format: str = "wav"
    emotion: TTSEmotion = TTSEmotion.NEUTRAL

    # Piper-specific
    piper_model_path: Optional[str] = None

    # VibeVoice-specific
    vibevoice_speaker_id: int = 0
    vibevoice_reference_audio: Optional[str] = None

    # Groq TTS specific
    groq_api_key: Optional[str] = None
    groq_model: str = "canopylabs/orpheus-v1-english"

    # ElevenLabs specific
    elevenlabs_api_key: Optional[str] = None
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0

    # IndexTTS specific
    # reference_audio: Path to a short WAV clip (3-10s) of target speaker.
    #   IndexTTS is zero-shot — it clones the voice from this reference.
    #   If None, falls back to a bundled default voice.
    index_tts_reference_audio: Optional[str] = None
    # reference_text: Transcript of reference_audio (improves quality, optional)
    index_tts_reference_text: Optional[str] = None
    # model_dir: Path to downloaded IndexTTS model weights.
    #   Download: git clone https://huggingface.co/IndexTeam/IndexTTS
    index_tts_model_dir: str = "./checkpoints"
    index_tts_device: str = "cuda"  # "cuda" or "cpu"
    # cfg_scale: Classifier-free guidance strength (1.0 = no guidance, 3.0 = strong)
    index_tts_cfg_scale: float = 3.0

    def __post_init__(self):
        if not 0.25 <= self.speed <= 4.0:
            raise ValueError("Speed must be between 0.25 and 4.0")

        if not self.voice:
            defaults = {
                TTSBackend.PIPER: "de_DE-thorsten-high",
                TTSBackend.VIBEVOICE: "Carter",
                TTSBackend.GROQ_TTS: "Fritz-PlayAI",
                TTSBackend.ELEVENLABS: "21m00Tcm4TlvDq8ikWAM",
                TTSBackend.INDEX_TTS: "",  # Uses reference audio, no named voice
            }
            object.__setattr__(self, "voice", defaults.get(self.backend, "default"))


@dataclass
class TTSResult:
    """Result from TTS synthesis."""

    audio: bytes
    format: str = "wav"
    sample_rate: int = 22050
    duration: Optional[float] = None
    channels: int = 1
    emotion: TTSEmotion = TTSEmotion.NEUTRAL

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_bytes(self.audio)

    def to_numpy(self):
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
    import struct

    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
        16, 1, channels, sample_rate, byte_rate,
        block_align, bits_per_sample, b"data", data_size,
    )


def _estimate_duration(text: str, speed: float = 1.0) -> float:
    chars_per_second = (150 * 5) / 60
    return len(text) / (chars_per_second * speed)


def _apply_emotion_prefix(text: str, emotion: TTSEmotion) -> str:
    """Prepend emotion hint to text for models that support it."""
    prefix = _EMOTION_PREFIXES.get(emotion, "")
    return prefix + text if prefix else text


# =============================================================================
# PIPER BACKEND (LOCAL CPU)
# =============================================================================


def _synthesize_piper(text: str, config: TTSConfig) -> TTSResult:
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise ImportError(
            "piper-tts not installed. Install with: pip install piper-tts"
        )

    model_path = _resolve_piper_model_path(config)
    voice = PiperVoice.load(model_path)

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        # Piper sets header itself when set_wav_format=True (default).
        voice.synthesize_wav(text, wav_file)

    audio_bytes = audio_buffer.getvalue()

    # Guard against silent failures — e.g. an unsupported API ending up
    # with only the 44-byte WAV header. Raise loudly instead of returning
    # a corrupt result that confuses downstream clients (icli_web, etc).
    if len(audio_bytes) <= 44:
        raise RuntimeError(
            f"Piper produced no audio samples for text={text!r}. "
            f"Got {len(audio_bytes)} bytes (header-only). "
            f"Check piper-tts version (need 1.3+) and model file integrity."
        )

    with io.BytesIO(audio_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            sample_rate = wav.getframerate()
            duration = wav.getnframes() / sample_rate

    return TTSResult(
        audio=audio_bytes, format="wav",
        sample_rate=sample_rate, duration=duration, channels=1,
        emotion=config.emotion,
    )


def _stream_piper(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    """Stream raw int16 PCM bytes from Piper in chunks.

    Note: these are RAW PCM bytes, not WAV — the caller is responsible for
    framing. If you need WAV framing per chunk, wrap each yielded blob in
    _create_wav_header(voice.config.sample_rate, ..., data_size=len(blob)).
    """
    try:
        from piper.voice import PiperVoice
    except ImportError:
        raise ImportError("piper-tts not installed")

    model_path = _resolve_piper_model_path(config)
    voice = PiperVoice.load(model_path)
    any_chunk = False
    for chunk in voice.synthesize(text):
        payload = getattr(chunk, "audio_int16_bytes", None)
        if payload:
            any_chunk = True
            yield payload
    if not any_chunk:
        raise RuntimeError(
            f"Piper synthesize() yielded no audio chunks for text={text!r}"
        )


def _resolve_piper_model_path(config: TTSConfig) -> str:
    """Build the absolute .onnx path Piper can load.

    Rules:
      - If piper_model_path is given and absolute, use it as-is.
      - Otherwise join config.voice (or model_path) onto PIPER_MODELS_FOLDER.
      - Always ensure the path ends in '.onnx' so Piper's auto-derived
        config lookup finds '<name>.onnx.json'.
    """
    raw = config.piper_model_path or config.voice
    if not raw.endswith(".onnx"):
        raw = raw + ".onnx"
    # Absolute paths pass through; relative paths resolve against the
    # models folder env var (default: current directory).
    if os.path.isabs(raw):
        return raw
    folder = os.getenv("PIPER_MODELS_FOLDER", ".")
    return os.path.join(folder, raw)


# =============================================================================
# VIBEVOICE BACKEND (LOCAL GPU)
# =============================================================================


def _synthesize_vibevoice(text: str, config: TTSConfig) -> TTSResult:
    try:
        import torch
        from vibevoice import VibeVoice
    except ImportError:
        raise ImportError("vibevoice not installed.")

    if not torch.cuda.is_available():
        raise RuntimeError("VibeVoice requires NVIDIA GPU with CUDA.")

    model = VibeVoice.from_pretrained("microsoft/VibeVoice-Streaming-0.5B")
    speaker = (
        model.load_speaker(config.vibevoice_reference_audio)
        if config.vibevoice_reference_audio
        else config.voice
    )
    audio = model.synthesize(text=text, speaker=speaker, speed=config.speed)

    audio_buffer = io.BytesIO()
    with wave.open(audio_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio.cpu().numpy().tobytes())

    return TTSResult(
        audio=audio_buffer.getvalue(), format="wav",
        sample_rate=24000, duration=len(audio) / 24000, channels=1,
    )


def _stream_vibevoice(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    try:
        import torch
        from vibevoice import VibeVoice
    except ImportError:
        raise ImportError("vibevoice not installed")
    if not torch.cuda.is_available():
        raise RuntimeError("VibeVoice requires NVIDIA GPU")
    model = VibeVoice.from_pretrained("microsoft/VibeVoice-Streaming-0.5B")
    for chunk in model.synthesize_stream(text=text, speaker=config.voice):
        yield chunk.cpu().numpy().tobytes()


# =============================================================================
# GROQ TTS BACKEND (API)
# =============================================================================


def _synthesize_groq_tts(text: str, config: TTSConfig) -> TTSResult:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed.")

    api_key = config.groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key required.")

    client = Groq(api_key=api_key)
    response = client.audio.speech.create(
        model=config.groq_model,
        voice=config.voice,
        input=text,
        response_format="wav",
        speed=config.speed,
    )

    audio_bytes = b"".join(response.iter_bytes())
    with io.BytesIO(audio_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            sample_rate = wav.getframerate()
            duration = wav.getnframes() / sample_rate
            channels = wav.getnchannels()

    return TTSResult(audio=audio_bytes, format="wav",
                     sample_rate=sample_rate, duration=duration, channels=channels)


def _stream_groq_tts(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    yield _synthesize_groq_tts(text, config).audio


# =============================================================================
# ELEVENLABS BACKEND (API)
# =============================================================================


def _synthesize_elevenlabs(text: str, config: TTSConfig) -> TTSResult:
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        raise ImportError("elevenlabs not installed.")

    api_key = config.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key required.")

    client = ElevenLabs(api_key=api_key)
    format_map = {
        TTSQuality.LOW: "mp3_22050_32",
        TTSQuality.MEDIUM: "mp3_44100_128",
        TTSQuality.HIGH: "pcm_44100",
    }
    output_format = format_map.get(config.quality, "mp3_44100_128")

    audio_generator = client.text_to_speech.convert(
        text=text, voice_id=config.voice, model_id=config.elevenlabs_model,
        output_format=output_format,
        voice_settings={
            "stability": config.elevenlabs_stability,
            "similarity_boost": config.elevenlabs_similarity_boost,
            "style": config.elevenlabs_style,
        },
    )
    audio_bytes = b"".join(audio_generator)

    if "pcm" in output_format:
        fmt = "pcm"
        sample_rate = int(output_format.split("_")[1])
    else:
        fmt = "mp3"
        sample_rate = int(output_format.split("_")[1])

    return TTSResult(audio=audio_bytes, format=fmt, sample_rate=sample_rate,
                     duration=_estimate_duration(text, config.speed), channels=1)


def _stream_elevenlabs(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        raise ImportError("elevenlabs not installed")

    api_key = config.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ElevenLabs API key required")

    client = ElevenLabs(api_key=api_key)
    audio_stream = client.text_to_speech.stream(
        text=text, voice_id=config.voice, model_id=config.elevenlabs_model,
    )
    for chunk in audio_stream:
        if isinstance(chunk, bytes):
            yield chunk


# =============================================================================
# INDEX TTS BACKEND (LOCAL GPU, ZERO-SHOT VOICE CLONING)
# =============================================================================
#
# IndexTTS is an industry-level zero-shot TTS system.
# It clones voice from a short reference audio (3-10 seconds) and supports
# emotion/style control via text prompts injected before the synthesis text.
#
# Setup:
#   pip install indextts
#   # or from source:
#   git clone https://github.com/index-tts/index-tts && cd index-tts
#   pip install -e .
#
# Download model weights:
#   # Option A: huggingface-cli
#   huggingface-cli download IndexTeam/IndexTTS --local-dir ./checkpoints
#   # Option B: python
#   from huggingface_hub import snapshot_download
#   snapshot_download("IndexTeam/IndexTTS", local_dir="./checkpoints")
#
# Reference audio:
#   - WAV format, 16kHz or 24kHz mono preferred
#   - 3-10 seconds of clean speech, no background noise
#   - The model clones this voice for all synthesis
#
# Emotion control:
#   The _apply_emotion_prefix() injects style hints before the text.
#   IndexTTS interprets these naturally from its training distribution.
#   Fine-grained control is possible with more detailed prompts.

# Module-level cache to avoid reloading the (large) model per call
_index_tts_model_cache: dict[str, object] = {}


def _get_index_tts_model(config: TTSConfig):
    """
    Load and cache IndexTTS model.

    IndexTTS requires significant VRAM (~6GB for the base model).
    The model is cached globally so repeated calls don't reload it.
    """
    try:
        from indextts.infer import IndexTTS
    except ImportError:
        raise ImportError(
            "indextts not installed.\n"
            "Install with: pip install indextts\n"
            "Or from source: git clone https://github.com/index-tts/index-tts && pip install -e index-tts"
        )

    cache_key = f"{config.index_tts_model_dir}:{config.index_tts_device}"
    if cache_key not in _index_tts_model_cache:
        model = IndexTTS(
            model_dir=config.index_tts_model_dir,
            cfg_scale=config.index_tts_cfg_scale,
        )
        _index_tts_model_cache[cache_key] = model

    return _index_tts_model_cache[cache_key]


def _synthesize_index_tts(text: str, config: TTSConfig) -> TTSResult:
    """
    Synthesize speech using IndexTTS (local zero-shot voice cloning).

    IndexTTS generates audio that matches the voice, prosody, and style
    of the reference_audio speaker. The emotion prefix modulates tone.

    Args:
        text: Text to synthesize
        config: TTSConfig with index_tts_* fields populated

    Requirements:
        - NVIDIA GPU with 6GB+ VRAM recommended (CPU works but is slow)
        - Model weights downloaded to index_tts_model_dir
        - Reference audio WAV file for voice cloning
    """
    if not config.index_tts_reference_audio:
        raise ValueError(
            "IndexTTS requires index_tts_reference_audio. "
            "Provide a path to a 3-10s WAV file of the target speaker."
        )

    model = _get_index_tts_model(config)

    # Inject emotion prefix into the text (IndexTTS interprets style hints)
    styled_text = _apply_emotion_prefix(text, config.emotion)

    # Synthesize to temp file (IndexTTS writes to file natively)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    try:
        model.infer(
            audio_prompt=config.index_tts_reference_audio,
            text=styled_text,
            output_path=output_path,
        )

        audio_bytes = Path(output_path).read_bytes()

    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

    # Parse WAV metadata
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
        emotion=config.emotion,
    )


def _stream_index_tts(text: str, config: TTSConfig) -> Generator[bytes, None, None]:
    """
    Stream IndexTTS synthesis.

    IndexTTS does not natively stream — we synthesize the full audio
    and yield it in a single chunk. For streaming UX, use sentence
    splitting at a higher level (see audioIo.py).
    """
    result = _synthesize_index_tts(text, config)
    yield result.audio


# =============================================================================
# PUBLIC API
# =============================================================================


def synthesize(text: str, config: Optional[TTSConfig] = None, **kwargs) -> TTSResult:
    """
    Synthesize speech from text.

    Main entry point for TTS. Routes to the configured backend.

    Args:
        text: Text to convert to speech
        config: TTSConfig object with all settings
        **kwargs: Override config settings (e.g. emotion=TTSEmotion.CALM)

    Returns:
        TTSResult with audio bytes and metadata

    Examples:
        # Local IndexTTS with custom voice
        result = synthesize(
            "Hello! I can speak with emotion.",
            config=TTSConfig(
                backend=TTSBackend.INDEX_TTS,
                index_tts_reference_audio="./my_voice.wav",
                index_tts_model_dir="./checkpoints",
                emotion=TTSEmotion.FRIENDLY,
            )
        )

        # Groq with excited tone (prefix injected)
        result = synthesize(
            "This is amazing!",
            config=TTSConfig(
                backend=TTSBackend.GROQ_TTS,
                voice="autumn",
                emotion=TTSEmotion.EXCITED,
            )
        )
    """
    if config is None:
        config = TTSConfig(**kwargs)
    elif kwargs:
        config_dict = {k: v for k, v in config.__dict__.items()}
        config_dict.update(kwargs)
        config = TTSConfig(**config_dict)

    backends = {
        TTSBackend.PIPER: _synthesize_piper,
        TTSBackend.VIBEVOICE: _synthesize_vibevoice,
        TTSBackend.GROQ_TTS: _synthesize_groq_tts,
        TTSBackend.ELEVENLABS: _synthesize_elevenlabs,
        TTSBackend.INDEX_TTS: _synthesize_index_tts,
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
    For IndexTTS, yields the full audio in one chunk.
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
        TTSBackend.INDEX_TTS: _stream_index_tts,
    }

    handler = stream_handlers.get(config.backend)
    if handler is None:
        raise ValueError(f"Unknown backend: {config.backend}")

    yield from handler(text, config)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def synthesize_piper(text: str, voice: str = "de_DE-thorsten-high", **kwargs) -> TTSResult:
    return synthesize(text, config=TTSConfig(backend=TTSBackend.PIPER, voice=voice, **kwargs))


def synthesize_groq(
    text: str, api_key: Optional[str] = None, voice: str = "Fritz-PlayAI", **kwargs
) -> TTSResult:
    return synthesize(
        text, config=TTSConfig(backend=TTSBackend.GROQ_TTS, groq_api_key=api_key, voice=voice, **kwargs)
    )


def synthesize_elevenlabs(
    text: str, api_key: Optional[str] = None, voice: str = "21m00Tcm4TlvDq8ikWAM", **kwargs
) -> TTSResult:
    return synthesize(
        text, config=TTSConfig(backend=TTSBackend.ELEVENLABS, elevenlabs_api_key=api_key, voice=voice, **kwargs)
    )


def synthesize_index_tts(
    text: str,
    reference_audio: str,
    reference_text: Optional[str] = None,
    model_dir: str = "./checkpoints",
    emotion: TTSEmotion = TTSEmotion.NEUTRAL,
    **kwargs,
) -> TTSResult:
    """
    Synthesize using IndexTTS (zero-shot voice cloning).

    Convenience function. Clones voice from reference_audio.

    Args:
        text: Text to speak
        reference_audio: Path to 3-10s WAV of target speaker
        reference_text: Optional transcript of reference audio
        model_dir: Path to IndexTTS model weights
        emotion: Tone/emotion preset

    Example:
        result = synthesize_index_tts(
            "Guten Morgen! Wie kann ich helfen?",
            reference_audio="./voice_samples/agent_voice.wav",
            emotion=TTSEmotion.FRIENDLY,
        )
        result.save("response.wav")
    """
    return synthesize(
        text,
        config=TTSConfig(
            backend=TTSBackend.INDEX_TTS,
            index_tts_reference_audio=reference_audio,
            index_tts_reference_text=reference_text,
            index_tts_model_dir=model_dir,
            emotion=emotion,
            **kwargs,
        ),
    )
