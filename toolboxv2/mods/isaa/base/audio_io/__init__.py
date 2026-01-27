"""
OmniCore Audio Module
=====================

Unified Speech-to-Text (STT) and Text-to-Speech (TTS) interface
with multiple backend support for local and cloud processing.

Quick Start:
------------
    from omnicore_audio import transcribe, synthesize

    # Transcribe audio
    result = transcribe("recording.wav")
    print(result.text)

    # Synthesize speech
    audio = synthesize("Hello, world!")
    audio.save("output.wav")

Backends:
---------
STT:
    - faster_whisper: Local CPU/GPU (default)
    - groq_whisper: Groq Cloud API

TTS:
    - piper: Local CPU (default)
    - vibevoice: Local GPU (requires NVIDIA)
    - groq_tts: Groq Cloud API
    - elevenlabs: ElevenLabs API

Configuration:
--------------
    # STT with specific backend
    from omnicore_audio import transcribe, STTConfig, STTBackend

    result = transcribe(
        "audio.wav",
        config=STTConfig(
            backend=STTBackend.GROQ_WHISPER,
            language="de"
        )
    )

    # TTS with specific backend
    from omnicore_audio import synthesize, TTSConfig, TTSBackend

    audio = synthesize(
        "Text to speak",
        config=TTSConfig(
            backend=TTSBackend.ELEVENLABS,
            voice="Rachel"
        )
    )

Author: OmniCore Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "OmniCore Team"

# =============================================================================
# STT EXPORTS
# =============================================================================

# =============================================================================
# ALL EXPORTS
# =============================================================================
# =============================================================================
# AUDIO I/O EXPORTS
# =============================================================================
from .audioIo import (
    # Configuration
    AudioIOConfig,
    AudioIOResult,
    AudioQuality,
    ProcessingMode,
    process_audio_native,
    # Convenience functions
    process_audio_pipeline,
    # Main functions
    process_audio_raw,
    process_audio_stream,
)

# =============================================================================
# NATIVE AUDIO MODEL EXPORTS
# =============================================================================
from .native.LiquidAI import (
    GenerationMode,
    # Configuration
    NativeAudioConfig,
    # Model classes (for type hints)
    NativeAudioModel,
    NativeAudioOutput,
    NativeModelBackend,
    # Model loader
    load_native_audio_model,
)
from .Stt import (
    STTBackend,
    # Configuration
    STTConfig,
    STTResult,
    STTSegment,
    # Main functions
    transcribe,
    transcribe_groq,
    # Convenience functions
    transcribe_local,
    transcribe_stream,
)

# =============================================================================
# TTS EXPORTS
# =============================================================================
from .Tts import (
    TTSBackend,
    # Configuration
    TTSConfig,
    TTSQuality,
    TTSResult,
    # Main functions
    synthesize,
    synthesize_elevenlabs,
    synthesize_groq,
    # Convenience functions
    synthesize_piper,
    synthesize_stream,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # STT
    "transcribe",
    "transcribe_stream",
    "transcribe_local",
    "transcribe_groq",
    "STTConfig",
    "STTBackend",
    "STTResult",
    "STTSegment",
    # TTS
    "synthesize",
    "synthesize_stream",
    "synthesize_piper",
    "synthesize_groq",
    "synthesize_elevenlabs",
    "TTSConfig",
    "TTSBackend",
    "TTSQuality",
    "TTSResult",
    # Audio I/O
    "process_audio_raw",
    "process_audio_stream",
    "process_audio_pipeline",
    "process_audio_native",
    "AudioIOConfig",
    "AudioIOResult",
    "ProcessingMode",
    "AudioQuality",
    # Native Audio Models
    "load_native_audio_model",
    "NativeAudioConfig",
    "NativeAudioOutput",
    "GenerationMode",
    "NativeModelBackend",
    "NativeAudioModel",
]


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================


def list_backends() -> dict:
    """List all available backends with their requirements."""
    return {
        "stt": {
            "faster_whisper": {
                "type": "local",
                "requirements": ["faster-whisper"],
                "device": "cpu/gpu",
                "notes": "Best for privacy, runs offline",
            },
            "groq_whisper": {
                "type": "api",
                "requirements": ["groq", "GROQ_API_KEY"],
                "notes": "Fastest API, 216x realtime speed",
            },
        },
        "tts": {
            "piper": {
                "type": "local",
                "requirements": ["piper-tts"],
                "device": "cpu",
                "notes": "Fast local TTS, good quality",
            },
            "vibevoice": {
                "type": "local",
                "requirements": ["vibevoice", "NVIDIA GPU 8GB+"],
                "device": "gpu",
                "notes": "High quality, multi-speaker, voice cloning",
            },
            "groq_tts": {
                "type": "api",
                "requirements": ["groq", "GROQ_API_KEY"],
                "notes": "Fast API TTS with Orpheus model",
            },
            "elevenlabs": {
                "type": "api",
                "requirements": ["elevenlabs", "ELEVENLABS_API_KEY"],
                "notes": "Highest quality, voice cloning",
            },
        },
    }


def check_requirements(backend: str) -> dict:
    """Check if requirements for a backend are satisfied."""
    checks = {"available": True, "missing": [], "warnings": []}

    if backend in ["faster_whisper"]:
        try:
            import faster_whisper
        except ImportError:
            checks["available"] = False
            checks["missing"].append("faster-whisper")

    elif backend in ["groq_whisper", "groq_tts"]:
        try:
            import groq
        except ImportError:
            checks["available"] = False
            checks["missing"].append("groq")

        import os

        if not os.environ.get("GROQ_API_KEY"):
            checks["warnings"].append("GROQ_API_KEY not set in environment")

    elif backend == "piper":
        try:
            from piper.voice import PiperVoice
        except ImportError:
            checks["available"] = False
            checks["missing"].append("piper-tts")

    elif backend == "vibevoice":
        try:
            import vibevoice
        except ImportError:
            checks["available"] = False
            checks["missing"].append("vibevoice")

        try:
            import torch

            if not torch.cuda.is_available():
                checks["warnings"].append("CUDA not available - VibeVoice requires GPU")
        except ImportError:
            checks["available"] = False
            checks["missing"].append("torch")

    elif backend == "elevenlabs":
        try:
            import elevenlabs
        except ImportError:
            checks["available"] = False
            checks["missing"].append("elevenlabs")

        import os

        if not os.environ.get("ELEVENLABS_API_KEY"):
            checks["warnings"].append("ELEVENLABS_API_KEY not set in environment")

    return checks
