"""
Audio Test Utilities
====================

Provider availability detection with install instructions.
Import this in every test file.
"""

import importlib
import io
import os
import struct
import unittest
import wave


# =============================================================================
# PROVIDER DETECTION
# =============================================================================

def _check(package: str) -> bool:
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False


# Runtime availability flags
HAS_FASTER_WHISPER = _check("faster_whisper")
HAS_GROQ           = _check("groq") and bool(os.environ.get("GROQ_API_KEY"))
HAS_GROQ_PACKAGE   = _check("groq")
HAS_ELEVENLABS     = _check("elevenlabs") and bool(os.environ.get("ELEVENLABS_API_KEY"))
HAS_PIPER          = _check("piper")
HAS_TORCH          = _check("torch")
HAS_VIBEVOICE      = _check("vibevoice") and HAS_TORCH
HAS_SOUNDDEVICE    = _check("sounddevice")
HAS_NUMPY          = _check("numpy")
HAS_LAVASR         = _check("models.lavasr") or os.path.exists("./LavaSR")
HAS_INDEXTTS       = _check("indextts")
HAS_RESAMPY        = _check("resampy")

# Index-TTS also needs weights on disk
_INDEX_TTS_MODEL_DIR = os.environ.get("INDEX_TTS_MODEL_DIR", "./checkpoints")
HAS_INDEX_TTS_WEIGHTS = (
    HAS_INDEXTTS
    and os.path.isdir(_INDEX_TTS_MODEL_DIR)
    and len(os.listdir(_INDEX_TTS_MODEL_DIR)) > 0
)


# =============================================================================
# SKIP DECORATORS WITH INSTALL HINTS
# =============================================================================

def require_faster_whisper(fn):
    msg = (
        "faster-whisper not installed.\n"
        "  pip install faster-whisper\n"
        "  (downloads model ~500MB on first run)"
    )
    return unittest.skipUnless(HAS_FASTER_WHISPER, msg)(fn)


def require_groq(fn):
    if not HAS_GROQ_PACKAGE:
        msg = "groq package not installed. pip install groq"
        return unittest.skip(msg)(fn)
    if not os.environ.get("GROQ_API_KEY"):
        msg = "GROQ_API_KEY not set. export GROQ_API_KEY=gsk_..."
        return unittest.skip(msg)(fn)
    return fn


def require_elevenlabs(fn):
    if not _check("elevenlabs"):
        msg = "elevenlabs not installed. pip install elevenlabs"
        return unittest.skip(msg)(fn)
    if not os.environ.get("ELEVENLABS_API_KEY"):
        msg = "ELEVENLABS_API_KEY not set. export ELEVENLABS_API_KEY=..."
        return unittest.skip(msg)(fn)
    return fn


def require_piper(fn):
    msg = (
        "piper-tts not installed.\n"
        "  pip install piper-tts\n"
        "  Models download automatically from Hugging Face."
    )
    return unittest.skipUnless(HAS_PIPER, msg)(fn)


def require_sounddevice(fn):
    msg = (
        "sounddevice not installed (required for LocalPlayer hardware output).\n"
        "  pip install sounddevice\n"
        "  Also needs: apt-get install libportaudio2  (Linux)"
    )
    return unittest.skipUnless(HAS_SOUNDDEVICE and HAS_NUMPY, msg)(fn)


def require_index_tts(fn):
    if not HAS_INDEXTTS:
        msg = (
            "indextts not installed.\n"
            "  pip install indextts\n"
            "  OR: git clone https://github.com/index-tts/index-tts && pip install -e index-tts"
        )
        return unittest.skip(msg)(fn)
    if not HAS_INDEX_TTS_WEIGHTS:
        msg = (
            f"IndexTTS model weights not found at '{_INDEX_TTS_MODEL_DIR}'.\n"
            "  huggingface-cli download IndexTeam/IndexTTS --local-dir ./checkpoints\n"
            "  OR set INDEX_TTS_MODEL_DIR env var to your weights directory."
        )
        return unittest.skip(msg)(fn)
    return fn


def require_lavasr(fn):
    msg = (
        "LavaSR not found.\n"
        "  git clone https://github.com/ysharma3501/LavaSR\n"
        "  pip install -r LavaSR/requirements.txt\n"
        "  Download weights per README, set LAVASR_MODEL_PATH and LAVASR_CONFIG_PATH."
    )
    return unittest.skipUnless(HAS_LAVASR, msg)(fn)


# =============================================================================
# AUDIO TEST HELPERS
# =============================================================================

def make_silent_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV buffer for testing."""
    n_samples = int(sample_rate * duration_s)
    pcm_data = b"\x00\x00" * n_samples
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buf.getvalue()


def make_sine_wav(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.3,
) -> bytes:
    """
    Generate a sine wave WAV buffer.
    Better than silence for testing STT (gives whisper something to process).
    """
    import math
    n_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(n_samples):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        samples.append(struct.pack("<h", max(-32768, min(32767, value))))
    pcm_data = b"".join(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buf.getvalue()


def assert_valid_wav(test_case: unittest.TestCase, wav_bytes: bytes, context: str = ""):
    """Assert that wav_bytes is a structurally valid WAV file."""
    prefix = f"[{context}] " if context else ""
    test_case.assertGreater(len(wav_bytes), 44, f"{prefix}WAV too short to have header")
    test_case.assertEqual(wav_bytes[:4], b"RIFF", f"{prefix}Missing RIFF header")
    test_case.assertEqual(wav_bytes[8:12], b"WAVE", f"{prefix}Missing WAVE marker")

    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            sr = wav_file.getframerate()
            channels = wav_file.getnchannels()

    test_case.assertGreater(n_frames, 0, f"{prefix}WAV has zero frames")
    test_case.assertGreater(sr, 0, f"{prefix}WAV sample rate is 0")
    test_case.assertIn(channels, [1, 2], f"{prefix}Unexpected channel count: {channels}")


def wav_info(wav_bytes: bytes) -> dict:
    """Return dict with sample_rate, channels, n_frames, duration_s."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            sr = wav.getframerate()
            n_frames = wav.getnframes()
            channels = wav.getnchannels()
    return {
        "sample_rate": sr,
        "channels": channels,
        "n_frames": n_frames,
        "duration_s": n_frames / sr if sr > 0 else 0,
    }


def print_provider_status():
    """Print availability status for all audio providers. Call at test start."""
    print("\n" + "=" * 60)
    print("AUDIO PROVIDER STATUS")
    print("=" * 60)
    providers = [
        ("faster-whisper (STT local)",    HAS_FASTER_WHISPER, "pip install faster-whisper"),
        ("groq (STT+TTS API)",            HAS_GROQ,           "pip install groq + GROQ_API_KEY"),
        ("elevenlabs (TTS API)",          HAS_ELEVENLABS,     "pip install elevenlabs + ELEVENLABS_API_KEY"),
        ("piper-tts (TTS local CPU)",     HAS_PIPER,          "pip install piper-tts"),
        ("vibevoice (TTS local GPU)",     HAS_VIBEVOICE,      "pip install vibevoice (needs CUDA)"),
        ("indextts (TTS zero-shot)",      HAS_INDEXTTS,       "pip install indextts"),
        ("indextts weights",              HAS_INDEX_TTS_WEIGHTS, f"huggingface-cli download IndexTeam/IndexTTS --local-dir {_INDEX_TTS_MODEL_DIR}"),
        ("sounddevice (LocalPlayer)",     HAS_SOUNDDEVICE,    "pip install sounddevice"),
        ("numpy",                         HAS_NUMPY,          "pip install numpy"),
        ("LavaSR (enhancer)",             HAS_LAVASR,         "git clone https://github.com/ysharma3501/LavaSR"),
    ]
    for name, available, install_hint in providers:
        status = "✓" if available else "✗"
        print(f"  {status} {name}")
        if not available:
            print(f"      → {install_hint}")
    print("=" * 60 + "\n")
