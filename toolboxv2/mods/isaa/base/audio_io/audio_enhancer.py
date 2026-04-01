"""
OmniCore Audio Enhancer
========================

Post-processing layer for synthesized speech audio.

Supported enhancers:
- LavaSR: Speech super-resolution (16kHz → 48kHz, noise reduction, clarity)
  Repo: https://github.com/ysharma3501/LavaSR

Usage:
    from audio_enhancer import AudioEnhancer, EnhancerConfig

    enhancer = AudioEnhancer()
    enhanced_bytes = enhancer.enhance(raw_wav_bytes)

Version: 1.0.0
"""

import io
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np


class EnhancerBackend(Enum):
    LAVASR = "lavasr"   # Speech super-resolution (ysharma3501/LavaSR)
    NONE = "none"       # Passthrough, no enhancement


@dataclass
class EnhancerConfig:
    """
    Configuration for audio enhancement.

    Attributes:
        backend: Which enhancer to use
        target_sr: Target sample rate after enhancement (LavaSR: 48000)
        device: "cuda" or "cpu"
        lavasr_model_path: Path to LavaSR checkpoint (.pth).
            Download: https://github.com/ysharma3501/LavaSR#installation
        lavasr_config_path: Path to LavaSR config YAML.
        enabled: Quick toggle without changing backend
    """

    backend: EnhancerBackend = EnhancerBackend.LAVASR
    target_sr: int = 48000
    device: str = "cuda"
    lavasr_model_path: str = "./lavasr_checkpoints/lavasr.pth"
    lavasr_config_path: str = "./lavasr_checkpoints/config.yaml"
    enabled: bool = True


# =============================================================================
# LAVASR BACKEND
# =============================================================================
#
# LavaSR is a GAN-based speech super-resolution model.
# It upsamples degraded/low-sample-rate speech to 48kHz with:
#   - Harmonic reconstruction
#   - Noise suppression
#   - Perceptual quality enhancement
#
# Installation:
#   git clone https://github.com/ysharma3501/LavaSR
#   cd LavaSR && pip install -r requirements.txt
#
#   # Download checkpoint:
#   # See README for link to pretrained weights (lavasr.pth + config.yaml)
#
# Input:  WAV bytes (any sample rate, mono preferred)
# Output: WAV bytes at target_sr (48000 recommended)
#
# Typical latency: 20-80ms on GPU for <5s audio


# Module-level model cache
_lavasr_cache: dict[str, object] = {}


def _load_lavasr(config: EnhancerConfig):
    """Load LavaSR model, cached globally."""
    cache_key = f"{config.lavasr_model_path}:{config.device}"
    if cache_key in _lavasr_cache:
        return _lavasr_cache[cache_key]

    try:
        import sys
        import torch
        # LavaSR is not pip-installable; add its directory to sys.path
        # Expected structure: ./LavaSR/models/lavasr.py
        lavasr_dir = str(Path(config.lavasr_model_path).parent.parent)
        if lavasr_dir not in sys.path:
            sys.path.insert(0, lavasr_dir)
        from models.lavasr import LavaSR  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"LavaSR not found: {e}\n"
            "Clone the repo: git clone https://github.com/ysharma3501/LavaSR\n"
            "Then set lavasr_model_path to point inside the cloned directory."
        )

    import yaml
    import torch

    with open(config.lavasr_config_path) as f:
        cfg = yaml.safe_load(f)

    model = LavaSR(**cfg.get("model", {}))
    checkpoint = torch.load(config.lavasr_model_path, map_location=config.device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    if config.device == "cuda":
        import torch
        if torch.cuda.is_available():
            model = model.cuda()

    _lavasr_cache[cache_key] = model
    return model


def _wav_bytes_to_numpy(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Extract PCM float32 array and sample rate from WAV bytes."""
    with io.BytesIO(wav_bytes) as buf:
        with wave.open(buf, "rb") as wav:
            sr = wav.getframerate()
            n_channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        # Downmix to mono
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sr


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 PCM array to WAV bytes."""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def _enhance_lavasr(wav_bytes: bytes, config: EnhancerConfig) -> bytes:
    """
    Run LavaSR super-resolution on WAV bytes.

    Input audio is resampled to the model's expected input rate (16kHz),
    then upsampled to target_sr (48kHz) by LavaSR.
    """
    import torch

    model = _load_lavasr(config)
    audio, input_sr = _wav_bytes_to_numpy(wav_bytes)

    # Resample to 16kHz if necessary (LavaSR input requirement)
    if input_sr != 16000:
        try:
            import resampy
            audio = resampy.resample(audio, input_sr, 16000)
        except ImportError:
            # Fallback: simple linear interpolation (lower quality)
            target_len = int(len(audio) * 16000 / input_sr)
            audio = np.interp(
                np.linspace(0, len(audio), target_len),
                np.arange(len(audio)),
                audio,
            )

    # Prepare tensor: shape [1, 1, T] (batch, channels, time)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
    if config.device == "cuda" and torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()

    with torch.no_grad():
        enhanced_tensor = model(audio_tensor)

    enhanced = enhanced_tensor.squeeze().cpu().numpy()
    return _numpy_to_wav_bytes(enhanced, config.target_sr)


# =============================================================================
# PUBLIC API
# =============================================================================


class AudioEnhancer:
    """
    Post-processing wrapper for synthesized speech audio.

    Stateless after initialization. Thread-safe (model is read-only during inference).

    Example:
        from audio_enhancer import AudioEnhancer, EnhancerConfig

        # Default: LavaSR on GPU
        enhancer = AudioEnhancer()
        enhanced = enhancer.enhance(tts_result.audio)

        # CPU fallback
        enhancer = AudioEnhancer(EnhancerConfig(device="cpu"))
        enhanced = enhancer.enhance(tts_result.audio)

        # Disabled (passthrough for testing)
        enhancer = AudioEnhancer(EnhancerConfig(enabled=False))
    """

    def __init__(self, config: Optional[EnhancerConfig] = None):
        self.config = config or EnhancerConfig()

    def enhance(self, wav_bytes: bytes) -> bytes:
        """
        Enhance WAV audio bytes.

        Returns enhanced WAV bytes, or original if disabled/passthrough.

        Args:
            wav_bytes: Input WAV audio (any sample rate)

        Returns:
            Enhanced WAV bytes at target_sr
        """
        if not self.config.enabled or self.config.backend == EnhancerBackend.NONE:
            return wav_bytes

        try:
            if self.config.backend == EnhancerBackend.LAVASR:
                return _enhance_lavasr(wav_bytes, self.config)
        except Exception as e:
            # Enhancement is best-effort — never block the audio pipeline
            print(f"[AudioEnhancer] Enhancement failed, using original: {e}")
            return wav_bytes

        return wav_bytes

    def enhance_result(self, tts_result) -> "bytes":
        """
        Convenience: enhance a TTSResult's audio in-place.

        Returns the enhanced WAV bytes.
        """
        return self.enhance(tts_result.audio)

    @property
    def is_available(self) -> bool:
        """Check if the enhancer backend can be loaded."""
        if self.config.backend == EnhancerBackend.NONE:
            return True
        if self.config.backend == EnhancerBackend.LAVASR:
            return (
                Path(self.config.lavasr_model_path).exists()
                and Path(self.config.lavasr_config_path).exists()
            )
        return False


# =============================================================================
# SINGLETON CONVENIENCE
# =============================================================================

_default_enhancer: Optional[AudioEnhancer] = None


def get_enhancer(config: Optional[EnhancerConfig] = None) -> AudioEnhancer:
    """
    Get or create the global AudioEnhancer instance.

    Call once at startup with config, then use get_enhancer() everywhere.
    """
    global _default_enhancer
    if _default_enhancer is None or config is not None:
        _default_enhancer = AudioEnhancer(config)
    return _default_enhancer
