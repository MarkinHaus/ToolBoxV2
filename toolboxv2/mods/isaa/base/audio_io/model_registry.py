"""
model_registry.py
=================

Process-wide singleton cache for heavy audio models.

Every TTS backend that loads a model from disk / HF hub goes through
this module. Reasons:
  - avoid reloading multi-GB checkpoints on every synthesize() call
  - one place for download-and-ensure logic
  - one place for device resolution + GPU-capability fallbacks
  - future STT entries land here too (same contract)

Public API per backend:
    get_piper(model_path_or_name, models_folder=None) -> PiperVoice
    get_vibevoice(repo_id) -> VibeVoice
    get_qwen3(model_id, device="cuda") -> Qwen3TTSModel

Download helpers (idempotent, safe to call repeatedly):
    download_piper_voice(voice_name, models_folder) -> absolute .onnx path
    ensure_hf_snapshot(repo_id) -> local path

Management:
    clear()                      -> drop caches, free GPU memory
    clear_backend(name)          -> drop one backend's cache
    cached_ids() -> dict         -> introspection

Thread-safe via a single module-level lock.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_caches: dict[str, dict[str, Any]] = {
    "piper": {},
    "vibevoice": {},
    "qwen3": {},
    "stt": {},
}


# ---------------------------------------------------------------------------
# Device / capability helpers
# ---------------------------------------------------------------------------

def _resolve_device(requested: str) -> tuple[str, str]:
    """
    Return (effective_device, dtype_str).

    Falls back to CPU with a warning when CUDA was asked for but unavailable.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "torch is required. Install with: pip install torch"
        ) from e

    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "model_registry: device=%r requested but CUDA unavailable. "
            "Falling back to CPU — inference will be slow.",
            requested,
        )
        return "cpu", "float32"
    if requested == "cpu":
        return "cpu", "float32"
    return requested, "bfloat16"


def _pick_attn_impl(device: str) -> str:
    """
    Pick the best attention implementation available on this machine.

    Order of preference:
      flash_attention_2 — fastest, needs Ampere+ GPU AND importable flash_attn.
                          Linux-only in practice — upstream does not publish
                          Windows wheels, source builds are brittle.
      sdpa              — PyTorch native scaled-dot-product attention. Works
                          on every platform (Linux / Windows / macOS / CPU),
                          ~10–20% slower than FA2 but always importable.
      eager             — pure-Python fallback. Only picked when SDPA is
                          somehow missing (torch < 2.0).

    The import-probe matters: a user can be on Linux+Ampere but not have
    flash_attn installed (e.g. just ran `pip install qwen-tts` without the
    flash-attn extra). Probing here prevents a crash deep inside
    Qwen3TTSModel.from_pretrained.
    """
    import platform

    if not device.startswith("cuda"):
        return "sdpa"

    # OS gate — flash_attn has no official Windows wheels.
    if platform.system() == "Windows":
        logger.info("flash_attn: skipped (Windows — using SDPA).")
        return "sdpa"

    # GPU capability gate — flash_attn needs SM80+ (Ampere+).
    try:
        import torch
        major, _ = torch.cuda.get_device_capability()
    except Exception:
        return "sdpa"
    if major < 8:
        return "sdpa"

    # Import probe — importable means it was built & installed correctly.
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.info(
            "flash_attn not importable — falling back to SDPA. "
            "Install with: pip install flash-attn --no-build-isolation",
        )
        return "sdpa"
    except Exception as e:
        logger.warning("flash_attn import raised %s — falling back to SDPA.", e)
        return "sdpa"

    return "flash_attention_2"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_piper_voice(
    voice_name: str,
    models_folder: Optional[str] = None,
) -> str:
    """
    Ensure a Piper voice (.onnx + .onnx.json) exists locally. Return
    absolute path to the .onnx.

    Strategy: piper-tts's built-in download via piper.download would be
    used if available; otherwise we rely on the user having placed the
    files in models_folder manually. We don't re-download if both files
    are already present.
    """
    folder = models_folder or os.getenv("PIPER_MODELS_FOLDER", ".")
    folder_path = Path(folder).expanduser().resolve()
    folder_path.mkdir(parents=True, exist_ok=True)

    name = voice_name if voice_name.endswith(".onnx") else voice_name + ".onnx"
    onnx_path = folder_path / name
    json_path = folder_path / (name + ".json")

    if onnx_path.exists() and json_path.exists():
        return str(onnx_path)

    # Attempt download via piper's own downloader if available
    try:
        from piper.download_voices import download_voice  # type: ignore
        stem = name[:-len(".onnx")]
        logger.info("Downloading Piper voice %s into %s", stem, folder_path)
        download_voice(stem, str(folder_path))
    except ImportError:
        logger.warning(
            "piper.download_voices not available. "
            "Place %s and %s manually into %s.",
            name, name + ".json", folder_path,
        )
    except Exception as e:
        logger.warning("Piper auto-download failed: %s", e)

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"Piper voice not found at {onnx_path}. "
            f"Place the .onnx and .onnx.json manually or enable auto-download."
        )
    return str(onnx_path)


def ensure_hf_snapshot(repo_id: str) -> str:
    """
    Make sure a HF repo is present in the local cache, return the path.

    Uses huggingface_hub.snapshot_download — respects HF_HOME env var.
    Safe to call repeatedly; becomes a no-op after first pull.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e
    return snapshot_download(repo_id)


# ---------------------------------------------------------------------------
# Piper
# ---------------------------------------------------------------------------

def get_piper(
    model_path_or_name: str,
    models_folder: Optional[str] = None,
) -> Any:
    """
    Load-or-return a cached PiperVoice.

    Accepts an absolute .onnx path or a voice-name; in the latter case
    we resolve via download_piper_voice().
    """
    try:
        from piper.voice import PiperVoice
    except ImportError as e:
        raise ImportError(
            "piper-tts not installed. pip install piper-tts"
        ) from e

    if os.path.isabs(model_path_or_name) and model_path_or_name.endswith(".onnx"):
        abs_path = model_path_or_name
    else:
        abs_path = download_piper_voice(model_path_or_name, models_folder)

    with _lock:
        cache = _caches["piper"]
        if abs_path in cache:
            return cache[abs_path]
        logger.info("Loading Piper voice %s", abs_path)
        voice = PiperVoice.load(abs_path)
        cache[abs_path] = voice
        return voice


# ---------------------------------------------------------------------------
# VibeVoice
# ---------------------------------------------------------------------------

def get_vibevoice(repo_id: str) -> Any:
    """Load-or-return a cached VibeVoice model."""
    try:
        import torch
        from vibevoice import VibeVoice
    except ImportError as e:
        raise ImportError("vibevoice not installed.") from e

    if not torch.cuda.is_available():
        raise RuntimeError("VibeVoice requires NVIDIA GPU with CUDA.")

    with _lock:
        cache = _caches["vibevoice"]
        if repo_id in cache:
            return cache[repo_id]
        logger.info("Loading VibeVoice %s", repo_id)
        model = VibeVoice.from_pretrained(repo_id)
        cache[repo_id] = model
        return model


# ---------------------------------------------------------------------------
# Qwen3-TTS
# ---------------------------------------------------------------------------

def get_qwen3(model_id: str, device: str = "cuda") -> Any:
    """
    Load-or-return a cached Qwen3TTSModel.

    Auto-falls back to CPU with a warning if CUDA was requested but
    unavailable. BF16 on GPU, FP32 on CPU. FlashAttention-2 on Ampere+.
    """
    effective_device, dtype_str = _resolve_device(device)
    cache_key = f"{model_id}:{effective_device}:{dtype_str}"

    with _lock:
        cache = _caches["qwen3"]
        if cache_key in cache:
            return cache[cache_key]

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "qwen-tts not installed. pip install -U qwen-tts"
            ) from e

        dtype = getattr(torch, dtype_str)
        attn_impl = _pick_attn_impl(effective_device)

        logger.info(
            "Loading Qwen3-TTS model=%s device=%s dtype=%s attn=%s",
            model_id, effective_device, dtype_str, attn_impl,
        )
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=effective_device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        cache[cache_key] = model
        return model


# ---------------------------------------------------------------------------
# STT — reserved
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Faster-Whisper (STT)
# ---------------------------------------------------------------------------

def get_faster_whisper(
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
) -> Any:
    """
    Load-or-return a cached faster_whisper.WhisperModel.

    Args:
        model: size ("tiny"/"base"/"small"/"medium"/"large-v3"/"distil-large-v3")
               or a local path.
        device: "cpu" / "cuda" / "auto". Auto falls back to CPU with warning
                if CUDA requested but unavailable.
        compute_type: "int8" (CPU default) / "float16" (GPU) / "float32".
    """
    effective_device = device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "faster-whisper: CUDA requested but unavailable. "
                    "Falling back to CPU.",
                )
                effective_device = "cpu"
                if compute_type == "float16":
                    compute_type = "int8"
        except ImportError:
            effective_device = "cpu"

    cache_key = f"{model}:{effective_device}:{compute_type}"
    with _lock:
        cache = _caches["stt"]
        if cache_key in cache:
            return cache[cache_key]

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper not installed. pip install faster-whisper"
            ) from e

        logger.info(
            "Loading faster-whisper model=%s device=%s compute=%s",
            model, effective_device, compute_type,
        )
        m = WhisperModel(model, device=effective_device, compute_type=compute_type)
        cache[cache_key] = m
        return m


# ---------------------------------------------------------------------------
# Parakeet-TDT (STT, CPU-optimized)
# ---------------------------------------------------------------------------

def get_parakeet(
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
    device: str = "cpu",
    config_preset: str = "balanced",
) -> Any:
    """
    Load-or-return a cached parakeet_stream.Parakeet.

    Parakeet-TDT via ONNX-Runtime: CPU-first streaming STT,
    typically ~30x realtime on modern CPUs with INT8.

    Args:
        model_name: HF model id (default: parakeet-tdt-0.6b-v3, 25 langs).
        device: "cpu" / "cuda" / "mps". Auto-fallback CUDA→CPU with warning.
        config_preset: parakeet-stream preset — "fast"/"balanced"/"accurate".
    """
    effective_device = device
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "parakeet: CUDA requested but unavailable. Falling back to CPU.",
                )
                effective_device = "cpu"
        except ImportError:
            effective_device = "cpu"

    cache_key = f"{model_name}:{effective_device}:{config_preset}"
    with _lock:
        cache = _caches["stt"]
        if cache_key in cache:
            return cache[cache_key]

        try:
            from parakeet_stream import Parakeet
        except ImportError as e:
            raise ImportError(
                "parakeet-stream not installed. pip install parakeet-stream"
            ) from e

        logger.info(
            "Loading Parakeet model=%s device=%s preset=%s",
            model_name, effective_device, config_preset,
        )
        pk = Parakeet(
            model_name=model_name,
            device=effective_device,
            config=config_preset,
        )
        cache[cache_key] = pk
        return pk


def get_stt_model(model_id: str, device: str = "cuda") -> Any:
    """
    Legacy dispatcher. Prefer get_faster_whisper() / get_parakeet() directly.
    Routes by model_id prefix.
    """
    if "parakeet" in model_id.lower():
        return get_parakeet(model_id, device=device)
    return get_faster_whisper(model_id, device=device)


# ---------------------------------------------------------------------------
# Management
# ---------------------------------------------------------------------------

def clear_backend(name: str) -> None:
    """Drop cache for a single backend: 'piper' | 'vibevoice' | 'qwen3' | 'stt'."""
    with _lock:
        if name not in _caches:
            raise KeyError(f"Unknown backend: {name!r}")
        _caches[name].clear()
    _free_gpu_if_possible()


def clear() -> None:
    """Drop every cached model. Next get_*() call reloads from disk."""
    with _lock:
        for c in _caches.values():
            c.clear()
    _free_gpu_if_possible()


def _free_gpu_if_possible() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def cached_ids() -> dict[str, list[str]]:
    """Introspection: what's loaded per backend."""
    with _lock:
        return {name: list(cache.keys()) for name, cache in _caches.items()}
