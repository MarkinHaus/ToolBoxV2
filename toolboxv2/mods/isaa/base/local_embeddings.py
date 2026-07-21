"""
Local Embeddings — HF/ONNX fallback backend for AISemanticMemory.

No Ollama, no API cost: uses fastembed (ONNX Runtime, no torch, ~50 MB,
Windows + Linux + macOS, CPU-fast). Models auto-download from HuggingFace
into a local cache on first use.

Selection order (see resolve_mode()):
  TB_EMBED_LOCAL=1     → always local
  TB_EMBED_LOCAL=0     → never local (cloud only, hard-fail on error)
  TB_EMBED_LOCAL=auto  → local when no cloud model is configured, and as
                         automatic fallback when the cloud call fails
                         (default)

Model selection:
  TB_EMBED_LOCAL_MODEL       → any model from the validated registry below,
                               any other fastembed-supported model, or an
                               entry from app.manifest.isaa.embedding.models
  app.manifest.isaa.embedding → ISAA section of the TB app manifest: holds
      mode / local_model / cache_dir / auto_install and a `models` list with
      extra local embedding models. Env vars override manifest values.

Auto-setup:
  fastembed is pip-installed on first use when missing
  (disable with TB_EMBED_AUTOINSTALL=0).

Author: Markin / ToolBoxV2
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from toolboxv2 import get_logger

logger = get_logger()

# =============================================================================
# VALIDATED MODEL REGISTRY (all fastembed-supported, verified 2026)
# =============================================================================
# name → {dim, tags} — tags are informational for pick-guidance.

# NOTE licenses: jina-embeddings-v5-* are CC BY-NC 4.0 — NON-COMMERCIAL only.
# Everything else below is Apache-2.0/MIT-class. Check before commercial use.
VALIDATED_MODELS: dict[str, dict] = {
    # ── 2026 additions (last-token pooling → served by the built-in
    #    LastTokenOnnxEmbedder, not fastembed) ──
    # 239M, 768 dim (= store default), MTEB v2 71.0 — best sub-500M model.
    # ⚠ CC BY-NC 4.0 (non-commercial).
    "jinaai/jina-embeddings-v5-text-nano-retrieval": {
        "dim": 768, "tags": ["2026", "multilingual", "de", "quality", "nc-license"],
        "pooling": "last_token", "hf": "jinaai/jina-embeddings-v5-text-nano-retrieval",
        "model_file": "onnx/model.onnx",
    },
    # 677M, 1024 dim (MRL-truncate to 768 is safe), MTEB v2 71.7 — best <1B.
    # ⚠ CC BY-NC 4.0 (non-commercial).
    "jinaai/jina-embeddings-v5-text-small-retrieval": {
        "dim": 1024, "tags": ["2026", "multilingual", "quality", "nc-license"],
        "pooling": "last_token", "hf": "jinaai/jina-embeddings-v5-text-small-retrieval",
        "model_file": "onnx/model.onnx",
    },
    # default: quantized nomic — 768 dim (matches store default), MRL-trained
    # (truncation-safe), fast on CPU
    "nomic-ai/nomic-embed-text-v1.5-Q": {"dim": 768, "tags": ["default", "quantized", "mrl", "en"]},
    "nomic-ai/nomic-embed-text-v1.5":   {"dim": 768, "tags": ["mrl", "en"]},
    # multilingual SOTA (89 languages), MRL — best quality choice
    "jinaai/jina-embeddings-v3":        {"dim": 1024, "tags": ["multilingual", "mrl", "quality"]},
    # German/English bilingual — ideal for Denglish content
    "jinaai/jina-embeddings-v2-base-de": {"dim": 768, "tags": ["de", "en"]},
    # code-specialised
    "jinaai/jina-embeddings-v2-base-code": {"dim": 768, "tags": ["code"]},
    # minimal footprint (~34 MB) — weakest quality, fastest cold start
    "BAAI/bge-small-en-v1.5":           {"dim": 384, "tags": ["tiny", "en"]},
    "intfloat/multilingual-e5-large":   {"dim": 1024, "tags": ["multilingual"]},
    "mixedbread-ai/mxbai-embed-large-v1": {"dim": 1024, "tags": ["quality", "en"]},
    "snowflake/snowflake-arctic-embed-m": {"dim": 768, "tags": ["en"]},
}

DEFAULT_LOCAL_MODEL = "nomic-ai/nomic-embed-text-v1.5-Q"

ENV_MODE = "TB_EMBED_LOCAL"              # 1 | 0 | auto (default auto)
ENV_MODEL = "TB_EMBED_LOCAL_MODEL"
ENV_CACHE = "TB_EMBED_CACHE_DIR"
ENV_AUTOINSTALL = "TB_EMBED_AUTOINSTALL"  # default 1


# =============================================================================
# MANIFEST (app.manifest.isaa.embedding) + RESOLUTION
# =============================================================================


def _data_dir() -> Path:
    try:
        from toolboxv2 import get_app

        return Path(get_app().data_dir)
    except Exception:
        return Path.home() / ".toolboxv2"


def _embedding_manifest():
    """Return app.manifest.isaa.embedding or None (isaa may be absent)."""
    try:
        from toolboxv2 import get_app

        manifest = get_app().manifest
        if manifest and manifest.isaa:
            return manifest.isaa.embedding
    except Exception:
        pass
    return None


def _mv(cfg, key, default=""):
    """Read a manifest field, treating unresolved ${...} templates as unset."""
    v = getattr(cfg, key, default) if cfg else default
    v = (v if isinstance(v, str) else v)
    if isinstance(v, str) and v.strip().startswith("${"):
        return default
    return v


def load_manifest() -> dict[str, dict]:
    """Extra local models from app.manifest.isaa.embedding.models."""
    out: dict[str, dict] = {}
    cfg = _embedding_manifest()
    if cfg is None:
        return out
    try:
        for m in getattr(cfg, "models", None) or []:
            get = (lambda k, d=None: m.get(k, d)) if isinstance(m, dict) \
                else (lambda k, d=None: getattr(m, k, d))
            name = get("name")
            if not name:
                continue
            entry = {"dim": int(get("dim", 768)),
                     "tags": list(get("tags", []) or ["manifest"])}
            for k in ("hf", "model_file", "pooling", "normalization"):
                v = get(k)
                if v is not None:
                    entry[k] = v
            out[name] = entry
    except Exception as e:
        logger.warning(f"[local_embeddings] embedding manifest unreadable: {e}")
    return out


def _register_custom_if_needed(name: str) -> None:
    """Manifest entries with an 'hf' source get registered as fastembed
    custom models so any ONNX embedding repo on HF is usable."""
    entry = load_manifest().get(name)
    if not entry or "hf" not in entry:
        return
    from fastembed import TextEmbedding

    try:
        supported = {m["model"] if isinstance(m, dict) else m.model
                     for m in TextEmbedding.list_supported_models()}
        if name in supported:
            return
    except Exception:
        pass
    from fastembed.common.model_description import ModelSource, PoolingType

    pooling = {"mean": PoolingType.MEAN, "cls": PoolingType.CLS,
               "disabled": PoolingType.DISABLED}.get(
        str(entry.get("pooling", "mean")).lower(), PoolingType.MEAN)
    TextEmbedding.add_custom_model(
        model=name,
        pooling=pooling,
        normalization=bool(entry.get("normalization", True)),
        sources=ModelSource(hf=str(entry["hf"])),
        dim=int(entry["dim"]),
        model_file=str(entry.get("model_file", "onnx/model.onnx")),
    )
    logger.info(f"[local_embeddings] registered custom model '{name}' "
                f"(hf: {entry['hf']})")


def model_registry() -> dict[str, dict]:
    """Validated registry + user manifest (manifest wins on name clash)."""
    reg = dict(VALIDATED_MODELS)
    reg.update(load_manifest())
    return reg


def resolve_mode() -> str:
    """'always' | 'never' | 'auto' — env wins over manifest."""
    v = os.getenv(ENV_MODE, "").strip().lower()
    if not v:
        v = str(_mv(_embedding_manifest(), "mode", "")).strip().lower()
    if v in ("1", "true", "always", "local"):
        return "always"
    if v in ("0", "false", "never", "cloud"):
        return "never"
    return "auto"


def resolve_model() -> tuple[str, int]:
    """Returns (model_name, native_dim). Env > manifest > default.
    Unknown models get dim probed at load."""
    reg = model_registry()
    name = os.getenv(ENV_MODEL, "").strip()
    if not name:
        name = str(_mv(_embedding_manifest(), "local_model", "")).strip()
    if not name:
        name = DEFAULT_LOCAL_MODEL
    if name in reg:
        return name, int(reg[name]["dim"])
    logger.warning(
        f"[local_embeddings] model '{name}' not in validated registry/manifest — "
        f"passing through to fastembed (dim probed at load). "
        f"Known: {', '.join(sorted(reg))}"
    )
    return name, 0


def _cache_dir() -> str:
    c = os.getenv(ENV_CACHE, "").strip()
    if not c:
        c = str(_mv(_embedding_manifest(), "cache_dir", "")).strip()
    return c or str(_data_dir() / "embed_cache")


def _autoinstall_allowed() -> bool:
    env = os.getenv(ENV_AUTOINSTALL, "").strip()
    if env:
        return env != "0"
    cfg = _embedding_manifest()
    return bool(getattr(cfg, "auto_install", True)) if cfg else True


def _ensure_fastembed() -> bool:
    """Import fastembed, auto-installing it once if allowed."""
    try:
        import fastembed  # noqa: F401

        return True
    except ImportError:
        pass
    if not _autoinstall_allowed():
        logger.error(
            "[local_embeddings] fastembed missing and autoinstall disabled. "
            "pip install fastembed"
        )
        return False
    logger.info("[local_embeddings] installing fastembed (one-time auto-setup)…")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "fastembed", "-q"]
        # system pythons on debian need the override; harmless elsewhere
        subprocess.run(cmd + ["--break-system-packages"], check=False,
                       capture_output=True, timeout=600)
        import fastembed  # noqa: F401

        return True
    except Exception:
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            import fastembed  # noqa: F401

            return True
        except Exception as e:
            logger.error(f"[local_embeddings] fastembed install failed: {e}")
            return False


# =============================================================================
# LAST-TOKEN ONNX LOADER
# =============================================================================
# fastembed (≤0.8.x) only supports CLS/MEAN/DISABLED pooling. 2026 models
# built on decoder backbones (jina-v5, Qwen3-Embedding) require LAST-TOKEN
# pooling — routing them through fastembed would silently produce wrong
# (mean-pooled) vectors. This loader uses onnxruntime + tokenizers directly
# (both are fastembed dependencies, so no extra install) and:
#   - prefers a graph-provided 'sentence_embedding' output when present
#   - otherwise applies correct last-non-padding-token pooling
#   - L2-normalizes the result


class LastTokenOnnxEmbedder:
    def __init__(self, name: str, hf_repo: str, model_file: str, cache: str):
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_path = hf_hub_download(hf_repo, model_file, cache_dir=cache)
        # external-weight sidecars (large exports): best-effort fetch
        for sidecar in (model_file + ".data", model_file + "_data"):
            try:
                hf_hub_download(hf_repo, sidecar, cache_dir=cache)
            except Exception:
                pass
        tok_path = hf_hub_download(hf_repo, "tokenizer.json", cache_dir=cache)
        self._tok = Tokenizer.from_file(tok_path)
        self._tok.enable_padding()
        self._tok.enable_truncation(max_length=8192)
        so = ort.SessionOptions()
        self._sess = ort.InferenceSession(
            model_path, so, providers=["CPUExecutionProvider"])
        self._out_names = [o.name for o in self._sess.get_outputs()]
        self._in_names = {i.name for i in self._sess.get_inputs()}
        logger.info(f"[local_embeddings] last-token loader ready: {name} "
                    f"(outputs: {self._out_names})")

    def embed(self, texts: list[str]):
        enc = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        feed = {k: v for k, v in feed.items() if k in self._in_names}
        outs = self._sess.run(None, feed)
        by_name = dict(zip(self._out_names, outs))
        if "sentence_embedding" in by_name:
            vecs = np.asarray(by_name["sentence_embedding"], dtype=np.float32)
        else:
            hidden = np.asarray(outs[0], dtype=np.float32)  # (b, seq, dim)
            last_idx = mask.sum(axis=1) - 1                  # last real token
            vecs = hidden[np.arange(hidden.shape[0]), last_idx]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms)


# =============================================================================
# EMBEDDER (lazy singleton, thread-safe)
# =============================================================================


class LocalEmbedder:
    _instance: "LocalEmbedder | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._model = None
        self._model_name = ""
        self._native_dim = 0
        self._load_lock = threading.Lock()

    def _load(self):
        """Load (or reload after env change) the ONNX model. Blocking."""
        name, dim = resolve_model()
        with self._load_lock:
            if self._model is not None and self._model_name == name:
                return
            cache = _cache_dir()
            Path(cache).mkdir(parents=True, exist_ok=True)

            # last-token models (jina-v5, qwen3-embedding, …) bypass fastembed
            entry = model_registry().get(name, {})
            if str(entry.get("pooling", "")).lower() in ("last_token", "last-token"):
                if not _ensure_fastembed():  # supplies onnxruntime+tokenizers+hub
                    raise RuntimeError("fastembed (deps) unavailable")
                self._model = LastTokenOnnxEmbedder(
                    name=name,
                    hf_repo=str(entry.get("hf", name)),
                    model_file=str(entry.get("model_file", "onnx/model.onnx")),
                    cache=cache,
                )
                self._model_name = name
                self._native_dim = dim or 768
                logger.info(f"[local_embeddings] ready: {name} dim={self._native_dim}")
                return

            if not _ensure_fastembed():
                raise RuntimeError("fastembed unavailable")
            from fastembed import TextEmbedding

            try:
                _register_custom_if_needed(name)
            except Exception as e:
                logger.warning(f"[local_embeddings] custom registration failed: {e}")
            logger.info(f"[local_embeddings] loading '{name}' (cache: {cache})")
            self._model = TextEmbedding(model_name=name, cache_dir=cache)
            self._model_name = name
            if not dim:  # probe unknown model
                dim = len(next(iter(self._model.embed(["dim probe"]))))
            self._native_dim = dim
            logger.info(f"[local_embeddings] ready: {name} dim={dim}")

    def embed_sync(self, texts: list[str], dimensions: int | None = None) -> np.ndarray:
        """Blocking embed. Adapts native dim to `dimensions`:
        larger → truncate + renormalize (MRL-style); smaller → zero-pad."""
        self._load()
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        if dimensions and vecs.shape[1] != dimensions:
            if vecs.shape[1] > dimensions:
                vecs = vecs[:, :dimensions]
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vecs = vecs / norms
            else:
                pad = np.zeros((vecs.shape[0], dimensions - vecs.shape[1]),
                               dtype=np.float32)
                vecs = np.concatenate([vecs, pad], axis=1)
        return vecs

    @property
    def info(self) -> dict:
        return {"model": self._model_name or resolve_model()[0],
                "native_dim": self._native_dim,
                "loaded": self._model is not None,
                "mode": resolve_mode()}


async def local_embed(texts: list[str], dimensions: int | None = None) -> np.ndarray:
    """Async wrapper — runs ONNX inference in a worker thread so the event
    loop (and the agent) never blocks on embedding."""
    embedder = LocalEmbedder()
    return await asyncio.to_thread(embedder.embed_sync, texts, dimensions)


def local_embed_available() -> bool:
    """Cheap check without loading a model."""
    try:
        import fastembed  # noqa: F401

        return True
    except ImportError:
        return _autoinstall_allowed()
