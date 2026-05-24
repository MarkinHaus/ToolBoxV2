"""
Universal Embedding Functions — via CompletionRouter (Layer 2)

Replaces litellm.aembedding() with router.embed().
OpenRouter workaround no longer needed — OpenAICompatAdapter handles it natively.

Usage:
    embeddings = await embed(["Hello world"], model="openrouter/qwen/qwen3-embedding-8b")
    embeddings = await embed(["Hello world"], model="ollama/nomic-embed-text")
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Markin Hausmanns"

import os
import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

try:
    from toolboxv2 import get_logger
except ImportError:
    import logging
    get_logger = lambda: logging.getLogger("llm_router.embeddings")

# ---------------------------------------------------------------------------
# Router singleton (lazy)
# ---------------------------------------------------------------------------

_embed_router = None


def _get_router():
    global _embed_router
    if _embed_router is None:
        try:
            from toolboxv2.mods.isaa.base.llm_router.router import CompletionRouter
            from toolboxv2.mods.isaa.base.llm_router.adapters.setup import setup_default_adapters
        except ImportError:
            from .router import CompletionRouter
            from .adapters.setup import setup_default_adapters
        _embed_router = CompletionRouter(strict_mode=False)
        setup_default_adapters(_embed_router)
    return _embed_router


def _resolve_api_key(model: str, api_key: str | None = None) -> str:
    """Resolve API key from model prefix → env var."""
    if api_key:
        return api_key
    _MAP = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "nvidia_nim": "NVIDIA_NIM_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "voyage": "VOYAGE_API_KEY",
        "ollama": "",
    }
    prefix = model.split("/", 1)[0] if "/" in model else ""
    env_key = _MAP.get(prefix, "")
    return os.environ.get(env_key, "") if env_key else ""


# =========================================================================
# MAIN EMBEDDING FUNCTION
# =========================================================================

async def embed(
    texts: list[str],
    model: str = "ollama/nomic-embed-text",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    input_type: str | None = None,
) -> np.ndarray:
    """
    Universal embeddings via CompletionRouter.

    Supports all providers registered in setup_default_adapters:
    - Ollama:      model="ollama/nomic-embed-text"
    - OpenRouter:  model="openrouter/qwen/qwen3-embedding-8b"
    - OpenAI:      model="openai/text-embedding-3-small"
    - Cohere:      model="cohere/embed-english-v3.0"
    - Gemini:      model="gemini/text-embedding-004"
    - Any OpenAI-compat: prefix/model-name

    Args:
        texts: List of texts to embed
        model: Router model string (prefix/model-name)
        dimensions: Target dimensions (MRL/Matryoshka support)
        api_base: Ignored (router handles base URLs)
        api_key: Optional API key override
        input_type: For Cohere: "search_document", "search_query", etc.

    Returns:
        np.ndarray: Embeddings matrix (n_texts, dimensions)
    """
    router = _get_router()
    key = _resolve_api_key(model, api_key)

    kwargs: dict[str, Any] = {}
    if dimensions:
        kwargs["dimensions"] = dimensions
    if input_type:
        kwargs["input_type"] = input_type

    result = await router.embed(model=model, texts=texts, api_key=key, **kwargs)

    embeddings = np.array(result.embeddings)

    # Post-hoc MRL/Matryoshka dimension reduction if provider didn't handle it
    if dimensions and embeddings.shape[1] > dimensions:
        embeddings = embeddings[:, :dimensions]

    return embeddings


# =========================================================================
# CONVENIENCE WRAPPERS (backward compat)
# =========================================================================

async def litellm_embed(
    texts: list[str],
    model: str = "openrouter/qwen/qwen3-embedding-8b",
    dimensions: int = 256,
    base_url: str = None,
    api_key: str = None,
    input_type: str | None = None,
    process_media: bool = True,
) -> np.ndarray:
    """Drop-in replacement for the old litellm_embed(). Delegates to embed()."""
    return await embed(
        texts=texts,
        model=model,
        dimensions=dimensions,
        api_base=base_url,
        api_key=api_key,
        input_type=input_type,
    )


async def smart_embed(
    texts: list[str],
    model: str = "ollama/nomic-embed-text",
    dimensions: int | None = None,
    api_key: str | None = None,
    input_type: str | None = None,
    prefer_direct: bool = True,
) -> np.ndarray:
    """Smart embed — router handles provider resolution, no special casing needed."""
    return await embed(
        texts=texts,
        model=model,
        dimensions=dimensions,
        api_key=api_key,
        input_type=input_type,
    )


# =========================================================================
# SYNC VERSION (deprecated — kept for backward compat)
# =========================================================================

def embed_sync(
    texts: list[str],
    model: str = "ollama/nomic-embed-text",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    input_type: str | None = None,
) -> np.ndarray:
    """Synchronous embedding via asyncio.run(). Deprecated — use embed() in async code."""
    import asyncio
    return asyncio.run(embed(
        texts=texts, model=model, dimensions=dimensions,
        api_base=api_base, api_key=api_key, input_type=input_type,
    ))


# =========================================================================
# OPENROUTER DIRECT (standalone, no router — kept as alternative)
# =========================================================================

async def embed_openrouter_direct(
    texts: list[str],
    model: str = "qwen/qwen3-embedding-8b",
    dimensions: int | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """
    Direct OpenRouter embeddings via aiohttp (no router, no litellm).

    Kept as standalone alternative for environments where the router isn't available.
    """
    import aiohttp

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")

    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OR_SITE_URL", ""),
        "X-Title": os.environ.get("OR_APP_NAME", "ToolBoxV2"),
    }

    payload: dict[str, Any] = {"model": model, "input": texts}
    if dimensions:
        payload["dimensions"] = dimensions

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

    embeddings = np.array([d["embedding"] for d in data["data"]])
    if dimensions and embeddings.shape[1] > dimensions:
        embeddings = embeddings[:, :dimensions]
    return embeddings


def embed_openrouter_direct_sync(
    texts: list[str],
    model: str = "qwen/qwen3-embedding-8b",
    dimensions: int | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """Sync version of embed_openrouter_direct()."""
    import asyncio
    return asyncio.run(embed_openrouter_direct(
        texts=texts, model=model, dimensions=dimensions, api_key=api_key,
    ))


# =========================================================================
# SIMILARITY HELPERS
# =========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between embedding vectors."""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


# =========================================================================
# KNOWN OPENROUTER EMBEDDING MODELS
# =========================================================================

OPENROUTER_EMBEDDING_MODELS = [
    "qwen/qwen3-embedding-0.6b",
    "qwen/qwen3-embedding-4b",
    "qwen/qwen3-embedding-8b",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "openai/text-embedding-ada-002",
    "cohere/embed-english-v3.0",
    "cohere/embed-multilingual-v3.0",
    "cohere/embed-english-light-v3.0",
    "cohere/embed-multilingual-light-v3.0",
    "google/gemini-embedding-001",
    "voyage/voyage-3",
    "voyage/voyage-3-lite",
]
