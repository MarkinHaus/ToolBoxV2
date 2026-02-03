"""
LiteLLM LLM Interface Module
============================

This module provides interfaces for interacting with LiteLLM's language models,
including text generation and embedding capabilities.

Author: Lightrag Team
Created: 2025-02-04
License: MIT License
Version: 1.0.0

Change Log:
- 1.0.0 (2025-02-04): Initial LiteLLM release
    * Ported OpenAI logic to use litellm async client
    * Updated error types and environment variable names
    * Preserved streaming and embedding support

Dependencies:
    - litellm
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.litellm import logging
if not hasattr(logging, 'NONE'):
    logging.NONE = 100

import litellm_complete, litellm_embed
"""

__version__ = "1.0.0"
__author__ = "Markin Hausmanns"
__status__ = "Demo"

import logging
import os

# Ensure AsyncIterator is imported correctly depending on Python version
from collections.abc import AsyncIterator

if not hasattr(logging, 'NONE'):
    logging.NONE = 100

import litellm

# lightRag utilities and types
import numpy as np

# Use pipmaster to ensure the litellm dependency is installed
from litellm import APIConnectionError, RateLimitError, Timeout, acompletion

# Import litellm's asynchronous client and error classes
# Retry handling for transient errors
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from toolboxv2 import get_logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, Timeout, APIConnectionError)),
)
async def litellm_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    base_url=None,
    api_key=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    """
    Core function to query the LiteLLM model. It builds the message context,
    invokes the completion API, and returns either a complete result string or
    an async iterator for streaming responses.
    """
    # Set the API key if provided
    if api_key:
        os.environ["LITELLM_API_KEY"] = api_key

    # Remove internal keys not needed for the client call
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    fallbacks_ = kwargs.pop("fallbacks", [])
    # Build the messages list from system prompt, conversation history, and the new prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages is not None:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Log query details for debugging purposes
    try:
        # Depending on the response format, choose the appropriate API call
        if "response_format" in kwargs:
            response = await acompletion(
                model=model, messages=messages,
                fallbacks=fallbacks_+os.getenv("FALLBACKS_MODELS", '').split(','),
                **kwargs
            )
        else:
            response = await acompletion(
                model=model, messages=messages,
                fallbacks=os.getenv("FALLBACKS_MODELS", '').split(','),
                **kwargs
            )
    except Exception as e:
        print(f"\n{model=}\n{prompt=}\n{system_prompt=}\n{history_messages=}\n{base_url=}\n{api_key=}\n{kwargs=}")
        get_logger().error(f"Failed to litellm memory work {e}")
        return ""

    # Check if the response is a streaming response (i.e. an async iterator)
    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                # Assume LiteLLM response structure is similar to OpenAI's
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                yield content

        return inner()
    else:
        # Non-streaming: extract and return the full content string

        content = response.choices[0].message.content
        if content is None:
            content = response.choices[0].message.tool_calls[0].function.arguments
        return content

def enforce_no_additional_properties(schema: dict) -> dict:
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        for prop in schema.get("properties", {}).values():
            enforce_no_additional_properties(prop)
    elif schema.get("type") == "array":
        enforce_no_additional_properties(schema.get("items", {}))
    return schema

async def litellm_complete(
    prompt, system_prompt=None, history_messages=None, keyword_extraction=False, model_name = "groq/gemma2-9b-it", **kwargs
) -> str | AsyncIterator[str]:
    """
    Public completion interface using the model name specified in the global configuration.
    Optionally extracts keywords if requested.
    """
    if history_messages is None:
        history_messages = []
    # Check and set response format for keyword extraction if needed
    keyword_extraction_flag = kwargs.pop("keyword_extraction", None)
    if keyword_extraction_flag:
        kwargs["response_format"] = "json"

    if "response_format" in kwargs:
        if isinstance(kwargs["response_format"], dict):
            kwargs["response_format"] = enforce_no_additional_properties(kwargs["response_format"])
        elif isinstance(kwargs["response_format"], str):
            pass
        else:
            kwargs["response_format"] = enforce_no_additional_properties(kwargs["response_format"].model_json_schema())  # oder .schema() in v1
     # kwargs["hashing_kv"].global_config["llm_model_name"]

    if any(x in model_name for x in ["mistral", "mixtral"]):
        kwargs.pop("response_format", None)

    return await litellm_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, Timeout, APIConnectionError)),
)
async def litellm_embed(
    texts: list[str],
    model: str = "openrouter/qwen/qwen3-embedding-8b",
    dimensions: int = 256,
    base_url: str = None,
    api_key: str = None,
    input_type: str | None = None,
    process_media: bool = True,
) -> np.ndarray:
    """
    Generates embeddings for the given list of texts using LiteLLM.
    """
    sto = litellm.drop_params
    litellm.drop_params = True
    res = await embed(
        model=model,
        texts=texts,
        dimensions=dimensions,
        api_key=api_key,
        input_type=input_type,
    )
    litellm.drop_params = sto
    return res


"""
Universal Embedding Function - Mit OpenRouter Workaround

BEKANNTER BUG: LiteLLM routet OpenRouter Embeddings nicht korrekt!
Fehler: "Unmapped LLM provider for this endpoint"

WORKAROUND: OpenRouter als OpenAI-kompatiblen Endpoint aufrufen
"""

import numpy as np
import os
import re
from pathlib import Path
from typing import Any

try:
    import litellm
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
    from litellm.exceptions import RateLimitError, Timeout, APIConnectionError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    retry_if_exception_type = lambda x: None
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kw: None
    RateLimitError = Exception
    Timeout = Exception
    APIConnectionError = Exception


# =========================================================================
# OPENROUTER EMBEDDING WORKAROUND
# =========================================================================

def _is_openrouter_model(model: str) -> bool:
    """Prüft ob es ein OpenRouter Modell ist"""
    return model.startswith("openrouter/") or "OPENROUTER" in os.environ.get("LITELLM_MODEL", "")


def _convert_openrouter_to_openai_compatible(model: str) -> tuple[str, dict]:
    """
    Konvertiert OpenRouter Modell zu OpenAI-kompatiblem Format.

    OpenRouter ist OpenAI-kompatibel, also können wir es als custom OpenAI endpoint nutzen.

    Args:
        model: z.B. "openrouter/qwen/qwen3-embedding-8b"

    Returns:
        tuple: (converted_model, extra_kwargs)
    """
    # Entferne "openrouter/" prefix
    if model.startswith("openrouter/"):
        actual_model = model[len("openrouter/"):]
    else:
        actual_model = model

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable required")

    # OpenRouter als OpenAI-kompatiblen Endpoint
    extra_kwargs = {
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": api_key,
    }

    # Model format: "openai/<model>" für LiteLLM OpenAI-kompatibel
    converted_model = f"openai/{actual_model}"

    return converted_model, extra_kwargs


# =========================================================================
# MAIN EMBEDDING FUNCTION
# =========================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, Timeout, APIConnectionError)) if LITELLM_AVAILABLE else (Exception,)
)
async def embed(
    texts: list[str],
    model: str = "ollama/nomic-embed-text",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    input_type: str | None = None,
) -> np.ndarray:
    """
    Universal embeddings via LiteLLM - MIT OPENROUTER WORKAROUND.

    Supports:
    - Ollama:      model="ollama/nomic-embed-text", "ollama/qwen3-embedding"
    - OpenRouter:  model="openrouter/qwen/qwen3-embedding-8b" (WORKAROUND!)
    - OpenAI:      model="text-embedding-3-small"
    - Cohere:      model="embed-english-v3.0"

    Args:
        texts: Liste von Texten
        model: LiteLLM Modell-String
        dimensions: Ziel-Dimensionen (MRL/Matryoshka Support)
        api_base: Optionale API Base URL
        api_key: Optionaler API Key
        input_type: Für Cohere: "search_document", "search_query", etc.

    Returns:
        np.ndarray: Embeddings Matrix (n_texts, dimensions)
    """
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM required - pip install litellm")

    # WORKAROUND: OpenRouter als OpenAI-kompatiblen Endpoint
    extra_kwargs: dict[str, Any] = {}

    if _is_openrouter_model(model):
        model, extra_kwargs = _convert_openrouter_to_openai_compatible(model)
        print(f"[embed] OpenRouter Workaround: Using {model} via OpenAI-compatible endpoint")

    # Baue LiteLLM kwargs
    kwargs: dict[str, Any] = {
        "model": model,
        "input": texts,
        **extra_kwargs,
    }

    # Überschreibe mit expliziten Parametern
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if dimensions:
        kwargs["dimensions"] = dimensions
    if input_type:
        kwargs["input_type"] = input_type

    # API Call
    response = await litellm.aembedding(**kwargs)

    # Extrahiere Embeddings
    embeddings = np.array([d["embedding"] for d in response.data])

    # Dimensionsreduktion für Matryoshka/MRL wenn nötig
    if dimensions and embeddings.shape[1] > dimensions:
        embeddings = embeddings[:, :dimensions]

    return embeddings


# =========================================================================
# SYNC VERSION
# =========================================================================

def embed_sync(
    texts: list[str],
    model: str = "ollama/nomic-embed-text",
    dimensions: int | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    input_type: str | None = None,
) -> np.ndarray:
    """Synchrone Version von embed()"""
    if not LITELLM_AVAILABLE:
        raise RuntimeError("LiteLLM required - pip install litellm")

    extra_kwargs: dict[str, Any] = {}

    if _is_openrouter_model(model):
        model, extra_kwargs = _convert_openrouter_to_openai_compatible(model)
        print(f"[embed] OpenRouter Workaround: Using {model} via OpenAI-compatible endpoint")

    kwargs: dict[str, Any] = {
        "model": model,
        "input": texts,
        **extra_kwargs,
    }

    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if dimensions:
        kwargs["dimensions"] = dimensions
    if input_type:
        kwargs["input_type"] = input_type

    response = litellm.embedding(**kwargs)

    embeddings = np.array([d["embedding"] for d in response.data])

    if dimensions and embeddings.shape[1] > dimensions:
        embeddings = embeddings[:, :dimensions]

    return embeddings


# =========================================================================
# OPENROUTER DIRECT (ohne LiteLLM)
# =========================================================================

async def embed_openrouter_direct(
    texts: list[str],
    model: str = "qwen/qwen3-embedding-8b",
    dimensions: int | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    """
    Direkte OpenRouter Embeddings ohne LiteLLM.

    Nutzt httpx direkt für maximale Kontrolle.

    OpenRouter Embedding Models:
    - qwen/qwen3-embedding-0.6b
    - qwen/qwen3-embedding-4b
    - qwen/qwen3-embedding-8b
    - openai/text-embedding-3-small
    - openai/text-embedding-3-large
    - cohere/embed-english-v3.0
    - cohere/embed-multilingual-v3.0

    Siehe: https://openrouter.ai/models?fmt=cards&output_modalities=embeddings
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required - pip install httpx")

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")

    # Entferne prefix falls vorhanden
    if model.startswith("openrouter/"):
        model = model[len("openrouter/"):]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OR_SITE_URL", ""),
        "X-Title": os.environ.get("OR_APP_NAME", "ToolBoxV2"),
    }

    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
    }

    if dimensions:
        payload["dimensions"] = dimensions

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

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
    """Synchrone Version von embed_openrouter_direct()"""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required - pip install httpx")

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

    payload: dict[str, Any] = {
        "model": model,
        "input": texts,
    }

    if dimensions:
        payload["dimensions"] = dimensions

    with httpx.Client() as client:
        response = client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

    embeddings = np.array([d["embedding"] for d in data["data"]])

    if dimensions and embeddings.shape[1] > dimensions:
        embeddings = embeddings[:, :dimensions]

    return embeddings


# =========================================================================
# SMART EMBED - Wählt automatisch die beste Methode
# =========================================================================

async def smart_embed(
    texts: list[str],
    model: str = "ollama/nomic-embed-text",
    dimensions: int | None = None,
    api_key: str | None = None,
    input_type: str | None = None,
    prefer_direct: bool = True,
) -> np.ndarray:
    """
    Intelligente Embedding-Funktion die automatisch die beste Methode wählt.

    - OpenRouter: Nutzt direkte API (umgeht LiteLLM Bug)
    - Ollama: Nutzt LiteLLM
    - Andere: Nutzt LiteLLM

    Args:
        texts: Texte zum Embedden
        model: Modell-String
        dimensions: Ziel-Dimensionen
        api_key: API Key (optional)
        input_type: Input-Typ für bestimmte Provider
        prefer_direct: Bei OpenRouter direkte API bevorzugen (default: True)
    """
    if _is_openrouter_model(model) and prefer_direct:
        # Direkte OpenRouter API (umgeht LiteLLM Bug komplett)
        return await embed_openrouter_direct(
            texts,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
        )
    else:
        # LiteLLM (mit Workaround für OpenRouter falls nötig)
        return await embed(
            texts,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            input_type=input_type,
        )


# =========================================================================
# SIMILARITY HELPERS
# =========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Berechnet Cosine Similarity zwischen Embedding-Vektoren"""
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


# =========================================================================
# OPENROUTER AVAILABLE MODELS
# =========================================================================

OPENROUTER_EMBEDDING_MODELS = [
    # Qwen3 Embedding (Text-only, MRL support)
    "qwen/qwen3-embedding-0.6b",
    "qwen/qwen3-embedding-4b",
    "qwen/qwen3-embedding-8b",

    # OpenAI via OpenRouter
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "openai/text-embedding-ada-002",

    # Cohere via OpenRouter
    "cohere/embed-english-v3.0",
    "cohere/embed-multilingual-v3.0",
    "cohere/embed-english-light-v3.0",
    "cohere/embed-multilingual-light-v3.0",

    # Google via OpenRouter
    "google/gemini-embedding-001",

    # Voyage via OpenRouter
    "voyage/voyage-3",
    "voyage/voyage-3-lite",
]


# =========================================================================
# EXAMPLE / TEST
# =========================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=== OpenRouter Embedding Test ===\n")

        # Test mit direkter API
        print("Testing embed_openrouter_direct()...")
        print("Model: qwen/qwen3-embedding-0.6b")

        # Simuliere (ohne echten API call)
        print("\nVerfügbare OpenRouter Embedding Modelle:")
        for m in OPENROUTER_EMBEDDING_MODELS:
            print(f"  - {m}")

        print("\n=== Usage Examples ===\n")

        print("# Option 1: Direkte OpenRouter API (empfohlen)")
        print('embeddings = await embed_openrouter_direct(')
        print('    ["Hello world"],')
        print('    model="qwen/qwen3-embedding-8b"')
        print(')')

        print("\n# Option 2: Smart Embed (wählt automatisch)")
        print('embeddings = await smart_embed(')
        print('    ["Hello world"],')
        print('    model="openrouter/qwen/qwen3-embedding-8b"')
        print(')')

        print("\n# Option 3: LiteLLM mit Workaround")
        print('embeddings = await embed(')
        print('    ["Hello world"],')
        print('    model="openrouter/qwen/qwen3-embedding-8b"')
        print(')')
        print("# -> Konvertiert zu: openai/qwen/qwen3-embedding-8b mit api_base=openrouter")

        print("\n# Ollama (funktioniert normal)")
        print('embeddings = await embed(')
        print('    ["Hello world"],')
        print('    model="ollama/nomic-embed-text"')
        print(')')

        embeddings = await embed(["Hello world"], model="ollama/embeddinggemma")
        print(embeddings)
        print("###################")
        embeddings = await embed(["Hello world"], model="openrouter/qwen/qwen3-embedding-8b")
        print(embeddings)

    asyncio.run(test())
