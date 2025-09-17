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
    model: str = "gemini/text-embedding-004",
    dimensions: int = 256,
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    """
    Generates embeddings for the given list of texts using LiteLLM.
    """
    response = await litellm.aembedding(
        model=model, input=texts,
        dimensions=dimensions,
        # encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
