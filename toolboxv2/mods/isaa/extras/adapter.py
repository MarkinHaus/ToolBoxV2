"""
LiteLLM LLM Interface Module
============================

This module provides interfaces for interacting with LiteLLM's language models,
including text generation and embedding capabilities.

Author: Lightrag Team
Created: 2025-02-04
License: MIT License

Copyright (c) 2025 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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
    from llm_interfaces.litellm import litellm_complete, litellm_embed
"""

__version__ = "1.0.0"
__author__ = "Markin Hausmanns"
__status__ = "Demo"

import sys
import os
from typing import Union

import litellm
# Use pipmaster to ensure the litellm dependency is installed
import pipmaster as pm
from litellm import RateLimitError, acompletion, Timeout, APIConnectionError, fallbacks

if not pm.is_installed("litellm"):
    pm.install("litellm")

# Import litellm's asynchronous client and error classes



# Retry handling for transient errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# lightRag utilities and types
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from lightrag.types import GPTKeywordExtractionFormat

import numpy as np

# Ensure AsyncIterator is imported correctly depending on Python version
if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator




@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, Timeout, APIConnectionError)),
)
async def litellm_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
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

    # Build the messages list from system prompt, conversation history, and the new prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Log query details for debugging purposes
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    logger.debug(f"System prompt: {system_prompt}")
    logger.debug("Full context:")

    # Depending on the response format, choose the appropriate API call
    if "response_format" in kwargs:
        response = await acompletion(
            model=model, messages=messages,
            fallbacks=os.getenv("FALLBACKS_MODELS", '').split(','),
            **kwargs
        )
    else:
        response = await acompletion(
            model=model, messages=messages,
            fallbacks=os.getenv("FALLBACKS_MODELS", '').split(','),
            **kwargs
        )

    # Check if the response is a streaming response (i.e. an async iterator)
    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                # Assume LiteLLM response structure is similar to OpenAI's
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        # Non-streaming: extract and return the full content string
        content = response.choices[0].message.content
        if content is None:
            content = response.choices[0].message.tool_calls[0].function.arguments
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


async def litellm_complete(
    prompt, system_prompt=None, history_messages=None, keyword_extraction=False,model_name = "groq/gemma2-9b-it", **kwargs
) -> Union[str, AsyncIterator[str]]:
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
     # kwargs["hashing_kv"].global_config["llm_model_name"]

    return await litellm_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, Timeout, APIConnectionError)),
)
async def litellm_embed(
    texts: list[str],
    model: str = "gemini/text-embedding-004",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    """
    Generates embeddings for the given list of texts using LiteLLM.
    """
    response = await litellm.aembedding(
        model=model, input=texts,
        # encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
