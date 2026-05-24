"""Factory for registering default provider adapters."""
from __future__ import annotations
import os

from ..router import CompletionRouter
from .openai_compat import OpenAICompatAdapter
from .zai import ZAIAdapter
from .anthropic import AnthropicAdapter
from .minimax import MiniMaxAdapter


def setup_default_adapters(router: CompletionRouter, env: dict | None = None):
    """Register all known providers from environment variables.

    Only registers adapters for providers where keys are present.
    Ollama is always registered (localhost).
    """
    env = env or dict(os.environ)

    # Groq
    if env.get("GROQ_API_KEY"):
        router.register("groq", OpenAICompatAdapter("https://api.groq.com/openai/v1"))

    # Gemini (OpenAI-compat endpoint)
    if env.get("GEMINI_API_KEY"):
        router.register("gemini", OpenAICompatAdapter(
            "https://generativelanguage.googleapis.com/v1beta/openai"))

    # ZAI Free + Coding
    if env.get("ZAI_API_KEY"):
        router.register("zai", ZAIAdapter(use_coding_plan=False))
        router.register("zglm", ZAIAdapter(use_coding_plan=True))

    # Anthropic
    if env.get("ANTHROPIC_API_KEY"):
        router.register("anthropic", AnthropicAdapter())

    # MiniMax
    if env.get("MINIMAX_API_KEY"):
        router.register("minimax", MiniMaxAdapter("https://api.minimax.io/v1"))

    # Ollama (always available)
    ollama_base = env.get("OLLAMA_BASE_URL", "http://localhost:11434")
    router.register("ollama", OpenAICompatAdapter(ollama_base.rstrip("/") + "/v1"))

    # All other OpenAI-compatible providers
    OPENAI_COMPAT: dict[str, tuple[str, str]] = {
        "cerebras":    ("CEREBRAS_API_KEY",    "https://api.cerebras.ai/v1"),
        "mistral":     ("MISTRAL_API_KEY",     "https://api.mistral.ai/v1"),
        "openrouter":  ("OPENROUTER_API_KEY",  "https://openrouter.ai/api/v1"),
        "deepseek":    ("DEEPSEEK_API_KEY",    "https://api.deepseek.com/v1"),
        "xai":         ("XAI_API_KEY",         "https://api.x.ai/v1"),
        "nvidia_nim":  ("NVIDIA_NIM_API_KEY",  "https://integrate.api.nvidia.com/v1"),
        "together_ai": ("TOGETHER_API_KEY",    "https://api.together.xyz/v1"),
        "cohere":      ("COHERE_API_KEY",      "https://api.cohere.com/v2"),
    }
    for prefix, (env_key, base_url) in OPENAI_COMPAT.items():
        if env.get(env_key):
            router.register(prefix, OpenAICompatAdapter(base_url))

    # Gateway (custom LLM proxy)
    gw_url = env.get("TB_LLM_GATEWAY_URL")
    if gw_url:
        router.register("gateway", OpenAICompatAdapter(gw_url))
