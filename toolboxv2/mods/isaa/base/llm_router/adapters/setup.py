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

    # 9Router
    if os.getenv("NINEROUTER_KEY"):
        adapter = OpenAICompatAdapter(os.getenv("NINEROUTER_URL", "http://localhost:20128/v1"))
        if os.getenv("NINEROUTER_URL", "").count("https://9router.") == 1:

            def build_headers(self, api_key: str) -> dict:
                """Default: Bearer token. Subclass overrides for x-api-key etc."""
                import base64
                auth_str = f"{os.getenv("NINEROUTER_USER", "")}:{os.getenv("NINEROUTER_PASSWORD", "")}"
                auth_b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")

                headers = {"Content-Type": "application/json", **self.default_headers}

                headers["Authorization"] = f"Basic {auth_b64}"
                if api_key:
                    headers["X-API-Key"] = f"Bearer {api_key}"

                return headers

            adapter.build_headers = build_headers

        router.register("9rou", adapter=adapter, env_key_name="NINEROUTER_KEY")

    # Groq
    if os.getenv("GROQ_API_KEY"):
        router.register("groq", OpenAICompatAdapter("https://api.groq.com/openai/v1"), env_key_name="GROQ_API_KEY")

    # Gemini (OpenAI-compat endpoint)
    if os.getenv("GEMINI_API_KEY"):
        router.register("gemini", OpenAICompatAdapter(
            "https://generativelanguage.googleapis.com/v1beta/openai"), env_key_name="GEMINI_API_KEY")

    # ZAI Free + Coding
    if os.getenv("ZAI_API_KEY"):
        router.register("zai", ZAIAdapter(use_coding_plan=False), env_key_name="ZAI_API_KEY")
        router.register("zglm", ZAIAdapter(use_coding_plan=True), env_key_name="ZAI_API_KEY")

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        router.register("anthropic", AnthropicAdapter(), env_key_name="ANTHROPIC_API_KEY")

    # MiniMax
    if os.getenv("MINIMAX_API_KEY"):
        router.register("minimax", MiniMaxAdapter("https://api.minimax.io/v1"), env_key_name="MINIMAX_API_KEY")

    # Ollama (always available)
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    router.register("ollama", OpenAICompatAdapter(ollama_base.rstrip("/") + "/v1"), env_key_name="OLLAMA_API_KEY")

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
        if os.getenv(env_key):
            router.register(prefix, OpenAICompatAdapter(base_url), env_key_name=env_key)

    # Gateway (custom LLM proxy)
    gw_url = os.getenv("TB_LLM_GATEWAY_URL")
    if gw_url:
        router.register("gateway", OpenAICompatAdapter(gw_url), env_key_name="TB_LLM_GATEWAY_URL")
