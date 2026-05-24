"""Static model info — replaces litellm.get_model_info()."""
from __future__ import annotations

MODEL_INFO: dict[str, dict] = {
    "groq/llama-3.3-70b-versatile": {"ctx": 128_000, "tools": True},
    "groq/llama-3.1-8b-instant": {"ctx": 128_000, "tools": True},
    "gemini/gemini-2.5-pro": {"ctx": 1_000_000, "tools": True},
    "gemini/gemini-2.5-flash": {"ctx": 1_000_000, "tools": True},
    "gemini/gemini-2.5-flash-lite": {"ctx": 1_000_000, "tools": True},
    "zai/glm-4.7-flash": {"ctx": 128_000, "tools": True},
    "zai/glm-4.5-flash": {"ctx": 128_000, "tools": True},
    "zglm/glm-5": {"ctx": 200_000, "tools": True},
    "zglm/glm-4.7": {"ctx": 200_000, "tools": True},
    "zglm/glm-4.6": {"ctx": 200_000, "tools": True},
    "anthropic/claude-sonnet-4-6": {"ctx": 200_000, "tools": True},
    "anthropic/claude-haiku-4-5": {"ctx": 200_000, "tools": True},
    "cerebras/llama-3.3-70b": {"ctx": 128_000, "tools": True},
    "mistral/mistral-large-latest": {"ctx": 128_000, "tools": True},
    "deepseek/deepseek-chat": {"ctx": 128_000, "tools": True},
    # Wildcards — prefix match
    "ollama/": {"ctx": 128_000, "tools": True},
    "openrouter/": {"ctx": 128_000, "tools": True},
}

_DEFAULT = {"ctx": 128_000, "tools": False}


def _lookup(model: str) -> dict:
    """Exact match first, then wildcard prefix match."""
    if model in MODEL_INFO:
        return MODEL_INFO[model]
    # Wildcard: check if model starts with any prefix key ending in '/'
    for key, info in MODEL_INFO.items():
        if key.endswith("/") and model.startswith(key):
            return info
    return _DEFAULT


def ctx_limit(model: str) -> int:
    return _lookup(model)["ctx"]


def supports_tools(model: str) -> bool:
    return _lookup(model)["tools"]
