"""Static model info — replaces litellm.get_model_info()."""
from __future__ import annotations

MODEL_INFO: dict[str, dict] = {
    # -------------------------------------------------
    # Groq & Cerebras (Hardware-beschleunigte Backends)
    # -------------------------------------------------
    "groq/llama-3.3-70b-versatile": {"ctx": 128_000, "tools": True},
    "groq/llama-3.1-8b-instant": {"ctx": 128_000, "tools": True},
    "cerebras/llama-3.3-70b": {"ctx": 128_000, "tools": True},

    # -------------------------------------------------
    # Google Gemini Series (Bis zu Gemini 3 Pro / 10M)
    # -------------------------------------------------
    "gemini/gemini-3.1-pro": {"ctx": 10_000_000, "tools": True},
    "gemini/gemini-3-pro": {"ctx": 10_000_000, "tools": True},
    "gemini/gemini-2.5-pro": {"ctx": 2_000_000, "tools": True},
    "gemini/gemini-2.5-flash": {"ctx": 1_000_000, "tools": True},
    "gemini/gemini-2.5-flash-lite": {"ctx": 1_000_000, "tools": True},

    # -------------------------------------------------
    # Z.ai / GLM Series (GLM-5 Generation bis 1M)
    # -------------------------------------------------
    "zai/glm-5.2": {"ctx": 1_000_000, "tools": True},
    "zai/glm-5.1": {"ctx": 198_000, "tools": True},
    "zai/glm-5": {"ctx": 200_000, "tools": True},
    "zai/glm-4.7-flash": {"ctx": 128_000, "tools": True},
    "zai/glm-4.5-flash": {"ctx": 128_000, "tools": True},
    "zglm/glm-5.2": {"ctx": 1_000_000, "tools": True},
    "zglm/glm-5": {"ctx": 200_000, "tools": True},
    "zglm/glm-4.7": {"ctx": 200_000, "tools": True},
    "zglm/glm-4.6": {"ctx": 200_000, "tools": True},

    # -------------------------------------------------
    # Anthropic Claude Series (Claude 5 Generation)
    # -------------------------------------------------
    "anthropic/claude-fable-5": {"ctx": 1_000_000, "tools": True},
    "anthropic/claude-mythos-5": {"ctx": 1_000_000, "tools": True},
    "anthropic/claude-opus-5": {"ctx": 1_000_000, "tools": True},
    "anthropic/claude-sonnet-5": {"ctx": 1_000_000, "tools": True},
    "anthropic/claude-haiku-4-5": {"ctx": 200_000, "tools": True},
    "anthropic/claude-sonnet-4-6": {"ctx": 200_000, "tools": True},

    # -------------------------------------------------
    # OpenAI GPT-5 Series
    # -------------------------------------------------
    "openai/gpt-5.6-sol": {"ctx": 1_050_000, "tools": True},
    "openai/gpt-5.5": {"ctx": 1_000_000, "tools": True},
    "openai/gpt-5": {"ctx": 400_000, "tools": True},
    "openai/gpt-5.2": {"ctx": 128_000, "tools": True},

    # -------------------------------------------------
    # Alibaba Qwen Series (Qwen 3.5 / 3.6 / 3.8)
    # -------------------------------------------------
    "qwen/qwen-3.8-max": {"ctx": 1_000_000, "tools": True},
    "qwen/qwen-3.8-27b": {"ctx": 262_144, "tools": True},
    "qwen/qwen-3.8-instruct": {"ctx": 128_000, "tools": True},
    "qwen/qwen3.6-plus": {"ctx": 256_000, "tools": True},
    "qwen/qwen3.6-35b-a3b": {"ctx": 262_144, "tools": True},
    "qwen/qwen3.6-27b": {"ctx": 262_144, "tools": True},
    "qwen/qwen3.5-plus": {"ctx": 1_000_000, "tools": True},
    "qwen/qwen3.5-397b-a17b": {"ctx": 262_144, "tools": True},
    "qwen/qwen3.5-9b": {"ctx": 262_144, "tools": True},
    "qwen/qwen3.5-4b": {"ctx": 262_144, "tools": True},
    "qwen/qwen3.5-2b": {"ctx": 262_144, "tools": True},
    "qwen/qwen3.5-0.8b": {"ctx": 262_144, "tools": True},

    # -------------------------------------------------
    # DeepSeek Series
    # -------------------------------------------------
    "deepseek/deepseek-v4": {"ctx": 1_000_000, "tools": True},
    "deepseek/deepseek-v4-flash": {"ctx": 1_000_000, "tools": True},
    "deepseek/deepseek-chat": {"ctx": 128_000, "tools": True},

    # -------------------------------------------------
    # Sonstige Open Source / Moonshot / MiniMax
    # -------------------------------------------------
    "mistral/mistral-large-latest": {"ctx": 128_000, "tools": True},
    "moonshot/kimi-k3": {"ctx": 1_000_000, "tools": True},
    "minimax/minimax-m3": {"ctx": 1_000_000, "tools": True},

    # Wildcards — Prefix Match
    "ollama/": {"ctx": 262_144, "tools": True},
    "openrouter/": {"ctx": 262_144, "tools": True},
    "9rou/": {"ctx": 1_000_000, "tools": True},
}

_DEFAULT = {"ctx": 262_144, "tools": False}


def _normalize(s: str) -> str:
    """Normalisiert Trennzeichen und Case für robusten Modell-Vergleich."""
    return s.lower().replace(".", "-").replace("_", "-").replace(":", "-")


def _lookup(model: str) -> dict:
    """
    Sucht das passende Modell-Limit.
    Reihenfolge:
      1. Exakter Match
      2. Spezifischer Substring-Match (längste Schlüssel zuerst)
      3. Basis-Modell-Match (ohne Provider-Präfixe von rechts nach links)
      4. Wildcard/Präfix-Match (z. B. 'ollama/', 'openrouter/')
    """
    # 1. Exakter Match (Fast-Path)
    if model in MODEL_INFO:
        return MODEL_INFO[model]

    norm_model = _normalize(model)

    # 2. Spezifische Keys (ohne Wildcard-Slashes am Ende) sortiert nach Länge absteigend
    specific_keys = [k for k in MODEL_INFO if not k.endswith("/")]
    specific_keys.sort(key=lambda x: len(_normalize(x)), reverse=True)

    for key in specific_keys:
        norm_key = _normalize(key)

        # Substring Match (z. B. 'anthropic/claude-sonnet-5' in 'openrouter/anthropic/claude-sonnet-5')
        if norm_key in norm_model:
            return MODEL_INFO[key]

        # Suffix/Base-Match (z. B. 'gemini-2-5-pro' in 'openrouter/google/gemini-2.5-pro')
        if "/" in key:
            base_key = _normalize(key.split("/", 1)[1])
            if base_key in norm_model:
                return MODEL_INFO[key]

    # 3. Wildcard / Provider Fallback (z. B. 'openrouter/', 'ollama/')
    for key, info in MODEL_INFO.items():
        if key.endswith("/") and norm_model.startswith(_normalize(key)):
            return info

    return _DEFAULT


def ctx_limit(model: str) -> int:
    return _lookup(model)["ctx"]


def supports_tools(model: str) -> bool:
    return _lookup(model)["tools"]
