"""Prompt-Caching helper — injects cache_control breakpoints for supported providers."""
from __future__ import annotations
from typing import Literal

# Min tokens per provider for caching to activate (approximate char-based heuristic)
_MIN_CHARS_FOR_CACHE = {
    "anthropic": 1024 * 3,   # ~1024 tokens
}

CacheTTL = Literal["5m", "1h"]


def supports_explicit_cache(prefix: str) -> bool:
    """True if provider needs explicit cache_control markers."""
    return prefix == "anthropic"


def supports_implicit_cache(prefix: str) -> bool:
    """True if provider auto-caches stable prefixes (no markers needed)."""
    return prefix in {
        "openrouter", "groq", "gemini", "cerebras", "deepseek",
        "xai", "together_ai", "mistral", "nvidia_nim", "9rou",
        "zai", "zglm", "minimax", "gateway",
    }


def inject_cache_breakpoints(
    messages: list[dict],
    tools: list[dict] | None,
    prefix: str,
    ttl: CacheTTL = "5m",
) -> tuple[list[dict], list[dict] | None]:
    """Adds cache_control markers for Anthropic. No-op for others.

    Anthropic supports up to 4 breakpoints. Strategy:
      1. Last `system` message       → caches the static system prompt
      2. Last tool definition         → caches the tool schema
      3. Last message before final user → caches the conversation history

    Returns (new_messages, new_tools). Originals are not mutated.
    """
    if not supports_explicit_cache(prefix):
        return messages, tools

    cache_marker = {"type": "ephemeral"}
    if ttl == "1h":
        cache_marker = {"type": "ephemeral", "ttl": "1h"}

    new_messages = [dict(m) for m in messages]
    new_tools = [dict(t) for t in tools] if tools else None
    breakpoints_used = 0
    MAX_BREAKPOINTS = 4

    # 1. System message → structured content with cache_control on last block
    total_chars = sum(len(str(m.get("content", ""))) for m in new_messages
                       if m.get("role") == "system")
    if total_chars >= _MIN_CHARS_FOR_CACHE["anthropic"]:
        for m in reversed(new_messages):
            if m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str):
                    m["content"] = [{
                        "type": "text",
                        "text": content,
                        "cache_control": cache_marker,
                    }]
                elif isinstance(content, list) and content:
                    last = dict(content[-1])
                    last["cache_control"] = cache_marker
                    m["content"] = content[:-1] + [last]
                breakpoints_used += 1
                break

    # 2. Last tool definition → tool-schema cache
    if new_tools and breakpoints_used < MAX_BREAKPOINTS:
        tools_chars = sum(len(str(t)) for t in new_tools)
        if tools_chars >= _MIN_CHARS_FOR_CACHE["anthropic"]:
            new_tools[-1]["cache_control"] = cache_marker
            breakpoints_used += 1

    # 3. History breakpoint — cache everything up to last user/tool turn
    if breakpoints_used < MAX_BREAKPOINTS and len(new_messages) >= 4:
        # Find second-to-last user message → cache the conversation up to that point
        user_indices = [i for i, m in enumerate(new_messages)
                        if m.get("role") in ("user", "tool")]
        if len(user_indices) >= 2:
            cache_at = user_indices[-2]
            target = new_messages[cache_at]
            content = target.get("content")
            if isinstance(content, str):
                target["content"] = [{
                    "type": "text",
                    "text": content,
                    "cache_control": cache_marker,
                }]
            elif isinstance(content, list) and content:
                last = dict(content[-1])
                last["cache_control"] = cache_marker
                target["content"] = content[:-1] + [last]
            new_messages[cache_at] = target
            breakpoints_used += 1

    return new_messages, new_tools
