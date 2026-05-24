"""MiniMax adapter with sanitization quirks. Extends OpenAI-compat."""
from __future__ import annotations
import re
from typing import TYPE_CHECKING

from .openai_compat import OpenAICompatAdapter
from ..types import CompletionResult

if TYPE_CHECKING:
    import aiohttp


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from MiniMax output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _extract_think_content(text: str) -> str | None:
    """Extract content from <think>...</think> blocks."""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if matches:
        return "\n".join(m.strip() for m in matches if m.strip())
    return None


class MiniMaxAdapter(OpenAICompatAdapter):
    """MiniMax via OpenAI-compat endpoint with sanitization quirks."""

    MODEL_MAP: dict[str, str] = {
        "minimax-m2.5": "MiniMax-M2.5",
        "minimax-m2.5-highspeed": "MiniMax-M2.5-highspeed",
        "minimax-m2.1": "MiniMax-M2.1",
        "minimax-m2": "MiniMax-M2",
        "codex-minimax-m2.5": "MiniMax-M2.5",
        "codex-minimax-m2.1": "MiniMax-M2.1",
    }

    def __init__(self, base_url: str = "https://api.minimax.io/v1", **kw):
        super().__init__(base_url=base_url, **kw)

    def _map_model(self, model: str) -> str:
        """Case-insensitive model name mapping."""
        return self.MODEL_MAP.get(model.lower(), model)

    def build_payload(self, model: str, messages: list,
                      tools: list | None, **kwargs) -> dict:
        mapped = self._map_model(model)
        clean = self._sanitize_messages(messages)
        payload = super().build_payload(mapped, clean, tools, **kwargs)
        payload["model"] = mapped
        return payload

    def _sanitize_messages(self, messages: list) -> list:
        """Sanitize messages for MiniMax: merge system, normalize content, fix tool sequences."""
        system_parts: list[str] = []
        non_system: list[dict] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content")

            # Normalize content to str
            if content is None:
                content = ""
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts) if text_parts else ""
            content = str(content)

            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
                continue

            if role not in ("user", "assistant", "tool"):
                continue

            out: dict = {"role": role, "content": content}
            if "tool_calls" in msg and msg["tool_calls"]:
                out["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                out["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                out["name"] = msg["name"]
            non_system.append(out)

        # Strict tool sequence validation
        strict = self._fix_tool_sequence(non_system)

        # Build final — system at index 0
        system_content = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."
        return [{"role": "system", "content": system_content}] + strict

    @staticmethod
    def _fix_tool_sequence(messages: list) -> list:
        """Fix orphaned tool results that MiniMax rejects with error 2013."""
        strict = []
        for m in messages:
            if m["role"] == "tool":
                # Find last non-tool message
                last_non_tool = None
                for sm in reversed(strict):
                    if sm["role"] != "tool":
                        last_non_tool = sm
                        break
                valid = False
                if (last_non_tool and last_non_tool["role"] == "assistant"
                        and "tool_calls" in last_non_tool):
                    req_id = m.get("tool_call_id")
                    if req_id:
                        for tc in last_non_tool["tool_calls"]:
                            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            if tc_id == req_id:
                                valid = True
                                break
                if not valid:
                    # Orphaned tool result → convert to user message
                    m = dict(m)
                    m["role"] = "user"
                    m["content"] = f"[Tool Result]\n{m.get('content', '')}"
                    m.pop("tool_call_id", None)
                    m.pop("name", None)
            strict.append(m)
        return strict

    def _parse_response(self, raw: dict, model: str) -> CompletionResult:
        """Parse response with think-tag stripping."""
        result = super()._parse_response(raw, model)
        if result.content:
            reasoning = _extract_think_content(result.content)
            if reasoning:
                result.raw["reasoning_content"] = reasoning
            result.content = _strip_think_tags(result.content)
            if not result.content:
                result.content = None
        return result
