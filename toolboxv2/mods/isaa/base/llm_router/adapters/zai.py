"""Z.AI GLM adapter. Two base URLs depending on plan type."""
from __future__ import annotations
from typing import TYPE_CHECKING

from .openai_compat import OpenAICompatAdapter
from ..types import CompletionResult

if TYPE_CHECKING:
    import aiohttp


class ZAIAdapter(OpenAICompatAdapter):
    """Z.AI GLM models. OpenAI-compatible, two base URLs for free vs coding plan."""

    FREE_BASE = "https://api.z.ai/api/paas/v4"
    CODING_BASE = "https://api.z.ai/api/coding/paas/v4"

    def __init__(self, use_coding_plan: bool = False, **kw):
        base = self.CODING_BASE if use_coding_plan else self.FREE_BASE
        super().__init__(base_url=base, **kw)
        self.use_coding_plan = use_coding_plan

    def _parse_response(self, raw: dict, model: str) -> CompletionResult:
        """Parse response, extracting reasoning_content if present."""
        result = super()._parse_response(raw, model)
        # Optionally extract reasoning_content from the raw response
        choices = raw.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            reasoning = msg.get("reasoning_content")
            if reasoning:
                result.raw["reasoning_content"] = reasoning
        return result
