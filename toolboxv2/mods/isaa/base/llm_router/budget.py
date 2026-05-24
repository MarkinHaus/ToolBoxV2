"""BudgetTracker — track and limit LLM usage per scope."""
from __future__ import annotations

from .types import UsageData, BudgetExceededError


class BudgetTracker:
    """Track LLM usage. Limits enforceable per scope."""

    def __init__(self):
        self._usage: dict[str, UsageData] = {}
        self._limits: dict[str, dict] = {}  # scope -> {max_tokens, max_cost_usd}
        self._cost_rates: dict[str, tuple[float, float]] = {}  # model -> (in_$/Mtok, out_$/Mtok)
        self._costs: dict[str, float] = {}  # scope -> cumulative USD

    def set_limit(self, scope: str, max_tokens: int = 0, max_cost_usd: float = 0.0):
        """scope: 'global' | 'groq' | 'groq/llama-3.3-70b' etc."""
        self._limits[scope] = {"max_tokens": max_tokens, "max_cost_usd": max_cost_usd}

    def set_cost_rate(self, model: str, input_per_mtok: float, output_per_mtok: float):
        self._cost_rates[model] = (input_per_mtok, output_per_mtok)

    def _scopes_for(self, model: str) -> list[str]:
        """Return all matching scopes for a model string like 'groq/llama-3.3-70b'."""
        scopes = ["global", model]
        if "/" in model:
            scopes.append(model.split("/", 1)[0])
        return scopes

    def track(self, model: str, usage: UsageData):
        """Called after every completion. Updates all matching scopes."""
        for scope in self._scopes_for(model):
            existing = self._usage.get(scope, UsageData())
            self._usage[scope] = UsageData(
                prompt_tokens=existing.prompt_tokens + usage.prompt_tokens,
                completion_tokens=existing.completion_tokens + usage.completion_tokens,
                total_tokens=existing.total_tokens + usage.total_tokens,
                cache_read_tokens=existing.cache_read_tokens + usage.cache_read_tokens,
                cache_creation_tokens=existing.cache_creation_tokens + usage.cache_creation_tokens,
            )
        # Cost tracking
        if model in self._cost_rates:
            in_rate, out_rate = self._cost_rates[model]
            cost = (usage.prompt_tokens * in_rate + usage.completion_tokens * out_rate) / 1_000_000
            for scope in self._scopes_for(model):
                self._costs[scope] = self._costs.get(scope, 0.0) + cost

    def check(self, model: str, estimated_tokens: int):
        """Pre-call check. Raises BudgetExceededError if any scope would be exceeded."""
        for scope in self._scopes_for(model):
            limit = self._limits.get(scope)
            if not limit:
                continue
            current = self._usage.get(scope, UsageData())
            if limit["max_tokens"] > 0:
                projected = current.total_tokens + estimated_tokens
                if projected > limit["max_tokens"]:
                    raise BudgetExceededError(scope, current.total_tokens, limit["max_tokens"])
            if limit["max_cost_usd"] > 0:
                current_cost = self._costs.get(scope, 0.0)
                if current_cost >= limit["max_cost_usd"]:
                    raise BudgetExceededError(scope, int(current_cost * 100), int(limit["max_cost_usd"] * 100))

    def get_stats(self, scope: str = "global") -> dict:
        usage = self._usage.get(scope, UsageData())
        limit = self._limits.get(scope, {})
        return {
            "total_tokens": usage.total_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_cost_usd": self._costs.get(scope, 0.0),
            "limit_tokens": limit.get("max_tokens", 0),
            "limit_cost_usd": limit.get("max_cost_usd", 0.0),
        }

    def reset(self, scope: str | None = None):
        """Reset usage counters. None = reset all."""
        if scope is None:
            self._usage.clear()
            self._costs.clear()
        else:
            self._usage.pop(scope, None)
            self._costs.pop(scope, None)
