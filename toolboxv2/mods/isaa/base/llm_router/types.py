"""Unified LLM result types. No external dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class UsageData:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class ToolCallData:
    id: str
    name: str
    arguments: dict  # already parsed, not JSON string


@dataclass
class CompletionResult:
    content: str | None
    tool_calls: list[ToolCallData] | None
    finish_reason: str
    usage: UsageData
    model: str
    raw: dict = field(default_factory=dict)


@dataclass
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class StreamChunk:
    content: str | None = None
    tool_call_delta: ToolCallDelta | None = None
    finish_reason: str | None = None
    usage: UsageData | None = None


@dataclass
class EmbedResult:
    embeddings: list[list[float]]
    usage: UsageData
    model: str


class ProviderError(Exception):
    def __init__(self, message: str, status_code: int, body: str, model: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.model = model


class BudgetExceededError(Exception):
    def __init__(self, scope: str, current: int, limit: int):
        super().__init__(f"Budget exceeded for scope '{scope}': {current}/{limit} tokens")
        self.scope = scope
        self.current = current
        self.limit = limit
