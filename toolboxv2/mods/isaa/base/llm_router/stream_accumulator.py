"""StreamAccumulator — accumulate StreamChunks into a CompletionResult."""
from __future__ import annotations
import json

from .types import CompletionResult, StreamChunk, ToolCallData, UsageData


class StreamAccumulator:
    """Accumulate StreamChunks into a CompletionResult."""

    def __init__(self):
        self.reset()

    def feed(self, chunk: StreamChunk):
        if chunk.content:
            self._content.append(chunk.content)
        if chunk.tool_call_delta:
            d = chunk.tool_call_delta
            idx = d.index
            # Grow list if needed
            while len(self._tool_calls) <= idx:
                self._tool_calls.append({"id": "", "name": "", "args": ""})
            tc = self._tool_calls[idx]
            if d.id:
                tc["id"] = d.id
            if d.name:
                tc["name"] = d.name
            tc["args"] += d.arguments_delta
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason
        if chunk.usage:
            self._usage = chunk.usage

    def build(self, model: str = "") -> CompletionResult:
        content = "".join(self._content) or None
        tool_calls = None
        if self._tool_calls:
            tool_calls = []
            for tc in self._tool_calls:
                try:
                    args = json.loads(tc["args"]) if tc["args"] else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc["args"]}
                tool_calls.append(ToolCallData(id=tc["id"], name=tc["name"], arguments=args))
        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._finish_reason or "stop",
            usage=self._usage or UsageData(),
            model=model,
        )

    def reset(self):
        self._content: list[str] = []
        self._tool_calls: list[dict] = []
        self._finish_reason: str | None = None
        self._usage: UsageData | None = None
