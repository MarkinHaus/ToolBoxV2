"""Compatibility layer: CompletionResult ↔ litellm-style Message objects.

Allows existing FlowAgent/ExecutionEngine code to work with CompletionRouter
without changing tool-call parsing, auto-resume logic, or streaming consumers.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any

from .types import CompletionResult, StreamChunk, ToolCallData, UsageData


# ---------------------------------------------------------------------------
# Shim classes that quack like litellm.types.utils.Message / Function / etc.
# ---------------------------------------------------------------------------

@dataclass
class _Function:
    name: str
    arguments: str  # JSON string, NOT parsed dict


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _Function

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass
class _Message:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None
    function_call: Any = None
    reasoning_content: str | None = None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str
    index: int = 0


@dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def _asdict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class _ModelResponse:
    """Quacks like litellm.ModelResponse — enough for a_run_llm_completion consumers."""
    choices: list[_Choice]
    usage: _Usage
    model: str = ""
    id: str = ""


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _tc_data_to_shim(tc: ToolCallData) -> _ToolCall:
    """ToolCallData (parsed dict args) → _ToolCall (JSON string args)."""
    args_str = json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
    return _ToolCall(
        id=tc.id,
        type="function",
        function=_Function(name=tc.name, arguments=args_str),
    )


def completion_result_to_message(result: CompletionResult) -> _Message:
    """CompletionResult → _Message (drop-in for litellm Message)."""
    tool_calls = None
    if result.tool_calls:
        tool_calls = [_tc_data_to_shim(tc) for tc in result.tool_calls]
    return _Message(
        role="assistant",
        content=result.content,
        tool_calls=tool_calls,
        reasoning_content=result.raw.get("reasoning_content"),
    )


def completion_result_to_model_response(result: CompletionResult) -> _ModelResponse:
    """CompletionResult → _ModelResponse (drop-in for litellm ModelResponse)."""
    msg = completion_result_to_message(result)
    return _ModelResponse(
        choices=[_Choice(message=msg, finish_reason=result.finish_reason)],
        usage=_Usage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        model=result.model,
    )


def usage_to_shim(u: UsageData) -> _Usage:
    return _Usage(
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        total_tokens=u.total_tokens,
    )


# ---------------------------------------------------------------------------
# Stream chunk → litellm-style streaming delta
# ---------------------------------------------------------------------------

@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list | None = None
    reasoning_content: str | None = None


@dataclass
class _StreamChoice:
    delta: _Delta
    finish_reason: str | None = None
    index: int = 0


@dataclass
class _StreamChunkShim:
    """Quacks like a litellm streaming chunk."""
    choices: list[_StreamChoice]
    usage: _Usage | None = None


def stream_chunk_to_shim(chunk: StreamChunk) -> _StreamChunkShim:
    """StreamChunk → litellm-style streaming chunk for _process_streaming_response."""
    tc_list = None
    if chunk.tool_call_delta:
        d = chunk.tool_call_delta
        tc_entry = {
            "index": d.index,
            "id": d.id,
            "function": {
                "name": d.name,
                "arguments": d.arguments_delta,
            },
        }
        tc_list = [tc_entry]

    delta = _Delta(
        content=chunk.content,
        tool_calls=tc_list,
    )
    usage = None
    if chunk.usage:
        usage = usage_to_shim(chunk.usage)

    return _StreamChunkShim(
        choices=[_StreamChoice(delta=delta, finish_reason=chunk.finish_reason)],
        usage=usage,
    )
