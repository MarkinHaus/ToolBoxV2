"""Direct Anthropic API adapter via raw aiohttp. /v1/messages format."""
from __future__ import annotations
import json
import time
from typing import AsyncIterator, TYPE_CHECKING

from ..adapter import ProviderAdapter
from ..types import (
    CompletionResult, StreamChunk, EmbedResult,
    ToolCallData, ToolCallDelta, UsageData, ProviderError,
)
from ..stream_metrics import StreamMetrics

if TYPE_CHECKING:
    import aiohttp

_ANTHROPIC_VERSION = "2023-06-01"

# stop_reason mapping
_STOP_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}


class AnthropicAdapter(ProviderAdapter):
    """Direct Anthropic API via raw aiohttp. /v1/messages format."""

    ALLOWED_PARAMS: frozenset = frozenset({
        'temperature', 'top_p', 'max_tokens', 'stop_sequences',
        'tools', 'tool_choice', 'metadata',
    })

    def __init__(self, base_url: str = "https://api.anthropic.com", **kw):
        super().__init__(base_url=base_url, **kw)

    def build_headers(self, api_key: str) -> dict:
        headers = {"Content-Type": "application/json",
                    "anthropic-version": _ANTHROPIC_VERSION,
                    **self.default_headers}
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def build_payload(self, model: str, messages: list,
                      tools: list | None, **kwargs) -> dict:
        """Convert OpenAI-format messages to Anthropic /v1/messages format."""
        system_parts, anthropic_messages = self._convert_messages(messages)

        payload: dict = {"model": model, "messages": anthropic_messages}
        if system_parts:
            payload["system"] = system_parts

        filtered = self.filter_params(kwargs)
        if tools:
            payload["tools"] = self._convert_tools(tools)
            tc = filtered.pop("tool_choice", None)
            if tc is not None:
                payload["tool_choice"] = self._convert_tool_choice(tc)
        filtered.pop("tools", None)
        filtered.pop("tool_choice", None)

        # max_tokens is required for Anthropic
        if "max_tokens" not in filtered:
            filtered["max_tokens"] = 4096
        payload.update(filtered)
        return payload

    # --- Message conversion ---

    def _convert_messages(self, messages: list) -> tuple[list | str, list]:
        """OpenAI messages → (system, anthropic_messages)."""
        system_parts: list[str] = []
        result: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            if role == "system":
                if isinstance(content, str) and content.strip():
                    system_parts.append(content.strip())
                elif isinstance(content, list):
                    # Pass through structured system content (for cache_control)
                    system_parts.append(content)
                continue

            if role == "assistant":
                blocks = []
                # Text content
                if content:
                    if isinstance(content, str):
                        blocks.append({"type": "text", "text": content})
                    elif isinstance(content, list):
                        blocks.extend(content)

                # Tool calls → tool_use blocks
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args_raw = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {"_raw": args_raw}
                    block = {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    }
                    blocks.append(block)

                if blocks:
                    result.append({"role": "assistant", "content": blocks})
                else:
                    result.append({"role": "assistant", "content": ""})
                continue

            if role == "tool":
                # Tool result → tool_result content block
                tool_result_content = content if content else ""
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": tool_result_content,
                }
                # Merge into existing user message or create new one
                if result and result[-1]["role"] == "user":
                    if isinstance(result[-1]["content"], list):
                        result[-1]["content"].append(block)
                    else:
                        result[-1]["content"] = [
                            {"type": "text", "text": result[-1]["content"]} if result[-1]["content"] else block,
                            block,
                        ]
                        if not result[-1]["content"][0].get("text"):
                            result[-1]["content"] = [block]
                else:
                    result.append({"role": "user", "content": [block]})
                continue

            if role == "user":
                anthropic_content = self._convert_user_content(content)
                # Check for cache_control on the message level
                if isinstance(msg.get("cache_control"), dict):
                    # Apply cache_control to the last content block
                    if isinstance(anthropic_content, list) and anthropic_content:
                        anthropic_content[-1]["cache_control"] = msg["cache_control"]
                result.append({"role": "user", "content": anthropic_content})

        # Build system — if structured content with cache_control, keep as list
        system: list | str = ""
        if len(system_parts) == 1 and isinstance(system_parts[0], str):
            system = system_parts[0]
        elif system_parts:
            flat = []
            for p in system_parts:
                if isinstance(p, str):
                    flat.append({"type": "text", "text": p})
                elif isinstance(p, list):
                    flat.extend(p)
            system = flat

        return system, result

    def _convert_user_content(self, content) -> str | list:
        """Convert user message content to Anthropic format."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            blocks = []
            for part in content:
                if isinstance(part, str):
                    blocks.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype == "text":
                        blocks.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "image_url":
                        url_data = part.get("image_url", {})
                        url = url_data.get("url", "") if isinstance(url_data, dict) else url_data
                        if url.startswith("data:"):
                            media_type, b64 = self._parse_data_url(url)
                            blocks.append({
                                "type": "image",
                                "source": {"type": "base64",
                                           "media_type": media_type, "data": b64},
                            })
                        else:
                            blocks.append({
                                "type": "image",
                                "source": {"type": "url", "url": url},
                            })
                    else:
                        blocks.append(part)  # pass through
            return blocks
        return str(content)

    @staticmethod
    def _parse_data_url(data_url: str) -> tuple[str, str]:
        if not data_url.startswith("data:"):
            return "image/png", data_url
        header, data = data_url.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "")
        return media_type, data

    def _convert_tools(self, tools: list) -> list:
        """OpenAI tool format → Anthropic tool format."""
        result = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                func = t["function"]
                entry = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters",
                                             {"type": "object", "properties": {}}),
                }
            else:
                # Already Anthropic format or flat
                entry = {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema") or t.get("parameters",
                                   {"type": "object", "properties": {}}),
                }
            if "cache_control" in t:
                entry["cache_control"] = t["cache_control"]
            result.append(entry)
        return result

    @staticmethod
    def _convert_tool_choice(tc) -> dict:
        """OpenAI tool_choice → Anthropic tool_choice."""
        if tc == "auto" or (isinstance(tc, dict) and tc.get("type") == "auto"):
            return {"type": "auto"}
        if tc == "none":
            return {"type": "none"}
        if tc == "required" or tc == "any":
            return {"type": "any"}
        if isinstance(tc, dict) and tc.get("type") == "function":
            return {"type": "tool", "name": tc.get("function", {}).get("name", "")}
        if isinstance(tc, dict):
            return tc  # pass through
        return {"type": "auto"}

    # --- Completion ---

    async def complete(self, session: 'aiohttp.ClientSession', api_key: str,
                       model: str, messages: list, tools: list | None,
                       **kwargs) -> CompletionResult:
        url = self.build_url("/v1/messages")
        headers = self.build_headers(api_key)
        payload = self.build_payload(model, messages, tools, **kwargs)

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ProviderError(
                    f"Anthropic HTTP {resp.status}: {body[:300]}",
                    status_code=resp.status, body=body, model=model,
                )
            raw = await resp.json()

        return self._parse_response(raw, model)

    # --- Streaming ---

    async def stream(self, session: 'aiohttp.ClientSession', api_key: str,
                     model: str, messages: list, tools: list | None,
                     metrics: StreamMetrics | None = None,
                     **kwargs) -> AsyncIterator[StreamChunk]:
        url = self.build_url("/v1/messages")
        headers = self.build_headers(api_key)
        payload = self.build_payload(model, messages, tools, **kwargs)
        payload["stream"] = True

        if metrics:
            metrics.t_start = time.perf_counter()

        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ProviderError(
                    f"Anthropic HTTP {resp.status}: {body[:300]}",
                    status_code=resp.status, body=body, model=model,
                )

            # State for Anthropic SSE events
            current_block_index = -1
            current_block_type = ""
            tool_blocks: dict[int, dict] = {}  # index -> {id, name, args}

            buffer = ""
            async for raw_bytes in resp.content.iter_any():
                buffer += raw_bytes.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if line.startswith("event: "):
                        continue  # we parse data lines only

                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    sc = self._handle_sse_event(
                        event, tool_blocks,
                        current_block_index, current_block_type,
                    )

                    # Update block tracking from event
                    etype = event.get("type", "")
                    if etype == "content_block_start":
                        current_block_index = event.get("index", current_block_index + 1)
                        cb = event.get("content_block", {})
                        current_block_type = cb.get("type", "")

                    if sc is None:
                        continue

                    if metrics:
                        metrics.chunk_count += 1
                        if metrics.chunk_count == 1:
                            metrics.t_first_token = time.perf_counter()
                        if sc.usage and sc.usage.completion_tokens:
                            metrics.token_count = sc.usage.completion_tokens

                    yield sc

        if metrics:
            metrics.t_end = time.perf_counter()

    def _handle_sse_event(self, event: dict, tool_blocks: dict,
                          block_index: int, block_type: str) -> StreamChunk | None:
        """Process a single Anthropic SSE event → StreamChunk or None."""
        etype = event.get("type", "")

        if etype == "message_start":
            msg = event.get("message", {})
            usage_raw = msg.get("usage", {})
            return StreamChunk(
                usage=UsageData(
                    prompt_tokens=usage_raw.get("input_tokens", 0),
                    cache_read_tokens=usage_raw.get("cache_read_input_tokens", 0),
                    cache_creation_tokens=usage_raw.get("cache_creation_input_tokens", 0),
                ),
            )

        if etype == "content_block_start":
            cb = event.get("content_block", {})
            idx = event.get("index", 0)
            if cb.get("type") == "tool_use":
                tool_blocks[idx] = {
                    "id": cb.get("id", ""),
                    "name": cb.get("name", ""),
                    "args": "",
                }
                return StreamChunk(
                    tool_call_delta=ToolCallDelta(
                        index=len(tool_blocks) - 1,
                        id=cb.get("id", ""),
                        name=cb.get("name", ""),
                    ),
                )
            return None

        if etype == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type", "")
            idx = event.get("index", block_index)

            if delta_type == "text_delta":
                return StreamChunk(content=delta.get("text", ""))

            if delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                if idx in tool_blocks:
                    tool_blocks[idx]["args"] += partial
                    tool_idx = list(tool_blocks.keys()).index(idx)
                    return StreamChunk(
                        tool_call_delta=ToolCallDelta(
                            index=tool_idx,
                            arguments_delta=partial,
                        ),
                    )
            return None

        if etype == "message_delta":
            delta = event.get("delta", {})
            usage_raw = event.get("usage", {})
            stop = delta.get("stop_reason")
            return StreamChunk(
                finish_reason=_STOP_MAP.get(stop, stop),
                usage=UsageData(
                    completion_tokens=usage_raw.get("output_tokens", 0),
                ),
            )

        # message_stop, content_block_stop, ping — ignore
        return None

    # --- Response parsing ---

    def _parse_response(self, raw: dict, model: str) -> CompletionResult:
        """Parse non-streaming Anthropic /v1/messages response."""
        content_parts = []
        tool_calls = []

        for block in raw.get("content", []):
            btype = block.get("type", "")
            if btype == "text":
                content_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(ToolCallData(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        stop = raw.get("stop_reason", "end_turn")
        usage_raw = raw.get("usage", {})

        return CompletionResult(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=_STOP_MAP.get(stop, stop),
            usage=UsageData(
                prompt_tokens=usage_raw.get("input_tokens", 0),
                completion_tokens=usage_raw.get("output_tokens", 0),
                total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
                cache_read_tokens=usage_raw.get("cache_read_input_tokens", 0),
                cache_creation_tokens=usage_raw.get("cache_creation_input_tokens", 0),
            ),
            model=raw.get("model", model),
            raw=raw,
        )

    # --- Embeddings not supported ---

    async def embed(self, session, api_key, model, texts, **kw):
        raise NotImplementedError("Anthropic does not provide an embeddings API")
