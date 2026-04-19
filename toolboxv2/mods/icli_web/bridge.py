"""
Shared IPC between icli and icli_web subprocess.

Topics (ZMQ pub/sub):
    icli.task.*                  — existing from icli.ingest_chunk
    icli.web.query               — subprocess → icli: run agent
    icli.web.stream.<cid>        — icli → subprocess: streamed chunks
    icli.web.config.update       — either direction: push TTS/STT/VAD config

Messages are JSON. Binary audio goes over WebSocket only, never ZMQ.
"""
from __future__ import annotations

import dataclasses
import json
import uuid
from typing import Any


TOPIC_TASK = "icli.task."
TOPIC_QUERY = "icli.web.query"
TOPIC_STREAM = "icli.web.stream."
TOPIC_CONFIG = "icli.web.config.update"


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def encode(obj: Any) -> bytes:
    """JSON-encode with dataclass support."""
    def default(o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if hasattr(o, "value"):  # Enum
            return o.value
        return str(o)
    return json.dumps(obj, default=default).encode()


def decode(data: bytes) -> Any:
    return json.loads(data.decode())
