# file: toolboxv2/mods/llama_lab/omni/types.py
"""Event types exchanged between an OmniBackend and the OmniSession."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OmniEventType(str, Enum):
    AUDIO = "audio"        # model speech out — PCM int16 16k mono (bytes)
    TEXT = "text"          # text token / segment
    TOOL_CALL = "tool_call"
    TURN_END = "turn_end"  # model finished its turn
    ERROR = "error"


@dataclass
class OmniEvent:
    type: OmniEventType
    data: Any = None                       # text str | pcm bytes | error str
    call_id: str = ""                      # TOOL_CALL: id to answer via send_tool_result
    name: str = ""                         # TOOL_CALL: function name
    arguments: dict = field(default_factory=dict)
    final: bool = False                    # TEXT: last chunk of the segment
    encoding: str = "pcm16;rate=16000"     # AUDIO: wire encoding hint

    def wire(self) -> dict:
        """JSON-safe shape for the client WebSocket (audio -> base64)."""
        import base64
        d = {"type": self.type.value, "final": self.final}
        if self.type is OmniEventType.AUDIO and isinstance(self.data, (bytes, bytearray)):
            d["data"] = base64.b64encode(bytes(self.data)).decode()
            d["encoding"] = self.encoding
        elif self.type is OmniEventType.TOOL_CALL:
            d.update(call_id=self.call_id, name=self.name, arguments=self.arguments)
        else:
            d["data"] = self.data
        return d
