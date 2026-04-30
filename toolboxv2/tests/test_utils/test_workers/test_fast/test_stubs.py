"""Minimal stubs for testing fast_tb without the full ToolBoxV2 stack."""

import sys
from types import ModuleType
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any, Dict


# --- event_manager stub ---
em = ModuleType("toolboxv2.utils.workers.event_manager")
em.ZMQEventManager = MagicMock
em.Event = MagicMock
em.EventType = MagicMock()
sys.modules["toolboxv2.utils.workers.event_manager"] = em

# --- system.types stub ---
sys_pkg = ModuleType("toolboxv2.utils.system")
sys.modules.setdefault("toolboxv2.utils.system", sys_pkg)

types_mod = ModuleType("toolboxv2.utils.system.types")


@dataclass
class RequestData:
    """Minimal stub for RequestData."""
    request: Dict[str, Any] = None
    session: Dict[str, Any] = None
    session_id: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "RequestData":
        return cls(
            request=data.get("request", {}),
            session=data.get("session", {}),
            session_id=data.get("session_id", ""),
        )


types_mod.RequestData = RequestData
sys.modules["toolboxv2.utils.system.types"] = types_mod

# --- requests stub (server_worker imports it at module level) ---
requests_stub = ModuleType("requests")
requests_stub.get = MagicMock(return_value=MagicMock(json=lambda: {}))
sys.modules.setdefault("requests", requests_stub)

# --- multipart stub ---
mp_stub = ModuleType("multipart")
mp_stub.parse_form_data = MagicMock()
mp_stub.is_form_request = MagicMock()
mp_stub.MultipartPart = MagicMock
sys.modules.setdefault("multipart", mp_stub)
