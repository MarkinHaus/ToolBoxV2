# core/utils.py
import json
import logging
import threading
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from toolboxv2 import get_logger

# --- Observability Setup (Placeholder) ---
# In a real application, initialize OpenTelemetry here
# from opentelemetry import trace, metrics
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
# ... more setup ...

# For now, provide dummy objects if OpenTelemetry is not installed
try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import SpanKind

    tracer = trace.get_tracer("enhanced_agent")
    meter = metrics.get_meter("enhanced_agent")
    # Basic console exporter for demonstration
    # trace.set_tracer_provider(TracerProvider())
    # trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    print("WARN: opentelemetry-api, opentelemetry-sdk not found. Observability disabled.")


    class DummyTracer:
        def start_as_current_span(self, *args, **kwargs):
            return DummySpan()


    class DummySpan:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_attribute(self, key, value):
            pass

        def record_exception(self, exception):
            pass

        def set_status(self, status):
            pass

        def end(self):
            pass


    class DummyMeter:
        def create_counter(self, *args, **kwargs):
            return DummyCounter()

        def create_histogram(self, *args, **kwargs):
            return DummyHistogram()


    class DummyCounter:
        def add(self, value, attributes=None):
            pass


    class DummyHistogram:
        def record(self, value, attributes=None):
            pass


    tracer = DummyTracer()
    meter = DummyMeter()
    SpanKind = None  # Define dummy SpanKind if needed
    OBSERVABILITY_AVAILABLE = False

# --- World Model (Moved from agent.py for potential reuse) ---
logger_wm = logging.getLogger("WorldModel")
logger_wm.setLevel(get_logger().level)

@dataclass
class WorldModel:
    """Thread-safe representation of the agent's persistent understanding of the world."""
    data: dict[str, Any] = dataclass_field(default_factory=dict)
    _lock: threading.Lock = dataclass_field(default_factory=threading.Lock)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any):
        with self._lock:
            logger_wm.debug(f"WorldModel SET: {key} = {value}")
            self.data[key] = value

    def remove(self, key: str):
        with self._lock:
            if key in self.data:
                logger_wm.debug(f"WorldModel REMOVE: {key}")
                del self.data[key]

    def show(self) -> str:
        with self._lock:
            if not self.data:
                return "[empty]"
            try:
                items = [f"- {k}: {json.dumps(v, indent=None, ensure_ascii=False, default=str)}"
                         for k, v in self.data.items()]
                return "\n".join(items)
            except Exception:
                items = [f"- {k}: {str(v)}" for k, v in self.data.items()]
                return "\n".join(items)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            # Deep copy might be needed if values are mutable and modified externally
            # For simplicity, shallow copy is used here.
            return self.data.copy()

    def update_from_dict(self, data_dict: dict[str, Any]):
        with self._lock:
            self.data.update(data_dict)
            logger_wm.debug(f"WorldModel updated from dict: {list(data_dict.keys())}")


# --- LLM Message (Moved from agent.py) ---
@dataclass
class LLMMessage:
    """Represents a message in a conversation with the LLM."""
    role: str  # "user", "assistant", "system", "tool"
    # Content can be string or list (e.g., multimodal with text/image dicts)
    # Conforms to LiteLLM/OpenAI structure
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None  # For tool responses
    name: str | None = None  # For tool calls/responses (function name)

    def to_dict(self) -> dict:
        """Convert to dictionary, handling potential dataclass nuances."""
        d = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


# --- Internal Agent State (Moved from agent.py) ---
class InternalAgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_A2A = "waiting_for_a2a"
    WAITING_FOR_HUMAN = "waiting_for_human"
    ERROR = "error"




class SafeExecutionResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    error: str | None = None # For execution framework errors, not code errors
    exit_code: int | None = None # 0 for success, non-zero for error in code
