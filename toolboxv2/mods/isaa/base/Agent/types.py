import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

class CheckpointConfig(BaseModel):
    """Checkpoint configuration"""
    enabled: bool = True
    interval_seconds: int = 300  # 5 minutes
    max_checkpoints: int = 10
    checkpoint_dir: str = "./checkpoints"
    auto_save_on_exit: bool = True
    auto_load_on_start: bool = True
    max_age_hours: int = 24

class ResponseFormat(Enum):
    FREE_TEXT = "free-text"
    WITH_TABLES = "with-tables"
    WITH_BULLET_POINTS = "with-bullet-points"
    WITH_LISTS = "with-lists"
    TEXT_ONLY = "text-only"
    MD_TEXT = "md-text"
    YAML_TEXT = "yaml-text"
    JSON_TEXT = "json-text"
    PSEUDO_CODE = "pseudo-code"
    CODE_STRUCTURE = "code-structure"


class TextLength(Enum):
    MINI_CHAT = "mini-chat"
    CHAT_CONVERSATION = "chat-conversation"
    TABLE_CONVERSATION = "table-conversation"
    DETAILED_INDEPTH = "detailed-indepth"
    PHD_LEVEL = "phd-level"

class NodeStatus(Enum):
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

@dataclass
class ProgressEvent:

    """Enhanced progress event with better error handling"""

    # === 1. Kern-Attribute (Für jedes Event) ===
    event_type: str
    node_name: str
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None

    # === 2. Status und Ergebnis-Attribute ===
    status: Optional[NodeStatus] = None
    success: Optional[bool] = None
    duration: Optional[float] = None
    error_details: dict[str, Any] = field(default_factory=dict)  # Strukturiert: message, type, traceback

    # === 3. LLM-spezifische Attribute ===
    llm_model: Optional[str] = None
    llm_prompt_tokens: Optional[int] = None
    llm_completion_tokens: Optional[int] = None
    llm_total_tokens: Optional[int] = None
    llm_cost: Optional[float] = None
    llm_input: Optional[Any] = None  # Optional für Debugging, kann groß sein
    llm_output: Optional[str] = None # Optional für Debugging, kann groß sein

    # === 4. Tool-spezifische Attribute ===
    tool_name: Optional[str] = None
    is_meta_tool: Optional[bool] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[Any] = None
    tool_error: Optional[str] = None
    llm_temperature: Optional[float]  = None

    # === 5. Strategie- und Kontext-Attribute ===
    agent_name: Optional[str] = None
    task_id: Optional[str] = None
    plan_id: Optional[str] = None


    # Node/Routing data
    routing_decision: Optional[str] = None
    node_phase: Optional[str] = None
    node_duration: Optional[float] = None

    # === 6. Metadaten (Für alles andere) ===
    metadata: dict[str, Any] = field(default_factory=dict)


    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = time.time()

        if self.metadata is None:
            self.metadata = {}
        if not self.event_id:
            self.event_id = f"{self.node_name}_{self.event_type}_{int(self.timestamp * 1000000)}"
        if 'error' in self.metadata or 'error_type' in self.metadata:
            if self.error_details is None:
                self.error_details = {}
            self.error_details['error'] = self.metadata.get('error')
            self.error_details['error_type'] = self.metadata.get('error_type')
            self.status = NodeStatus.FAILED
        if self.status == NodeStatus.FAILED:
            self.success = False
        if self.status == NodeStatus.COMPLETED:
            self.success = True

    def _to_dict(self) -> dict[str, Any]:
        """Convert ProgressEvent to dictionary with proper handling of all field types"""
        result = {}

        # Get all fields from the dataclass
        for field in fields(self):
            value = getattr(self, field.name)

            # Handle None values
            if value is None:
                result[field.name] = None
                continue

            # Handle NodeStatus enum
            if isinstance(value, NodeStatus | Enum):
                result[field.name] = value.value
            # Handle dataclass objects
            elif is_dataclass(value):
                result[field.name] = asdict(value)
            # Handle dictionaries (recursively process nested enums/dataclasses)
            elif isinstance(value, dict):
                result[field.name] = self._process_dict(value)
            # Handle lists (recursively process nested items)
            elif isinstance(value, list):
                result[field.name] = self._process_list(value)
            # Handle primitive types
            else:
                result[field.name] = value

        return result

    def _process_dict(self, d: dict[str, Any]) -> dict[str, Any]:
        """Recursively process dictionary values"""
        result = {}
        for k, v in d.items():
            if isinstance(v, Enum):
                result[k] = v.value
            elif is_dataclass(v):
                result[k] = asdict(v)
            elif isinstance(v, dict):
                result[k] = self._process_dict(v)
            elif isinstance(v, list):
                result[k] = self._process_list(v)
            else:
                result[k] = v
        return result

    def _process_list(self, lst: list[Any]) -> list[Any]:
        """Recursively process list items"""
        result = []
        for item in lst:
            if isinstance(item, Enum):
                result.append(item.value)
            elif is_dataclass(item):
                result.append(asdict(item))
            elif isinstance(item, dict):
                result.append(self._process_dict(item))
            elif isinstance(item, list):
                result.append(self._process_list(item))
            else:
                result.append(item)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ProgressEvent':
        """Create ProgressEvent from dictionary"""
        # Create a copy to avoid modifying the original
        data_copy = dict(data)

        # Handle NodeStatus enum conversion from string back to enum
        if 'status' in data_copy and data_copy['status'] is not None:
            if isinstance(data_copy['status'], str):
                try:
                    data_copy['status'] = NodeStatus(data_copy['status'])
                except (ValueError, TypeError):
                    # If invalid status value, set to None
                    data_copy['status'] = None

        # Filter out any keys that aren't valid dataclass fields
        field_names = {field.name for field in fields(cls)}
        filtered_data = {k: v for k, v in data_copy.items() if k in field_names}

        # Ensure metadata is properly initialized
        if 'metadata' not in filtered_data or filtered_data['metadata'] is None:
            filtered_data['metadata'] = {}

        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Return event data with None values removed for compact display"""
        data = self._to_dict()

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()
                        if v is not None and v != {} and v != [] and v != ''}
            elif isinstance(d, list):
                cleaned_list = [clean_dict(item) for item in d if item is not None]
                return [item for item in cleaned_list if item != {} and item != []]
            return d

        return clean_dict(data)

    def get_chat_display_data(self) -> dict[str, Any]:
        """Get data optimized for chat view display"""
        filtered = self.filter_none_values()

        # Core fields always shown
        core_data = {
            'event_type': filtered.get('event_type'),
            'node_name': filtered.get('node_name'),
            'timestamp': filtered.get('timestamp'),
            'event_id': filtered.get('event_id'),
            'status': filtered.get('status')
        }

        # Add specific fields based on event type
        if self.event_type == 'outline_created':
            if 'metadata' in filtered:
                core_data['outline_steps'] = len(filtered['metadata'].get('outline', []))
        elif self.event_type == 'reasoning_loop':
            if 'metadata' in filtered:
                core_data.update({
                    'loop_number': filtered['metadata'].get('loop_number'),
                    'outline_step': filtered['metadata'].get('outline_step'),
                    'context_size': filtered['metadata'].get('context_size')
                })
        elif self.event_type == 'tool_call':
            core_data.update({
                'tool_name': filtered.get('tool_name'),
                'is_meta_tool': filtered.get('is_meta_tool')
            })
        elif self.event_type == 'llm_call':
            core_data.update({
                'llm_model': filtered.get('llm_model'),
                'llm_total_tokens': filtered.get('llm_total_tokens'),
                'llm_cost': filtered.get('llm_cost')
            })

        # Remove None values from core_data
        return {k: v for k, v in core_data.items() if v is not None}

    def get_detailed_display_data(self) -> dict[str, Any]:
        """Get complete filtered data for detailed popup view"""
        return self.filter_none_values()

    def get_progress_summary(self) -> str:
        """Get a brief summary for progress sidebar"""
        if self.event_type == 'reasoning_loop' and 'metadata' in self.filter_none_values():
            metadata = self.filter_none_values()['metadata']
            loop_num = metadata.get('loop_number', '?')
            step = metadata.get('outline_step', '?')
            return f"Loop {loop_num}, Step {step}"
        elif self.event_type == 'tool_call':
            tool_name = self.tool_name or 'Unknown Tool'
            return f"{'Meta ' if self.is_meta_tool else ''}{tool_name}"
        elif self.event_type == 'llm_call':
            model = self.llm_model or 'Unknown Model'
            tokens = self.llm_total_tokens
            return f"{model} ({tokens} tokens)" if tokens else model
        else:
            return self.event_type.replace('_', ' ').title()

class ProgressTracker:
    """Advanced progress tracking with cost calculation and memory leak prevention"""

    def __init__(self, progress_callback: callable  = None, agent_name="unknown", max_events: int = 1000):
        self.progress_callback = progress_callback
        self.events: list[ProgressEvent] = []
        self.active_timers: dict[str, float] = {}
        self.max_events = max_events  # Sliding window limit to prevent memory leak

        # Cost tracking (simplified - would need actual provider pricing)
        self.token_costs = {
            "input": 0.00001,  # $0.01/1K tokens input
            "output": 0.00003,  # $0.03/1K tokens output
        }
        self.agent_name = agent_name

    async def emit_event(self, event: ProgressEvent):
        """Emit progress event with callback and storage (sliding window to prevent memory leak)"""
        self.events.append(event)
        event.agent_name = self.agent_name

        # Sliding window: keep only last max_events to prevent memory leak
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(event)
                else:
                    self.progress_callback(event)
            except Exception:
                import traceback
                print(traceback.format_exc())


    def start_timer(self, key: str) -> float:
        """Start timing operation"""
        start_time = time.perf_counter()
        self.active_timers[key] = start_time
        return start_time

    def end_timer(self, key: str) -> float:
        """End timing operation and return duration"""
        if key not in self.active_timers:
            return 0.0
        duration = time.perf_counter() - self.active_timers[key]
        del self.active_timers[key]
        return duration

    def calculate_llm_cost(self, model: str, input_tokens: int, output_tokens: int,completion_response:Any=None) -> float:
        """Calculate approximate LLM cost"""
        cost = (input_tokens / 1000) * self.token_costs["input"] + (output_tokens / 1000) * self.token_costs["output"]
        if hasattr(completion_response, "_hidden_params"):
            cost = completion_response._hidden_params.get("response_cost", 0)
        try:
            import litellm
            cost = litellm.completion_cost(model=model, completion_response=completion_response)
        except ImportError:
            pass
        except Exception as e:
            try:
                import litellm
                cost = litellm.completion_cost(model=model.split('/')[-1], completion_response=completion_response)
            except Exception:
                pass
        return cost or (input_tokens / 1000) * self.token_costs["input"] + (output_tokens / 1000) * self.token_costs["output"]

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive progress summary"""
        summary = {
            "total_events": len(self.events),
            "llm_calls": len([e for e in self.events if e.event_type == "llm_call"]),
            "tool_calls": len([e for e in self.events if e.event_type == "tool_call"]),
            "total_cost": sum(e.llm_cost for e in self.events if e.llm_cost),
            "total_tokens": sum(e.llm_total_tokens for e in self.events if e.llm_total_tokens),
            "total_duration": sum(e.node_duration for e in self.events if e.node_duration),
            "nodes_visited": list(set(e.node_name for e in self.events)),
            "tools_used": list(set(e.tool_name for e in self.events if e.tool_name)),
            "models_used": list(set(e.llm_model for e in self.events if e.llm_model))
        }
        return summary


@dataclass
class FormatConfig:
    """Konfiguration für Response-Format und -Länge"""
    response_format: ResponseFormat = ResponseFormat.FREE_TEXT
    text_length: TextLength = TextLength.CHAT_CONVERSATION
    custom_instructions: str = ""
    strict_format_adherence: bool = True
    quality_threshold: float = 0.7

    def get_format_instructions(self) -> str:
        """Generiere Format-spezifische Anweisungen"""
        format_instructions = {
            ResponseFormat.FREE_TEXT: "Use natural continuous text without special formatting.",
            ResponseFormat.WITH_TABLES: "Integrate tables for structured data representation. Use Markdown tables.",
            ResponseFormat.WITH_BULLET_POINTS: "Structure information with bullet points (•, -, *) for better readability.",
            ResponseFormat.WITH_LISTS: "Use numbered and unnumbered lists to organize content.",
            ResponseFormat.TEXT_ONLY: "Plain text only without formatting, symbols, or structural elements.",
            ResponseFormat.MD_TEXT: "Full Markdown formatting with headings, code blocks, links, etc.",
            ResponseFormat.YAML_TEXT: "Structure responses in YAML format for machine-readable output.",
            ResponseFormat.JSON_TEXT: "Format responses as a JSON structure for API integration.",
            ResponseFormat.PSEUDO_CODE: "Use pseudocode structure for algorithmic or logical explanations.",
            ResponseFormat.CODE_STRUCTURE: "Structure like code with indentation, comments, and logical blocks."
        }
        return format_instructions.get(self.response_format, "Standard-Formatierung.")

    def get_length_instructions(self) -> str:
        """Generiere Längen-spezifische Anweisungen"""
        length_instructions = {
            TextLength.MINI_CHAT: "Very short, concise answers (1–2 sentences, max 50 words). Chat style.",
            TextLength.CHAT_CONVERSATION: "Moderate conversation length (2–4 sentences, 50–150 words). Natural conversational style.",
            TextLength.TABLE_CONVERSATION: "Structured, tabular presentation with compact explanations (100–250 words).",
            TextLength.DETAILED_INDEPTH: "Comprehensive, detailed explanations (300–800 words) with depth and context.",
            TextLength.PHD_LEVEL: "Academic depth with extensive explanations (800+ words), references, and technical terminology."
        }
        return length_instructions.get(self.text_length, "Standard-Länge.")

    def get_combined_instructions(self) -> str:
        """Kombiniere Format- und Längen-Anweisungen"""
        instructions = []
        instructions.append("## Format-Anforderungen:")
        instructions.append(self.get_format_instructions())
        instructions.append("\n## Längen-Anforderungen:")
        instructions.append(self.get_length_instructions())

        if self.custom_instructions:
            instructions.append("\n## Zusätzliche Anweisungen:")
            instructions.append(self.custom_instructions)

        if self.strict_format_adherence:
            instructions.append("\n## ATTENTION: STRICT FORMAT ADHERENCE REQUIRED!")

        return "\n".join(instructions)

    def get_expected_word_range(self) -> tuple[int, int]:
        """Erwartete Wortanzahl für Qualitätsbewertung"""
        ranges = {
            TextLength.MINI_CHAT: (10, 50),
            TextLength.CHAT_CONVERSATION: (50, 150),
            TextLength.TABLE_CONVERSATION: (100, 250),
            TextLength.DETAILED_INDEPTH: (300, 800),
            TextLength.PHD_LEVEL: (800, 2000)
        }
        return ranges.get(self.text_length, (50, 200))

@dataclass
class Task:
    id: str
    type: str
    description: str
    status: str = "pending"  # pending, running, completed, failed, paused
    priority: int = 1
    dependencies: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    result: Any = None
    error: str = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime  = None
    completed_at: datetime  = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    critical: bool = False

    task_identification_attr: bool = True


    def __post_init__(self):
        """Ensure all mutable defaults are properly initialized"""
        if self.metadata is None:
            self.metadata = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.subtasks is None:
            self.subtasks = []

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

@dataclass
class TaskPlan:
    id: str
    name: str
    description: str
    tasks: list[Task] = field(default_factory=list)
    status: str = "created"  # created, running, paused, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_strategy: str = "sequential"  # sequential, parallel, mixed

@dataclass
class LLMTask(Task):
    """Spezialisierter Task für LLM-Aufrufe"""
    llm_config: dict[str, Any] = field(default_factory=lambda: {
        "model_preference": "fast",  # "fast" | "complex"
        "temperature": 0.7,
        "max_tokens": 1024
    })
    prompt_template: str = ""
    context_keys: list[str] = field(default_factory=list)  # Keys aus shared state
    output_schema: dict  = None  # JSON Schema für Validierung


@dataclass
class ToolTask(Task):
    """Spezialisierter Task für Tool-Aufrufe"""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)  # Kann {{ }} Referenzen enthalten
    hypothesis: str = ""  # Was erwarten wir von diesem Tool?
    validation_criteria: str = ""  # Wie validieren wir das Ergebnis?
    expectation: str = ""  # Wie sollte das Ergebnis aussehen?


@dataclass
class DecisionTask(Task):
    """Task für dynamisches Routing"""
    decision_prompt: str = ""  # Kurze Frage an LLM
    routing_map: dict[str, str] = field(default_factory=dict)  # Ergebnis -> nächster Task
    decision_model: str = "fast"  # Welches LLM für Entscheidung


class PlanData(BaseModel):
    """Dataclass for plan data"""
    plan_name: str = Field(..., discription="Name of the plan")
    description: str = Field(..., discription="Description of the plan")
    execution_strategy: str = Field(..., discription="Execution strategy for the plan")
    tasks: list[LLMTask | ToolTask | DecisionTask] = Field(..., discription="List of tasks in the plan")


# Erweiterte Task-Erstellung
def create_task(task_type: str, **kwargs) -> Task:
    """Factory für Task-Erstellung mit korrektem Typ"""
    task_classes = {
        "llm_call": LLMTask,
        "tool_call": ToolTask,
        "decision": DecisionTask,
        "generic": Task,
        "LLMTask": LLMTask,
        "ToolTask": ToolTask,
        "DecisionTask": DecisionTask,
        "Task": Task,
    }

    task_class = task_classes.get(task_type, Task)

    # Standard-Felder setzen
    if "id" not in kwargs:
        kwargs["id"] = str(uuid.uuid4())
    if "type" not in kwargs:
        kwargs["type"] = task_type
    if "critical" not in kwargs:
        kwargs["critical"] = task_type in ["llm_call", "decision"]

    # Ensure metadata is initialized
    if "metadata" not in kwargs:
        kwargs["metadata"] = {}

    # Create task and ensure post_init is called
    task = task_class(**kwargs)

    # Double-check metadata initialization
    if not hasattr(task, 'metadata') or task.metadata is None:
        task.metadata = {}

    return task


@dataclass
class AgentCheckpoint:
    """Enhanced AgentCheckpoint with UnifiedContextManager and ChatSession integration"""
    timestamp: datetime
    agent_state: dict[str, Any]
    task_state: dict[str, Any]
    world_model: dict[str, Any]
    active_flows: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    # NEUE: Enhanced checkpoint data for UnifiedContextManager integration
    session_data: dict[str, Any] = field(default_factory=dict)
    context_manager_state: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    variable_system_state: dict[str, Any] = field(default_factory=dict)
    results_store: dict[str, Any] = field(default_factory=dict)
    tool_capabilities: dict[str, Any] = field(default_factory=dict)
    variable_scopes: dict[str, Any] = field(default_factory=dict)

    # Session-restricted tools map: {tool_name: {session_id: allowed (bool), '*': default_allowed (bool)}}
    session_tool_restrictions: dict[str, dict[str, bool]] = field(default_factory=dict)

    # Optional: Additional system state
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    execution_history: list[dict[str, Any]] = field(default_factory=list)

    def get_checkpoint_summary(self) -> str:
        """Get human-readable checkpoint summary"""
        try:
            summary_parts = []

            # Basic info
            if self.session_data:
                session_count = len([s for s in self.session_data.values() if s.get("status") != "failed"])
                summary_parts.append(f"{session_count} sessions")

            # Task info
            if self.task_state:
                completed_tasks = len([t for t in self.task_state.values() if t.get("status") == "completed"])
                total_tasks = len(self.task_state)
                summary_parts.append(f"{completed_tasks}/{total_tasks} tasks")

            # Conversation info
            if self.conversation_history:
                summary_parts.append(f"{len(self.conversation_history)} messages")

            # Context info
            if self.context_manager_state:
                cache_count = self.context_manager_state.get("cache_entries", 0)
                if cache_count > 0:
                    summary_parts.append(f"{cache_count} cached contexts")

            # Variable system info
            if self.variable_system_state:
                scopes = len(self.variable_system_state.get("scopes", {}))
                summary_parts.append(f"{scopes} variable scopes")

            # Tool capabilities
            if self.tool_capabilities:
                summary_parts.append(f"{len(self.tool_capabilities)} analyzed tools")

            return "; ".join(summary_parts) if summary_parts else "Basic checkpoint"

        except Exception as e:
            return f"Summary generation failed: {str(e)}"

    def get_storage_size_estimate(self) -> dict[str, int]:
        """Estimate storage size of different checkpoint components"""
        try:
            sizes = {}

            # Calculate sizes in bytes (approximate)
            sizes["agent_state"] = len(str(self.agent_state))
            sizes["task_state"] = len(str(self.task_state))
            sizes["world_model"] = len(str(self.world_model))
            sizes["conversation_history"] = len(str(self.conversation_history))
            sizes["session_data"] = len(str(self.session_data))
            sizes["context_manager_state"] = len(str(self.context_manager_state))
            sizes["variable_system_state"] = len(str(self.variable_system_state))
            sizes["results_store"] = len(str(self.results_store))
            sizes["tool_capabilities"] = len(str(self.tool_capabilities))

            sizes["total_bytes"] = sum(sizes.values())
            sizes["total_kb"] = sizes["total_bytes"] / 1024
            sizes["total_mb"] = sizes["total_kb"] / 1024

            return sizes

        except Exception as e:
            return {"error": str(e)}

    def validate_checkpoint_integrity(self) -> dict[str, Any]:
        """Validate checkpoint integrity and completeness"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "completeness_score": 0.0,
            "components_present": []
        }

        try:
            # Check required components
            required_components = ["timestamp", "agent_state", "task_state", "world_model", "active_flows"]
            for component in required_components:
                if hasattr(self, component) and getattr(self, component) is not None:
                    validation["components_present"].append(component)
                else:
                    validation["errors"].append(f"Missing required component: {component}")
                    validation["is_valid"] = False

            # Check optional enhanced components
            enhanced_components = ["session_data", "context_manager_state", "conversation_history",
                                   "variable_system_state", "results_store", "tool_capabilities"]

            for component in enhanced_components:
                if hasattr(self, component) and getattr(self, component):
                    validation["components_present"].append(component)

            # Calculate completeness score
            total_possible = len(required_components) + len(enhanced_components)
            validation["completeness_score"] = len(validation["components_present"]) / total_possible

            # Check timestamp validity
            if isinstance(self.timestamp, datetime):
                age_hours = (datetime.now() - self.timestamp).total_seconds() / 3600
                if age_hours > 24:
                    validation["warnings"].append(f"Checkpoint is {age_hours:.1f} hours old")
            else:
                validation["errors"].append("Invalid timestamp format")
                validation["is_valid"] = False

            # Check session data consistency
            if self.session_data and self.conversation_history:
                session_ids_in_data = set(self.session_data.keys())
                session_ids_in_conversation = set(
                    msg.get("session_id") for msg in self.conversation_history
                    if msg.get("session_id")
                )

                if session_ids_in_data != session_ids_in_conversation:
                    validation["warnings"].append("Session data and conversation history session IDs don't match")

            return validation

        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")
            validation["is_valid"] = False
            return validation

    def get_version_info(self) -> dict[str, str]:
        """Get checkpoint version information"""
        return {
            "checkpoint_version": self.metadata.get("checkpoint_version", "1.0"),
            "data_format": "enhanced" if self.session_data or self.context_manager_state else "basic",
            "context_system": "unified" if self.context_manager_state else "legacy",
            "variable_system": "integrated" if self.variable_system_state else "basic",
            "session_management": "chatsession" if self.session_data else "memory_only",
            "created_with": "FlowAgent v2.0 Enhanced Context System"
        }

@dataclass
class PersonaConfig:
    name: str
    style: str = "professional"
    personality_traits: list[str] = field(default_factory=lambda: ["helpful", "concise"])
    tone: str = "friendly"
    response_format: str = "direct"
    custom_instructions: str = ""

    format_config: FormatConfig  = None

    apply_method: str = "system_prompt"  # "system_prompt" | "post_process" | "both"
    integration_level: str = "light"  # "light" | "medium" | "heavy"

    def to_system_prompt_addition(self) -> str:
        """Convert persona to system prompt addition with format integration"""
        if self.apply_method in ["system_prompt", "both"]:
            additions = []
            additions.append(f"You are {self.name}.")
            additions.append(f"Your communication style is {self.style} with a {self.tone} tone.")

            if self.personality_traits:
                traits_str = ", ".join(self.personality_traits)
                additions.append(f"Your key traits are: {traits_str}.")

            if self.custom_instructions:
                additions.append(self.custom_instructions)

            # Format-spezifische Anweisungen hinzufügen
            if self.format_config:
                additions.append("\n" + self.format_config.get_combined_instructions())

            return " ".join(additions)
        return ""

    def update_format(self, response_format: ResponseFormat|str, text_length: TextLength|str, custom_instructions: str = ""):
        """Dynamische Format-Aktualisierung"""
        try:
            format_enum = ResponseFormat(response_format) if isinstance(response_format, str) else response_format
            length_enum = TextLength(text_length) if isinstance(text_length, str) else text_length

            if not self.format_config:
                self.format_config = FormatConfig()

            self.format_config.response_format = format_enum
            self.format_config.text_length = length_enum

            if custom_instructions:
                self.format_config.custom_instructions = custom_instructions


        except ValueError:
            raise ValueError(f"Invalid format '{response_format}' or length '{text_length}'")

    def should_post_process(self) -> bool:
        """Check if post-processing should be applied"""
        return self.apply_method in ["post_process", "both"]

class AgentModelData(BaseModel):
    name: str = "FlowAgent"
    fast_llm_model: str = "openrouter/anthropic/claude-3-haiku"
    complex_llm_model: str = "openrouter/openai/gpt-4o"
    system_message: str = "You are a production-ready autonomous agent."
    temperature: float = 0.7
    max_tokens: int = 2048
    max_input_tokens: int = 32768
    api_key: str | None  = None
    api_base: str | None  = None
    budget_manager: Any  = None
    caching: bool = True
    persona: PersonaConfig | None = True
    use_fast_response: bool = True
    handler_path_or_dict: str | dict[str, Any] | None = None

    def get_system_message_with_persona(self) -> str:
        """Get system message with persona integration"""
        base_message = self.system_message

        if self.persona and self.persona.apply_method in ["system_prompt", "both"]:
            persona_addition = self.persona.to_system_prompt_addition()
            if persona_addition:
                base_message += f"\n## Persona Instructions\n{persona_addition}"

        return base_message


class ToolAnalysis(BaseModel):
    """Defines the structure for a valid tool analysis."""
    primary_function: str = Field(..., description="The main purpose of the tool.")
    use_cases: list[str] = Field(..., description="Specific use cases for the tool.")
    trigger_phrases: list[str] = Field(..., description="Phrases that should trigger the tool.")
    indirect_connections: list[str] = Field(..., description="Non-obvious connections or applications.")
    complexity_scenarios: list[str] = Field(..., description="Complex scenarios where the tool can be applied.")
    user_intent_categories: list[str] = Field(..., description="Categories of user intent the tool addresses.")
    confidence_triggers: dict[str, float] = Field(..., description="Phrases mapped to confidence scores.")
    tool_complexity: str = Field(..., description="The complexity of the tool, rated as low, medium, or high.")
    args_schema: dict[str, Any] | None = Field(..., description="The schema for the tool's arguments.")


@dataclass
class ChainMetadata:
    """Metadata for stored chains"""
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    author: str = ""
    complexity: str = "simple"  # simple, medium, complex
    agent_count: int = 0
    has_conditionals: bool = False
    has_parallels: bool = False
    has_error_handling: bool = False
