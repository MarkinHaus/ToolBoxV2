from enum import Enum
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid
import time


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


@dataclass
class ProgressEvent:
    """Enhanced progress event with better error handling"""
    event_type: str
    timestamp: float
    node_name: str
    event_id: str = ""

    agent_name: Optional[str] = None
    # Status information
    status: Optional[NodeStatus] = None
    success: Optional[bool] = None
    error_details: Optional[Dict[str, Any]] = None

    # LLM-specific data
    llm_model: Optional[str] = None
    llm_prompt_tokens: Optional[int] = None
    llm_completion_tokens: Optional[int] = None
    llm_total_tokens: Optional[int] = None
    llm_cost: Optional[float] = None
    llm_duration: Optional[float] = None
    llm_temperature: Optional[float] = None

    # Tool-specific data
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    tool_duration: Optional[float] = None
    tool_success: Optional[bool] = None
    tool_error: Optional[str] = None

    # Node/Routing data
    routing_decision: Optional[str] = None
    routing_from: Optional[str] = None
    routing_to: Optional[str] = None
    node_phase: Optional[str] = None
    node_duration: Optional[float] = None

    # Context data
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    plan_id: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
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


class ProgressTracker:
    """Advanced progress tracking with cost calculation"""

    def __init__(self, progress_callback: Optional[callable] = None, agent_name="unknown"):
        self.progress_callback = progress_callback
        self.events: List[ProgressEvent] = []
        self.active_timers: Dict[str, float] = {}

        # Cost tracking (simplified - would need actual provider pricing)
        self.token_costs = {
            "input": 0.00001,  # $0.01/1K tokens input
            "output": 0.00003,  # $0.03/1K tokens output
        }
        self.agent_name = agent_name

    async def emit_event(self, event: ProgressEvent):
        """Emit progress event with callback and storage"""
        self.events.append(event)
        event.agent_name = self.agent_name

        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(event)
                else:
                    self.progress_callback(event)
            except Exception as e:
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

    def calculate_llm_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate approximate LLM cost"""
        # Simplified cost calculation - would need actual provider pricing
        input_cost = (input_tokens / 1000) * self.token_costs["input"]
        output_cost = (output_tokens / 1000) * self.token_costs["output"]
        return input_cost + output_cost

    def get_summary(self) -> Dict[str, Any]:
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
            instructions.append(f"\n## Zusätzliche Anweisungen:")
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
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    result: Any = None
    error: str = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
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
    tasks: List[Task] = field(default_factory=list)
    status: str = "created"  # created, running, paused, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_strategy: str = "sequential"  # sequential, parallel, mixed

@dataclass
class LLMTask(Task):
    """Spezialisierter Task für LLM-Aufrufe"""
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_preference": "fast",  # "fast" | "complex"
        "temperature": 0.7,
        "max_tokens": 1024
    })
    prompt_template: str = ""
    context_keys: List[str] = field(default_factory=list)  # Keys aus shared state
    output_schema: Optional[Dict] = None  # JSON Schema für Validierung


@dataclass
class ToolTask(Task):
    """Spezialisierter Task für Tool-Aufrufe"""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)  # Kann {{ }} Referenzen enthalten
    hypothesis: str = ""  # Was erwarten wir von diesem Tool?
    validation_criteria: str = ""  # Wie validieren wir das Ergebnis?
    expectation: str = ""  # Wie sollte das Ergebnis aussehen?


@dataclass
class DecisionTask(Task):
    """Task für dynamisches Routing"""
    decision_prompt: str = ""  # Kurze Frage an LLM
    routing_map: Dict[str, str] = field(default_factory=dict)  # Ergebnis -> nächster Task
    decision_model: str = "fast"  # Welches LLM für Entscheidung


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
    agent_state: Dict[str, Any]
    task_state: Dict[str, Any]
    world_model: Dict[str, Any]
    active_flows: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEUE: Enhanced checkpoint data for UnifiedContextManager integration
    session_data: Dict[str, Any] = field(default_factory=dict)
    context_manager_state: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    variable_system_state: Dict[str, Any] = field(default_factory=dict)
    results_store: Dict[str, Any] = field(default_factory=dict)
    tool_capabilities: Dict[str, Any] = field(default_factory=dict)
    variable_scopes: Dict[str, Any] = field(default_factory=dict)

    # Optional: Additional system state
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)

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

    def get_storage_size_estimate(self) -> Dict[str, int]:
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

    def validate_checkpoint_integrity(self) -> Dict[str, Any]:
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

    def get_version_info(self) -> Dict[str, str]:
        """Get checkpoint version information"""
        return {
            "checkpoint_version": self.metadata.get("checkpoint_version", "1.0"),
            "data_format": "enhanced" if self.session_data or self.context_manager_state else "basic",
            "context_system": "unified" if self.context_manager_state else "legacy",
            "variable_system": "integrated" if self.variable_system_state else "basic",
            "session_management": "chatsession" if self.session_data else "memory_only",
            "created_with": f"FlowAgent v2.0 Enhanced Context System"
        }

@dataclass
class PersonaConfig:
    name: str
    style: str = "professional"
    personality_traits: List[str] = field(default_factory=lambda: ["helpful", "concise"])
    tone: str = "friendly"
    response_format: str = "direct"
    custom_instructions: str = ""

    format_config: Optional[FormatConfig] = None

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


        except ValueError as e:
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
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    budget_manager: Optional[Any] = None
    caching: bool = True
    persona: Optional[PersonaConfig] = None
    use_fast_response: bool = True

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
    use_cases: List[str] = Field(..., description="Specific use cases for the tool.")
    trigger_phrases: List[str] = Field(..., description="Phrases that should trigger the tool.")
    indirect_connections: List[str] = Field(..., description="Non-obvious connections or applications.")
    complexity_scenarios: List[str] = Field(..., description="Complex scenarios where the tool can be applied.")
    user_intent_categories: List[str] = Field(..., description="Categories of user intent the tool addresses.")
    confidence_triggers: Dict[str, float] = Field(..., description="Phrases mapped to confidence scores.")
    tool_complexity: str = Field(..., description="The complexity of the tool, rated as low, medium, or high.")
    args_schema: Optional[Dict[str, Any]] = Field(..., description="The schema for the tool's arguments.")
