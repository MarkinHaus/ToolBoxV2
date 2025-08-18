from enum import Enum
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import uuid
import time


class ResponseFormat(Enum):
    FREI_TEXT = "frei-text"
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
    response_format: ResponseFormat = ResponseFormat.FREI_TEXT
    text_length: TextLength = TextLength.CHAT_CONVERSATION
    custom_instructions: str = ""
    strict_format_adherence: bool = True
    quality_threshold: float = 0.7

    def get_format_instructions(self) -> str:
        """Generiere Format-spezifische Anweisungen"""
        format_instructions = {
            ResponseFormat.FREI_TEXT: "Verwende natürlichen Fließtext ohne spezielle Formatierung.",
            ResponseFormat.WITH_TABLES: "Integriere Tabellen zur strukturierten Darstellung von Daten. Verwende Markdown-Tabellen.",
            ResponseFormat.WITH_BULLET_POINTS: "Strukturiere Informationen mit Bullet Points (•, -, *) für bessere Lesbarkeit.",
            ResponseFormat.WITH_LISTS: "Verwende nummerierte und unnummerierte Listen zur Organisation von Inhalten.",
            ResponseFormat.TEXT_ONLY: "Nur reiner Text ohne Formatierung, Symbole oder Strukturelemente.",
            ResponseFormat.MD_TEXT: "Vollständige Markdown-Formatierung mit Headings, Code-Blocks, Links etc.",
            ResponseFormat.YAML_TEXT: "Strukturiere Antworten als YAML-Format für maschinenlesbare Ausgabe.",
            ResponseFormat.JSON_TEXT: "Formatiere Antworten als JSON-Struktur für API-Integration.",
            ResponseFormat.PSEUDO_CODE: "Verwende Pseudocode-Struktur für algorithmische oder logische Erklärungen.",
            ResponseFormat.CODE_STRUCTURE: "Strukturiere wie Code mit Einrückungen, Kommentaren und logischen Blöcken."
        }
        return format_instructions.get(self.response_format, "Standard-Formatierung.")

    def get_length_instructions(self) -> str:
        """Generiere Längen-spezifische Anweisungen"""
        length_instructions = {
            TextLength.MINI_CHAT: "Sehr kurze, prägnante Antworten (1-2 Sätze, max 50 Wörter). Chat-Style.",
            TextLength.CHAT_CONVERSATION: "Moderate Gesprächslänge (2-4 Sätze, 50-150 Wörter). Natürlicher Unterhaltungsstil.",
            TextLength.TABLE_CONVERSATION: "Strukturierte, tabellarische Darstellung mit kompakten Erklärungen (100-250 Wörter).",
            TextLength.DETAILED_INDEPTH: "Ausführliche, detaillierte Erklärungen (300-800 Wörter) mit Tiefe und Kontext.",
            TextLength.PHD_LEVEL: "Akademische Tiefe mit umfassenden Erklärungen (800+ Wörter), Quellenangaben und Fachterminologie."
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
            instructions.append("\n## WICHTIG: Halte dich strikt an diese Format- und Längen-Vorgaben!")

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


@dataclass
class CompoundTask(Task):
    """Task der Sub-Tasks gruppiert"""
    sub_task_ids: List[str] = field(default_factory=list)
    execution_strategy: str = "sequential"  # "sequential" | "parallel"
    success_criteria: str = ""  # Wann ist der Compound-Task erfolgreich?


# Erweiterte Task-Erstellung
def create_task(task_type: str, **kwargs) -> Task:
    """Factory für Task-Erstellung mit korrektem Typ"""
    task_classes = {
        "llm_call": LLMTask,
        "tool_call": ToolTask,
        "decision": DecisionTask,
        "compound": CompoundTask,
        "generic": Task,
        "LLMTask": LLMTask,
        "ToolTask": ToolTask,
        "DecisionTask": DecisionTask,
        "CompoundTask": CompoundTask,
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
    timestamp: datetime
    agent_state: Dict[str, Any]
    task_state: Dict[str, Any]
    world_model: Dict[str, Any]
    active_flows: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


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
                base_message += f"\n\n## Persona Instructions\n{persona_addition}"

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
