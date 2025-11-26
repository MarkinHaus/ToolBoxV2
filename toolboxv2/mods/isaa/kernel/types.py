"""
ProA Kernel - Proactive Autonomous Kernel
Version: 1.0.0

Transforms the FlowAgent from a reactive tool into a persistent,
event-driven, always-on companion with proactive capabilities.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from collections import deque

VALID_TASK_TYPES = {"reminder", "query", "action", "notification"}

# ===== SIGNAL TYPES AND STRUCTURES =====

class SignalType(Enum):
    """Types of signals that can be processed by the kernel"""
    USER_INPUT = "user_input"  # Direct user interaction
    SYSTEM_EVENT = "system_event"  # Tool results, timers, file changes
    HEARTBEAT = "heartbeat"  # Internal maintenance signal
    ERROR = "error"  # Error conditions
    TOOL_RESULT = "tool_result"  # Specific tool execution results
    CALENDAR_EVENT = "calendar_event"  # Calendar/scheduling events
    EXTERNAL_TRIGGER = "external_trigger"  # External system triggers


@dataclass
class Signal:
    """Unified signal structure for all kernel inputs"""
    id: str
    type: SignalType
    content: Any
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    priority: int = 5  # 0 (low) to 10 (critical)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Enable priority queue sorting (higher priority first)"""
        return self.priority > other.priority


# ===== USER STATE TRACKING =====

class UserState(Enum):
    """Possible states of user engagement"""
    ACTIVE = "active"  # Recently interacted (< 60s)
    IDLE = "idle"  # Connected but quiet (> 5min)
    AWAY = "away"  # No connection / long inactivity
    BUSY = "busy"  # Do Not Disturb mode


@dataclass
class UserContext:
    """Track user state and context"""
    user_id: str
    state: UserState = UserState.IDLE
    last_interaction: float = field(default_factory=time.time)
    location: str = "web"  # web, mobile, desktop, etc.
    do_not_disturb: bool = False
    activity_history: list[tuple[float, str]] = field(default_factory=list)

    def update_interaction(self, activity: str = "input"):
        """Record user interaction"""
        self.last_interaction = time.time()
        self.state = UserState.ACTIVE
        self.activity_history.append((self.last_interaction, activity))

        # Keep only last 100 activities
        if len(self.activity_history) > 100:
            self.activity_history = self.activity_history[-100:]

    def get_idle_time(self) -> float:
        """Get seconds since last interaction"""
        return time.time() - self.last_interaction

    def update_state(self):
        """Update state based on idle time"""
        idle_time = self.get_idle_time()

        if self.do_not_disturb:
            self.state = UserState.BUSY
        elif idle_time < 60:
            self.state = UserState.ACTIVE
        elif idle_time < 300:  # 5 minutes
            self.state = UserState.IDLE
        else:
            self.state = UserState.AWAY


# ===== PROACTIVITY DECISION ENGINE =====

@dataclass
class ProactivityContext:
    """Context for making proactivity decisions"""
    user_state: UserState
    signal: Signal
    last_proactive_time: float
    cooldown_period: float = 300.0  # 5 minutes default
    recent_proactive_count: int = 0


class ProactivityDecision(Enum):
    """Possible proactivity decisions"""
    INTERRUPT = "interrupt"  # Proactively notify user
    QUEUE = "queue"  # Store for later
    SILENT = "silent"  # Process silently
    IGNORE = "ignore"  # Skip processing


class IDecisionEngine(ABC):
    """Abstract interface for proactivity decision making"""

    @abstractmethod
    async def evaluate_proactivity(
        self,
        context: ProactivityContext
    ) -> ProactivityDecision:
        """
        Decide if and how to handle a signal proactively

        Args:
            context: Context containing signal, user state, and history

        Returns:
            ProactivityDecision indicating how to handle the signal
        """
        pass

    @abstractmethod
    async def should_interrupt_user(
        self,
        signal: Signal,
        user_state: UserState
    ) -> bool:
        """
        Quick check if user should be interrupted

        Args:
            signal: The signal to potentially interrupt with
            user_state: Current user state

        Returns:
            True if interruption is warranted
        """
        pass


class DefaultDecisionEngine(IDecisionEngine):
    """Default implementation of proactivity decision logic"""

    # Priority thresholds
    CRITICAL_PRIORITY = 8
    HIGH_PRIORITY = 6
    MEDIUM_PRIORITY = 4

    async def evaluate_proactivity(
        self,
        context: ProactivityContext
    ) -> ProactivityDecision:
        """Evaluate if proactive action is needed"""
        signal = context.signal
        user_state = context.user_state

        # Critical priority always interrupts
        if signal.priority >= self.CRITICAL_PRIORITY:
            return ProactivityDecision.INTERRUPT

        # Never interrupt when busy
        if user_state == UserState.BUSY:
            return ProactivityDecision.QUEUE

        # Don't interrupt active users (unless high priority)
        if user_state == UserState.ACTIVE:
            if signal.priority >= self.HIGH_PRIORITY:
                return ProactivityDecision.INTERRUPT
            return ProactivityDecision.QUEUE

        # For idle users, check cooldown
        if user_state == UserState.IDLE:
            time_since_last = time.time() - context.last_proactive_time

            if time_since_last < context.cooldown_period:
                return ProactivityDecision.QUEUE

            # Too many recent proactive actions?
            if context.recent_proactive_count > 3:
                return ProactivityDecision.QUEUE

            # Good time for medium+ priority
            if signal.priority >= self.MEDIUM_PRIORITY:
                return ProactivityDecision.INTERRUPT

        # Away users: only critical
        if user_state == UserState.AWAY:
            if signal.priority >= self.CRITICAL_PRIORITY:
                return ProactivityDecision.QUEUE
            return ProactivityDecision.SILENT

        return ProactivityDecision.SILENT

    async def should_interrupt_user(
        self,
        signal: Signal,
        user_state: UserState
    ) -> bool:
        """Quick interrupt check"""
        # Critical always interrupts (except busy)
        if signal.priority >= self.CRITICAL_PRIORITY:
            return user_state != UserState.BUSY

        # High priority interrupts idle users
        if signal.priority >= self.HIGH_PRIORITY:
            return user_state == UserState.IDLE

        return False


# ===== STATE MONITOR =====

class IStateMonitor(ABC):
    """Abstract interface for monitoring user and system state"""

    user_contexts: dict[str, UserContext] = {}

    @abstractmethod
    async def get_user_state(self, user_id: str) -> UserState:
        """Get current user state"""
        pass

    @abstractmethod
    async def update_user_activity(
        self,
        user_id: str,
        activity: str = "input"
    ):
        """Record user activity"""
        pass

    @abstractmethod
    async def set_user_location(self, user_id: str, location: str):
        """Update user's current interface location"""
        pass

    @abstractmethod
    async def set_do_not_disturb(self, user_id: str, enabled: bool):
        """Set do-not-disturb mode"""
        pass


class StateMonitor(IStateMonitor):
    """Implementation of state monitoring"""

    def __init__(self):
        self.user_contexts: dict[str, UserContext] = {}

    def _get_or_create_context(self, user_id: str) -> UserContext:
        """Get or create user context"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(user_id=user_id)
        return self.user_contexts[user_id]

    async def get_user_state(self, user_id: str) -> UserState:
        """Get current user state"""
        context = self._get_or_create_context(user_id)
        context.update_state()
        return context.state

    async def update_user_activity(
        self,
        user_id: str,
        activity: str = "input"
    ):
        """Record user activity"""
        context = self._get_or_create_context(user_id)
        context.update_interaction(activity)

    async def set_user_location(self, user_id: str, location: str):
        """Update user's interface location"""
        context = self._get_or_create_context(user_id)
        context.location = location

    async def set_do_not_disturb(self, user_id: str, enabled: bool):
        """Set do-not-disturb mode"""
        context = self._get_or_create_context(user_id)
        context.do_not_disturb = enabled
        context.update_state()

    def get_context(self, user_id: str) -> Optional[UserContext]:
        """Get full user context"""
        return self.user_contexts.get(user_id)


# ===== SIGNAL BUS (Central Entry Point) =====

class ISignalBus(ABC):
    """Abstract interface for signal ingestion and routing"""

    @abstractmethod
    async def emit_signal(self, signal: Signal):
        """Emit a signal into the kernel"""
        pass

    @abstractmethod
    async def get_next_signal(self, timeout: float = None) -> Optional[Signal]:
        """Get next prioritized signal"""
        pass

    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size"""
        pass


class SignalBus(ISignalBus):
    """Implementation of signal bus with priority queue"""

    def __init__(self, max_queue_size: int = 1000):
        self.queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.signal_history: deque = deque(maxlen=100)

    async def emit_signal(self, signal: Signal):
        """Emit a signal into the kernel"""
        try:
            await self.queue.put(signal)
            self.signal_history.append({
                "id": signal.id,
                "type": signal.type.value,
                "priority": signal.priority,
                "timestamp": signal.timestamp
            })
        except asyncio.QueueFull:
            # Drop lowest priority signal if queue is full
            print(f"WARNING: Signal queue full, dropping signal {signal.id}")

    async def get_next_signal(
        self,
        timeout: float = None
    ) -> Optional[Signal]:
        """Get next prioritized signal"""
        try:
            return await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    def get_signal_history(self) -> list[dict]:
        """Get recent signal history"""
        return list(self.signal_history)


# ===== KERNEL CONFIGURATION =====

@dataclass
class KernelConfig:
    """Configuration for ProA Kernel"""
    # Timing
    heartbeat_interval: float = 60.0  # seconds
    idle_threshold: float = 300.0  # 5 minutes
    active_threshold: float = 60.0  # 1 minute

    # Proactivity
    proactive_cooldown: float = 300.0  # 5 minutes between proactive actions
    max_proactive_per_hour: int = 5

    # Queue management
    max_signal_queue_size: int = 1000
    signal_timeout: float = 1.0  # Wait time for signals

    # Resource limits
    max_concurrent_tasks: int = 10
    task_timeout: float = 300.0  # 5 minutes per task


# ===== MAIN KERNEL INTERFACE =====

class IProAKernel(ABC):
    """
    Abstract interface for the ProA Kernel

    The kernel wraps the FlowAgent and provides:
    - Event-driven architecture
    - Proactive capabilities
    - User state awareness
    - Signal prioritization
    - Always-on lifecycle
    """

    @abstractmethod
    async def start(self):
        """Start the kernel lifecycle loop"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the kernel gracefully"""
        pass

    @abstractmethod
    async def handle_user_input(
        self,
        user_id: str,
        content: str,
        metadata: dict = None
    ) -> str:
        """
        Handle direct user input

        Args:
            user_id: User identifier
            content: User's input text
            metadata: Optional metadata (voice flags, etc.)

        Returns:
            Agent's response
        """
        pass

    @abstractmethod
    async def trigger_event(
        self,
        event_name: str,
        payload: dict,
        priority: int = 5,
        source: str = "external"
    ):
        """
        Trigger a system event

        Args:
            event_name: Name of the event
            payload: Event data
            priority: Event priority (0-10)
            source: Event source identifier
        """
        pass

    @abstractmethod
    async def set_user_location(self, user_id: str, location: str):
        """Update user's interface location (web, mobile, etc.)"""
        pass

    @abstractmethod
    async def set_do_not_disturb(self, user_id: str, enabled: bool):
        """Enable/disable do-not-disturb mode"""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Get kernel status and metrics"""
        pass


# ===== KERNEL LIFECYCLE STATES =====

class KernelState(Enum):
    """Possible kernel states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


# ===== EXAMPLE OUTPUT ROUTER =====

class IOutputRouter(ABC):
    """Abstract interface for routing agent outputs"""

    @abstractmethod
    async def send_response(
        self,
        user_id: str,
        content: str,
        role: str = "assistant",
        metadata: dict = None
    ):
        """Send a response to the user"""
        pass

    @abstractmethod
    async def send_notification(
        self,
        user_id: str,
        content: str,
        priority: int = 5,
        metadata: dict = None
    ):
        """Send a proactive notification"""
        pass


class ConsoleOutputRouter(IOutputRouter):
    """Simple console-based output router for testing"""

    async def send_response(
        self,
        user_id: str,
        content: str,
        role: str = "assistant",
        metadata: dict = None
    ):
        """Send response to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {role} -> {user_id}: {content}")

    async def send_notification(
        self,
        user_id: str,
        content: str,
        priority: int = 5,
        metadata: dict = None
    ):
        """Send notification to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        priority_label = "🔴" if priority >= 8 else "🟡" if priority >= 5 else "🟢"
        print(f"[{timestamp}] {priority_label} PROACTIVE -> {user_id}: {content}")


# ===== KERNEL METRICS =====

@dataclass
class KernelMetrics:
    """Metrics for kernel operation"""
    start_time: float = field(default_factory=time.time)
    signals_processed: int = 0
    user_inputs_handled: int = 0
    system_events_handled: int = 0
    proactive_actions: int = 0
    errors: int = 0
    average_response_time: float = 0.0

    def update_response_time(self, response_time: float):
        """Update average response time"""
        n = self.signals_processed
        self.average_response_time = (
            (self.average_response_time * n + response_time) / (n + 1)
        )

    def get_uptime(self) -> float:
        """Get kernel uptime in seconds"""
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "uptime_seconds": self.get_uptime(),
            "signals_processed": self.signals_processed,
            "user_inputs": self.user_inputs_handled,
            "system_events": self.system_events_handled,
            "proactive_actions": self.proactive_actions,
            "errors": self.errors,
            "avg_response_time": self.average_response_time
        }

import asyncio
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
from collections import defaultdict
import traceback

# ===== LEARNING SYSTEM =====

class InteractionType(Enum):
    """Types of interactions to learn from"""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    TOOL_USAGE = "tool_usage"
    ERROR = "error"
    FEEDBACK = "feedback"
    PREFERENCE = "preference"


class LearningRecord(BaseModel):
    """Pydantic model for learning records"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    user_id: str
    interaction_type: InteractionType
    content: dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)
    outcome: Optional[str] = None
    feedback_score: Optional[float] = None  # -1.0 to 1.0


class UserPreferences(BaseModel):
    """Learned user preferences"""
    user_id: str
    communication_style: str = "balanced"  # concise, detailed, balanced
    response_format: str = "text"  # text, bullet-points, structured
    proactivity_level: str = "medium"  # low, medium, high
    preferred_tools: list[str] = Field(default_factory=list)
    time_preferences: dict[str, Any] = Field(default_factory=dict)
    language_preference: str = "en"
    topic_interests: list[str] = Field(default_factory=list)
    learned_patterns: dict[str, Any] = Field(default_factory=dict)
    last_updated: float = Field(default_factory=time.time)



# ===== MEMORY INJECTION SYSTEM =====

class MemoryType(Enum):
    """Types of memories"""
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"
    CONTEXT = "context"
    RELATIONSHIP = "relationship"


class Memory(BaseModel):
    """Individual memory item"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    memory_type: MemoryType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = Field(default_factory=list)


# ===== TASK SCHEDULER =====

class TaskStatus(Enum):
    """Status of scheduled tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledTask(BaseModel):
    """Model for scheduled tasks"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    task_type: str  # reminder, query, action, etc.
    content: str
    scheduled_time: float
    created_at: float = Field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=0, le=10)
    recurrence: Optional[dict[str, Any]] = None  # For recurring tasks
    metadata: dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
