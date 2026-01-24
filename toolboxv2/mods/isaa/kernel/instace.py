"""
ProA Kernel - Autonomous Event-Driven System
Version: 2.1.0

Architecture:
- Perception Layer: Event → PerceivedEvent (RuleSet.get_groups_for_intent)
- World Model: User + Environment State (SessionManager.sessions)
- Attention System: Salience Scoring (RuleSet.match_rules)
- Decision Engine: Act/Schedule/Queue/Observe (RuleSet.rule_on_action)
- Learning Loop: Pattern Detection (RuleSet.learn_pattern, add_rule)

Uses FlowAgent + SessionManager + RuleSet - no duplication.
"""

import asyncio
import pickle
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from toolboxv2.mods.isaa.kernel.types import (
    Signal, SignalType, SignalBus, ISignalBus,
    UserState, UserContext, StateMonitor, IStateMonitor,
    ProactivityContext, ProactivityDecision,
    IDecisionEngine, DefaultDecisionEngine,
    IProAKernel, IOutputRouter, ConsoleOutputRouter,
    KernelConfig, KernelState, KernelMetrics,
    InteractionType, LearningRecord, UserPreferences,
    Memory, MemoryType, ScheduledTask, TaskStatus
)

from toolboxv2.mods.isaa.kernel.models import (
    ContextStore, ProactiveActionTracker,
    LearningEngine, MemoryStore, TaskScheduler,
    AgentIntegrationLayer
)

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.agent_session import AgentSession
    from toolboxv2.mods.isaa.base.Agent.rule_set import RuleSet


# =============================================================================
# PERCEPTION LAYER
# =============================================================================

@dataclass
class PerceivedEvent:
    """Normalized event with extracted features."""
    raw_signal: Signal
    user_id: str

    # Extracted
    intent: str = ""
    entities: list[str] = field(default_factory=list)
    urgency: float = 0.5
    topic_tags: list[str] = field(default_factory=list)

    # From RuleSet
    matching_tool_groups: list[str] = field(default_factory=list)
    matching_rules: list[str] = field(default_factory=list)
    relevant_patterns: list[str] = field(default_factory=list)


class PerceptionLayer:
    """Transforms raw signals into perceived events using RuleSet."""

    def __init__(self, kernel: 'Kernel'):
        self.kernel = kernel

    async def perceive(self, signal: Signal, session: 'AgentSession') -> PerceivedEvent:
        """Extract features from signal using session's RuleSet."""
        user_id = signal.metadata.get("user_id", "default")
        content = str(signal.content)

        # Quick intent extraction via heuristics
        intent = self._extract_intent(content)
        entities = self._extract_entities(content)
        urgency = self._compute_urgency(signal, content)

        # Use RuleSet for tool group matching
        rule_set = session.rule_set
        tool_groups = [g.name for g in rule_set.get_groups_for_intent(intent)]

        # Get matching rules
        situation = rule_set.current_situation or "general"
        matched = rule_set.match_rules(situation, intent)
        matching_rules = [r.id for r in matched[:3]]

        # Get relevant patterns
        patterns = [p.pattern for p in rule_set.get_relevant_patterns(intent, limit=5)]

        return PerceivedEvent(
            raw_signal=signal,
            user_id=user_id,
            intent=intent,
            entities=entities,
            urgency=urgency,
            topic_tags=self._extract_topics(content),
            matching_tool_groups=tool_groups,
            matching_rules=matching_rules,
            relevant_patterns=patterns
        )

    def _extract_intent(self, content: str) -> str:
        """Heuristic intent extraction."""
        content_lower = content.lower()

        # Question detection
        if any(q in content_lower for q in ["?", "what", "how", "why", "when", "where", "who"]):
            return "question"

        # Command detection
        if any(c in content_lower for c in ["create", "make", "build", "generate"]):
            return "create"
        if any(c in content_lower for c in ["delete", "remove", "clear"]):
            return "delete"
        if any(c in content_lower for c in ["update", "change", "modify", "edit"]):
            return "update"
        if any(c in content_lower for c in ["find", "search", "look", "get"]):
            return "search"
        if any(c in content_lower for c in ["remind", "schedule", "later", "tomorrow"]):
            return "schedule"
        if any(c in content_lower for c in ["help", "assist", "support"]):
            return "help"

        return "general"

    def _extract_entities(self, content: str) -> list[str]:
        """Extract key entities (simple word extraction)."""
        import re
        entities = []

        # Extract quoted strings first
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', content)
        for match in quoted:
            entities.append(match[0] or match[1])

        # Remove quoted parts for word extraction
        clean = re.sub(r'"[^"]+"|\'[^\']+\'', '', content)

        # Extract capitalized words
        for w in clean.split():
            w_clean = w.strip('.,!?;:')
            if len(w_clean) > 2 and w_clean[0].isupper() and w_clean not in entities:
                entities.append(w_clean)

        return entities[:10]

    def _extract_topics(self, content: str) -> list[str]:
        """Extract topic tags."""
        topics = []
        keywords = {
            "code": ["code", "program", "script", "function", "class"],
            "file": ["file", "document", "folder", "directory"],
            "api": ["api", "endpoint", "request", "response"],
            "data": ["data", "database", "sql", "query"],
            "web": ["web", "website", "html", "css", "javascript"],
        }
        content_lower = content.lower()
        for topic, kws in keywords.items():
            if any(kw in content_lower for kw in kws):
                topics.append(topic)
        return topics

    def _compute_urgency(self, signal: Signal, content: str) -> float:
        """Compute urgency score 0.0-1.0."""
        urgency = signal.priority / 10.0

        # Boost for urgent keywords
        urgent_words = ["urgent", "asap", "immediately", "critical", "emergency", "now"]
        if any(w in content.lower() for w in urgent_words):
            urgency = min(1.0, urgency + 0.3)

        return urgency


# =============================================================================
# WORLD MODEL
# =============================================================================

@dataclass
class UserModel:
    """Dynamic model of a user."""
    user_id: str

    # Activity tracking
    activity_rhythm: dict[int, float] = field(default_factory=dict)  # hour -> activity_prob
    interaction_count: int = 0
    last_interaction: float = field(default_factory=time.time)

    # Inferred
    preferred_response_style: str = "balanced"
    topics_of_interest: list[str] = field(default_factory=list)
    engagement_level: float = 0.5

    def update_activity(self, hour: int = None):
        """Update activity rhythm."""
        hour = hour or datetime.now().hour
        current = self.activity_rhythm.get(hour, 0.0)
        self.activity_rhythm[hour] = current * 0.9 + 0.1
        self.interaction_count += 1
        self.last_interaction = time.time()

    def is_likely_available(self) -> bool:
        """Check if user is likely available based on rhythm."""
        hour = datetime.now().hour
        return self.activity_rhythm.get(hour, 0.5) > 0.3


class WorldModel:
    """Maintains world state from SessionManager."""

    def __init__(self, kernel: 'Kernel'):
        self.kernel = kernel
        self.users: dict[str, UserModel] = {}

    def get_user(self, user_id: str) -> UserModel:
        """Get or create user model."""
        if user_id not in self.users:
            self.users[user_id] = UserModel(user_id=user_id)
            self.users[user_id].update_activity()
        return self.users[user_id]

    def get_active_sessions(self) -> list[str]:
        """Get active session IDs from SessionManager."""
        return self.kernel.agent.session_manager.list_sessions()

    def get_session_stats(self, user_id: str) -> dict:
        """Get session statistics."""
        session = self.kernel.agent.session_manager.get(user_id)
        if session:
            return session.get_stats()
        return {}

    async def update_from_event(self, event: PerceivedEvent, success: bool):
        """Update world model after interaction."""
        user = self.get_user(event.user_id)
        user.update_activity()

        # Update topics
        for topic in event.topic_tags:
            if topic not in user.topics_of_interest:
                user.topics_of_interest.append(topic)
        user.topics_of_interest = user.topics_of_interest[-20:]  # Keep last 20

        # Update engagement
        if success:
            user.engagement_level = min(1.0, user.engagement_level + 0.05)
        else:
            user.engagement_level = max(0.0, user.engagement_level - 0.1)


# =============================================================================
# ATTENTION SYSTEM
# =============================================================================

@dataclass
class SalienceScore:
    """How important is this event?"""
    score: float
    reasons: list[str] = field(default_factory=list)
    should_interrupt: bool = False


class AttentionSystem:
    """Determines event importance using RuleSet."""

    def __init__(self, kernel: 'Kernel'):
        self.kernel = kernel

    def compute_salience(
        self,
        event: PerceivedEvent,
        user_model: UserModel,
        session: 'AgentSession'
    ) -> SalienceScore:
        """Compute salience score."""
        score = 0.0
        reasons = []

        # 1. Explicit urgency
        if event.urgency > 0.7:
            score += 0.3 + (event.urgency/10)
            reasons.append(f"High urgency: {event.urgency:.0%}")

        # 2. Rule match strength (from RuleSet)
        if event.matching_rules:
            rule = session.rule_set.get_rule(event.matching_rules[0])
            if rule and rule.confidence > 0.7:
                score += 0.25
                reasons.append(f"Strong rule match: {rule.id}")

        # 3. User engagement
        if user_model.engagement_level > 0.6:
            score += 0.15
            reasons.append("High user engagement")

        # 4. Topic relevance
        if any(t in user_model.topics_of_interest for t in event.topic_tags):
            score += 0.15
            reasons.append("Matches user interests")

        # 5. Tool group availability
        if event.matching_tool_groups:
            score += 0.1
            reasons.append(f"Tools available: {event.matching_tool_groups[:2]}")


        return SalienceScore(
            score=min(1.0, max(0.0, score)),
            reasons=reasons,
            should_interrupt=score > 0.5 or event.raw_signal.type.value == "user_input"
        )


# =============================================================================
# AUTONOMOUS DECISION ENGINE
# =============================================================================

class AutonomousDecision(Enum):
    """Kernel decision types."""
    ACT_NOW = "act_now"
    SCHEDULE = "schedule"
    QUEUE = "queue"
    OBSERVE = "observe"
    IGNORE = "ignore"


class ActionType(Enum):
    """How to act."""
    RESPOND = "respond"
    NOTIFY = "notify"
    INITIATE = "initiate"
    BACKGROUND = "background"
    ASK = "ask"


@dataclass
class ActionPlan:
    """What to do and how."""
    decision: AutonomousDecision
    action_type: ActionType
    content: str
    tool_groups: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    schedule_at: Optional[float] = None
    confidence: float = 0.5
    reasoning: list[str] = field(default_factory=list)


class AutonomousDecisionEngine:
    """Decision engine using RuleSet."""

    def __init__(self, kernel: 'Kernel'):
        self.kernel = kernel

    async def decide(
        self,
        event: PerceivedEvent,
        salience: SalienceScore,
        user_model: UserModel,
        session: 'AgentSession'
    ) -> ActionPlan:
        """Decide what to do."""
        reasoning = []
        rule_set = session.rule_set

        # Check RuleSet for action permission
        rule_result = rule_set.rule_on_action(
            action=event.intent,
            context={"urgency": event.urgency, "user_available": user_model.is_likely_available()}
        )

        if not rule_result.allowed:
            reasoning.append(f"Blocked by rule: {rule_result.required_steps}")
            return ActionPlan(
                decision=AutonomousDecision.OBSERVE,
                action_type=ActionType.BACKGROUND,
                content="",
                instructions=rule_result.instructions,
                confidence=0.9,
                reasoning=reasoning
            )

        # Determine proactivity based on salience
        if salience.should_interrupt:
            decision = AutonomousDecision.ACT_NOW
            reasoning.extend(salience.reasons)
        elif salience.score > 0.4:
            decision = AutonomousDecision.QUEUE
            reasoning.append("Medium salience - queue for later")
        elif salience.score > 0.2:
            decision = AutonomousDecision.OBSERVE
            reasoning.append("Low salience - observe only")
        else:
            decision = AutonomousDecision.IGNORE
            reasoning.append("Very low salience")

        # Determine action type
        signal_type = event.raw_signal.type
        if signal_type == SignalType.USER_INPUT:
            action_type = ActionType.RESPOND
        elif signal_type == SignalType.SYSTEM_EVENT:
            action_type = ActionType.NOTIFY if salience.should_interrupt else ActionType.BACKGROUND
        else:
            action_type = ActionType.RESPOND

        # Collect instructions from matched rules
        instructions = list(rule_result.instructions)
        for rule_id in event.matching_rules[:2]:
            rule = rule_set.get_rule(rule_id)
            if rule:
                instructions.extend(rule.instructions)

        return ActionPlan(
            decision=decision,
            action_type=action_type,
            content=str(event.raw_signal.content),
            tool_groups=event.matching_tool_groups,
            instructions=list(set(instructions)),
            confidence=salience.score,
            reasoning=reasoning
        )


# =============================================================================
# LEARNING LOOP
# =============================================================================

@dataclass
class InteractionOutcome:
    """Outcome of an action for learning."""
    event: PerceivedEvent
    plan: ActionPlan
    success: bool
    response_time: float
    user_feedback: Optional[str] = None
    feedback_score: float = 0.0


class LearningLoop:
    """Learns from outcomes using RuleSet."""

    def __init__(self, kernel: 'Kernel'):
        self.kernel = kernel
        self._buffer: list[InteractionOutcome] = []
        self._pattern_threshold = 10

    async def record_outcome(
        self,
        outcome: InteractionOutcome,
        session: 'AgentSession'
    ):
        """Record and learn from outcome."""
        rule_set = session.rule_set

        # Update rule confidence
        for rule_id in outcome.event.matching_rules:
            if outcome.success:
                rule_set.record_rule_success(rule_id)
            else:
                rule_set.record_rule_failure(rule_id)

        # Buffer for pattern detection
        self._buffer.append(outcome)

        # Trigger pattern detection periodically
        if len(self._buffer) >= self._pattern_threshold:
            await self._detect_patterns(session)

    async def _detect_patterns(self, session: 'AgentSession'):
        """Detect patterns and generate rules."""
        rule_set = session.rule_set

        # Group by intent
        by_intent: dict[str, list[InteractionOutcome]] = {}
        for outcome in self._buffer:
            by_intent.setdefault(outcome.event.intent, []).append(outcome)

        for intent, outcomes in by_intent.items():
            successes = [o for o in outcomes if o.success]

            if len(successes) >= 3:
                # Find common tool groups in successes
                tool_counts: dict[str, int] = {}
                for o in successes:
                    for tg in o.plan.tool_groups:
                        tool_counts[tg] = tool_counts.get(tg, 0) + 1

                if tool_counts:
                    best_tool = max(tool_counts.items(), key=lambda x: x[1])[0]

                    # Learn pattern
                    pattern = f"For '{intent}' tasks, {best_tool} tools are effective"
                    rule_set.learn_pattern(
                        pattern=pattern,
                        source_situation=intent,
                        confidence=min(0.8, len(successes) * 0.1),
                        category="tool_preference",
                        tags=[intent, best_tool]
                    )

                # Generate rule if high confidence
                if len(successes) >= 5:
                    rule_set.add_rule(
                        situation="general",
                        intent=intent,
                        instructions=[f"Consider using {best_tool} tools", "Approach has proven effective"],
                        required_tool_groups=[best_tool] if tool_counts else [],
                        learned=True,
                        confidence=min(0.8, len(successes) * 0.1)
                    )

        # Detect time-based patterns
        await self._detect_time_patterns(session)

        # Clear buffer
        self._buffer.clear()

    async def _detect_time_patterns(self, session: 'AgentSession'):
        """Detect temporal patterns."""
        rule_set = session.rule_set

        # Group by hour
        by_hour: dict[int, list[InteractionOutcome]] = {}
        for outcome in self._buffer:
            ts = outcome.event.raw_signal.timestamp
            hour = datetime.fromtimestamp(ts).hour
            by_hour.setdefault(hour, []).append(outcome)

        for hour, outcomes in by_hour.items():
            if len(outcomes) >= 3:
                # Find common intents at this hour
                intent_counts: dict[str, int] = {}
                for o in outcomes:
                    intent_counts[o.event.intent] = intent_counts.get(o.event.intent, 0) + 1

                if intent_counts:
                    common_intent = max(intent_counts.items(), key=lambda x: x[1])
                    if common_intent[1] >= 2:
                        pattern = f"User often requests '{common_intent[0]}' around {hour}:00"
                        rule_set.learn_pattern(
                            pattern=pattern,
                            source_situation="temporal",
                            confidence=min(0.7, common_intent[1] * 0.15),
                            category="temporal",
                            tags=["time", f"hour_{hour}", common_intent[0]]
                        )


# =============================================================================
# MAIN KERNEL
# =============================================================================

class Kernel(IProAKernel):
    """Autonomous event-driven kernel."""

    def __init__(
        self,
        agent: 'FlowAgent',
        config: KernelConfig = None,
        decision_engine: IDecisionEngine = None,
        output_router: IOutputRouter = None
    ):
        self.agent = agent

        self.config = config or KernelConfig()
        self.legacy_decision_engine = decision_engine or DefaultDecisionEngine()
        self.output_router = output_router or ConsoleOutputRouter()

        # Core systems
        self.signal_bus: ISignalBus = SignalBus(max_queue_size=self.config.max_signal_queue_size)
        self.state_monitor: IStateMonitor = StateMonitor()
        self.context_store = ContextStore()

        # Autonomous layers
        self.perception = PerceptionLayer(self)
        self.world_model = WorldModel(self)
        self.attention = AttentionSystem(self)
        self.decision_engine = AutonomousDecisionEngine(self)
        self.learning_loop = LearningLoop(self)

        # Extended systems (from models.py)
        self.learning_engine = LearningEngine(agent)
        self.memory_store = MemoryStore()
        self.scheduler = TaskScheduler(self)
        self.integration = AgentIntegrationLayer(self)

        # State
        self.state = KernelState.STOPPED
        self.metrics = KernelMetrics()
        self.proactive_tracker = ProactiveActionTracker()
        self.running = False
        self._current_user_id: Optional[str] = None
        self._pending_questions: dict[str, asyncio.Future] = {}

        # Tasks
        self.main_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

        self._register_tools()

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self):
        """Start kernel."""

        if self.state == KernelState.RUNNING:
            return

        self.state = KernelState.STARTING
        self.running = True
        await self.scheduler.start()
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.main_task = asyncio.create_task(self._main_loop())

        self.state = KernelState.RUNNING

    async def stop(self):
        """Stop kernel."""
        if self.state == KernelState.STOPPED:
            return

        self.state = KernelState.STOPPING
        self.running = False

        await self.scheduler.stop()
        for task in [self.heartbeat_task, self.main_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await self.agent.close()
        self.state = KernelState.STOPPED

    def _register_tools(self):
        """Register kernel tools with proper category and flags."""
        print("Registering kernel tools...")
        kernel_tools = [
            (
                self.integration.schedule_task,
                "kernel_schedule_task",
                (
                    "Plant eine Aufgabe oder Erinnerung im Kernel-Scheduler.\n\n"
                    "MUSS:\n"
                    "- task_type (str): Typ der Aufgabe, z. B. 'reminder', 'job', 'follow_up'\n"
                    "- content (str): Inhalt oder Beschreibung der Aufgabe\n\n"
                    "OPTIONAL:\n"
                    "- delay_seconds (float): Verzögerung in Sekunden ab jetzt\n"
                    "- scheduled_time (float): Absoluter Unix-Timestamp für die Ausführung\n"
                    "- priority (int, Standard=5): Priorität (höher = wichtiger)\n\n"
                    "HINWEISE:\n"
                    "- Entweder delay_seconds ODER scheduled_time verwenden\n"
                    "- Erzeugt persistente Seiteneffekte (Task wird gespeichert)\n"
                    "- Gibt eine task_id zurück"
                ),
                ["kernel", "scheduling"],
                {"side_effect": True, "persistent": True}
            ),

            (
                self.integration.send_intermediate_response,
                "kernel_send_intermediate",
                (
                    "Sendet eine Zwischenmeldung an den Nutzer während laufender Verarbeitung.\n\n"
                    "MUSS:\n"
                    "- content (str): Text der Statusmeldung\n\n"
                    "OPTIONAL:\n"
                    "- stage (str, Standard='processing'): Verarbeitungsphase "
                    "(z. B. 'analysis', 'loading', 'thinking')\n\n"
                    "HINWEISE:\n"
                    "- Unterbricht die Ausführung NICHT\n"
                    "- Wird bevorzugt für lange oder mehrstufige Agentenprozesse genutzt\n"
                    "- Fallback auf Notification, falls kein Intermediate-Channel existiert"
                ),
                ["kernel", "communication"],
                {"intermediate": True}
            ),

            (
                self.integration.ask_user,
                "kernel_ask_user",
                (
                    "Stellt dem Nutzer eine explizite Frage und wartet auf eine Antwort.\n\n"
                    "MUSS:\n"
                    "- question (str): Die zu stellende Frage\n\n"
                    "OPTIONAL:\n"
                    "- timeout (float, Standard=300.0): Maximale Wartezeit in Sekunden\n\n"
                    "HINWEISE:\n"
                    "- Pausiert die Agentenausführung bis Antwort oder Timeout\n"
                    "- Gibt die Nutzerantwort als String zurück\n"
                    "- Gibt None zurück, wenn das Timeout erreicht wird\n"
                    "- Sollte sparsam eingesetzt werden (User-Interaktion!)"
                ),
                ["kernel", "communication"],
                {"pauses_execution": True}
            ),

            (
                self.integration.inject_memory,
                "kernel_inject_memory",
                (
                    "Speichert gezielt Wissen über den Nutzer im Memory-System.\n\n"
                    "MUSS:\n"
                    "- content (str): Zu speichernde Information (Fakt, Präferenz, Kontext)\n\n"
                    "OPTIONAL:\n"
                    "- memory_type (str, Standard='fact'): 'fact', 'preference', 'context'\n"
                    "- importance (float, Standard=0.5): Relevanz (0.0 – 1.0)\n"
                    "- tags (list[str]): Freie Tags zur späteren Filterung\n\n"
                    "HINWEISE:\n"
                    "- Erzeugt persistente Seiteneffekte\n"
                    "- Wird für Personalisierung und Langzeitlernen verwendet\n"
                    "- Sollte nur bei stabilen, verlässlichen Informationen genutzt werden"
                ),
                ["kernel", "memory"],
                {"side_effect": True}
            ),

            (
                self.integration.get_user_preferences,
                "kernel_get_preferences",
                (
                    "Liest die aktuell gelernten Nutzerpräferenzen aus dem Learning-System.\n\n"
                    "MUSS:\n"
                    "- Keine Argumente\n\n"
                    "RÜCKGABE:\n"
                    "- dict mit Präferenzen (z. B. Kommunikationsstil, Detailgrad)\n\n"
                    "HINWEISE:\n"
                    "- Read-only (keine Seiteneffekte)\n"
                    "- Sollte vor Antwortgenerierung zur Personalisierung genutzt werden"
                ),
                ["kernel", "memory"],
                {"readonly": True}
            ),

            (
                self.integration.record_feedback,
                "kernel_record_feedback",
                (
                    "Speichert explizites Feedback zur Verbesserung des Lernsystems.\n\n"
                    "MUSS:\n"
                    "- feedback (str): Textuelles Feedback\n"
                    "- score (float): Bewertung (z. B. -1.0 schlecht bis +1.0 gut)\n\n"
                    "HINWEISE:\n"
                    "- Erzeugt Seiteneffekte im Learning-System\n"
                    "- Wird zur Anpassung zukünftiger Antworten genutzt\n"
                    "- Sollte Feedback zur Qualität oder Relevanz widerspiegeln"
                ),
                ["kernel", "learning"],
                {"side_effect": True}
            ),
        ]

        for func, name, desc, category, flags in kernel_tools:
            print("Registering tool:", self.agent.tool_manager.register)
            self.agent.tool_manager.register(func, name, description=desc, category=category, flags=flags)

    # =========================================================================
    # MAIN LOOPS
    # =========================================================================

    async def _main_loop(self):
        """Process signals with autonomous pipeline."""
        print("Kernel started")
        while self.running:
            try:
                signal = await self.signal_bus.get_next_signal(timeout=self.config.signal_timeout)
                if signal:
                    await self._process_signal_autonomous(signal)
                else:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:

                print("CancelledError in main loop:")
                break
            except Exception as e:
                print("Error in main loop:", e)
                self.metrics.errors += 1
        print("Kernel stopped")

    async def _heartbeat_loop(self):
        """Maintenance heartbeat."""
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                await self._handle_heartbeat()
            except asyncio.CancelledError:
                break

    # =========================================================================
    # AUTONOMOUS SIGNAL PROCESSING
    # =========================================================================

    async def _process_signal_autonomous(self, signal: Signal):
        """Full autonomous pipeline: Perceive → Attend → Decide → Act → Learn."""
        start = time.time()
        self.metrics.signals_processed += 1

        user_id = signal.metadata.get("user_id", "default")
        self._current_user_id = user_id

        try:
            # Get or create session
            session = await self.agent.session_manager.get_or_create(user_id)
            # 1. PERCEIVE
            event = await self.perception.perceive(signal, session)
            # 2. GET USER MODEL
            user_model = self.world_model.get_user(user_id)
            if signal.source.startswith("user_"):
                user_model.update_activity()
            # 3. ATTEND (compute salience)
            salience = self.attention.compute_salience(event, user_model, session)
            # 4. DECIDE
            plan = await self.decision_engine.decide(event, salience, user_model, session)
            print("Plan:", plan)
            # 5. ACT
            self.agent.init_session_tools(session)
            print("Session init")
            success, response = await self._execute_plan(event, plan, session)
            print(f"Success: {success}, Response: {response}")
            # 6. LEARN
            outcome = InteractionOutcome(
                event=event,
                plan=plan,
                success=success,
                response_time=time.time() - start
            )
            await self.learning_loop.record_outcome(outcome, session)

            # Update world model
            await self.world_model.update_from_event(event, success)

            self.metrics.update_response_time(time.time() - start)

        except Exception as e:
            self.metrics.errors += 1
            import traceback
            traceback.print_exc()
        finally:
            self._current_user_id = None

    async def _execute_plan(
        self,
        event: PerceivedEvent,
        plan: ActionPlan,
        session: 'AgentSession'
    ) -> tuple[bool, str]:
        """Execute action plan."""
        user_id = event.user_id

        if plan.decision == AutonomousDecision.IGNORE and plan.content == "":
            return True, ""

        if plan.decision == AutonomousDecision.OBSERVE:
            # Store context only
            self.context_store.store_event(event.raw_signal.id, {
                "intent": event.intent,
                "entities": event.entities,
                "observed_at": time.time()
            })


        if plan.decision == AutonomousDecision.SCHEDULE:
            # Schedule for later
            task_id = await self.scheduler.schedule_task(
                user_id=user_id,
                task_type="query",
                content=plan.content,
                delay_seconds=300,  # 5 minutes
                priority=int(plan.confidence * 10)
            )
            return True, f"Scheduled: {task_id}"

        if plan.decision == AutonomousDecision.QUEUE:
            # Queue for when user is available
            self.context_store.store_event(f"queued_{event.raw_signal.id}", {
                "content": plan.content,
                "intent": event.intent,
                "status": "pending"
            })
            return True, ""

        # ACT_NOW - Execute via FlowAgent
        try:
            # Set situation in RuleSet
            session.rule_set.set_situation("kernel_action", event.intent)

            # Activate tool groups
            for group in plan.tool_groups[:3]:
                session.rule_set.activate_tool_group(group)

            # Inject memory context
            memories = await self.memory_store.get_relevant_memories(user_id, plan.content, limit=5)
            if memories:
                memory_ctx = self.memory_store.format_memories_for_context(memories)
                session.vfs.create("user_memories", memory_ctx)

            # Build query with instructions
            query = plan.content
            if plan.instructions:
                instructions_text = "\n".join(f"- {i}" for i in plan.instructions[:5])
                query = f"{plan.content}\n\n[Instructions]\n{instructions_text}"


            # Execute
            response = await self.agent.a_run(
                query=query,
                session_id=session.session_id,
                user_id=user_id
            )

            # Ensure response is a string (a_run can return various types)
            if response is None:
                response = ""
            elif not isinstance(response, str):
                # Handle Message objects, dicts, or other types
                if hasattr(response, 'content'):
                    response = str(response.content)
                elif hasattr(response, 'text'):
                    response = str(response.text)
                else:
                    response = str(response)

            # Handle special states
            if response.startswith("__NEEDS_HUMAN__:"):
                question = response.replace("__NEEDS_HUMAN__:", "")
                await self.output_router.send_notification(
                    user_id, f"❓ {question}", priority=8
                )
                return True, response
            elif response.startswith("__PAUSED__"):
                return True, response

            # Record interaction
            await self.learning_engine.record_interaction(
                user_id=user_id,
                interaction_type=InteractionType.AGENT_RESPONSE,
                content={"response": response[:500]},
                outcome="success"
            )

            # Send response based on action type
            if plan.action_type == ActionType.RESPOND:
                await self.output_router.send_response(user_id, response, "assistant")
            elif plan.action_type == ActionType.NOTIFY:
                self.metrics.proactive_actions += 1
                self.proactive_tracker.record_action()
                await self.output_router.send_notification(user_id, response, int(plan.confidence * 10))

            return True, response

        except Exception as e:
            await self.output_router.send_response(user_id, f"Error: {e}", "assistant")
            return False, str(e)

    async def _handle_heartbeat(self):
        """Heartbeat maintenance."""
        # Update user states
        for ctx in self.state_monitor.user_contexts.values():
            ctx.update_state()

        # Clean old context
        self.context_store.clear_old_events(max_age_seconds=3600)

        # Execute overdue tasks
        now = time.time()
        overdue = [t for t in self.scheduler.tasks.values()
                   if t.status == TaskStatus.PENDING and t.scheduled_time < now - 60]
        for task in overdue[:5]:
            asyncio.create_task(self.scheduler._execute_task(task))

        # Prune low confidence patterns in active sessions
        for session in self.agent.session_manager.get_all_active():
            session.rule_set.prune_low_confidence_patterns(threshold=0.2)

    # =========================================================================
    # PUBLIC API (IProAKernel)
    # =========================================================================

    async def handle_user_input(self, user_id: str, content: str, metadata: dict = None) -> str:
        """Handle user input."""
        print("Handling user input:", user_id, content)
        await self.signal_bus.emit_signal(Signal(
            id=str(uuid.uuid4()),
            type=SignalType.USER_INPUT,
            priority=10,
            content=content,
            source=f"user_{user_id}",
            metadata={"user_id": user_id, **(metadata or {})}
        ))
        return ""

    async def trigger_event(self, event_name: str, payload: dict, priority: int = 5, source: str = "external"):
        """Trigger system event."""
        await self.signal_bus.emit_signal(Signal(
            id=str(uuid.uuid4()),
            type=SignalType.SYSTEM_EVENT,
            priority=priority,
            content=payload,
            source=source,
            metadata={"event_name": event_name}
        ))

    async def set_user_location(self, user_id: str, location: str):
        await self.state_monitor.set_user_location(user_id, location)

    async def set_do_not_disturb(self, user_id: str, enabled: bool):
        await self.state_monitor.set_do_not_disturb(user_id, enabled)

    def get_status(self) -> dict[str, Any]:
        """Get kernel status."""
        return {
            "state": self.state.value,
            "running": self.running,
            "agent": self.agent.amd.name if self.agent and self.agent.amd else None,
            "metrics": self.metrics.to_dict(),
            "world_model": {"users": len(self.world_model.users), "sessions": len(self.world_model.get_active_sessions())},
            "learning": {"records": len(self.learning_engine.records), "preferences": len(self.learning_engine.preferences)},
            "memory": {"total": len(self.memory_store.memories)},
            "scheduler": {"pending": sum(1 for t in self.scheduler.tasks.values() if t.status == TaskStatus.PENDING)}
        }

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def save_to_file(self, filepath: str = None) -> dict:
        """Save kernel state."""
        try:
            if not filepath:
                from toolboxv2 import get_app
                folder = Path(get_app().data_dir) / 'Agents' / 'kernel' / self.agent.amd.name
                folder.mkdir(parents=True, exist_ok=True)
                filepath = str(folder / f"kernel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

            state = {
                "version": "2.1.0",
                "agent": self.agent.amd.name,
                "saved_at": datetime.now().isoformat(),
                "metrics": self.metrics.to_dict(),
                "world_model": {u: {"activity": m.activity_rhythm, "topics": m.topics_of_interest, "engagement": m.engagement_level}
                               for u, m in self.world_model.users.items()},
                "learning": {"records": [r.model_dump() for r in self.learning_engine.records],
                            "preferences": {u: p.model_dump() for u, p in self.learning_engine.preferences.items()}},
                "memory": {"memories": {m: mem.model_dump() for m, mem in self.memory_store.memories.items()},
                          "user_memories": dict(self.memory_store.user_memories)},
                "scheduler": {"tasks": {t: task.model_dump() for t, task in self.scheduler.tasks.items()}}
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            return {"success": True, "filepath": filepath}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def load_from_file(self, filepath: str) -> dict:
        """Load kernel state."""
        try:
            if not Path(filepath).exists():
                return {"success": False, "error": "File not found"}

            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Restore world model
            for user_id, data in state.get("world_model", {}).items():
                user = self.world_model.get_user(user_id)
                user.activity_rhythm = data.get("activity", {})
                user.topics_of_interest = data.get("topics", [])
                user.engagement_level = data.get("engagement", 0.5)

            # Restore learning
            l = state.get("learning", {})
            self.learning_engine.records = [LearningRecord(**r) for r in l.get("records", [])]
            self.learning_engine.preferences = {u: UserPreferences(**p) for u, p in l.get("preferences", {}).items()}

            # Restore memory
            mem = state.get("memory", {})
            self.memory_store.memories = {m: Memory(**d) for m, d in mem.get("memories", {}).items()}
            self.memory_store.user_memories = defaultdict(list, mem.get("user_memories", {}))

            # Restore scheduler
            for tid, td in state.get("scheduler", {}).get("tasks", {}).items():
                self.scheduler.tasks[tid] = ScheduledTask(**td)

            return {"success": True, "loaded": filepath}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def process_signal(self, signal: Signal):
        return await self.signal_bus.emit_signal(signal)
