"""
Kernel Unit Tests
Tests for ProA Kernel autonomous system.

Uses unittest (not pytest).
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any

from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import (
    Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter,
    KernelState, UserState, ProactivityDecision, TaskStatus,
    InteractionType, MemoryType
)


# =============================================================================
# MOCK CLASSES
# =============================================================================

class MockOutputRouter(IOutputRouter):
    """Mock output router for testing."""

    def __init__(self):
        self.responses: list[dict] = []
        self.notifications: list[dict] = []

    async def send_response(self, user_id: str, content: str, role: str = "assistant", metadata: dict = None):
        self.responses.append({
            "user_id": user_id,
            "content": content,
            "role": role,
            "metadata": metadata
        })

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        self.notifications.append({
            "user_id": user_id,
            "content": content,
            "priority": priority,
            "metadata": metadata
        })

    def clear(self):
        self.responses.clear()
        self.notifications.clear()


class MockRuleSet:
    """Mock RuleSet for session."""

    def __init__(self):
        self.current_situation = None
        self.current_intent = None
        self._active_groups = set()
        self.learned_patterns = []
        self.rules = {}

    def get_groups_for_intent(self, intent: str):
        return [MagicMock(name=f"{intent}_tools")]

    def match_rules(self, situation: str, intent: str, min_score: float = 0.3):
        return []

    def get_rule(self, rule_id: str):
        return self.rules.get(rule_id)

    def get_relevant_patterns(self, situation: str = None, limit: int = 10):
        return self.learned_patterns[:limit]

    def rule_on_action(self, action: str, context: dict = None):
        return MagicMock(allowed=True, instructions=[], warnings=[], required_steps=[])

    def set_situation(self, situation: str, intent: str):
        self.current_situation = situation
        self.current_intent = intent

    def activate_tool_group(self, group: str):
        self._active_groups.add(group)

    def learn_pattern(self, pattern: str, source_situation: str = None, confidence: float = 0.5,
                      category: str = "general", tags: list = None):
        self.learned_patterns.append(MagicMock(pattern=pattern, confidence=confidence))
        return self.learned_patterns[-1]

    def add_rule(self, situation: str, intent: str, instructions: list, required_tool_groups: list = None,
                 learned: bool = False, confidence: float = 1.0):
        rule_id = f"rule_{len(self.rules)}"
        self.rules[rule_id] = MagicMock(
            id=rule_id, situation=situation, intent=intent,
            instructions=instructions, confidence=confidence
        )
        return self.rules[rule_id]

    def record_rule_success(self, rule_id: str):
        pass

    def record_rule_failure(self, rule_id: str):
        pass

    def prune_low_confidence_patterns(self, threshold: float = 0.2):
        self.learned_patterns = [p for p in self.learned_patterns if p.confidence >= threshold]
        return 0


class MockVFS:
    """Mock VFS for session."""

    def __init__(self):
        self.files = {}

    def create(self, filename: str, content: str):
        self.files[filename] = content
        return {"success": True}


class MockSession:
    """Mock AgentSession."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.rule_set = MockRuleSet()
        self.vfs = MockVFS()
        self._initialized = True
        self._closed = False

    def get_stats(self):
        return {"session_id": self.session_id, "initialized": self._initialized}


class MockSessionManager:
    """Mock SessionManager."""

    def __init__(self):
        self.sessions: dict[str, MockSession] = {}

    async def get_or_create(self, session_id: str, **kwargs) -> MockSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = MockSession(session_id)
        return self.sessions[session_id]

    def get(self, session_id: str):
        return self.sessions.get(session_id)

    def list_sessions(self):
        return list(self.sessions.keys())

    def get_all_active(self):
        return [s for s in self.sessions.values() if s._initialized and not s._closed]


class MockAMD:
    """Mock AgentModelData."""
    name = "TestAgent"
    system_message = "You are a test agent."


class MockCheckpointManager:
    """Mock CheckpointManager."""

    async def save_current(self):
        return "checkpoint_123"


class MockToolManager:
    """Mock ToolManager for testing."""

    test_name = "onlein"

    def __init__(self):
        self._registry = {}

    def register(self, func, name: str, description: str = "", category: list = None, flags: dict = None):
        """Register a tool (matches ToolManager.register signature)."""
        self._registry[name] = {
            "func": func,
            "description": description,
            "category": category or [],
            "flags": flags or {}
        }
        print("Registered tool: MOCK", name)

    def get(self, name: str):
        return self._registry.get(name)

    def get_stats(self):
        return {"total_tools": len(self._registry)}


class MockFlowAgent:
    """Mock FlowAgent for testing."""

    def __init__(self):
        self.amd = MockAMD()
        self.session_manager = MockSessionManager()
        self.checkpoint_manager = MockCheckpointManager()
        self.tool_manager = MockToolManager()  # Required by Kernel._register_tools()
        self._first_class_tools = {}
        self._tools = {}
        self._run_calls = []
        self.tool_manager.register = MagicMock()

    def add_first_class_tool(self, func, name: str, description: str = ""):
        self._first_class_tools[name] = {"func": func, "description": description}

    async def add_tool(self, func, name: str, description: str = "", category: list = None, flags: dict = None):
        self._tools[name] = {
            "func": func,
            "description": description,
            "category": category or [],
            "flags": flags or {}
        }

    def init_session_tools(self, session):
        """Initialize session-specific tools (no-op for mock)."""
        pass

    async def a_run(self, query: str, session_id: str = "default", user_id: str = None,
                    remember: bool = True, **kwargs) -> str:
        self._run_calls.append({"query": query, "session_id": session_id, "user_id": user_id})
        return f"Response to: {query[:50]}"

    async def restore(self):
        pass

    async def close(self):
        pass


# =============================================================================
# TEST CASES
# =============================================================================

class TestKernelInit(unittest.TestCase):
    """Test Kernel initialization."""

    def test_init_default_config(self):
        """Test kernel initializes with default config."""
        agent = MockFlowAgent()
        kernel = Kernel(agent=agent)

        self.assertEqual(kernel.state, KernelState.STOPPED)
        self.assertFalse(kernel.running)
        self.assertIsNotNone(kernel.config)
        self.assertIsNotNone(kernel.signal_bus)
        self.assertIsNotNone(kernel.state_monitor)

    def test_init_custom_config(self):
        """Test kernel initializes with custom config."""
        agent = MockFlowAgent()
        config = KernelConfig(
            heartbeat_interval=5.0,
            proactive_cooldown=60.0
        )
        kernel = Kernel(agent=agent, config=config)

        self.assertEqual(kernel.config.heartbeat_interval, 5.0)
        self.assertEqual(kernel.config.proactive_cooldown, 60.0)

    def test_init_custom_output_router(self):
        """Test kernel with custom output router."""
        agent = MockFlowAgent()
        router = MockOutputRouter()
        kernel = Kernel(agent=agent, output_router=router)

        self.assertIs(kernel.output_router, router)

    def test_init_components(self):
        """Test all components are initialized."""
        agent = MockFlowAgent()
        kernel = Kernel(agent=agent)

        # Core components
        self.assertIsNotNone(kernel.perception)
        self.assertIsNotNone(kernel.world_model)
        self.assertIsNotNone(kernel.attention)
        self.assertIsNotNone(kernel.decision_engine)
        self.assertIsNotNone(kernel.learning_loop)

        # Extended components
        self.assertIsNotNone(kernel.learning_engine)
        self.assertIsNotNone(kernel.memory_store)
        self.assertIsNotNone(kernel.scheduler)
        self.assertIsNotNone(kernel.integration)


class TestKernelLifecycle(unittest.TestCase):
    """Test Kernel lifecycle (start/stop)."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.router = MockOutputRouter()
        self.kernel = Kernel(agent=self.agent, output_router=self.router)

    def test_start(self):
        """Test kernel start."""

        async def run_test():
            await self.kernel.start()

            self.assertEqual(self.kernel.state, KernelState.RUNNING)
            self.assertTrue(self.kernel.running)
            self.assertIsNotNone(self.kernel.main_task)
            self.assertIsNotNone(self.kernel.heartbeat_task)

            # Cleanup
            await self.kernel.stop()

        asyncio.run(run_test())

    def test_stop(self):
        """Test kernel stop."""

        async def run_test():
            await self.kernel.start()
            await self.kernel.stop()

            self.assertEqual(self.kernel.state, KernelState.STOPPED)
            self.assertFalse(self.kernel.running)

        asyncio.run(run_test())

    def test_double_start_noop(self):
        """Test double start is no-op."""

        async def run_test():
            await self.kernel.start()
            state_before = self.kernel.state
            await self.kernel.start()
            state_after = self.kernel.state

            self.assertEqual(state_before, state_after)

            await self.kernel.stop()

        asyncio.run(run_test())

    def test_double_stop_noop(self):
        """Test double stop is no-op."""

        async def run_test():
            await self.kernel.start()
            await self.kernel.stop()
            await self.kernel.stop()  # Should not raise

            self.assertEqual(self.kernel.state, KernelState.STOPPED)

        asyncio.run(run_test())


class TestSignalProcessing(unittest.TestCase):
    """Test signal processing."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.router = MockOutputRouter()
        self.kernel = Kernel(agent=self.agent, output_router=self.router)

    def test_handle_user_input(self):
        """Test handling user input."""

        async def run_test():
            await self.kernel.start()

            await self.kernel.handle_user_input(
                user_id="user123",
                content="Hello, how are you?",
                metadata={"source": "test"}
            )

            # Give time for signal to be processed
            await asyncio.sleep(0.2)

            # Check signal was emitted
            self.assertGreater(self.kernel.metrics.signals_processed, 0)

            await self.kernel.stop()

        asyncio.run(run_test())

    def test_trigger_event(self):
        """Test triggering system event."""

        async def run_test():
            await self.kernel.start()

            await self.kernel.trigger_event(
                event_name="test_event",
                payload={"data": "test"},
                priority=7
            )

            await asyncio.sleep(0.2)

            self.assertGreater(self.kernel.metrics.signals_processed, 0)

            await self.kernel.stop()

        asyncio.run(run_test())

    def test_signal_creates_session(self):
        """Test that processing signal creates session."""

        async def run_test():
            await self.kernel.start()

            await self.kernel.handle_user_input(
                user_id="new_user",
                content="Test message"
            )

            await asyncio.sleep(0.2)

            # Session should be created
            session = self.agent.session_manager.get("new_user")
            self.assertIsNotNone(session)

            await self.kernel.stop()

        asyncio.run(run_test())


class TestPerceptionLayer(unittest.TestCase):
    """Test Perception Layer."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)
        self.session = MockSession("test_session")

    def test_extract_intent_question(self):
        """Test intent extraction for questions."""
        signal = KernelSignal(
            id="sig1", type=SignalType.USER_INPUT, priority=5,
            content="What is the weather today?",
            metadata={"user_id": "user1"}
        )

        async def run_test():
            event = await self.kernel.perception.perceive(signal, self.session)
            self.assertEqual(event.intent, "question")

        asyncio.run(run_test())

    def test_extract_intent_create(self):
        """Test intent extraction for create commands."""
        signal = KernelSignal(
            id="sig2", type=SignalType.USER_INPUT, priority=5,
            content="Create a new file for the project",
            metadata={"user_id": "user1"}
        )

        async def run_test():
            event = await self.kernel.perception.perceive(signal, self.session)
            self.assertEqual(event.intent, "create")

        asyncio.run(run_test())

    def test_extract_intent_schedule(self):
        """Test intent extraction for scheduling."""
        signal = KernelSignal(
            id="sig3", type=SignalType.USER_INPUT, priority=5,
            content="Remind me tomorrow about the meeting",
            metadata={"user_id": "user1"}
        )

        async def run_test():
            event = await self.kernel.perception.perceive(signal, self.session)
            self.assertEqual(event.intent, "schedule")

        asyncio.run(run_test())

    def test_compute_urgency_high(self):
        """Test high urgency detection."""
        signal = KernelSignal(
            id="sig4", type=SignalType.USER_INPUT, priority=9,
            content="URGENT: Fix the bug immediately!",
            metadata={"user_id": "user1"}
        )

        async def run_test():
            event = await self.kernel.perception.perceive(signal, self.session)
            self.assertGreater(event.urgency, 0.8)

        asyncio.run(run_test())

    def test_extract_entities(self):
        """Test entity extraction."""
        entities = self.kernel.perception._extract_entities('Check "Project Alpha" for John')
        self.assertIn("Project Alpha", entities)
        self.assertIn("John", entities)

    def test_extract_topics(self):
        """Test topic extraction."""
        topics = self.kernel.perception._extract_topics("Write a python script to query the database")
        self.assertIn("code", topics)
        self.assertIn("data", topics)


class TestWorldModel(unittest.TestCase):
    """Test World Model."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_get_user_creates_model(self):
        """Test getting user creates model if not exists."""
        user = self.kernel.world_model.get_user("new_user")

        self.assertIsNotNone(user)
        self.assertEqual(user.user_id, "new_user")
        self.assertIn("new_user", self.kernel.world_model.users)

    def test_update_activity(self):
        """Test user activity update."""
        user = self.kernel.world_model.get_user("user1")
        initial_count = user.interaction_count

        user.update_activity()

        self.assertEqual(user.interaction_count, initial_count + 1)
        self.assertIn(time.localtime().tm_hour, user.activity_rhythm)

    def test_is_likely_available(self):
        """Test availability prediction."""
        user = self.kernel.world_model.get_user("user1")

        # Add activity for current hour
        current_hour = time.localtime().tm_hour
        user.activity_rhythm[current_hour] = 0.8

        self.assertTrue(user.is_likely_available())

        # Low activity
        user.activity_rhythm[current_hour] = 0.1
        self.assertFalse(user.is_likely_available())


class TestAttentionSystem(unittest.TestCase):
    """Test Attention System."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)
        self.session = MockSession("test")

    def test_salience_high_urgency(self):
        """Test salience for high urgency events."""
        from toolboxv2.mods.isaa.kernel.instace import PerceivedEvent

        event = PerceivedEvent(
            raw_signal=KernelSignal(id="s1", type=SignalType.USER_INPUT, priority=9, content="urgent"),
            user_id="user1",
            intent="help",
            urgency=0.9
        )
        user = self.kernel.world_model.get_user("user1")

        salience = self.kernel.attention.compute_salience(event, user, self.session)

        self.assertGreater(salience.score, 0.3)
        self.assertTrue(any("urgency" in r.lower() for r in salience.reasons))

    def test_salience_with_tool_groups(self):
        """Test salience boost for available tool groups."""
        from toolboxv2.mods.isaa.kernel.instace import PerceivedEvent

        event = PerceivedEvent(
            raw_signal=KernelSignal(id="s2", type=SignalType.USER_INPUT, priority=5, content="test"),
            user_id="user1",
            intent="create",
            matching_tool_groups=["file_tools", "code_tools"]
        )
        user = self.kernel.world_model.get_user("user1")

        salience = self.kernel.attention.compute_salience(event, user, self.session)

        self.assertGreater(salience.score, 0.0)


class TestDecisionEngine(unittest.TestCase):
    """Test Autonomous Decision Engine."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)
        self.session = MockSession("test")

    def test_decide_act_now(self):
        """Test ACT_NOW decision for high salience."""
        from toolboxv2.mods.isaa.kernel.instace import PerceivedEvent, SalienceScore, AutonomousDecision

        event = PerceivedEvent(
            raw_signal=KernelSignal(id="s1", type=SignalType.USER_INPUT, priority=9, content="test"),
            user_id="user1",
            intent="help"
        )
        salience = SalienceScore(score=0.8, reasons=["High urgency"], should_interrupt=True)
        user = self.kernel.world_model.get_user("user1")
        user.activity_rhythm[time.localtime().tm_hour] = 0.8  # Make available

        async def run_test():
            plan = await self.kernel.decision_engine.decide(event, salience, user, self.session)
            self.assertEqual(plan.decision, AutonomousDecision.ACT_NOW)

        asyncio.run(run_test())

    def test_decide_queue(self):
        """Test QUEUE decision for medium salience."""
        from toolboxv2.mods.isaa.kernel.instace import PerceivedEvent, SalienceScore, AutonomousDecision

        event = PerceivedEvent(
            raw_signal=KernelSignal(id="s2", type=SignalType.SYSTEM_EVENT, priority=5, content="update"),
            user_id="user1",
            intent="notify"
        )
        salience = SalienceScore(score=0.45, reasons=["Medium priority"], should_interrupt=False)
        user = self.kernel.world_model.get_user("user1")

        async def run_test():
            plan = await self.kernel.decision_engine.decide(event, salience, user, self.session)
            self.assertEqual(plan.decision, AutonomousDecision.QUEUE)

        asyncio.run(run_test())

    def test_decide_blocked_by_rule(self):
        """Test decision blocked by RuleSet."""
        from toolboxv2.mods.isaa.kernel.instace import PerceivedEvent, SalienceScore, AutonomousDecision

        # Make rule_on_action return not allowed
        self.session.rule_set.rule_on_action = lambda action, context=None: MagicMock(
            allowed=False,
            instructions=["Need validation"],
            warnings=[],
            required_steps=["validate first"]
        )

        event = PerceivedEvent(
            raw_signal=KernelSignal(id="s3", type=SignalType.USER_INPUT, priority=5, content="delete all"),
            user_id="user1",
            intent="delete"
        )
        salience = SalienceScore(score=0.6, should_interrupt=True)
        user = self.kernel.world_model.get_user("user1")

        async def run_test():
            plan = await self.kernel.decision_engine.decide(event, salience, user, self.session)
            self.assertEqual(plan.decision, AutonomousDecision.OBSERVE)
            self.assertIn("Need validation", plan.instructions)

        asyncio.run(run_test())


class TestLearningLoop(unittest.TestCase):
    """Test Learning Loop."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)
        self.session = MockSession("test")

    def test_record_outcome_updates_buffer(self):
        """Test recording outcome adds to buffer."""
        from toolboxv2.mods.isaa.kernel.instace import (
            PerceivedEvent, ActionPlan, InteractionOutcome,
            AutonomousDecision, ActionType
        )

        event = PerceivedEvent(
            raw_signal=KernelSignal(id="s1", type=SignalType.USER_INPUT, priority=5, content="test"),
            user_id="user1",
            intent="create",
            matching_rules=[]
        )
        plan = ActionPlan(
            decision=AutonomousDecision.ACT_NOW,
            action_type=ActionType.RESPOND,
            content="test",
            tool_groups=["file_tools"]
        )
        outcome = InteractionOutcome(event=event, plan=plan, success=True, response_time=0.5)

        async def run_test():
            initial_len = len(self.kernel.learning_loop._buffer)
            await self.kernel.learning_loop.record_outcome(outcome, self.session)
            self.assertEqual(len(self.kernel.learning_loop._buffer), initial_len + 1)

        asyncio.run(run_test())

    def test_pattern_detection_triggered(self):
        """Test pattern detection is triggered after threshold."""
        from toolboxv2.mods.isaa.kernel.instace import (
            PerceivedEvent, ActionPlan, InteractionOutcome,
            AutonomousDecision, ActionType
        )

        self.kernel.learning_loop._pattern_threshold = 3

        async def run_test():
            for i in range(5):
                event = PerceivedEvent(
                    raw_signal=KernelSignal(id=f"s{i}", type=SignalType.USER_INPUT, priority=5, content="create file"),
                    user_id="user1",
                    intent="create",
                    matching_rules=[]
                )
                plan = ActionPlan(
                    decision=AutonomousDecision.ACT_NOW,
                    action_type=ActionType.RESPOND,
                    content="test",
                    tool_groups=["file_tools"]
                )
                outcome = InteractionOutcome(event=event, plan=plan, success=True, response_time=0.3)
                await self.kernel.learning_loop.record_outcome(outcome, self.session)

            # Buffer should be cleared after pattern detection
            self.assertEqual(len(self.kernel.learning_loop._buffer), 2)  # 5 % 3 = 2 remaining

        asyncio.run(run_test())


class TestMemoryStore(unittest.TestCase):
    """Test Memory Store integration."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_inject_memory(self):
        """Test injecting memory."""

        async def run_test():
            memory_id = await self.kernel.memory_store.inject_memory(
                user_id="user1",
                memory_type=MemoryType.FACT,
                content="User works at Anthropic",
                importance=0.8,
                tags=["work", "company"]
            )

            self.assertIsNotNone(memory_id)
            self.assertIn(memory_id, self.kernel.memory_store.memories)

        asyncio.run(run_test())

    def test_get_relevant_memories(self):
        """Test getting relevant memories."""

        async def run_test():
            # Inject some memories
            await self.kernel.memory_store.inject_memory(
                user_id="user1",
                memory_type=MemoryType.PREFERENCE,
                content="Prefers concise responses",
                importance=0.9
            )
            await self.kernel.memory_store.inject_memory(
                user_id="user1",
                memory_type=MemoryType.FACT,
                content="Uses Python daily",
                importance=0.7
            )

            memories = await self.kernel.memory_store.get_relevant_memories(
                user_id="user1",
                query="code",
                limit=5
            )

            self.assertGreater(len(memories), 0)

        asyncio.run(run_test())


class TestScheduler(unittest.TestCase):
    """Test Task Scheduler."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_schedule_task(self):
        """Test scheduling a task."""

        async def run_test():
            await self.kernel.scheduler.start()

            task_id = await self.kernel.scheduler.schedule_task(
                user_id="user1",
                task_type="reminder",
                content="Test reminder",
                delay_seconds=60,
                priority=5
            )

            self.assertIsNotNone(task_id)
            self.assertIn(task_id, self.kernel.scheduler.tasks)
            self.assertEqual(self.kernel.scheduler.tasks[task_id].status, TaskStatus.PENDING)

            await self.kernel.scheduler.stop()

        asyncio.run(run_test())


class TestStateMonitor(unittest.TestCase):
    """Test State Monitor."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_get_user_state(self):
        """Test getting user state."""

        async def run_test():
            state = await self.kernel.state_monitor.get_user_state("user1")
            self.assertIn(state, [UserState.ACTIVE, UserState.IDLE, UserState.AWAY, UserState.BUSY])

        asyncio.run(run_test())

    def test_update_user_activity(self):
        """Test updating user activity."""

        async def run_test():
            await self.kernel.state_monitor.update_user_activity("user1", "input")
            state = await self.kernel.state_monitor.get_user_state("user1")
            self.assertEqual(state, UserState.ACTIVE)

        asyncio.run(run_test())

    def test_set_do_not_disturb(self):
        """Test setting DND mode."""

        async def run_test():
            await self.kernel.state_monitor.set_do_not_disturb("user1", True)
            state = await self.kernel.state_monitor.get_user_state("user1")
            self.assertEqual(state, UserState.BUSY)

        asyncio.run(run_test())


class TestKernelStatus(unittest.TestCase):
    """Test Kernel status reporting."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_get_status(self):
        """Test getting kernel status."""
        status = self.kernel.get_status()

        self.assertIn("state", status)
        self.assertIn("running", status)
        self.assertIn("metrics", status)
        self.assertIn("world_model", status)
        self.assertIn("learning", status)
        self.assertIn("memory", status)
        self.assertIn("scheduler", status)

    def test_status_after_start(self):
        """Test status after kernel start."""

        async def run_test():
            await self.kernel.start()

            status = self.kernel.get_status()
            self.assertEqual(status["state"], "running")
            self.assertTrue(status["running"])

            await self.kernel.stop()

        asyncio.run(run_test())


class TestPersistence(unittest.TestCase):
    """Test save/load functionality."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_save_creates_file(self):
        """Test saving creates a file."""
        import tempfile
        import os

        async def run_test():
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                filepath = f.name

            try:
                # Add some data
                await self.kernel.memory_store.inject_memory(
                    user_id="user1",
                    memory_type=MemoryType.FACT,
                    content="Test fact",
                    importance=0.5
                )

                result = await self.kernel.save_to_file(filepath)

                self.assertTrue(result["success"])
                self.assertTrue(os.path.exists(filepath))
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)

        asyncio.run(run_test())

    def test_load_restores_state(self):
        """Test loading restores state."""
        import tempfile
        import os

        async def run_test():
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                filepath = f.name

            try:
                # Add data and save
                await self.kernel.memory_store.inject_memory(
                    user_id="user1",
                    memory_type=MemoryType.FACT,
                    content="Test fact",
                    importance=0.8
                )
                await self.kernel.save_to_file(filepath)

                # Create new kernel and load
                new_kernel = Kernel(agent=self.agent)
                result = await new_kernel.load_from_file(filepath)

                self.assertTrue(result["success"])
                self.assertGreater(len(new_kernel.memory_store.memories), 0)
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)

        asyncio.run(run_test())


class TestIntegrationLayer(unittest.TestCase):
    """Test Agent Integration Layer."""

    def setUp(self):
        self.agent = MockFlowAgent()
        self.kernel = Kernel(agent=self.agent)

    def test_schedule_task_via_integration(self):
        """Test scheduling via integration layer."""

        async def run_test():
            await self.kernel.scheduler.start()
            self.kernel._current_user_id = "user1"

            task_id = await self.kernel.integration.schedule_task(
                task_type="reminder",
                content="Test reminder",
                delay_seconds=30
            )

            self.assertIsNotNone(task_id)

            await self.kernel.scheduler.stop()

        asyncio.run(run_test())

    def test_get_preferences_via_integration(self):
        """Test getting preferences via integration layer."""

        async def run_test():
            self.kernel._current_user_id = "user1"

            prefs = await self.kernel.integration.get_user_preferences()

            self.assertIsInstance(prefs, dict)
            self.assertIn("communication_style", prefs)

        asyncio.run(run_test())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

