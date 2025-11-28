"""
ProA Kernel - Complete Implementation
Version: 1.0.0

Full implementation of the Proactive Autonomous Kernel that wraps FlowAgent
and provides always-on, event-driven, proactive capabilities.
"""

import asyncio
import random
import time
import uuid
import json
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path
from collections import defaultdict
import traceback

from pydantic import BaseModel

from toolboxv2.mods.isaa.base.Agent.agent import FlowAgent
# Import all core interfaces from types
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

# Import model classes
from toolboxv2.mods.isaa.kernel.models import (
    ContextStore, ProactiveActionTracker,
    LearningEngine, MemoryStore, TaskScheduler,
    WebSocketOutputRouter, MultiChannelRouter,
    AgentIntegrationLayer
)


# ===== KERNEL IMPLEMENTATION =====

class Kernel(IProAKernel):
    """
    kernel with learning, memory, and scheduling
    """

    def __init__(
        self,
        agent: FlowAgent,
        config: KernelConfig = None,
        decision_engine: IDecisionEngine = None,
        output_router: IOutputRouter = None
    ):
        """Initialize kernel"""
        self.agent = agent
        self.config = config or KernelConfig()
        self.decision_engine = decision_engine or DefaultDecisionEngine()
        self.output_router = output_router or ConsoleOutputRouter()

        # Core components
        self.signal_bus: ISignalBus = SignalBus(
            max_queue_size=self.config.max_signal_queue_size
        )
        self.state_monitor: IStateMonitor = StateMonitor()
        self.context_store = ContextStore()

        # New advanced components
        self.learning_engine = LearningEngine(agent)
        self.memory_store = MemoryStore()
        self.scheduler = TaskScheduler(self)

        # Agent integration layer
        self.integration = AgentIntegrationLayer(self)

        # State
        self.state = KernelState.STOPPED
        self.metrics = KernelMetrics()
        self.proactive_tracker = ProactiveActionTracker()
        self.running = False

        # Lifecycle
        self.main_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None

        # Current context
        self._current_user_id: Optional[str] = None
        self._pending_questions: dict[str, asyncio.Future] = {}

        print(f"✓ ProA Kernel initialized for {(agent.amd.name if agent and agent.amd else None) or 'self'}")

    async def _export_functions_to_agent(self):
        """Export kernel functions to agent for use in tools"""
        # Make functions available as agent tools
        self.agent.add_first_class_tool(
            self.integration.schedule_task,
            "kernel_schedule_task",
            description="Schedule a task for future execution. Use for reminders, delayed queries, or scheduled actions. "
                       "Args: task_type (str: 'reminder'/'query'/'action'), content (str), "
                       "delay_seconds (float, optional), scheduled_time (float, optional), priority (int: 0-10, default 5). "
                       "Returns: task_id (str). Example: await kernel_schedule_task('reminder', 'Follow up on project X', delay_seconds=3600)"
        )

        self.agent.add_first_class_tool(
            self.integration.send_intermediate_response,
            "kernel_send_intermediate",
            description="Send intermediate status updates during long-running operations to keep user informed. "
                       "Args: content (str), stage (str: 'processing'/'analysis'/'synthesis'/etc., default 'processing'). "
                       "Example: await kernel_send_intermediate('Analyzing data...', stage='analysis')"
        )

        self.agent.add_first_class_tool(
            self.integration.ask_user,
            "kernel_ask_user",
            description="Ask the user a question and wait for their response. Use when you need clarification or user input during execution. "
                       "Args: question (str), timeout (float: seconds, default 300.0). "
                       "Returns: answer (str) or None if timeout. "
                       "Example: answer = await kernel_ask_user('Which option do you prefer: A or B?', timeout=60.0)"
        )

        self.agent.add_first_class_tool(
            self.integration.inject_memory,
            "kernel_inject_memory",
            description="Store important information about the user for future sessions. Use for preferences, facts, events, or context. "
                       "Args: content (str), memory_type (str: 'fact'/'event'/'preference'/'context', default 'fact'), "
                       "importance (float: 0.0-1.0, default 0.5), tags (list[str], optional). "
                       "Returns: memory_id (str). "
                       "Example: await kernel_inject_memory('User prefers concise responses', memory_type='preference', importance=0.8, tags=['communication'])"
        )

        self.agent.add_first_class_tool(
            self.integration.get_user_preferences,
            "kernel_get_preferences",
            description="Get the current user's learned preferences from previous interactions. "
                       "Returns: dict with keys: communication_style, response_format, proactivity_level, preferred_tools. "
                       "Example: prefs = await kernel_get_preferences(); style = prefs.get('communication_style')"
        )

        self.agent.add_first_class_tool(
            self.integration.record_feedback,
            "kernel_record_feedback",
            description="Record user feedback to improve future responses through learning. Use when user expresses satisfaction/dissatisfaction. "
                       "Args: feedback (str), score (float: -1.0 to 1.0, negative=bad, positive=good). "
                       "Example: await kernel_record_feedback('Response was too verbose', score=-0.5)"
        )

        # >>>>>>>>>>>>


        await self.agent.add_tool(
            self.integration.schedule_task,
            "kernel_schedule_task",
            description="Schedule a task for future execution. Use for reminders, delayed queries, or scheduled actions. "
                       "Args: task_type (str: 'reminder'/'query'/'action'), content (str), "
                       "delay_seconds (float, optional), scheduled_time (float, optional), priority (int: 0-10, default 5). "
                       "Returns: task_id (str). Example: await kernel_schedule_task('reminder', 'Follow up on project X', delay_seconds=3600)"
        )

        await self.agent.add_tool(
            self.integration.send_intermediate_response,
            "kernel_send_intermediate",
            description="Send intermediate status updates during long-running operations to keep user informed. "
                       "Args: content (str), stage (str: 'processing'/'analysis'/'synthesis'/etc., default 'processing'). "
                       "Example: await kernel_send_intermediate('Analyzing data...', stage='analysis')"
        )

        await self.agent.add_tool(
            self.integration.ask_user,
            "kernel_ask_user",
            description="Ask the user a question and wait for their response. Use when you need clarification or user input during execution. "
                       "Args: question (str), timeout (float: seconds, default 300.0). "
                       "Returns: answer (str) or None if timeout. "
                       "Example: answer = await kernel_ask_user('Which option do you prefer: A or B?', timeout=60.0)"
        )

        await self.agent.add_tool(
            self.integration.inject_memory,
            "kernel_inject_memory",
            description="Store important information about the user for future sessions. Use for preferences, facts, events, or context. "
                       "Args: content (str), memory_type (str: 'fact'/'event'/'preference'/'context', default 'fact'), "
                       "importance (float: 0.0-1.0, default 0.5), tags (list[str], optional). "
                       "Returns: memory_id (str). "
                       "Example: await kernel_inject_memory('User prefers concise responses', memory_type='preference', importance=0.8, tags=['communication'])"
        )

        await self.agent.add_tool(
            self.integration.get_user_preferences,
            "kernel_get_preferences",
            description="Get the current user's learned preferences from previous interactions. "
                       "Returns: dict with keys: communication_style, response_format, proactivity_level, preferred_tools. "
                       "Example: prefs = await kernel_get_preferences(); style = prefs.get('communication_style')"
        )

        await self.agent.add_tool(
            self.integration.record_feedback,
            "kernel_record_feedback",
            description="Record user feedback to improve future responses through learning. Use when user expresses satisfaction/dissatisfaction. "
                       "Args: feedback (str), score (float: -1.0 to 1.0, negative=bad, positive=good). "
                       "Example: await kernel_record_feedback('Response was too verbose', score=-0.5)"
        )

        print("✓ Exported 6 kernel functions to agent as tools")

    async def start(self):
        """Start the kernel"""
        if self.state == KernelState.RUNNING:
            return

        # Export functions to agent
        await self._export_functions_to_agent()
        await self.agent.load_latest_checkpoint()
        print("Starting ProA Kernel...")
        self.state = KernelState.STARTING
        self.running = True

        # Start scheduler
        await self.scheduler.start()

        # Start lifecycle tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.main_task = asyncio.create_task(self._lifecycle_loop())

        self.state = KernelState.RUNNING
        print(f"✓ Kernel running")

    async def stop(self):
        """Stop the kernel"""
        if self.state == KernelState.STOPPED:
            return

        print("Stopping Kernel...")
        self.state = KernelState.STOPPING
        self.running = False

        # Stop scheduler
        await self.scheduler.stop()

        # Stop tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.main_task:
            self.main_task.cancel()

        await self.agent.close()
        self.state = KernelState.STOPPED
        print("✓ Kernel stopped")

    async def _lifecycle_loop(self):
        """Main lifecycle loop"""
        while self.running:
            try:
                signal = await self.signal_bus.get_next_signal(
                    timeout=self.config.signal_timeout
                )

                if signal:
                    await self._process_signal(signal)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Lifecycle error: {e}")
                traceback.print_exc()

    async def _heartbeat_loop(self):
        """Heartbeat loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                # Emit heartbeat signal
                heartbeat = Signal(
                    id=str(uuid.uuid4()),
                    type=SignalType.HEARTBEAT,
                    priority=0,
                    content={"timestamp": time.time()},
                    source="kernel",
                    timestamp=time.time()
                )

                await self.signal_bus.emit_signal(heartbeat)


            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Heartbeat error: {e}")

        # ===== SIGNAL PROCESSING =====

    async def _process_signal(self, signal: Signal):
        """
        Process a signal based on its type

        Args:
            signal: The signal to process
        """
        start_time = time.time()

        try:
            self.metrics.signals_processed += 1

            # Route based on signal type
            if signal.type == SignalType.USER_INPUT:
                await self._handle_user_input(signal)

            elif signal.type == SignalType.SYSTEM_EVENT:
                await self._handle_system_event(signal)

            elif signal.type == SignalType.HEARTBEAT:
                await self._handle_heartbeat_signal(signal)

            elif signal.type == SignalType.TOOL_RESULT:
                await self._handle_tool_result_signal(signal)

            elif signal.type == SignalType.ERROR:
                await self._handle_error_signal(signal)

            else:
                signal.content += " System signal"
                await self._handle_user_input(signal)

            # Update metrics
            response_time = time.time() - start_time
            self.metrics.update_response_time(response_time)

        except Exception as e:
            self.metrics.errors += 1
            print(f"Error processing signal {signal.id}: {e}")
            traceback.print_exc()

    async def _handle_user_input(self, signal: Signal):
        """user input handling with learning"""
        user_id = signal.metadata.get("user_id", signal.id or "default")
        content = signal.content

        # Set current user context
        self._current_user_id = user_id

        # Update user state
        await self.state_monitor.update_user_activity(user_id, "input")

        # Record interaction
        await self.learning_engine.record_interaction(
            user_id=user_id,
            interaction_type=InteractionType.USER_INPUT,
            content={"query": content}
        )

        # Get relevant memories
        memories = await self.memory_store.get_relevant_memories(
            user_id=user_id,
            query=content,
            limit=5
        )

        # Apply preferences
        modified_query, hints = await self.learning_engine.apply_preferences_to_query(
            user_id, content
        )

        # Inject memory context
        if memories:
            memory_context = self.memory_store.format_memories_for_context(memories)
            # Inject into agent's variable system
            if hasattr(self.agent, 'variable_manager'):
                self.agent.variable_manager.set(
                    f'user_memories.{user_id}',
                    memory_context
                )

        # Get formatting instructions from metadata (set by Discord voice input)
        formatting_instructions = signal.metadata.get("formatting_instructions", "")

        # Get voice channel history from metadata (set by Discord voice input in group calls)
        voice_channel_history = signal.metadata.get("voice_channel_history", "")

        # Temporarily inject formatting instructions and voice history into system prompt
        original_system_message = None
        if hasattr(self.agent, 'amd'):
            original_system_message = self.agent.amd.system_message

            # Build additional context
            additional_context = ""
            if formatting_instructions:
                additional_context += f"\n\n{formatting_instructions}"
            if voice_channel_history:
                additional_context += f"\n\n{voice_channel_history}"
                print(f"📋 [KERNEL] Injecting voice channel history into agent context")

            if additional_context:
                self.agent.amd.system_message = original_system_message + additional_context

        try:
            # Check if fast response mode is enabled (for voice input)
            fast_response_mode = signal.metadata.get("fast_response", False)

            if fast_response_mode:
                print(f"🚀 [KERNEL] Fast Response Mode enabled for voice input")

                # PHASE 1: Single LLM call with full context for immediate response
                print(f"🚀 [KERNEL] Phase 1: Generating immediate response...")
                class ImmediateResponse(BaseModel):
                    response: str
                    needs_tools: bool

                response = await self.agent.a_format_class(
                    pydantic_model=ImmediateResponse,
                    prompt='Task generate an immediate response to the following USER REQUEST: '+modified_query,
                    session_id=user_id,
                    auto_context=True,
                    model_preference="fast",
                )

                # Record and send immediate response
                await self.learning_engine.record_interaction(
                    user_id=user_id,
                    interaction_type=InteractionType.AGENT_RESPONSE,
                    content={"response": response.get("response"), "phase": "immediate"},
                    outcome="success"
                )

                print(f"🚀 [KERNEL] Sending immediate response...")
                await self.output_router.send_response(
                    user_id=user_id,
                    content=response.get("response"),
                    role="assistant"
                )

                if not response.get("needs_tools"):
                    return


            # Normal mode: Standard agent run
            response = await self.agent.a_run(
                query=modified_query,
                session_id=user_id,
                user_id=user_id,
                remember=True,
                fast_run=True
            )

            # Record response
            await self.learning_engine.record_interaction(
                user_id=user_id,
                interaction_type=InteractionType.AGENT_RESPONSE,
                content={"response": response},
                outcome="success"
            )

            # Send response
            await self.output_router.send_response(
                user_id=user_id,
                content=response,
                role="assistant"
            )

        except Exception as e:
            # Restore original system message on error
            if original_system_message is not None and hasattr(self.agent, 'amd'):
                self.agent.amd.system_message = original_system_message
            error_msg = f"Error: {str(e)}"
            await self.output_router.send_response(
                user_id=user_id,
                content=error_msg,
                role="assistant"
            )

            await self.learning_engine.record_interaction(
                user_id=user_id,
                interaction_type=InteractionType.ERROR,
                content={"error": str(e)},
                outcome="error"
            )

        finally:
            self._current_user_id = None

    async def _handle_system_event(self, signal: Signal):
        """Handle SYSTEM_EVENT signal"""
        self.metrics.system_events_handled += 1

        # Store event in context
        self.context_store.store_event(signal.id, {
            "type": signal.type.value,
            "content": signal.content,
            "source": signal.source,
            "timestamp": signal.timestamp,
            "metadata": signal.metadata
        })

        # Check if proactive action is needed
        user_id = signal.metadata.get("user_id", signal.id or "default")
        user_state = await self.state_monitor.get_user_state(user_id)

        context = ProactivityContext(
            user_state=user_state,
            signal=signal,
            last_proactive_time=self.proactive_tracker.last_proactive_time,
            cooldown_period=self.config.proactive_cooldown,
            recent_proactive_count=self.proactive_tracker.get_recent_count()
        )

        decision = await self.decision_engine.evaluate_proactivity(context)

        if decision == ProactivityDecision.INTERRUPT:
            await self._proactive_notify(user_id, signal)
        elif decision == ProactivityDecision.QUEUE:
            # Store for later retrieval
            print(f"Queued event {signal.id} for later")
        elif decision == ProactivityDecision.SILENT:
            # Process silently - just stored in context
            print(f"Silently processed event {signal.id}")

    async def _handle_tool_result_signal(self, signal: Signal):
        """Handle TOOL_RESULT signal"""
        # Store tool result in context
        self.context_store.store_event(signal.id, {
            "type": "tool_result",
            "tool_name": signal.metadata.get("tool_name"),
            "result": signal.content,
            "timestamp": signal.timestamp
        })

        # Check if this result warrants proactive notification
        if signal.priority >= 7:
            user_id = signal.metadata.get("user_id", signal.id or "default")
            await self._proactive_notify(user_id, signal)

    async def _handle_heartbeat_signal(self, signal: Signal):
        """Handle HEARTBEAT signal with task recovery"""
        # Maintenance tasks
        # Update all user states
        if hasattr(self.state_monitor, 'user_contexts'):
            for user_id, context in self.state_monitor.user_contexts.items():
                context.update_state()

        # Clean old context
        self.context_store.clear_old_events(max_age_seconds=3600)

        # Prüfe auf verpasste Tasks
        now = time.time()
        overdue_tasks = [
            task for task in self.scheduler.tasks.values()
            if task.status == TaskStatus.PENDING
               and task.scheduled_time < now - 60  # Mehr als 1 Minute überfällig
        ]

        if overdue_tasks:
            print(f"⚠️ Found {len(overdue_tasks)} overdue tasks, executing now...")
            for task in overdue_tasks[:5]:  # Max 5 auf einmal
                if task.status == TaskStatus.PENDING:
                    asyncio.create_task(self.scheduler._execute_task(task))

        # Alte abgeschlossene Tasks bereinigen
        completed_cutoff = now - 86400  # 24 Stunden
        old_completed = [
            tid for tid, task in self.scheduler.tasks.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
               and task.scheduled_time < completed_cutoff
        ]

        for tid in old_completed[:100]:  # Max 100 auf einmal
            del self.scheduler.tasks[tid]

        if old_completed:
            print(f"🧹 Cleaned up {len(old_completed)} old tasks")

        # System health check
        queue_size = self.signal_bus.get_queue_size()
        if queue_size > 100:
            print(f"WARNING: High signal queue size: {queue_size}")

    async def _handle_error_signal(self, signal: Signal):
        """Handle ERROR signal"""
        self.metrics.errors += 1

        # Critical errors should notify immediately
        if signal.priority >= 8:
            user_id = signal.metadata.get("user_id", signal.id or "default")
            await self.output_router.send_notification(
                user_id=user_id,
                content=f"Critical error: {signal.content}",
                priority=signal.priority
            )

        # ===== PROACTIVE NOTIFICATIONS =====

    async def _proactive_notify(self, user_id: str, signal: Signal):
        """
        Send a proactive notification to the user

        Args:
            user_id: User to notify
            signal: Signal that triggered the notification
        """
        self.metrics.proactive_actions += 1
        self.proactive_tracker.record_action()

        # Build notification content
        content = self._build_notification_content(signal)

        # Send notification
        await self.output_router.send_notification(
            user_id=user_id,
            content=content,
            priority=signal.priority,
            metadata=signal.metadata
        )

    def _build_notification_content(self, signal: Signal) -> str:
        """Build human-readable notification from signal"""
        if isinstance(signal.content, str):
            return signal.content

        if isinstance(signal.content, dict):
            # Extract meaningful info from dict
            message = signal.content.get("message")
            if message:
                return message

            # Fallback to JSON representation
            return f"Event: {signal.content}"

        return str(signal.content)

    # Public API
    async def handle_user_input(
        self,
        user_id: str,
        content: str,
        metadata: dict = None
    ) -> str:
        """Handle user input"""
        signal = Signal(
            id=str(uuid.uuid4()),
            type=SignalType.USER_INPUT,
            priority=10,
            content=content,
            source=f"user_{user_id}",
            timestamp=time.time(),
            metadata={"user_id": user_id, **(metadata or {})}
        )

        await self.signal_bus.emit_signal(signal)
        return ""

    async def trigger_event(
        self,
        event_name: str,
        payload: dict,
        priority: int = 5,
        source: str = "external"
    ):
        """Trigger system event"""
        signal = Signal(
            id=str(uuid.uuid4()),
            type=SignalType.SYSTEM_EVENT,
            priority=priority,
            content=payload,
            source=source,
            timestamp=time.time(),
            metadata={"event_name": event_name}
        )

        await self.signal_bus.emit_signal(signal)

    async def process_signal(self, signal: Signal):
        """Process signal"""
        await self.signal_bus.emit_signal(signal)

    async def set_user_location(self, user_id: str, location: str):
        """Set user location"""
        await self.state_monitor.set_user_location(user_id, location)

    async def set_do_not_disturb(self, user_id: str, enabled: bool):
        """Set DND mode"""
        await self.state_monitor.set_do_not_disturb(user_id, enabled)

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status"""
        return {
            "state": self.state.value,
            "running": self.running,
            "agent_name": self.agent.amd.name,
            "metrics": self.metrics.to_dict(),
            "learning": {
                "total_records": len(self.learning_engine.records),
                "users_learned": len(self.learning_engine.preferences)
            },
            "memory": {
                "total_memories": len(self.memory_store.memories),
                "users_with_memory": len(self.memory_store.user_memories)
            },
            "scheduler": {
                "total_tasks": len(self.scheduler.tasks),
                "pending_tasks": sum(
                    1 for t in self.scheduler.tasks.values()
                    if t.status == TaskStatus.PENDING
                )
            }
        }


    # ===== SAVE/LOAD METHODS =====

    async def save_to_file(self, filepath: str = None) -> dict[str, Any]:
        """
        Save complete kernel state to file

        Args:
            filepath: Path to save file (default: auto-generated)

        Returns:
            dict with save statistics
        """
        try:
            if filepath is None:
                # Auto-generate path
                from toolboxv2 import get_app
                folder = Path(get_app().data_dir) / 'Agents' / 'kernel' / self.agent.amd.name
                folder.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = str(folder / f"kernel_state_{timestamp}.pkl")

            # Collect state
            state_data = {
                "version": "2.0.0",
                "agent_name": self.agent.amd.name,
                "saved_at": datetime.now().isoformat(),
                "config": {
                    "heartbeat_interval": self.config.heartbeat_interval,
                    "idle_threshold": self.config.idle_threshold,
                    "proactive_cooldown": self.config.proactive_cooldown,
                    "max_proactive_per_hour": self.config.max_proactive_per_hour,
                    "max_signal_queue_size": self.config.max_signal_queue_size
                },
                "metrics": self.metrics.to_dict(),
                "learning": {
                    "records": [r.model_dump() for r in self.learning_engine.records],
                    "preferences": {
                        uid: prefs.model_dump()
                        for uid, prefs in self.learning_engine.preferences.items()
                    }
                },
                "memory": {
                    "memories": {
                        mid: mem.model_dump()
                        for mid, mem in self.memory_store.memories.items()
                    },
                    "user_memories": dict(self.memory_store.user_memories)
                },
                "scheduler": {
                    "tasks": {
                        tid: task.model_dump()
                        for tid, task in self.scheduler.tasks.items()
                    }
                },
                "state_monitor": {
                    "user_contexts": {
                        uid: {
                            "user_id": ctx.user_id,
                            "state": ctx.state.value,
                            "last_interaction": ctx.last_interaction,
                            "location": ctx.location,
                            "do_not_disturb": ctx.do_not_disturb,
                            "activity_history": ctx.activity_history[-50:]  # Last 50
                        }
                        for uid, ctx in self.state_monitor.user_contexts.items()
                    }
                }
            }

            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)

            # Calculate statistics
            stats = {
                "success": True,
                "filepath": filepath,
                "file_size_kb": Path(filepath).stat().st_size / 1024,
                "learning_records": len(state_data["learning"]["records"]),
                "user_preferences": len(state_data["learning"]["preferences"]),
                "memories": len(state_data["memory"]["memories"]),
                "scheduled_tasks": len(state_data["scheduler"]["tasks"]),
                "user_contexts": len(state_data["state_monitor"]["user_contexts"]),
                "saved_at": state_data["saved_at"]
            }

            print(f"✓ Kernel state saved to {filepath}")
            print(f"  - Learning records: {stats['learning_records']}")
            print(f"  - User preferences: {stats['user_preferences']}")
            print(f"  - Memories: {stats['memories']}")
            print(f"  - Scheduled tasks: {stats['scheduled_tasks']}")

            return stats

        except Exception as e:
            print(f"❌ Failed to save kernel state: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


    async def load_from_file(self, filepath: str) -> dict[str, Any]:
        """
        Load kernel state from file

        Args:
            filepath: Path to saved state file

        Returns:
            dict with load statistics
        """
        try:
            if not Path(filepath).exists():
                return {
                    "success": False,
                    "error": f"File not found: {filepath}"
                }

            # Load state data
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)

            # Validate version
            version = state_data.get("version", "unknown")
            print(f"Loading kernel state version {version}...")

            # Restore config
            config_data = state_data.get("config", {})
            for key, value in config_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            # Restore metrics
            if "metrics" in state_data:
                metrics_data = state_data["metrics"]
                self.metrics.signals_processed = metrics_data.get("signals_processed", 0)
                self.metrics.user_inputs_handled = metrics_data.get("user_inputs", 0)
                self.metrics.system_events_handled = metrics_data.get("system_events", 0)
                self.metrics.proactive_actions = metrics_data.get("proactive_actions", 0)
                self.metrics.errors = metrics_data.get("errors", 0)
                self.metrics.average_response_time = metrics_data.get("avg_response_time", 0.0)

            # Restore learning engine
            if "learning" in state_data:
                learning_data = state_data["learning"]

                # Restore records
                self.learning_engine.records = [
                    LearningRecord(**record_data)
                    for record_data in learning_data.get("records", [])
                ]

                # Restore preferences
                self.learning_engine.preferences = {
                    uid: UserPreferences(**prefs_data)
                    for uid, prefs_data in learning_data.get("preferences", {}).items()
                }

            # Restore memory store
            if "memory" in state_data:
                memory_data = state_data["memory"]

                # Restore memories
                self.memory_store.memories = {
                    mid: Memory(**mem_data)
                    for mid, mem_data in memory_data.get("memories", {}).items()
                }

                # Restore user memory mappings
                self.memory_store.user_memories = defaultdict(
                    list,
                    memory_data.get("user_memories", {})
                )

            # Restore scheduler
            if "scheduler" in state_data:
                scheduler_data = state_data["scheduler"]

                # Restore tasks
                self.scheduler.tasks = {
                    tid: ScheduledTask(**task_data)
                    for tid, task_data in scheduler_data.get("tasks", {}).items()
                }

            # Restore state monitor
            if "state_monitor" in state_data:
                monitor_data = state_data["state_monitor"]

                # Restore user contexts
                for uid, ctx_data in monitor_data.get("user_contexts", {}).items():
                    context = UserContext(
                        user_id=ctx_data["user_id"],
                        state=UserState(ctx_data["state"]),
                        last_interaction=ctx_data["last_interaction"],
                        location=ctx_data["location"],
                        do_not_disturb=ctx_data["do_not_disturb"],
                        activity_history=ctx_data.get("activity_history", [])
                    )
                    self.state_monitor.user_contexts[uid] = context

            # Calculate statistics
            stats = {
                "success": True,
                "filepath": filepath,
                "version": version,
                "saved_at": state_data.get("saved_at"),
                "loaded_at": datetime.now().isoformat(),
                "learning_records": len(self.learning_engine.records),
                "user_preferences": len(self.learning_engine.preferences),
                "memories": len(self.memory_store.memories),
                "scheduled_tasks": len(self.scheduler.tasks),
                "user_contexts": len(self.state_monitor.user_contexts)
            }

            print(f"✓ Kernel state loaded from {filepath}")
            print(f"  - Learning records: {stats['learning_records']}")
            print(f"  - User preferences: {stats['user_preferences']}")
            print(f"  - Memories: {stats['memories']}")
            print(f"  - Scheduled tasks: {stats['scheduled_tasks']}")
            print(f"  - User contexts: {stats['user_contexts']}")

            return stats

        except Exception as e:
            print(f"❌ Failed to load kernel state: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


    def to_dict(self) -> dict[str, Any]:
        """
        Export kernel state to dictionary (for API/serialization)

        Returns:
            dict with complete kernel state
        """
        return {
            "version": "2.0.0",
            "agent_name": self.agent.amd.name,
            "state": self.state.value,
            "running": self.running,
            "exported_at": datetime.now().isoformat(),
            "config": {
                "heartbeat_interval": self.config.heartbeat_interval,
                "idle_threshold": self.config.idle_threshold,
                "proactive_cooldown": self.config.proactive_cooldown,
                "max_proactive_per_hour": self.config.max_proactive_per_hour
            },
            "metrics": self.metrics.to_dict(),
            "learning": {
                "total_records": len(self.learning_engine.records),
                "user_preferences": {
                    uid: prefs.model_dump()
                    for uid, prefs in self.learning_engine.preferences.items()
                }
            },
            "memory": {
                "total_memories": len(self.memory_store.memories),
                "user_memory_counts": {
                    uid: len(mids)
                    for uid, mids in self.memory_store.user_memories.items()
                }
            },
            "scheduler": {
                "total_tasks": len(self.scheduler.tasks),
                "pending_tasks": [
                    task.model_dump()
                    for task in self.scheduler.tasks.values()
                    if task.status.value == "pending"
                ]
            },
            "users": {
                uid: {
                    "state": ctx.state.value,
                    "last_interaction": ctx.last_interaction,
                    "location": ctx.location,
                    "do_not_disturb": ctx.do_not_disturb,
                    "idle_time": ctx.get_idle_time()
                }
                for uid, ctx in self.state_monitor.user_contexts.items()
            }
        }


    async def from_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Import kernel state from dictionary

        Args:
            data: Dictionary with kernel state (from to_dict or API)

        Returns:
            dict with import statistics
        """
        try:
            version = data.get("version", "unknown")
            print(f"Importing kernel state version {version}...")

            # Import config
            if "config" in data:
                config_data = data["config"]
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

            # Import learning preferences
            if "learning" in data and "user_preferences" in data["learning"]:
                self.learning_engine.preferences = {
                    uid: UserPreferences(**prefs_data)
                    for uid, prefs_data in data["learning"]["user_preferences"].items()
                }

            # Import scheduled tasks
            if "scheduler" in data and "pending_tasks" in data["scheduler"]:
                for task_data in data["scheduler"]["pending_tasks"]:
                    task = ScheduledTask(**task_data)
                    self.scheduler.tasks[task.id] = task

            stats = {
                "success": True,
                "version": version,
                "imported_at": datetime.now().isoformat(),
                "user_preferences": len(self.learning_engine.preferences),
                "scheduled_tasks": len(
                    [t for t in data.get("scheduler", {}).get("pending_tasks", [])]
                )
            }

            print(f"✓ Kernel state imported")
            print(f"  - User preferences: {stats['user_preferences']}")
            print(f"  - Scheduled tasks: {stats['scheduled_tasks']}")

            return stats

        except Exception as e:
            print(f"❌ Failed to import kernel state: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


    # ===== SYSTEM PROMPT EXTENSION =====

    def get_kernel_system_prompt_extension(self) -> str:
        """
        Generate system prompt extension that informs the agent about kernel capabilities

        This should be added to the agent's system prompt to enable kernel awareness.

        Returns:
            str: System prompt extension text
        """
        # Get current user preferences if available
        prefs_info = ""
        if self._current_user_id:
            prefs = self.learning_engine.get_preferences(self._current_user_id)
            prefs_info = f"""

## Current User Preferences (Learned)
- Communication Style: {prefs.communication_style}
- Response Format: {prefs.response_format}
- Proactivity Level: {prefs.proactivity_level}
- Preferred Tools: {', '.join(prefs.preferred_tools) if prefs.preferred_tools else 'None learned yet'}
"""

        # Get memory context if available
        memory_info = ""
        if self._current_user_id:
            memory_count = len(self.memory_store.user_memories.get(self._current_user_id, []))
            if memory_count > 0:
                memory_info = f"""

## User Memory Context
You have access to {memory_count} stored memories about this user.
These memories are automatically injected into your context when relevant.
"""

        prompt_extension = f"""

# ========== KERNEL CAPABILITIES ==========

You are running inside an Kernel that provides advanced capabilities beyond standard agent execution.

## Available Kernel Tools

You have access to the following kernel tools that you can call directly:

### 1. kernel_schedule_task
Schedule a task for future execution (reminders, delayed queries, or scheduled actions).

**Parameters:**
- task_type: "reminder", "query", or "action"
- content: Description of the task
- delay_seconds: (optional) Delay in seconds from now
- scheduled_time: (optional) Unix timestamp for exact scheduling
- priority: (optional) 0-10, default 5

**Returns:** task_id (string)

**Example usage:** When user says "Remind me tomorrow at 2pm to check the report", call kernel_schedule_task with task_type="reminder", content="Check the report", and appropriate scheduled_time.

### 2. kernel_send_intermediate
Send status updates during long-running operations to keep the user informed.

**Parameters:**
- content: Status message to send
- stage: (optional) "processing", "analysis", "synthesis", etc. (default: "processing")

**Example usage:** During multi-step analysis, call kernel_send_intermediate with content="Analyzing data..." and stage="analysis" to update the user.

### 3. kernel_ask_user
Ask the user a question and wait for their response.

**Parameters:**
- question: The question to ask
- timeout: (optional) Seconds to wait (default: 300.0)

**Returns:** User's answer (string) or None if timeout

**Example usage:** When you need clarification, call kernel_ask_user with question="Which option do you prefer: A or B?" and wait for the response.

### 4. kernel_inject_memory
Store important information about the user for future sessions.

**Parameters:**
- content: Information to remember
- memory_type: (optional) "fact", "event", "preference", or "context" (default: "fact")
- importance: (optional) 0.0 to 1.0 (default: 0.5)
- tags: (optional) List of tags for categorization

**Returns:** memory_id (string)

**Example usage:** When user states "I prefer concise responses", call kernel_inject_memory with content="User prefers concise responses", memory_type="preference", importance=0.8, tags=["communication", "style"].

### 5. kernel_get_preferences
Get the current user's learned preferences from previous interactions.

**Parameters:** None

**Returns:** Dictionary with keys:
- communication_style: "concise", "detailed", or "balanced"
- response_format: "text", "bullet-points", or "structured"
- proactivity_level: "low", "medium", or "high"
- preferred_tools: List of tool names

**Example usage:** Call kernel_get_preferences at the start of complex tasks to adapt your response style.

### 6. kernel_record_feedback
Record user feedback to improve future responses through learning.

**Parameters:**
- feedback: Description of the feedback
- score: -1.0 to 1.0 (negative = bad, positive = good)

**Example usage:** When user says "that was too verbose", call kernel_record_feedback with feedback="Response was too verbose", score=-0.5.

## When to Use Kernel Tools

**kernel_schedule_task** - Use when user mentions future actions, reminders, or scheduled queries:
- "Remind me tomorrow at 2pm"
- "Check the weather in 2 hours"
- "Follow up on this next week"

**kernel_send_intermediate** - Use for long-running operations to keep user informed:
- Multi-step analysis
- Large data processing
- Complex tool chains
- Any operation taking more than a few seconds

**kernel_ask_user** - Use when you need clarification or choices during execution:
- Ambiguous requests
- Multiple valid options
- Confirmation needed before taking action

**kernel_inject_memory** - Use when learning important facts about the user:
- User states preferences ("I prefer...", "I like...", "I don't like...")
- Personal information shared
- Important context for future interactions
- Recurring patterns you notice

**kernel_get_preferences** - Use to adapt your response style automatically:
- Call at the start of complex tasks
- Check before generating long responses
- Adjust verbosity based on user's preference
- Choose appropriate format

**kernel_record_feedback** - Use when user expresses satisfaction/dissatisfaction:
- Explicit feedback ("that's perfect", "too long", "not what I wanted")
- Corrections to your responses
- Style adjustment requests
{prefs_info}{memory_info}

## Important Guidelines

1. **Use these tools proactively** - They significantly enhance user experience
2. **Memory is persistent** - Information you store will be available in future sessions
3. **Learning is continuous** - The kernel learns from every interaction
4. **Don't ask permission** - Just use the tools when appropriate
5. **Tasks run independently** - Scheduled tasks execute even after the current session ends
6. **Call tools directly** - These are available in your toolkit, use them like any other tool

## Current Kernel Status
- State: {self.state.value}
- Total interactions processed: {self.metrics.signals_processed}
- Learning records: {len(self.learning_engine.records)}
- Stored memories: {len(self.memory_store.memories)}
- Scheduled tasks: {len(self.scheduler.tasks)}

# ==========================================
"""

        return prompt_extension


    def inject_kernel_prompt_to_agent(self):
        """
        Inject kernel capabilities into agent's system prompt

        This should be called after kernel initialization to make the agent
        aware of kernel functions.
        """
        try:
            # Get extension
            extension = self.get_kernel_system_prompt_extension()

            # Add to agent's system message
            if hasattr(self.agent, 'amd'):
                current_prompt = self.agent.amd.system_message or ""

                # Check if already injected
                if "KERNEL CAPABILITIES" not in current_prompt:
                    self.agent.amd.system_message = current_prompt + "\n\n" + extension
                    print("✓ Kernel capabilities injected into agent system prompt")
                else:
                    # Update existing section
                    parts = current_prompt.split("# ========== KERNEL CAPABILITIES ==========")
                    if len(parts) == 2:
                        self.agent.amd.system_message = parts[0] + extension
                        print("✓ Kernel capabilities updated in agent system prompt")
            else:
                print("⚠️  Agent does not have AMD - cannot inject prompt")

        except Exception as e:
            print(f"❌ Failed to inject kernel prompt: {e}")


# ===== EXAMPLE USAGE =====

async def example_usage():
    """Example of how to use the ProA Kernel"""
    print("\n" + "=" * 60)
    print("ProA Kernel - Example Usage")
    print("=" * 60 + "\n")

    # Note: In real usage, you would import FlowAgent from your actual module
    # from your_module import FlowAgent, AgentModelData

    # For this example, we'll create a mock agent
    class MockAgent:
        class MockAMD:
            name = "TestAgent"

        amd = MockAMD()

        async def a_run(self, query, session_id=None, user_id=None, remember=True):
            await asyncio.sleep(0.5)  # Simulate processing
            return f"Mock response to: {query}"

    # Create mock agent
    agent = MockAgent()

    # Create kernel
    config = KernelConfig(
        heartbeat_interval=10.0,
        proactive_cooldown=5.0,
        max_proactive_per_hour=10
    )

    kernel = Kernel(
        agent=agent,
        config=config
    )

    # Start kernel
    await kernel.start()

    try:
        # Simulate user interactions
        print("\n--- Simulating user interactions ---\n")

        # User input
        await kernel.handle_user_input(
            user_id="user123",
            content="Hello, how are you?"
        )

        await asyncio.sleep(2)

        # System event (low priority)
        await kernel.trigger_event(
            event_name="file_uploaded",
            payload={"filename": "document.pdf", "size": 1024},
            priority=3
        )

        await asyncio.sleep(2)

        # System event (high priority - should trigger notification)
        await kernel.trigger_event(
            event_name="critical_alert",
            payload={"message": "System backup completed successfully"},
            priority=8
        )

        await asyncio.sleep(2)

        # Set user as busy
        await kernel.set_do_not_disturb("user123", True)

        # This event should be queued, not notified
        await kernel.trigger_event(
            event_name="low_priority_update",
            payload={"message": "New feature available"},
            priority=5
        )

        await asyncio.sleep(2)

        # Check status
        status = kernel.get_status()
        print("\n--- Kernel Status ---")
        print(f"State: {status['state']}")
        print(f"Metrics: {status['metrics']}")
        print(f"Queue Size: {status['signal_queue_size']}")

        # Let it run for a bit
        print("\n--- Kernel running... (press Ctrl+C to stop) ---\n")
        await asyncio.sleep(10)

    finally:
        # Stop kernel
        await kernel.stop()
        print("\n✓ Example completed\n")


if __name__ == "__main__":
    # Run example
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

print("\n✓ ProA Kernel Implementation Complete")
print("Ready to wrap any FlowAgent and provide always-on capabilities")
