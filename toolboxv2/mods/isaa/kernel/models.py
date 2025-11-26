"""
ProA Kernel - Advanced Implementation with Learning & Scheduling
Version: 2.0.0

Extended implementation with:
- Memory injection and learning from interactions
- WebSocket and advanced output routers
- Task scheduling for user and agent
- Preference learning system
- Agent integration layer with exported functions
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from collections import defaultdict

# Import core interfaces
from toolboxv2.mods.isaa.kernel.types import (
    Signal, SignalType, IOutputRouter, LearningRecord, UserPreferences, InteractionType, Memory, MemoryType,
    ScheduledTask, TaskStatus, VALID_TASK_TYPES
)


# ===== CONTEXT STORE =====

class ContextStore:
    """
    Speichert System-Events und deren Ergebnisse für den Agent-Kontext
    """

    def __init__(self, max_size: int = 1000):
        self.events: dict[str, dict] = {}
        self.max_size = max_size
        self.access_count: dict[str, int] = {}

    def store_event(self, event_id: str, data: dict):
        """Store an event result"""
        if len(self.events) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.events[least_accessed]
            del self.access_count[least_accessed]

        self.events[event_id] = {
            **data,
            "stored_at": time.time()
        }
        self.access_count[event_id] = 0

    def get_event(self, event_id: str) -> Optional[dict]:
        """Get an event result"""
        if event_id in self.events:
            self.access_count[event_id] += 1
            return self.events[event_id]
        return None

    def get_recent_events(self, limit: int = 10) -> list[dict]:
        """Get recent events sorted by timestamp"""
        events = sorted(
            self.events.values(),
            key=lambda x: x.get("stored_at", 0),
            reverse=True
        )
        return events[:limit]

    def clear_old_events(self, max_age_seconds: float = 3600):
        """Clear events older than max_age"""
        now = time.time()
        to_delete = []

        for event_id, data in self.events.items():
            if now - data.get("stored_at", now) > max_age_seconds:
                to_delete.append(event_id)

        for event_id in to_delete:
            del self.events[event_id]
            if event_id in self.access_count:
                del self.access_count[event_id]


# ===== PROACTIVE ACTION TRACKER =====

class ProactiveActionTracker:
    """Tracks proactive actions to enforce rate limits"""

    def __init__(self):
        self.actions: list[tuple[float, str]] = []
        self.last_proactive_time: float = 0

    def record_action(self, action_type: str = "notification"):
        """Record a proactive action"""
        now = time.time()
        self.actions.append((now, action_type))
        self.last_proactive_time = now

        # Keep only last hour
        one_hour_ago = now - 3600
        self.actions = [a for a in self.actions if a[0] > one_hour_ago]

    def get_recent_count(self, window_seconds: float = 3600) -> int:
        """Get count of recent proactive actions"""
        now = time.time()
        cutoff = now - window_seconds
        return sum(1 for t, _ in self.actions if t > cutoff)

    def get_time_since_last(self) -> float:
        """Get seconds since last proactive action"""
        if self.last_proactive_time == 0:
            return float('inf')
        return time.time() - self.last_proactive_time


# ===== LEARNING SYSTEM =====


class LearningEngine:
    """
    Learning system that analyzes interactions and adapts behavior
    """

    def __init__(self, agent):
        self.agent = agent
        self.records: list[LearningRecord] = []
        self.preferences: dict[str, UserPreferences] = {}
        self.max_records = 10000

    async def record_interaction(
        self,
        user_id: str,
        interaction_type: InteractionType,
        content: dict,
        context: dict = None,
        outcome: str = None,
        feedback_score: float = None
    ):
        """Record an interaction for learning"""
        record = LearningRecord(
            user_id=user_id,
            interaction_type=interaction_type,
            content=content,
            context=context or {},
            outcome=outcome,
            feedback_score=feedback_score
        )

        self.records.append(record)

        # Limit records - FIX: Korrigierte Filter-Syntax
        if len(self.records) > self.max_records:
            # Behalte Records mit Feedback-Score (wichtiger für Learning)
            self.records = [r for r in self.records if r.feedback_score is not None]
            # Falls immer noch zu viele, behalte die neuesten
            if len(self.records) > self.max_records:
                self.records = self.records[-self.max_records:]

        if interaction_type != InteractionType.FEEDBACK:
            return

        # Trigger learning if enough data - FIX: Korrigierte Filter-Syntax
        records_with_feedback = [r for r in self.records if r.feedback_score is not None]
        if len(self.records) % 10 == 0 and records_with_feedback:
            from toolboxv2 import get_app
            get_app().run_bg_task_advanced(self.analyze_and_learn, user_id)

    async def analyze_and_learn(self, user_id: str):
        """Analyze interactions and update preferences"""
        user_records = [r for r in self.records if r.user_id == user_id]

        if len(user_records) < 5:
            return

        # Get or create preferences
        if user_id not in self.preferences:
            self.preferences[user_id] = UserPreferences(user_id=user_id)

        prefs = self.preferences[user_id]

        # Use agent's a_format_class for structured analysis
        class PreferenceAnalysis(BaseModel):
            """Analysis of user preferences"""
            communication_style: str = Field(
                description="concise, detailed, or balanced"
            )
            response_format: str = Field(
                description="text, bullet-points, or structured"
            )
            proactivity_level: str = Field(
                description="low, medium, or high"
            )
            preferred_tools: list[str] = Field(
                description="List of tools user frequently uses"
            )
            topic_interests: list[str] = Field(
                description="Topics user is interested in"
            )
            time_pattern: dict[str, str] = Field(
                description="When user is most active"
            )
            confidence: float = Field(
                description="Confidence in analysis (0-1)",
                ge=0.0,
                le=1.0
            )

        # Build analysis prompt
        recent_interactions = user_records[-20:]  # Last 20
        interaction_summary = "\n".join([
            f"- {r.interaction_type.value}: {r.content.get('summary', str(r.content)[:100])}"
            for r in recent_interactions
        ])

        prompt = f"""
Analyze these user interactions and infer preferences:

User ID: {user_id}
Recent Interactions:
{interaction_summary}

Current Preferences:
- Style: {prefs.communication_style}
- Format: {prefs.response_format}
- Proactivity: {prefs.proactivity_level}

Analyze patterns and suggest updated preferences.
Consider:
1. Length and detail of responses user prefers
2. Format preferences (lists, paragraphs, etc.)
3. When they interact most
4. Tools they use frequently
5. Topics they discuss

Provide confident analysis only if patterns are clear.
"""

        try:
            analysis = await self.agent.a_format_class(
                pydantic_model=PreferenceAnalysis,
                prompt=prompt,
                auto_context=False,
                max_retries=2
            )

            # Update preferences if confidence is high
            if analysis.get('confidence', 0) > 0.6:
                prefs.communication_style = analysis['communication_style']
                prefs.response_format = analysis['response_format']
                prefs.proactivity_level = analysis['proactivity_level']
                prefs.preferred_tools = analysis['preferred_tools']
                prefs.topic_interests = analysis['topic_interests']
                prefs.time_preferences = analysis['time_pattern']
                prefs.last_updated = time.time()

                print(f"✓ Updated preferences for {user_id} (confidence: {analysis['confidence']})")

        except Exception as e:
            print(f"Preference learning failed: {e}")

    def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences"""
        if user_id not in self.preferences:
            self.preferences[user_id] = UserPreferences(user_id=user_id)
        return self.preferences[user_id]

    async def apply_preferences_to_query(
        self,
        user_id: str,
        query: str
    ) -> tuple[str, dict]:
        """
        Apply learned preferences to modify query or execution

        Returns:
            (modified_query, execution_hints)
        """
        prefs = self.get_preferences(user_id)

        execution_hints = {
            "response_format": prefs.response_format,
            "communication_style": prefs.communication_style,
            "preferred_tools": prefs.preferred_tools,
            "proactivity_level": prefs.proactivity_level
        }

        # Add style guidance to query if needed
        style_guidance = ""
        if prefs.communication_style == "concise":
            style_guidance = " (Respond concisely)"
        elif prefs.communication_style == "detailed":
            style_guidance = " (Provide detailed explanation)"

        modified_query = query + style_guidance

        return modified_query, execution_hints


# ===== MEMORY INJECTION SYSTEM =====


class MemoryStore:
    """
    Advanced memory system for injecting context
    """

    def __init__(self, max_memories: int = 5000):
        self.memories: dict[str, Memory] = {}
        self.max_memories = max_memories
        self.user_memories: dict[str, list[str]] = defaultdict(list)

    async def inject_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: dict = None,
        importance: float = 0.5,
        tags: list[str] = None
    ) -> str:
        """Inject a new memory"""
        memory = Memory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            tags=tags or []
        )

        self.memories[memory.id] = memory
        self.user_memories[user_id].append(memory.id)

        # Cleanup if too many
        if len(self.memories) > self.max_memories:
            await self._cleanup_old_memories()

        return memory.id

    async def _cleanup_old_memories(self):
        """Remove least important/accessed memories with proper error handling"""
        # Sort by importance and access
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.importance * 0.5 + (m.access_count / 100) * 0.5)
        )

        # Remove bottom 10%
        to_remove = int(len(sorted_memories) * 0.1)

        for memory in sorted_memories[:to_remove]:
            memory_id = memory.id
            user_id = memory.user_id

            # Sichere Löschung mit Error-Handling
            if memory_id in self.memories:
                del self.memories[memory_id]

            # Sichere Entfernung aus user_memories
            if user_id in self.user_memories:
                try:
                    self.user_memories[user_id].remove(memory_id)
                except ValueError:
                    pass  # Already removed

                # Leere Listen entfernen
                if not self.user_memories[user_id]:
                    del self.user_memories[user_id]

    async def get_relevant_memories(
        self,
        user_id: str,
        query: str = None,
        limit: int = 10,
        min_importance: float = 0.3
    ) -> list[Memory]:
        """Get relevant memories for context"""
        user_memory_ids = self.user_memories.get(user_id, [])
        user_memories = [
            self.memories[mid] for mid in user_memory_ids
            if mid in self.memories
        ]

        # Filter by importance
        relevant = [
            m for m in user_memories
            if m.importance >= min_importance
        ]

        # Update access stats
        for memory in relevant:
            memory.last_accessed = time.time()
            memory.access_count += 1

        # Sort by importance and recency
        relevant.sort(
            key=lambda m: (m.importance * 0.7 +
                           (time.time() - m.created_at) / 86400 * 0.3),
            reverse=True
        )

        return relevant[:limit]

    def format_memories_for_context(
        self,
        memories: list[Memory]
    ) -> str:
        """Format memories for LLM context"""
        if not memories:
            return ""

        sections = {
            MemoryType.FACT: [],
            MemoryType.PREFERENCE: [],
            MemoryType.EVENT: [],
            MemoryType.CONTEXT: []
        }

        for memory in memories:
            sections[memory.memory_type].append(memory.content)

        formatted = "## User Memory Context\n\n"

        if sections[MemoryType.PREFERENCE]:
            formatted += "**User Preferences:**\n"
            for pref in sections[MemoryType.PREFERENCE]:
                formatted += f"- {pref}\n"
            formatted += "\n"

        if sections[MemoryType.FACT]:
            formatted += "**Known Facts:**\n"
            for fact in sections[MemoryType.FACT]:
                formatted += f"- {fact}\n"
            formatted += "\n"

        if sections[MemoryType.EVENT]:
            formatted += "**Past Events:**\n"
            for event in sections[MemoryType.EVENT]:
                formatted += f"- {event}\n"
            formatted += "\n"

        return formatted


# ===== TASK SCHEDULER =====


class TaskScheduler:
    """
    Advanced task scheduler for user and agent tasks
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.tasks: dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the scheduler"""
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        print("✓ Task Scheduler started")

    async def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        print("✓ Task Scheduler stopped")

    async def schedule_task(
        self,
        user_id: str,
        task_type: str,
        content: str,
        scheduled_time: float = None,
        delay_seconds: float = None,
        priority: int = 5,
        recurrence: dict = None,
        metadata: dict = None
    ) -> str:
        """
        Schedule a task for execution with validation
        """
        # Validiere task_type
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Invalid task_type '{task_type}'. Valid types: {VALID_TASK_TYPES}")

        # Validiere und berechne scheduled_time
        now = time.time()

        if scheduled_time is None:
            if delay_seconds is None:
                delay_seconds = 0
            scheduled_time = now + max(0, delay_seconds)  # Nicht in der Vergangenheit
        else:
            # Wenn scheduled_time in der Vergangenheit liegt, führe sofort aus
            if scheduled_time < now:
                print(f"⚠️ Warning: scheduled_time in past, executing immediately")
                scheduled_time = now + 1  # 1 Sekunde Verzögerung für Queue-Verarbeitung

        # Validiere priority
        priority = max(0, min(10, priority))

        # Validiere content
        if not content or not content.strip():
            raise ValueError("Task content cannot be empty")

        task = ScheduledTask(
            user_id=user_id,
            task_type=task_type,
            content=content.strip(),
            scheduled_time=scheduled_time,
            priority=priority,
            recurrence=recurrence,
            metadata=metadata or {}
        )

        self.tasks[task.id] = task

        scheduled_dt = datetime.fromtimestamp(scheduled_time)
        delay_info = f"in {scheduled_time - now:.1f}s" if scheduled_time > now else "immediately"
        print(f"✓ Scheduled {task_type} task {task.id} for {scheduled_dt} ({delay_info})")

        return task.id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
        return False

    async def _scheduler_loop(self):
        """Main scheduler loop with improved task handling"""
        while self.running:
            try:
                await asyncio.sleep(1)  # Check every second
                now = time.time()

                # Sammle alle fälligen Tasks auf einmal
                due_tasks = [
                    task for task_id, task in list(self.tasks.items())
                    if task.status == TaskStatus.PENDING and task.scheduled_time <= now
                ]

                # Sortiere nach Priorität (höchste zuerst)
                due_tasks.sort(key=lambda t: t.priority, reverse=True)

                # Limitiere gleichzeitige Ausführungen
                max_concurrent = getattr(self.kernel.config, 'max_concurrent_tasks', 5)
                running_count = sum(
                    1 for t in self.tasks.values()
                    if t.status == TaskStatus.RUNNING
                )

                available_slots = max_concurrent - running_count

                for task in due_tasks[:available_slots]:
                    # Doppelte Ausführung verhindern
                    if task.status == TaskStatus.PENDING:
                        task.status = TaskStatus.RUNNING  # Sofort markieren
                        asyncio.create_task(self._execute_task(task))

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler loop error: {e}")
                import traceback
                traceback.print_exc()

    async def _execute_task(self, task: ScheduledTask):
        """Execute a scheduled task with proper user notification"""
        task.status = TaskStatus.RUNNING
        print(f"Executing task {task.id} content: {task.content}")

        try:
            # Create signal for the task
            signal = Signal(
                id=str(uuid.uuid4()),
                type=SignalType.SYSTEM_EVENT,
                priority=task.priority,
                content={
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "content": task.content
                },
                source="task_scheduler",
                timestamp=time.time(),
                metadata={
                    "user_id": task.user_id,
                    "scheduled_task": True
                }
            )

            # Emit signal
            await self.kernel.signal_bus.emit_signal(signal)

            if task.task_type == "reminder":
                await self.kernel.output_router.send_notification(
                    user_id=task.user_id,
                    content=f"⏰ Reminder: {task.content}",
                    priority=task.priority
                )

            elif task.task_type == "query":
                # Execute as agent query
                response = await self.kernel.agent.a_run(
                    query=task.content,
                    session_id=task.user_id,
                    user_id=task.user_id,
                    remember=True
                )
                task.result = response

                # Sende das Ergebnis an den Benutzer!
                await self.kernel.output_router.send_notification(
                    user_id=task.user_id,
                    content=f"📋 Scheduled Query Result:\n{response}",
                    priority=task.priority,
                    metadata={"task_id": task.id, "task_type": "query_result"}
                )

            elif task.task_type == "action":
                # Neuer Task-Typ "action" für proaktive Aktionen
                response = await self.kernel.agent.a_run(
                    query=f"Execute action: {task.content}",
                    session_id=task.user_id,
                    user_id=task.user_id,
                    remember=True
                )
                task.result = response
                await self.kernel.output_router.send_notification(
                    user_id=task.user_id,
                    content=f"✅ Action completed: {response[:200]}{'...' if len(response) > 200 else ''}",
                    priority=task.priority
                )

            task.status = TaskStatus.COMPLETED

            # Handle recurrence
            if task.recurrence:
                interval = task.recurrence.get("interval", 3600)
                new_time = task.scheduled_time + interval

                # Validiere, dass new_time in der Zukunft liegt
                if new_time <= time.time():
                    new_time = time.time() + interval

                await self.schedule_task(
                    user_id=task.user_id,
                    task_type=task.task_type,
                    content=task.content,
                    scheduled_time=new_time,
                    priority=task.priority,
                    recurrence=task.recurrence,
                    metadata=task.metadata
                )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            print(f"Task execution failed: {e}")

            # Benachrichtige User über fehlgeschlagene Tasks
            await self.kernel.output_router.send_notification(
                user_id=task.user_id,
                content=f"❌ Scheduled task failed: {task.content[:50]}...\nError: {str(e)[:100]}",
                priority=max(task.priority, 6)  # Mindestens mittlere Priorität
            )

    def get_user_tasks(
        self,
        user_id: str,
        status: TaskStatus = None
    ) -> list[ScheduledTask]:
        """Get tasks for a user"""
        tasks = [
            t for t in self.tasks.values()
            if t.user_id == user_id
        ]

        if status:
            tasks = [t for t in tasks if t.status == status]

        return sorted(tasks, key=lambda t: t.scheduled_time)


# ===== ADVANCED OUTPUT ROUTERS =====

class WebSocketOutputRouter(IOutputRouter):
    """WebSocket-based output router"""

    def __init__(self):
        self.connections: dict[str, Any] = {}  # user_id -> websocket
        self.pending_messages: dict[str, list] = defaultdict(list)
        self.max_pending = 50

    def register_connection(self, user_id: str, websocket):
        """Register a WebSocket connection"""
        self.connections[user_id] = websocket
        print(f"✓ WebSocket registered for {user_id}")
        asyncio.create_task(self._flush_pending(user_id))

    async def _flush_pending(self, user_id: str):
        """Send pending messages after reconnection"""
        if user_id not in self.pending_messages:
            return

        pending = self.pending_messages[user_id]
        self.pending_messages[user_id] = []

        for message in pending:
            try:
                ws = self.connections.get(user_id)
                if ws:
                    await ws.send_json(message)
            except Exception:
                self.pending_messages[user_id].append(message)
                break  # Connection failed again

    def unregister_connection(self, user_id: str):
        """Unregister a WebSocket connection"""
        if user_id in self.connections:
            del self.connections[user_id]
            print(f"✓ WebSocket unregistered for {user_id}")

    async def send_response(
        self,
        user_id: str,
        content: str,
        role: str = "assistant",
        metadata: dict = None
    ):
        """Send response via WebSocket"""
        if user_id not in self.connections:
            print(f"No WebSocket for {user_id}")
            return

        message = {
            "type": "response",
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        try:
            ws = self.connections[user_id]
            await ws.send_json(message)
        except Exception as e:
            print(f"WebSocket send failed: {e}")

    async def send_notification(
        self,
        user_id: str,
        content: str,
        priority: int = 5,
        metadata: dict = None
    ):
        """Send notification via WebSocket with fallback"""
        message = {
            "type": "notification",
            "content": content,
            "priority": priority,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        if user_id not in self.connections:
            # Queue statt verwerfen
            if len(self.pending_messages[user_id]) < self.max_pending:
                self.pending_messages[user_id].append(message)
                print(f"📥 Queued notification for offline user {user_id}")
            return

        try:
            ws = self.connections[user_id]
            await ws.send_json(message)
        except Exception as e:
            print(f"WebSocket send failed: {e}")
            # Bei Fehler auch queuen
            if len(self.pending_messages[user_id]) < self.max_pending:
                self.pending_messages[user_id].append(message)
            # Connection ist wahrscheinlich tot
            self.unregister_connection(user_id)

    async def send_intermediate_response(
        self,
        user_id: str,
        content: str,
        stage: str = "processing"
    ):
        """Send intermediate status update"""
        if user_id not in self.connections:
            return

        message = {
            "type": "intermediate",
            "stage": stage,
            "content": content,
            "timestamp": time.time()
        }

        try:
            ws = self.connections[user_id]
            await ws.send_json(message)
        except Exception as e:
            print(f"WebSocket send failed: {e}")


class MultiChannelRouter(IOutputRouter):
    """Route to multiple channels (console, websocket, etc.)"""

    def __init__(self):
        self.routers: list[IOutputRouter] = []

    def add_router(self, router: IOutputRouter):
        """Add a router"""
        self.routers.append(router)

    async def send_response(
        self,
        user_id: str,
        content: str,
        role: str = "assistant",
        metadata: dict = None
    ):
        """Send via all routers"""
        for router in self.routers:
            try:
                await router.send_response(user_id, content, role, metadata)
            except Exception as e:
                print(f"Router failed: {e}")

    async def send_notification(
        self,
        user_id: str,
        content: str,
        priority: int = 5,
        metadata: dict = None
    ):
        """Send notification via all routers"""
        for router in self.routers:
            try:
                await router.send_notification(user_id, content, priority, metadata)
            except Exception as e:
                print(f"Router failed: {e}")


# ===== AGENT INTEGRATION LAYER =====

class AgentIntegrationLayer:
    """
    Provides exported functions for the agent to interact with kernel
    """

    def __init__(self, kernel):
        self.kernel = kernel

    async def schedule_task(
        self,
        task_type: str,
        content: str,
        delay_seconds: float = None,
        scheduled_time: float = None,
        priority: int = 5
    ) -> str:
        """
        Schedule a task (callable by agent)

        Example:
            task_id = await schedule_task(
                "reminder",
                "Follow up on project X",
                delay_seconds=3600
            )
        """
        user_id = self.kernel._current_user_id or "system"

        return await self.kernel.scheduler.schedule_task(
            user_id=user_id,
            task_type=task_type,
            content=content,
            scheduled_time=scheduled_time,
            delay_seconds=delay_seconds,
            priority=priority
        )

    async def send_intermediate_response(
        self,
        content: str,
        stage: str = "processing"
    ):
        """
        Send intermediate response while processing

        Example:
            await send_intermediate_response(
                "Analyzing data...",
                stage="analysis"
            )
        """
        user_id = self.kernel._current_user_id or "system"

        if hasattr(self.kernel.output_router, 'send_intermediate_response'):
            await self.kernel.output_router.send_intermediate_response(
                user_id, content, stage
            )
        else:
            # Fallback to notification
            await self.kernel.output_router.send_notification(
                user_id, f"[{stage}] {content}", priority=3
            )

    async def ask_user(
        self,
        question: str,
        timeout: float = 300.0
    ) -> str:
        """
        Ask user a question and wait for response

        Example:
            answer = await ask_user(
                "Which option do you prefer: A or B?",
                timeout=60.0
            )
        """
        user_id = self.kernel._current_user_id or "system"

        # Send question
        await self.kernel.output_router.send_notification(
            user_id=user_id,
            content=f"❓ {question}",
            priority=8,
            metadata={"requires_response": True}
        )

        # Wait for response
        response_future = asyncio.Future()
        question_id = str(uuid.uuid4())

        # Register response handler
        self.kernel._pending_questions[question_id] = response_future

        try:
            answer = await asyncio.wait_for(response_future, timeout=timeout)
            return answer
        except asyncio.TimeoutError:
            return None
        finally:
            del self.kernel._pending_questions[question_id]

    async def inject_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: list[str] = None
    ) -> str:
        """
        Inject a memory for current user

        Example:
            memory_id = await inject_memory(
                "User prefers concise responses",
                memory_type="preference",
                importance=0.8
            )
        """
        user_id = self.kernel._current_user_id or "system"

        from toolboxv2.mods.isaa.kernel.types import MemoryType
        mem_type = MemoryType[memory_type.upper()]

        return await self.kernel.memory_store.inject_memory(
            user_id=user_id,
            memory_type=mem_type,
            content=content,
            importance=importance,
            tags=tags or []
        )

    async def get_user_preferences(self) -> dict:
        """
        Get current user's learned preferences

        Example:
            prefs = await get_user_preferences()
            style = prefs.get('communication_style')
        """
        user_id = self.kernel._current_user_id or "system"
        prefs = self.kernel.learning_engine.get_preferences(user_id)
        return prefs.model_dump()

    async def record_feedback(
        self,
        feedback: str,
        score: float
    ):
        """
        Record feedback for learning

        Example:
            await record_feedback("Response was too long", -0.5)
        """
        user_id = self.kernel._current_user_id or "system"

        await self.kernel.learning_engine.record_interaction(
            user_id=user_id,
            interaction_type=InteractionType.FEEDBACK,
            content={"feedback": feedback},
            feedback_score=score
        )

