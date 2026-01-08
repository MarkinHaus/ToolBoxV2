"""
Voice-First Streaming System for FlowAgent V2

Intelligent a_stream that:
- Instantly responds to simple queries (streamed)
- Detects hard capabilities → spawns a_run in background
- Natural voice-like transitions: "Ok moment..." → [work] → "Here's the result..."
- Non-blocking: User can continue while agent works

Author: FlowAgent V2
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent


# =============================================================================
# CAPABILITY DETECTION
# =============================================================================

class QueryComplexity(str, Enum):
    """Query complexity levels"""
    INSTANT = "instant"          # Direct answer, stream immediately
    QUICK_TOOL = "quick_tool"    # Single fast tool call
    HARD_TASK = "hard_task"      # Complex, needs a_run


class CapabilityDetection(BaseModel):
    """Fast capability detection (optimized for 0.5B models)"""
    complexity: str = Field(description="instant|quick_tool|hard_task")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    tool_hint: str | None = Field(default=None, description="Tool name if quick_tool")
    reason: str = Field(default="", max_length=50)


# =============================================================================
# BACKGROUND TASK MANAGER
# =============================================================================

@dataclass
class BackgroundTask:
    """Track background a_run executions"""
    task_id: str
    query: str
    status: str = "running"  # running, completed, failed
    result: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Async task reference
    _task: asyncio.Task | None = None


class BackgroundTaskManager:
    """Manages background a_run executions"""

    def __init__(self):
        self.tasks: dict[str, BackgroundTask] = {}
        self._callbacks: dict[str, Callable] = {}

    def create_task(
        self,
        coro,
        query: str,
        callback: Callable[[str, str], None] | None = None
    ) -> str:
        """Create and start background task"""
        task_id = f"bg_{uuid.uuid4().hex[:8]}"

        bg_task = BackgroundTask(
            task_id=task_id,
            query=query
        )

        async def wrapped():
            try:
                result = await coro
                bg_task.result = result
                bg_task.status = "completed"
                bg_task.completed_at = datetime.now()

                if callback:
                    callback(task_id, result)

            except Exception as e:
                bg_task.error = str(e)
                bg_task.status = "failed"
                bg_task.completed_at = datetime.now()

                if callback:
                    callback(task_id, f"Error: {e}")

        bg_task._task = asyncio.create_task(wrapped())
        self.tasks[task_id] = bg_task

        if callback:
            self._callbacks[task_id] = callback

        return task_id

    def get_status(self, task_id: str) -> dict | None:
        """Get task status"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task_id,
            "query": task.query[:50],
            "status": task.status,
            "result": task.result,
            "error": task.error,
            "duration": (
                (task.completed_at or datetime.now()) - task.started_at
            ).total_seconds()
        }

    def get_result(self, task_id: str) -> str | None:
        """Get completed task result"""
        task = self.tasks.get(task_id)
        if task and task.status == "completed":
            return task.result
        return None

    def list_pending(self) -> list[dict]:
        """List all running tasks"""
        return [
            self.get_status(tid)
            for tid, t in self.tasks.items()
            if t.status == "running"
        ]

    async def wait_for(self, task_id: str, timeout: float = 60.0) -> str | None:
        """Wait for task completion"""
        task = self.tasks.get(task_id)
        if not task or not task._task:
            return None

        try:
            await asyncio.wait_for(task._task, timeout=timeout)
            return task.result
        except asyncio.TimeoutError:
            return None

    def cancel(self, task_id: str) -> bool:
        """Cancel a running task"""
        task = self.tasks.get(task_id)
        if task and task._task and not task._task.done():
            task._task.cancel()
            task.status = "cancelled"
            return True
        return False


# =============================================================================
# VOICE-AWARE PHRASES
# =============================================================================

VOICE_PHRASES = {
    "de": {
        "thinking": "Moment, ich denke nach...",
        "starting_task": "Ok, ich kümmere mich darum...",
        "quick_tool": "Kurz, ich schaue nach...",
        "working": "Ich arbeite daran...",
        "done": "Fertig! ",
        "result_ready": "Ok, hier ist das Ergebnis: ",
        "will_callback": "Das dauert etwas länger. Ich melde mich wenn ich fertig bin.",
        "background_done": "Ich bin fertig mit deiner Anfrage: ",
        "checking_status": "Ich prüfe den Status...",
        "still_working": "Ich arbeite noch daran, bitte warte kurz...",
        "error": "Entschuldigung, da ist etwas schiefgelaufen: ",
    },
    "en": {
        "thinking": "Moment, let me think...",
        "starting_task": "Ok, I'll take care of that...",
        "quick_tool": "Let me check...",
        "working": "Working on it...",
        "done": "Done! ",
        "result_ready": "Ok, here's the result: ",
        "will_callback": "This will take a moment. I'll let you know when I'm done.",
        "background_done": "I finished your request: ",
        "checking_status": "Checking the status...",
        "still_working": "Still working on it, please wait...",
        "error": "Sorry, something went wrong: ",
    }
}


# =============================================================================
# STREAMING ENGINE
# =============================================================================

class VoiceStreamEngine:
    """
    Voice-first streaming engine.

    Handles:
    - Instant responses (streamed)
    - Quick tool calls (inline)
    - Hard tasks (background a_run)
    """

    def __init__(
        self,
        agent: 'FlowAgent',
        language: str = "en",
        callback_on_complete: Callable[[str, str, str], None] | None = None
    ):
        self.agent = agent
        self.language = language
        self.phrases = VOICE_PHRASES.get(language, VOICE_PHRASES["en"])
        self.callback_on_complete = callback_on_complete

        self.background_manager = BackgroundTaskManager()

        # Quick tool patterns (regex-free, keyword based)
        # In __init__, quick_tool_patterns anpassen:
        self.quick_tool_patterns = {
            "time": ["time", "zeit", "uhrzeit", "wie spät", "what time", "current time"],
            "date": ["date", "datum", "today's date", "welches datum", "heute"],
            "weather": ["weather", "wetter"],
            "calculate": ["calculate", "berechne", "rechne", "math", "rechnung"],
            "search": ["search", "suche", "find", "finde", "research", "recherchieren"],
            "reminder": ["reminder", "remind", "erinnerung", "erinnere", "erinnere mich"],

        }
    def _get_phrase(self, key: str) -> str:
        return self.phrases.get(key, self.phrases.get("thinking", "..."))

    async def _detect_complexity(
        self,
        query: str,
        session_id: str
    ) -> CapabilityDetection:
        """
        Fast complexity detection with priority-based heuristics.

        Priority order:
        1. Quick tool patterns (highest specificity)
        2. Hard task action keywords
        3. Query structure analysis
        4. LLM fallback for ambiguous cases
        """
        query_lower = query.lower().strip()
        query_len = len(query_lower)
        words = query_lower.split()
        word_count = len(words)

        # =================================================================
        # PRIORITY 1: Quick tool patterns (most specific, check first)
        # =================================================================
        for tool_name, patterns in self.quick_tool_patterns.items():
            if any(p in query_lower for p in patterns):
                return CapabilityDetection(
                    complexity="quick_tool",
                    confidence=0.9,
                    tool_hint=tool_name,
                    reason=f"pattern_{tool_name}"
                )

        # =================================================================
        # PRIORITY 2: Hard task detection
        # =================================================================

        # Strong action verbs that almost always need a_run
        hard_action_verbs = {
            "create", "erstelle", "erstell",
            "build", "baue", "bau",
            "write", "schreib", "schreibe",
            "generate", "generiere", "generier",
            "analyze", "analysiere", "analysier",
            "research", "recherchiere", "recherchier",
            "summarize", "zusammenfassen", "fasse zusammen",
            "compare", "vergleiche", "vergleich",
            "fix", "repariere", "reparier",
            "debug", "solve", "löse", "lös",
            "implement", "implementiere", "implementier",
            "refactor", "optimize", "optimiere",
            "deploy", "install", "installiere",
            "configure", "konfiguriere", "konfigur",
            "migrate", "migriere", "migrier",
        }

        # Check if query starts with or contains action verb
        first_word = words[0] if words else ""
        has_hard_action = (
            first_word in hard_action_verbs or
            any(verb in query_lower for verb in hard_action_verbs if len(verb) > 4)
        )

        # Multi-step indicators
        multi_step_patterns = [
            " and ", " und ", " then ", " dann ",
            " also ", " auch ", " plus ",
            "step", "schritt", "first", "zuerst",
            "multiple", "mehrere", "all ", "alle ",
        ]
        has_multi_step = any(p in query_lower for p in multi_step_patterns)

        # Complex query structure
        is_structurally_complex = (
            query_len > 150 or
            word_count > 25 or
            query_lower.count('.') > 1 or
            query_lower.count(',') > 3
        )

        if has_hard_action and self.agent.tool_manager.entries:
            confidence = 0.85 if has_multi_step else 0.75
            return CapabilityDetection(
                complexity="hard_task",
                confidence=confidence,
                reason="action_verb"
            )

        if is_structurally_complex and has_multi_step:
            return CapabilityDetection(
                complexity="hard_task",
                confidence=0.7,
                reason="complex_structure"
            )

        # =================================================================
        # PRIORITY 3: Instant detection (simple queries)
        # =================================================================

        # Greetings and simple social
        greeting_patterns = [
            "hi", "hello", "hey", "hallo", "moin", "servus",
            "good morning", "guten morgen", "good evening",
            "how are you", "wie geht", "what's up", "was geht",
            "thanks", "danke", "thank you", "bye", "tschüss",
        ]
        if any(query_lower.startswith(g) or query_lower == g for g in greeting_patterns):
            return CapabilityDetection(
                complexity="instant",
                confidence=0.95,
                reason="greeting"
            )

        # Simple questions (what is, who is, why, how does)
        simple_question_starts = [
            "what is", "was ist", "wer ist", "who is",
            "why", "warum", "wieso", "weshalb",
            "how does", "wie funktioniert",
            "what does", "was bedeutet", "was macht",
            "can you explain", "erkläre", "explain",
            "tell me about", "erzähl mir",
            "define", "definiere",
        ]
        if any(query_lower.startswith(q) for q in simple_question_starts):
            # But not if combined with action request
            if not has_hard_action and not has_multi_step:
                return CapabilityDetection(
                    complexity="instant",
                    confidence=0.85,
                    reason="simple_question"
                )

        # Short queries without action intent
        if word_count <= 5 and not has_hard_action:
            # Check it's not a tool query we might have missed
            tool_indicators = ["get", "hol", "fetch", "show", "zeig", "check", "prüf"]
            if not any(t in query_lower for t in tool_indicators):
                return CapabilityDetection(
                    complexity="instant",
                    confidence=0.8,
                    reason="short_simple"
                )

        # =================================================================
        # PRIORITY 4: LLM fallback for ambiguous cases
        # =================================================================
        tools_available = len(self.agent.tool_manager.entries) > 0

        try:
            categories = self.agent.tool_manager.list_categories()[:5]

            prompt = f"""Classify query complexity:
Query: "{query[:100]}"
Tools: {tools_available} | Categories: {', '.join(categories) if categories else 'none'}

instant = direct answer, no tools needed
quick_tool = single fast tool call
hard_task = multiple steps or complex reasoning"""

            result = await self.agent.a_format_class(
                CapabilityDetection,
                prompt,
                model_preference="fast",
                max_retries=0,
                auto_context=False
            )

            return CapabilityDetection(**result)

        except Exception:
            # Safe fallback: instant for short, hard for long
            if word_count <= 8:
                return CapabilityDetection(
                    complexity="instant",
                    confidence=0.5,
                    reason="fallback_short"
                )
            return CapabilityDetection(
                complexity="hard_task" if tools_available else "instant",
                confidence=0.5,
                reason="fallback_long"
            )

    async def stream(
        self,
        query: str,
        session_id: str = "default",
        remember: bool = True,
        force_mode: str | None = None,  # "instant", "quick_tool", "hard_task", None=auto
        wait_for_hard: bool = False,    # Wait for hard tasks or return immediately
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Main streaming entry point.

        Yields:
            str: Streamed response chunks

        Special yields:
            "__TASK_STARTED__:{task_id}" - Background task started
            "__TASK_DONE__:{task_id}:{result}" - Background task completed (if subscribed)
        """
        self.agent.active_session = session_id
        self.agent.is_running = True

        try:
            session = await self.agent.session_manager.get_or_create(session_id)

            # Check for status query about background tasks
            status_check = self._check_status_query(query)
            if status_check:
                async for chunk in self._handle_status_query(status_check):
                    yield chunk
                return

            # Detect complexity
            if force_mode:
                detection = CapabilityDetection(
                    complexity=force_mode,
                    confidence=1.0,
                    reason="forced"
                )
            else:
                detection = await self._detect_complexity(query, session_id)

            # Route based on complexity
            if detection.complexity == "instant":
                async for chunk in self._handle_instant(
                    query, session, remember, **kwargs
                ):
                    yield chunk

            elif detection.complexity == "quick_tool":
                async for chunk in self._handle_quick_tool(
                    query, session, remember, detection.tool_hint, **kwargs
                ):
                    yield chunk

            elif detection.complexity == "hard_task":
                async for chunk in self._handle_hard_task(
                    query, session, remember, wait_for_hard, **kwargs
                ):
                    yield chunk

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield self._get_phrase("error") + str(e)

        finally:
            self.agent.is_running = False
            # self.agent.active_session = None

    def _check_status_query(self, query: str) -> str | None:
        """Check if query is asking about task status"""
        query_lower = query.lower()

        status_patterns = [
            "status", "fertig", "done", "ready", "bereit",
            "wie weit", "how far", "progress", "fortschritt"
        ]

        if any(p in query_lower for p in status_patterns):
            # Check for task ID pattern in query (bg_XXXXX)
            import re
            task_id_match = re.search(r'bg_[a-f0-9]+', query_lower)
            if task_id_match:
                return task_id_match.group(0)

            # Check registered tasks
            for task_id in self.background_manager.tasks.keys():
                parts = task_id.split("_")
                if task_id in query_lower or (len(parts) > 1 and parts[1] in query_lower):
                    return task_id

            # Return "any" to check all pending
            pending = self.background_manager.list_pending()
            if pending:
                return "any"

        return None

    async def _handle_status_query(self, task_ref: str) -> AsyncGenerator[str, None]:
        """Handle status check queries"""

        if task_ref == "any":
            pending = self.background_manager.list_pending()
            if pending:
                yield self._get_phrase("still_working")
                for task in pending:
                    yield f"\n- {task['query']}: {task['status']}"
            else:
                # Check recently completed
                completed = [
                    t for t in self.background_manager.tasks.values()
                    if t.status == "completed"
                ]
                if completed:
                    latest = max(completed, key=lambda t: t.completed_at or datetime.min)
                    yield self._get_phrase("background_done")
                    yield latest.result or ""
                else:
                    yield "Keine laufenden Aufgaben." if self.language == "de" else "No running tasks."
        else:
            status = self.background_manager.get_status(task_ref)
            if status:
                if status["status"] == "completed":
                    yield self._get_phrase("result_ready")
                    yield status["result"] or ""
                elif status["status"] == "running":
                    yield self._get_phrase("still_working")
                elif status["status"] == "failed":
                    yield self._get_phrase("error") + (status["error"] or "Unknown error")
            else:
                yield "Task nicht gefunden." if self.language == "de" else "Task not found."

    async def _handle_instant(
        self,
        query: str,
        session,
        remember: bool,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Handle instant response - direct streaming"""

        if remember:
            await session.add_message({"role": "user", "content": query})

        # Get context
        history = session.get_history_for_llm(last_n=5)

        # Check for RAG context (but don't wait long)
        rag_context = ""
        try:
            rag_task = asyncio.create_task(session.get_reference(query))
            rag_context = await asyncio.wait_for(rag_task, timeout=0.5)
        except (asyncio.TimeoutError, Exception):
            pass

        messages = []
        if rag_context:
            messages.append({"role": "system", "content": f"Context:\n{rag_context[:1000]}"})
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        # Stream response
        full_response = ""

        model = self.agent.amd.fast_llm_model
        llm_kwargs = {
            'model': model,
            'messages': messages,
            'stream': True,
            'stream_options': {"include_usage": True},
            'max_tokens': kwargs.get('max_tokens', 500),
            'temperature': kwargs.get('temperature', 0.7)
        }

        api_key = self.agent._get_api_key_for_model(model)
        if api_key:
            llm_kwargs['api_key'] = api_key

        try:
            import litellm
            response = await self.agent.llm_handler.completion_with_rate_limiting(
                litellm, **llm_kwargs
            )
            # Falls response ein coroutine ist, await es
            if asyncio.iscoroutine(response):
                response = await response

            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                if content:
                    yield content

        except Exception as e:
            yield self._get_phrase("error") + str(e)
            return

        if remember and full_response:
            await session.add_message({"role": "assistant", "content": full_response})

    async def _handle_quick_tool(
        self,
        query: str,
        session,
        remember: bool,
        tool_hint: str | None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Handle quick tool call - inline execution"""

        # Acknowledge
        yield self._get_phrase("quick_tool")

        if remember:
            await session.add_message({"role": "user", "content": query})

        # Find tool
        tool_name = None
        if tool_hint:
            # Check if we have this tool
            tool = self.agent.tool_manager.get(tool_hint)
            if tool:
                tool_name = tool_hint

        if not tool_name:
            # Let LLM pick tool
            tools = self.agent.tool_manager.get_all()
            if not tools:
                # No tools, fall back to instant
                yield " "
                async for chunk in self._handle_instant(query, session, False, **kwargs):
                    yield chunk
                return

            tool_list = [{"name": "get_task_result", "desc": "Get result of previous task", "args": {"task_id": "string"}},
                          {"name": "cancel_task", "desc": "Cancel a pending task", "args": {"task_id": "string"}},
                          {"name": "wait_for_task", "desc": "Wait for a task to complete", "args": {"task_id": "string", "timeout": "number"}},
                          {"name": "get_pending_tasks", "desc": "Get list of pending task", "args": {}},

                         ]+[{"name": t.name, "desc": t.description[:250], "args": t.args_schema} for t in tools]

            try:
                from pydantic import BaseModel, Field

                class QuickToolSelect(BaseModel):
                    tool: str = Field(description="Tool name to use")
                    args: dict = Field(default_factory=dict, description="Tool arguments")

                result = await self.agent.a_format_class(
                    QuickToolSelect,
                    f"Query: {query}\nTools: {json.dumps(tool_list)}\nSelect best tool and args.",
                    model_preference="fast",
                    max_retries=0
                )
                if asyncio.iscoroutine(result):
                    result = await result

                tool_name = result.get("tool")
                tool_args = result.get("args", {})

            except Exception:
                # Fall back to instant
                yield " "
                async for chunk in self._handle_instant(query, session, False, **kwargs):
                    yield chunk
                return
        else:
            tool_args = {}

        # Execute tool
        try:
            tool_result = await self.agent.arun_function(tool_name, **tool_args)
            result_str = json.dumps(tool_result, default=str, ensure_ascii=False)[:1000]

        except Exception as e:
            yield self._get_phrase("error") + str(e)
            return

        # Generate response with tool result
        yield self._get_phrase("done")

        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"[Called {tool_name}]"},
            {"role": "user", "content": f"Tool result: {result_str}\n\nRespond naturally to the original query."}
        ]

        full_response = ""

        try:
            import litellm
            response = await self.agent.llm_handler.completion_with_rate_limiting(
                litellm,
                model=self.agent.amd.fast_llm_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=300
            )

            async for chunk in response:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                if content:
                    yield content

        except Exception as e:
            yield str(tool_result)[:500]
            full_response = str(tool_result)[:500]

        if remember and full_response:
            await session.add_message({"role": "assistant", "content": full_response})

    async def _handle_hard_task(
        self,
        query: str,
        session,
        remember: bool,
        wait_for_result: bool,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Handle hard task - background a_run"""

        # Acknowledge immediately
        yield self._get_phrase("starting_task")

        if remember:
            await session.add_message({"role": "user", "content": query})

        # Define callback for completion
        completed_result = None
        completion_event = asyncio.Event()

        def on_complete(task_id: str, result: str):
            nonlocal completed_result
            completed_result = result
            completion_event.set()

            if self.callback_on_complete:
                self.callback_on_complete(task_id, query, result)

        # Start a_run in background
        coro = self.agent.a_run(
            query=query,
            session_id=session.session_id,
            remember=False,  # We already added the message
            intermediate_callback=None,  # Could wire up for live updates
            **kwargs
        )

        task_id = self.background_manager.create_task(
            coro,
            query=query,
            callback=on_complete
        )

        # Yield task started marker
        yield f"\n__TASK_STARTED__:{task_id}\n"

        if wait_for_result:
            # Wait for completion
            yield self._get_phrase("working")

            try:
                await asyncio.wait_for(completion_event.wait(), timeout=120.0)

                if completed_result:
                    yield "\n" + self._get_phrase("result_ready")
                    yield completed_result

                    if remember:
                        await session.add_message({
                            "role": "assistant",
                            "content": completed_result
                        })
                else:
                    yield "\n" + self._get_phrase("error") + "No result"

            except asyncio.TimeoutError:
                yield "\n" + self._get_phrase("will_callback")
                yield f"\nTask ID: {task_id}"
        else:
            # Return immediately
            yield self._get_phrase("will_callback")
            yield f"\nTask ID: {task_id}"

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_pending_tasks(self) -> list[dict]:
        """Get all pending background tasks"""
        return self.background_manager.list_pending()

    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> str | None:
        """Wait for a specific task to complete"""
        return await self.background_manager.wait_for(task_id, timeout)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        return self.background_manager.cancel(task_id)

    def get_task_result(self, task_id: str) -> str | None:
        """Get result of completed task"""
        return self.background_manager.get_result(task_id)


# =============================================================================
# INTEGRATION WITH FLOWAGENT
# =============================================================================

async def voice_stream(
    agent: 'FlowAgent',
    query: str,
    session_id: str = "default",
    language: str = "en",
    wait_for_hard: bool = False,
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Convenience function for voice-first streaming.

    Usage:
        async for chunk in voice_stream(agent, "What's the weather?"):
            print(chunk, end="", flush=True)
    """
    engine = VoiceStreamEngine(agent, language=language)

    async for chunk in engine.stream(
        query=query,
        session_id=session_id,
        wait_for_hard=wait_for_hard,
        **kwargs
    ):
        yield chunk
