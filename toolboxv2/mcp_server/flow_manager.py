"""
FlowSessionManager - Production-Ready Flow Execution for MCP
============================================================

Captures stdin/stdout from flows, enabling step-by-step execution
for AI agents. Acts as gateway between MCP protocol and ToolBoxV2 flows.

Architecture:
- FlowSession: Holds state, I/O buffers, execution context
- FlowExecutor: Runs flow coroutines with captured I/O
- FlowSessionManager: Orchestrates sessions, handles lifecycle
"""

from __future__ import annotations

import asyncio
import time
import uuid
import sys
import io
import json
import inspect
import logging
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Callable,
    Coroutine,
    Union,
    TYPE_CHECKING,
)
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from toolboxv2.utils.toolbox import App

logger = logging.getLogger("mcp.flows")


# =============================================================================
# ENUMS & DATA TYPES
# =============================================================================


class FlowState(Enum):
    """Flow execution states."""
    CREATED = "created"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class IOType(Enum):
    """I/O entry types."""
    STDOUT = "stdout"
    STDERR = "stderr"
    STDIN = "stdin"
    PROMPT = "prompt"
    RESULT = "result"


@dataclass
class IOEntry:
    """Single I/O entry with metadata."""
    type: IOType
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class FlowSession:
    """
    Represents an active flow execution session.
    
    Holds all state needed for step-by-step flow execution,
    including I/O buffers and execution context.
    """
    __slots__ = (
        "session_id", "flow_name", "created", "last_activity",
        "state", "context", "history", "io_buffer", "pending_input",
        "current_prompt", "result", "error", "_task", "_input_event",
        "_cancelled"
    )
    
    session_id: str
    flow_name: str
    created: float
    last_activity: float
    state: FlowState
    context: Dict[str, Any]
    history: List[str]
    io_buffer: List[IOEntry]
    pending_input: Optional[str]
    current_prompt: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    _task: Optional[asyncio.Task]
    _input_event: Optional[asyncio.Event]
    _cancelled: bool
    
    def __init__(
        self,
        session_id: str,
        flow_name: str,
        created: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.flow_name = flow_name
        self.created = created or time.time()
        self.last_activity = self.created
        self.state = FlowState.CREATED
        self.context = context or {}
        self.history = []
        self.io_buffer = []
        self.pending_input = None
        self.current_prompt = None
        self.result = None
        self.error = None
        self._task = None
        self._input_event = None
        self._cancelled = False
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def is_expired(self, timeout: int) -> bool:
        """Check if session has expired."""
        return (time.time() - self.last_activity) > timeout
    
    def add_output(self, content: str, io_type: IOType = IOType.STDOUT) -> None:
        """Add output to I/O buffer."""
        self.io_buffer.append(IOEntry(
            type=io_type,
            content=content,
            timestamp=time.time()
        ))
        self.history.append(f"[{io_type.value}] {content[:100]}...")
    
    def get_pending_output(self) -> List[Dict[str, Any]]:
        """Get and clear pending output."""
        output = [entry.to_dict() for entry in self.io_buffer]
        self.io_buffer.clear()
        return output
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "flow_name": self.flow_name,
            "state": self.state.value,
            "created": self.created,
            "last_activity": self.last_activity,
            "current_prompt": self.current_prompt,
            "history_count": len(self.history),
            "has_pending_output": len(self.io_buffer) > 0,
            "result": self.result,
            "error": self.error
        }


# =============================================================================
# CAPTURED I/O STREAMS
# =============================================================================


class CapturedInput:
    """
    Mock stdin that waits for async input from session.
    
    When flow calls input(), this blocks until the agent
    provides input via flow_continue().
    """
    
    def __init__(self, session: FlowSession, loop: asyncio.AbstractEventLoop):
        self._session = session
        self._loop = loop
        self._buffer = io.StringIO()
    
    def readline(self) -> str:
        """Block until input is provided."""
        # Set session to waiting state
        self._session.state = FlowState.WAITING_INPUT
        self._session._input_event = asyncio.Event()
        
        # Wait for input (blocking in sync context)
        future = asyncio.run_coroutine_threadsafe(
            self._wait_for_input(),
            self._loop
        )
        
        try:
            # Wait with timeout (5 minutes max)
            result = future.result(timeout=300)
            self._session.state = FlowState.RUNNING
            return result + "\n"
        except Exception as e:
            logger.error(f"Input wait error: {e}")
            return "\n"
    
    async def _wait_for_input(self) -> str:
        """Async wait for input event."""
        if self._session._input_event:
            await self._session._input_event.wait()
        
        input_val = self._session.pending_input or ""
        self._session.pending_input = None
        self._session._input_event = None
        
        # Record input in history
        self._session.add_output(input_val, IOType.STDIN)
        
        return input_val
    
    def read(self, size: int = -1) -> str:
        return self.readline()


class CapturedOutput:
    """
    Mock stdout/stderr that captures output to session buffer.
    """
    
    def __init__(self, session: FlowSession, io_type: IOType = IOType.STDOUT):
        self._session = session
        self._io_type = io_type
        self._buffer = io.StringIO()
    
    def write(self, text: str) -> int:
        if text and text.strip():
            self._session.add_output(text.rstrip(), self._io_type)
        return len(text)
    
    def flush(self) -> None:
        pass
    
    def isatty(self) -> bool:
        return False


# =============================================================================
# FLOW EXECUTOR
# =============================================================================


class FlowExecutor:
    """
    Executes flow functions with captured I/O.
    
    Handles both sync and async flows, capturing all stdin/stdout/stderr
    and converting interactive flows to step-by-step execution.
    """
    
    def __init__(self, app: "App", executor: Optional[ThreadPoolExecutor] = None):
        self._app = app
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    async def execute(
        self,
        session: FlowSession,
        flow_func: Callable,
        **kwargs
    ) -> Any:
        """
        Execute a flow function with captured I/O.
        
        Args:
            session: The flow session
            flow_func: The flow's run() function
            **kwargs: Arguments to pass to the flow
        
        Returns:
            The flow's return value
        """
        self._loop = asyncio.get_event_loop()
        
        # Create captured I/O streams
        captured_stdin = CapturedInput(session, self._loop)
        captured_stdout = CapturedOutput(session, IOType.STDOUT)
        captured_stderr = CapturedOutput(session, IOType.STDERR)
        
        # Patch builtins for input() capture
        original_input = __builtins__.get("input") if isinstance(__builtins__, dict) else getattr(__builtins__, "input", input)
        
        def captured_input(prompt: str = "") -> str:
            """Captured input function."""
            if prompt:
                session.current_prompt = prompt
                session.add_output(prompt, IOType.PROMPT)
            
            session.state = FlowState.WAITING_INPUT
            session._input_event = asyncio.Event()
            
            # Wait synchronously for async input
            future = asyncio.run_coroutine_threadsafe(
                self._wait_for_input(session),
                self._loop
            )
            
            try:
                result = future.result(timeout=300)
                session.state = FlowState.RUNNING
                session.current_prompt = None
                return result
            except asyncio.CancelledError:
                raise KeyboardInterrupt("Flow cancelled")
            except Exception as e:
                logger.error(f"Input error: {e}")
                raise
        
        # Execute with captured I/O
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            sys.stdin = captured_stdin
            sys.stdout = captured_stdout
            sys.stderr = captured_stderr
            
            # Patch input
            if isinstance(__builtins__, dict):
                __builtins__["input"] = captured_input
            else:
                setattr(__builtins__, "input", captured_input)
            
            session.state = FlowState.RUNNING
            
            # Execute flow
            if asyncio.iscoroutinefunction(flow_func):
                result = await flow_func(self._app, self._app.args_sto, **kwargs)
            else:
                # Run sync flow in executor
                result = await self._loop.run_in_executor(
                    self._executor,
                    lambda: flow_func(self._app, self._app.args_sto, **kwargs)
                )
            
            session.state = FlowState.COMPLETED
            session.result = result
            session.add_output(json.dumps(result) if result else "Flow completed", IOType.RESULT)
            
            return result
            
        except asyncio.CancelledError:
            session.state = FlowState.CANCELLED
            session.error = "Flow cancelled by user"
            raise
        except Exception as e:
            session.state = FlowState.ERROR
            session.error = str(e)
            session.add_output(f"Error: {e}", IOType.STDERR)
            logger.exception(f"Flow execution error: {e}")
            raise
        finally:
            # Restore I/O
            sys.stdin = old_stdin
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            if isinstance(__builtins__, dict):
                __builtins__["input"] = original_input
            else:
                setattr(__builtins__, "input", original_input)
    
    async def _wait_for_input(self, session: FlowSession) -> str:
        """Wait for input from agent."""
        if session._input_event:
            # Wait with periodic check for cancellation
            while not session._input_event.is_set():
                if session._cancelled:
                    raise asyncio.CancelledError()
                try:
                    await asyncio.wait_for(
                        session._input_event.wait(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
        
        result = session.pending_input or ""
        session.pending_input = None
        session._input_event = None
        session.add_output(result, IOType.STDIN)
        
        return result
    
    def close(self) -> None:
        """Cleanup executor."""
        self._executor.shutdown(wait=False)


# =============================================================================
# FLOW SESSION MANAGER
# =============================================================================


class FlowSessionManager:
    """
    Production-ready flow session management.
    
    Features:
    - Session lifecycle management (create, get, update, delete)
    - Automatic expiration cleanup
    - Max session limit enforcement
    - I/O capture and buffering
    - Step-by-step flow execution
    - Thread-safe operations
    """
    
    def __init__(
        self,
        app: Optional["App"] = None,
        max_sessions: int = 100,
        timeout: int = 3600
    ):
        self._app = app
        self._sessions: Dict[str, FlowSession] = {}
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._timeout = timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        self._executor: Optional[FlowExecutor] = None
    
    def set_app(self, app: "App") -> None:
        """Set the app reference (for lazy initialization)."""
        self._app = app
        self._executor = FlowExecutor(app)
    
    @property
    def count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)
    
    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------
    
    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Flow cleanup task started")
    
    async def stop_cleanup(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Flow cleanup task stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(300)
                count = await self.cleanup_expired()
                if count > 0:
                    logger.info(f"Cleaned up {count} expired flow sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        async with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired(self._timeout)
            ]
            for sid in expired:
                session = self._sessions[sid]
                # Cancel running task
                if session._task and not session._task.done():
                    session._cancelled = True
                    session._task.cancel()
                del self._sessions[sid]
            return len(expired)
    
    # -------------------------------------------------------------------------
    # SESSION OPERATIONS
    # -------------------------------------------------------------------------
    
    async def create(
        self,
        flow_name: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        auto_start: bool = False,
        **kwargs
    ) -> FlowSession:
        """
        Create a new flow session.
        
        Args:
            flow_name: Name of the flow to execute
            session_id: Optional custom session ID
            context: Optional initial context
            auto_start: If True, start execution immediately
            **kwargs: Arguments to pass to flow
        
        Returns:
            The created FlowSession
        """
        if not self._app:
            raise RuntimeError("App not initialized. Call set_app() first.")
        
        # Validate flow exists
        if flow_name not in self._app.flows:
            # Try to load it
            from toolboxv2.flows import flows_dict
            self._app.flows = {**self._app.flows, **flows_dict(s=flow_name, remote=True)}
            
            if flow_name not in self._app.flows:
                available = list(self._app.flows.keys())
                raise ValueError(
                    f"Flow '{flow_name}' not found. Available: {available[:10]}"
                )
        
        if session_id is None:
            session_id = f"flow_{uuid.uuid4().hex[:12]}"
        
        async with self._lock:
            # Enforce max sessions
            if len(self._sessions) >= self._max_sessions:
                # Remove oldest completed/error session first
                oldest_finished = None
                oldest_time = float('inf')
                
                for sid, sess in self._sessions.items():
                    if sess.state in (FlowState.COMPLETED, FlowState.ERROR, FlowState.CANCELLED):
                        if sess.last_activity < oldest_time:
                            oldest_time = sess.last_activity
                            oldest_finished = sid
                
                if oldest_finished:
                    del self._sessions[oldest_finished]
                else:
                    # Remove oldest any session
                    oldest_id = min(
                        self._sessions.keys(),
                        key=lambda k: self._sessions[k].last_activity
                    )
                    session = self._sessions[oldest_id]
                    if session._task and not session._task.done():
                        session._cancelled = True
                        session._task.cancel()
                    del self._sessions[oldest_id]
                
                logger.info(f"Removed session to make room for new one")
            
            session = FlowSession(
                session_id=session_id,
                flow_name=flow_name,
                context=context
            )
            
            self._sessions[session_id] = session
        
        logger.debug(f"Created flow session {session_id} for {flow_name}")
        
        # Auto-start if requested
        if auto_start:
            await self.start_execution(session_id, **kwargs)
        
        return session
    
    async def get(self, session_id: str) -> Optional[FlowSession]:
        """Get session by ID."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.update_activity()
            return session
    
    async def update(
        self,
        session_id: str,
        state: Optional[FlowState] = None,
        context: Optional[Dict[str, Any]] = None,
        history_entry: Optional[str] = None
    ) -> bool:
        """Update session state."""
        async with self._lock:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            session.update_activity()
            
            if state is not None:
                session.state = state
            if context is not None:
                session.context.update(context)
            if history_entry is not None:
                session.history.append(history_entry)
            
            return True
    
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            
            # Cancel running task
            if session._task and not session._task.done():
                session._cancelled = True
                session._task.cancel()
                try:
                    await session._task
                except (asyncio.CancelledError, Exception):
                    pass
            
            del self._sessions[session_id]
            logger.debug(f"Deleted session {session_id}")
            return True
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        async with self._lock:
            return [session.to_dict() for session in self._sessions.values()]
    
    # -------------------------------------------------------------------------
    # FLOW EXECUTION
    # -------------------------------------------------------------------------
    
    async def start_execution(
        self,
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start flow execution for a session.
        
        Args:
            session_id: The session ID
            **kwargs: Arguments to pass to the flow
        
        Returns:
            Status dict with initial output
        """
        session = await self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.state not in (FlowState.CREATED,):
            raise ValueError(f"Session already started (state: {session.state.value})")
        
        if not self._executor:
            self._executor = FlowExecutor(self._app)
        
        # Get flow function
        flow_func = self._app.flows.get(session.flow_name)
        if not flow_func:
            raise ValueError(f"Flow {session.flow_name} not found")
        
        # Store kwargs in context
        session.context["kwargs"] = kwargs
        
        # Start execution task
        session._task = asyncio.create_task(
            self._executor.execute(session, flow_func, **kwargs)
        )
        
        # Wait briefly for initial output
        await asyncio.sleep(0.1)
        
        return {
            "session_id": session_id,
            "state": session.state.value,
            "output": session.get_pending_output(),
            "waiting_input": session.state == FlowState.WAITING_INPUT,
            "prompt": session.current_prompt
        }
    
    async def provide_input(
        self,
        session_id: str,
        input_value: str
    ) -> Dict[str, Any]:
        """
        Provide input to a waiting flow.
        
        Args:
            session_id: The session ID
            input_value: The input string
        
        Returns:
            Status dict with new output
        """
        session = await self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.state != FlowState.WAITING_INPUT:
            return {
                "session_id": session_id,
                "state": session.state.value,
                "error": f"Session not waiting for input (state: {session.state.value})",
                "output": session.get_pending_output(),
                "result": session.result
            }
        
        # Provide input
        session.pending_input = input_value
        if session._input_event:
            session._input_event.set()
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Wait up to 5 seconds for state change
        for _ in range(50):
            if session.state != FlowState.RUNNING:
                break
            await asyncio.sleep(0.1)
        
        return {
            "session_id": session_id,
            "state": session.state.value,
            "output": session.get_pending_output(),
            "waiting_input": session.state == FlowState.WAITING_INPUT,
            "prompt": session.current_prompt,
            "result": session.result if session.state == FlowState.COMPLETED else None,
            "error": session.error if session.state == FlowState.ERROR else None
        }
    
    async def get_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current status of a flow session.
        
        Args:
            session_id: The session ID
        
        Returns:
            Complete status dict
        """
        session = await self.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return {
            "session_id": session_id,
            "flow_name": session.flow_name,
            "state": session.state.value,
            "created": session.created,
            "last_activity": session.last_activity,
            "output": session.get_pending_output(),
            "waiting_input": session.state == FlowState.WAITING_INPUT,
            "prompt": session.current_prompt,
            "history": session.history[-20:],  # Last 20 entries
            "context": session.context,
            "result": session.result,
            "error": session.error
        }
    
    async def cancel(self, session_id: str) -> bool:
        """
        Cancel a running flow.
        
        Args:
            session_id: The session ID
        
        Returns:
            True if cancelled successfully
        """
        session = await self.get(session_id)
        if not session:
            return False
        
        if session._task and not session._task.done():
            session._cancelled = True
            if session._input_event:
                session._input_event.set()  # Unblock waiting input
            session._task.cancel()
            
            try:
                await session._task
            except (asyncio.CancelledError, Exception):
                pass
            
            session.state = FlowState.CANCELLED
            session.error = "Cancelled by user"
            logger.info(f"Cancelled flow session {session_id}")
            return True
        
        return False
    
    def close(self) -> None:
        """Cleanup manager resources."""
        if self._executor:
            self._executor.close()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_flow_manager(
    app: Optional["App"] = None,
    max_sessions: int = 100,
    timeout: int = 3600
) -> FlowSessionManager:
    """
    Create a configured FlowSessionManager.
    
    Args:
        app: ToolBoxV2 App instance
        max_sessions: Maximum concurrent sessions
        timeout: Session timeout in seconds
    
    Returns:
        Configured FlowSessionManager
    """
    return FlowSessionManager(app=app, max_sessions=max_sessions, timeout=timeout)
