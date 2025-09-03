import asyncio
import builtins
import contextlib
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from toolboxv2 import App
from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent

from .types import (
    AgentRegistered,
    AgentRegistration,
    ExecutionError,
    ExecutionResult,
    RunRequest,
    WsMessage,
)

# Use a more robust websocket client library if available, or fallback
try:
    import websockets.client as ws_client
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ws_client = None
    ConnectionClosed = Exception


class RegistryClient:
    """Manages the client-side connection to the Registry Server with robust reconnection and long-running support."""

    def __init__(self, app: App):
        self.app = app

        # WebSocket connection
        self.ws: ws_client.WebSocketClientProtocol | None = None
        self.server_url: str | None = None

        # Task management
        self.connection_task: asyncio.Task | None = None
        self.ping_task: asyncio.Task | None = None
        self.message_handler_tasks: set[asyncio.Task] = set()
        self.progress_processor_task: asyncio.Task | None = None

        # Connection state
        self.is_connected = False
        self.should_reconnect = True
        self.reconnect_in_progress = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

        # Agent management
        self.local_agents: dict[str, Any] = {}
        self.registered_info: dict[str, AgentRegistered] = {}
        self.running_executions: dict[str, asyncio.Task] = {}
        self.persistent_callbacks: dict[str, Callable] = {}

        # Progress streaming (NO BATCHING - immediate streaming)
        self.progress_queues: dict[str, asyncio.Queue] = {}
        self.active_streams: set[str] = set()

        # Event handling
        self.custom_event_handlers: dict[str, Callable[[dict], Awaitable[None]]] = {}
        self.pending_registrations: dict[str, asyncio.Future] = {}
        self.registration_counter = 0

    # Utility Methods
    async def get_connection_status(self) -> dict[str, Any]:
        """Get detailed connection status information."""
        try:
            connection_status = {
                "is_connected": self.is_connected,
                "server_url": self.server_url,
                "reconnect_attempts": self.reconnect_attempts,
                "max_reconnect_attempts": self.max_reconnect_attempts,
                "should_reconnect": self.should_reconnect,
                "reconnect_in_progress": self.reconnect_in_progress,
                "websocket_state": None,
                "websocket_open": False,
                "tasks": {
                    "connection_task_running": self.connection_task and not self.connection_task.done(),
                    "ping_task_running": self.ping_task and not self.ping_task.done(),
                },
                "registered_agents_count": len(self.local_agents),
                "running_executions_count": len(self.running_executions),
                "pending_registrations_count": len(self.pending_registrations),
                "persistent_callbacks_count": len(self.persistent_callbacks),
                "last_ping_time": getattr(self, 'last_ping_time', None),
                "connection_uptime": None,
                "connection_established_at": getattr(self, 'connection_established_at', None),
            }

            # WebSocket specific status
            if self.ws:
                connection_status.update({
                    "websocket_state": str(self.ws.state.name) if hasattr(self.ws.state, 'name') else str(
                        self.ws.state),
                    "websocket_open": self.ws.open,
                    "websocket_closed": self.ws.closed,
                })

            # Calculate uptime
            if hasattr(self, 'connection_established_at') and self.connection_established_at:
                connection_status[
                    "connection_uptime"] = asyncio.get_event_loop().time() - self.connection_established_at

            return connection_status

        except Exception as e:
            self.app.print(f"Error getting connection status: {e}")
            return {
                "error": str(e),
                "is_connected": False,
                "server_url": self.server_url,
            }

    async def get_registered_agents(self) -> dict[str, AgentRegistered]:
        """Get all registered agents information."""
        try:
            agents_info = {}

            for agent_id, reg_info in self.registered_info.items():
                # Get agent instance if available
                agent_instance = self.local_agents.get(agent_id)

                # Create enhanced agent info
                agent_data = {
                    "registration_info": reg_info,
                    "agent_available": agent_instance is not None,
                    "agent_type": type(agent_instance).__name__ if agent_instance else "Unknown",
                    "has_progress_callback": hasattr(agent_instance, 'progress_callback') if agent_instance else False,
                    "supports_progress_callback": hasattr(agent_instance,
                                                          'set_progress_callback') if agent_instance else False,
                    "is_persistent_callback_active": agent_id in self.persistent_callbacks,
                    "registration_timestamp": getattr(reg_info, 'registration_timestamp', None),
                }

                # Add agent capabilities if available
                if agent_instance and hasattr(agent_instance, 'get_capabilities'):
                    try:
                        agent_data["capabilities"] = await agent_instance.get_capabilities()
                    except Exception as e:
                        agent_data["capabilities_error"] = str(e)

                agents_info[agent_id] = agent_data

            return agents_info

        except Exception as e:
            self.app.print(f"Error getting registered agents: {e}")
            return {}

    async def get_running_executions(self) -> dict[str, dict[str, Any]]:
        """Get information about currently running executions."""
        try:
            executions_info = {}

            for request_id, execution_task in self.running_executions.items():
                execution_info = {
                    "request_id": request_id,
                    "task_done": execution_task.done(),
                    "task_cancelled": execution_task.cancelled(),
                    "start_time": getattr(execution_task, 'start_time', None),
                    "running_time": None,
                    "task_exception": None,
                    "task_result": None,
                }

                # Calculate running time
                if hasattr(execution_task, 'start_time') and execution_task.start_time:
                    execution_info["running_time"] = asyncio.get_event_loop().time() - execution_task.start_time

                # Get task status details
                if execution_task.done():
                    try:
                        if execution_task.exception():
                            execution_info["task_exception"] = str(execution_task.exception())
                        else:
                            execution_info["task_result"] = "completed_successfully"
                    except Exception as e:
                        execution_info["task_status_error"] = str(e)

                executions_info[request_id] = execution_info

            return executions_info

        except Exception as e:
            self.app.print(f"Error getting running executions: {e}")
            return {}

    async def cancel_execution(self, request_id: str) -> bool:
        """Cancel a running execution."""
        try:
            if request_id not in self.running_executions:
                self.app.print(f"❌ Execution {request_id} not found")
                return False

            execution_task = self.running_executions[request_id]

            if execution_task.done():
                self.app.print(f"⚠️  Execution {request_id} already completed")
                return True

            # Cancel the task
            execution_task.cancel()

            try:
                # Wait a moment for graceful cancellation
                await asyncio.wait_for(execution_task, timeout=5.0)
            except asyncio.CancelledError:
                self.app.print(f"✅ Execution {request_id} cancelled successfully")
            except asyncio.TimeoutError:
                self.app.print(f"⚠️  Execution {request_id} cancellation timeout - may still be running")
            except Exception as e:
                self.app.print(f"⚠️  Execution {request_id} cancellation resulted in exception: {e}")

            # Send cancellation notice to server
            try:
                if self.is_connected and self.ws and self.ws.open:
                    cancellation_event = ProgressEvent(
                        event_type="execution_cancelled",
                        node_name="RegistryClient",
                        success=False,
                        metadata={
                            "request_id": request_id,
                            "cancellation_reason": "client_requested",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    )

                    cancellation_message = ExecutionResult(
                        request_id=request_id,
                        payload=cancellation_event.to_dict(),
                        is_final=True
                    )

                    await self._send_message('execution_result', cancellation_message.model_dump())

            except Exception as e:
                self.app.print(f"Failed to send cancellation notice to server: {e}")

            # Cleanup
            self.running_executions.pop(request_id, None)

            return True

        except Exception as e:
            self.app.print(f"Error cancelling execution {request_id}: {e}")
            return False

    async def health_check(self) -> bool:
        """Perform a health check of the connection."""
        try:
            # Basic connection checks
            if not self.is_connected:
                self.app.print("🔍 Health check: Not connected")
                return False

            if not self.ws or not self.ws.open:
                self.app.print("🔍 Health check: WebSocket not open")
                return False

            # Ping test
            try:
                pong_waiter = await self.ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=10.0)

                # Update last ping time
                self.last_ping_time = asyncio.get_event_loop().time()

                # Test message sending
                test_message = WsMessage(
                    event='health_check',
                    data={
                        "timestamp": self.last_ping_time,
                        "client_id": getattr(self, 'client_id', 'unknown'),
                        "registered_agents": list(self.local_agents.keys()),
                        "running_executions": list(self.running_executions.keys())
                    }
                )

                await self.ws.send(test_message.model_dump_json())

                self.app.print("✅ Health check: Connection healthy")
                return True

            except asyncio.TimeoutError:
                self.app.print("❌ Health check: Ping timeout")
                return False
            except Exception as ping_error:
                self.app.print(f"❌ Health check: Ping failed - {ping_error}")
                return False

        except Exception as e:
            self.app.print(f"❌ Health check: Error - {e}")
            return False

    async def get_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive diagnostic information."""
        try:
            diagnostics = {
                "connection_status": await self.get_connection_status(),
                "registered_agents": await self.get_registered_agents(),
                "running_executions": await self.get_running_executions(),
                "health_status": await self.health_check(),
                "system_info": {
                    "python_version": sys.version,
                    "asyncio_running": True,
                    "event_loop": str(asyncio.get_running_loop()),
                    "thread_name": threading.current_thread().name,
                },
                "performance_metrics": {
                    "total_messages_sent": getattr(self, 'total_messages_sent', 0),
                    "total_messages_received": getattr(self, 'total_messages_received', 0),
                    "total_reconnections": self.reconnect_attempts,
                    "total_registrations": len(self.registered_info),
                    "memory_usage": self._get_memory_usage(),
                },
                "error_log": getattr(self, 'recent_errors', []),
            }

            return diagnostics

        except Exception as e:
            return {
                "diagnostics_error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }

    def _get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage information."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
                "available": psutil.virtual_memory().available,
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}

    async def cleanup_completed_executions(self):
        """Clean up completed execution tasks."""
        try:
            completed_tasks = []

            for request_id, task in self.running_executions.items():
                if task.done():
                    completed_tasks.append(request_id)

            for request_id in completed_tasks:
                self.running_executions.pop(request_id, None)
                self.app.print(f"🧹 Cleaned up completed execution: {request_id}")

            return len(completed_tasks)

        except Exception as e:
            self.app.print(f"Error during cleanup: {e}")
            return 0

    async def connect(self, server_url: str, timeout: float = 30.0):
        """Connect and start all background tasks."""
        if not ws_client:
            self.app.print("Websockets library not installed. Please run 'pip install websockets'")
            return False

        if self.ws and self.ws.open:
            self.app.print("Already connected to the registry server.")
            return True

        self.server_url = server_url
        self.should_reconnect = True
        self.reconnect_in_progress = False

        try:
            self.app.print(f"Connecting to Registry Server at {server_url}...")
            self.ws = await asyncio.wait_for(
                ws_client.connect(server_url),
                timeout=timeout
            )

            self.is_connected = True
            self.reconnect_attempts = 0

            # Start all background tasks
            await self._start_all_background_tasks()

            self.app.print(f"✅ Successfully connected and started all tasks")
            return True

        except asyncio.TimeoutError:
            self.app.print(f"❌ Connection timeout after {timeout}s")
            return False
        except Exception as e:
            self.app.print(f"❌ Connection failed: {e}")
            return False

    async def _start_all_background_tasks(self):
        """Start all background tasks needed for operation."""
        # Start connection listener
        self.connection_task = asyncio.create_task(self._listen())

        # Start ping task
        self.ping_task = asyncio.create_task(self._ping_loop())

        self.app.print("🚀 All background tasks started")
    async def _start_ping_task(self):
        """Start the ping/heartbeat task in the background."""
        if self.ping_task and not self.ping_task.done():
            return  # Already running

        self.ping_task = asyncio.create_task(self._ping_loop())

    async def _ping_loop(self):
        """Dedicated ping task that never blocks and has highest priority."""
        ping_interval = 20  # Less aggressive than server's 5s interval
        consecutive_failures = 0
        max_failures = 2

        while self.is_connected and self.should_reconnect:
            try:
                await asyncio.sleep(ping_interval)

                # Double-check connection state
                if not self.ws or not self.ws.open or self.ws.closed:
                    self.app.print("Ping task detected closed connection")
                    break

                try:
                    # Send ping with short timeout
                    pong_waiter = await self.ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=8.0)  # Less than server's 10s timeout

                    consecutive_failures = 0
                    self.app.print("📡 Heartbeat successful")

                except asyncio.TimeoutError:
                    consecutive_failures += 1
                    self.app.print(f"⚠️ Ping timeout ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        self.app.print("❌ Multiple ping timeouts - connection dead")
                        break

                except Exception as ping_error:
                    consecutive_failures += 1
                    self.app.print(f"❌ Ping error ({consecutive_failures}/{max_failures}): {ping_error}")

                    if consecutive_failures >= max_failures:
                        break

            except Exception as e:
                self.app.print(f"Ping loop error: {e}")
                break

        self.app.print("Ping task stopped")
        # Trigger reconnect if we should still be connected
        if self.should_reconnect and self.is_connected:
            asyncio.create_task(self._trigger_reconnect())

    async def _trigger_reconnect(self):
        """Trigger a reconnection attempt."""
        if self.reconnect_in_progress:
            return

        self.reconnect_in_progress = True
        self.is_connected = False

        try:
            if self.ws:
                with contextlib.suppress(Exception):
                    await self.ws.close()
                self.ws = None

            # Stop current tasks
            if self.connection_task and not self.connection_task.done():
                self.connection_task.cancel()
            if self.ping_task and not self.ping_task.done():
                self.ping_task.cancel()

            self.app.print("🔄 Attempting to reconnect...")
            await self._reconnect_with_backoff()

        finally:
            self.reconnect_in_progress = False

    async def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff."""
        max_attempts = 10
        base_delay = 2
        max_delay = 300  # 5 minutes max

        for attempt in range(max_attempts):
            if not self.should_reconnect:
                break

            delay = min(base_delay * (2 ** attempt), max_delay)
            self.app.print(f"🔄 Reconnect attempt {attempt + 1}/{max_attempts} in {delay}s...")

            await asyncio.sleep(delay)

            try:
                if self.server_url:
                    self.ws = await ws_client.connect(self.server_url)
                    self.is_connected = True
                    self.reconnect_attempts = 0

                    # Restart tasks
                    self.connection_task = asyncio.create_task(self._listen())
                    await self._start_ping_task()

                    # Re-register agents
                    await self._reregister_agents()

                    self.app.print("✅ Reconnected successfully!")
                    return

            except Exception as e:
                self.app.print(f"❌ Reconnect attempt {attempt + 1} failed: {e}")

        self.app.print("❌ All reconnection attempts failed")
        self.should_reconnect = False

    async def _reregister_agents(self):
        """Re-register all local agents after reconnection."""
        if not self.registered_info:
            self.app.print("No agents to re-register")
            return

        self.app.print(f"Re-registering {len(self.registered_info)} agents...")

        for agent_id, reg_info in list(self.registered_info.items()):
            try:
                agent_instance = self.local_agents.get(agent_id)
                if not agent_instance:
                    continue

                # Create new registration (server will assign new IDs)
                new_reg_info = await self.register(
                    agent_instance,
                    reg_info.public_name,
                    self.local_agents.get(f"{agent_id}_description", "Re-registered agent")
                )

                if new_reg_info:
                    # Update stored information
                    old_agent_id = agent_id
                    new_agent_id = new_reg_info.public_agent_id

                    # Move agent to new ID
                    self.local_agents[new_agent_id] = self.local_agents.pop(old_agent_id)
                    self.registered_info[new_agent_id] = self.registered_info.pop(old_agent_id)

                    self.app.print(f"✅ Re-registered agent: {reg_info.public_name} (new ID: {new_agent_id})")
                else:
                    self.app.print(f"❌ Failed to re-register agent: {reg_info.public_name}")

            except Exception as e:
                self.app.print(f"Error re-registering agent {reg_info.public_name}: {e}")

        self.app.print("Agent re-registration completed")

    async def _create_persistent_progress_callback(self, request_id: str, agent_id: str):
        """Create progress callback with offline queuing capability."""
        progress_queue = asyncio.Queue(maxsize=100)  # Buffer for offline messages

        async def persistent_progress_callback(event: ProgressEvent):
            try:
                # Add to queue first
                try:
                    progress_queue.put_nowait((event, asyncio.get_event_loop().time()))
                except asyncio.QueueFull:
                    # Remove oldest item and add new one
                    try:
                        progress_queue.get_nowait()
                        progress_queue.put_nowait((event, asyncio.get_event_loop().time()))
                    except asyncio.QueueEmpty:
                        pass

                # Try to send immediately if connected
                if await self._check_connection_health():
                    try:
                        result = ExecutionResult(
                            request_id=request_id,
                            payload=event.to_dict(),
                            is_final=False
                        )
                        success = await self._send_message('execution_result', result.model_dump())
                        if success:
                            # Remove from queue since it was sent successfully
                            try:
                                progress_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            return
                    except Exception as e:
                        self.app.print(f"Progress send failed, queued: {e}")

                # If we get here, message is queued for later sending

            except Exception as e:
                self.app.print(f"Progress callback error: {e}")

        # Store queue for later processing
        self.progress_queues[request_id] = progress_queue
        return persistent_progress_callback
    async def _store_progress_callback_state(self, agent_id: str, callback_func):
        """Store progress callback for reconnection scenarios."""
        self.persistent_callbacks[agent_id] = callback_func

    async def _restore_progress_callbacks(self):
        """Restore progress callbacks after reconnection."""
        for agent_id, callback_func in self.persistent_callbacks.items():
            agent = self.local_agents.get(agent_id)
            if agent and hasattr(agent, 'set_progress_callback'):
                agent.set_progress_callback(callback_func)

    def on(self, event_name: str, handler: Callable[[dict], Awaitable[None]]):
        """Register an async callback function to handle a custom event from the server."""
        self.app.print(f"Handler for custom event '{event_name}' registered.")
        self.custom_event_handlers[event_name] = handler

    async def send_custom_event(self, event_name: str, data: dict[str, Any]):
        """Send a custom event with a JSON payload to the server."""
        if not self.is_connected or not self.ws or not self.ws.open:
            self.app.print("Cannot send custom event: Not connected.")
            return

        try:
            message = WsMessage(event=event_name, data=data)
            await self.ws.send(message.model_dump_json())
            self.app.print(f"Sent custom event '{event_name}' to server.")
        except Exception as e:
            self.app.print(f"Failed to send custom event: {e}")
            await self._handle_connection_error()

    async def _listen(self):
        """Robust message listening loop with immediate connection loss detection."""
        self.app.print("Registry client is now listening for incoming requests...")

        try:
            while self.is_connected and self.ws and self.ws.open:
                try:
                    # Check connection state before each recv attempt
                    if self.ws.closed:
                        self.app.print("WebSocket is closed - triggering reconnect")
                        break

                    message_raw = await asyncio.wait_for(self.ws.recv(), timeout=5.0)

                    # Handle different message types immediately
                    if isinstance(message_raw, bytes):
                        # Server ping - respond immediately
                        continue

                    # Process text messages
                    try:
                        message = WsMessage.model_validate_json(message_raw)
                        # Handle critical messages immediately, others in background
                        if message.event in ['agent_registered']:
                            await self._handle_message(message)
                        else:
                            # Handle non-critical messages in background to avoid blocking
                            task = asyncio.create_task(self._handle_message(message))
                            self.message_handler_tasks.add(task)
                            # Clean completed tasks
                            self.message_handler_tasks = {t for t in self.message_handler_tasks if not t.done()}

                    except Exception as e:
                        self.app.print(f"Error processing message: {e} | Raw: {message_raw[:200]}")

                except asyncio.TimeoutError:
                    # Normal timeout - check connection health
                    if not self.ws or not self.ws.open or self.ws.closed:
                        self.app.print("Connection health check failed during timeout")
                        break
                    continue

                except ConnectionClosed as e:
                    self.app.print(f"Connection closed by server: {e}")
                    break

                except Exception as e:
                    # Any other WebSocket error means connection is likely dead
                    if "ConnectionClosedError" in str(type(e)) or "IncompleteReadError" in str(type(e)):
                        self.app.print(f"Connection lost: {e}")
                        break
                    else:
                        self.app.print(f"Unexpected error in listen loop: {e}")
                        # Don't break on unexpected errors, but log them
                        await asyncio.sleep(0.1)

        except Exception as e:
            self.app.print(f"Fatal error in listen loop: {e}")
        finally:
            # Always trigger reconnection attempt
            if self.should_reconnect:
                asyncio.create_task(self._trigger_reconnect())

    async def _handle_message(self, message: WsMessage):
        """Handle incoming WebSocket messages with non-blocking execution."""
        try:
            if message.event == 'agent_registered':
                # Handle registration confirmation immediately
                reg_info = AgentRegistered.model_validate(message.data)
                reg_id = None
                for rid, future in self.pending_registrations.items():
                    if not future.done():
                        reg_id = rid
                        break

                if reg_id and reg_id in self.pending_registrations:
                    if not self.pending_registrations[reg_id].done():
                        self.pending_registrations[reg_id].set_result(reg_info)
                else:
                    self.app.print("Received agent_registered but no pending registration found")

            elif message.event == 'run_request':
                # Handle run requests in background - NEVER block here
                run_data = RunRequest.model_validate(message.data)
                asyncio.create_task(self._handle_run_request(run_data))

            elif message.event in self.custom_event_handlers:
                # Handle custom events in background
                self.app.print(f"Received custom event '{message.event}' from server.")
                handler = self.custom_event_handlers[message.event]
                asyncio.create_task(handler(message.data))

            else:
                self.app.print(f"Received unhandled event from server: '{message.event}'")

        except Exception as e:
            self.app.print(f"Error handling message: {e}")
            # Don't let message handling errors kill the connection

    async def register(self, agent_instance: Any, public_name: str, description: str | None = None) -> AgentRegistered | None:
        """Register an agent with the server."""
        if not self.is_connected or not self.ws:
            self.app.print("Not connected. Cannot register agent.")
            return None

        try:
            # Create registration request
            registration = AgentRegistration(public_name=public_name, description=description)
            message = WsMessage(event='register', data=registration.model_dump())

            # Create future for registration response
            reg_id = f"reg_{self.registration_counter}"
            self.registration_counter += 1
            self.pending_registrations[reg_id] = asyncio.Future()

            # Send registration request
            await self.ws.send(message.model_dump_json())
            self.app.print(f"Sent registration request for agent '{public_name}'")

            # Wait for registration confirmation
            try:
                reg_info = await asyncio.wait_for(self.pending_registrations[reg_id], timeout=30.0)

                # Store agent and registration info
                self.local_agents[reg_info.public_agent_id] = agent_instance
                self.registered_info[reg_info.public_agent_id] = reg_info

                self.app.print(f"Agent '{public_name}' registered successfully.")
                self.app.print(f"  Public URL: {reg_info.public_url}")
                self.app.print(f"  API Key: {reg_info.public_api_key}")

                return reg_info

            except TimeoutError:
                self.app.print("Timeout waiting for registration confirmation.")
                return None

        except Exception as e:
            self.app.print(f"Error during registration: {e}")
            return None
        finally:
            # Cleanup pending registration
            self.pending_registrations.pop(reg_id, None)

    async def _handle_run_request(self, run_request: RunRequest):
        """Handle run request - start agent in completely separate task."""
        agent_id = run_request.public_agent_id
        agent = self.local_agents.get(agent_id)

        if not agent:
            await self._stream_error(run_request.request_id, f"Agent with ID {agent_id} not found")
            return

        # Start agent execution in separate task - NEVER await here
        execution_task = asyncio.create_task(
            self._execute_agent_with_monitoring(agent, run_request)
        )

        # Store task but don't wait for it
        self.running_executions[run_request.request_id] = execution_task

        self.app.print(f"🚀 Agent execution started in background: {run_request.request_id}")
        # This method returns immediately - agent runs in background
    async def _execute_agent_with_monitoring(self, agent: Any, run_request: RunRequest):
        """Execute agent in completely separate task - never blocks main connection."""
        request_id = run_request.request_id
        agent_id = run_request.public_agent_id

        try:
            # Create progress streaming callback
            progress_callback = await self._create_streaming_progress_callback(request_id, agent_id)

            # Store original callback
            original_callback = getattr(agent, 'progress_callback', None)

            # Set streaming progress callback
            if hasattr(agent, 'set_progress_callback'):
                agent.set_progress_callback(progress_callback)
            elif hasattr(agent, 'progress_callback'):
                agent.progress_callback = progress_callback

            # Store for reconnection scenarios
            self.persistent_callbacks[agent_id] = progress_callback
            self.active_streams.add(request_id)

            self.app.print(f"🚀 Starting agent execution in separate task: {request_id}")

            # EXECUTE THE AGENT - this can run for hours/days
            final_result = await agent.a_run(
                query=run_request.query,
                session_id=run_request.session_id,
                **run_request.kwargs
            )

            # Send final result
            await self._stream_final_result(request_id, final_result, agent_id, run_request.session_id)

            self.app.print(f"✅ Agent execution completed: {request_id}")

        except Exception as e:
            self.app.print(f"❌ Agent execution failed: {e}")
            await self._stream_error(request_id, str(e))
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            await self.running_executions.pop(request_id, None)
            self.persistent_callbacks.pop(agent_id, None)
            self.active_streams.discard(request_id)

            # Close progress queue
            if request_id in self.progress_queues:
                queue = self.progress_queues.pop(request_id)
                # Signal queue processor to stop for this request
                try:
                    await queue.put(None)  # Sentinel value
                except:
                    pass

            # Restore original callback
            try:
                if hasattr(agent, 'set_progress_callback'):
                    agent.set_progress_callback(original_callback)
                elif hasattr(agent, 'progress_callback'):
                    agent.progress_callback = original_callback
            except Exception as cleanup_error:
                self.app.print(f"Warning: Callback cleanup failed: {cleanup_error}")

    async def _stream_final_result(self, request_id: str, final_result: Any, agent_id: str, session_id: str):
        """Stream final result immediately."""
        final_event = ProgressEvent(
            event_type="execution_complete",
            node_name="RegistryClient",
            success=True,
            metadata={
                "result": final_result,
                "agent_id": agent_id,
                "session_id": session_id
            }
        )

        final_message = ExecutionResult(
            request_id=request_id,
            payload=final_event.to_dict(),
            is_final=True
        )

        # Stream final result with high priority
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                if await self._check_connection_health():
                    success = await self._send_message('execution_result', final_message.model_dump())
                    if success:
                        self.app.print(f"✅ Final result streamed successfully")
                        return

                await asyncio.sleep(1.0 * (attempt + 1))  # Longer delays for final result

            except Exception as e:
                self.app.print(f"Final result stream attempt {attempt + 1} failed: {e}")

        self.app.print(f"❌ Failed to stream final result after {max_attempts} attempts")

    async def _stream_error(self, request_id: str, error_message: str):
        """Stream error immediately."""
        error_payload = ExecutionError(request_id=request_id, error=error_message)

        for attempt in range(5):
            try:
                if await self._check_connection_health():
                    success = await self._send_message('execution_error', error_payload.model_dump())
                    if success:
                        return
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                self.app.print(f"Error stream attempt {attempt + 1} failed: {e}")

    async def _create_streaming_progress_callback(self, request_id: str, agent_id: str):
        """Create callback that streams progress immediately as it comes."""
        # Create queue for this specific request
        progress_queue = asyncio.Queue()
        self.progress_queues[request_id] = progress_queue

        # Start dedicated processor for this request
        processor_task = asyncio.create_task(
            self._process_progress_stream(request_id, progress_queue)
        )

        async def streaming_progress_callback(event: ProgressEvent):
            """Stream progress immediately - no batching, no delays."""
            try:
                if request_id in self.active_streams:
                    # Put in queue for immediate processing
                    await progress_queue.put(event)
            except Exception as e:
                self.app.print(f"Progress streaming error: {e}")

        return streaming_progress_callback

    async def _process_progress_stream(self, request_id: str, progress_queue: asyncio.Queue):
        """Process progress stream in real-time - separate task per request."""
        self.app.print(f"📡 Started progress streaming for request: {request_id}")

        while request_id in self.active_streams:
            try:
                # Get next progress event (blocking)
                event = await progress_queue.get()

                # Sentinel value to stop
                if event is None:
                    break

                # Stream immediately - no batching
                await self._stream_progress_immediately(request_id, event)

            except Exception as e:
                self.app.print(f"Progress stream processing error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

        self.app.print(f"📡 Stopped progress streaming for request: {request_id}")

    async def _stream_progress_immediately(self, request_id: str, event: ProgressEvent):
        """Stream single progress event immediately."""
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                if await self._check_connection_health():
                    result = ExecutionResult(
                        request_id=request_id,
                        payload=event.to_dict(),
                        is_final=False
                    )

                    success = await self._send_message('execution_result', result.model_dump())
                    if success:
                        return  # Successfully streamed

                # Connection unhealthy - brief wait before retry
                await asyncio.sleep(0.2 * (attempt + 1))

            except Exception as e:
                self.app.print(f"Stream attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.2 * (attempt + 1))

        # All attempts failed - but don't crash, just log
        self.app.print(f"⚠️ Failed to stream progress after {max_attempts} attempts")


    async def send_ui_progress(self, progress_data: dict[str, Any], retry_count: int = 3):
        """Enhanced UI progress sender with retry logic."""
        if not self.is_connected or not self.ws or not self.ws.open:
            self.app.print("Registry client WebSocket not connected - queuing progress update")
            # Could implement a queue here for offline progress updates
            return False

        for attempt in range(retry_count):
            try:
                # Structure progress message for registry server
                ui_message = {
                    "timestamp": progress_data.get('timestamp', asyncio.get_event_loop().time()),
                    "agent_id": progress_data.get('agent_id', 'unknown'),
                    "event_type": progress_data.get('event_type', 'unknown'),
                    "status": progress_data.get('status', 'processing'),
                    "agent_name": progress_data.get('agent_name', 'Unknown'),
                    "node_name": progress_data.get('node_name', 'Unknown'),
                    "session_id": progress_data.get('session_id'),
                    "metadata": progress_data.get('metadata', {}),

                    # Enhanced progress data for UI panels
                    "outline_progress": progress_data.get('progress_data', {}).get('outline', {}),
                    "activity_info": progress_data.get('progress_data', {}).get('activity', {}),
                    "meta_tool_info": progress_data.get('progress_data', {}).get('meta_tool', {}),
                    "system_status": progress_data.get('progress_data', {}).get('system', {}),
                    "graph_info": progress_data.get('progress_data', {}).get('graph', {}),

                    # UI flags for selective updates
                    "ui_flags": progress_data.get('ui_flags', {}),

                    # Performance metrics
                    "performance": progress_data.get('performance', {}),

                    # Message metadata
                    "message_id": f"msg_{asyncio.get_event_loop().time()}_{attempt}",
                    "retry_count": attempt
                }

                # Send as WsMessage
                message = WsMessage(event='ui_progress_update', data=ui_message)
                await self.ws.send(message.model_dump_json())

                # Success - break retry loop
                self.app.print(
                    f"📤 Sent UI progress: {progress_data.get('event_type')} | {progress_data.get('status')} (attempt {attempt + 1})")
                return True

            except Exception as e:
                self.app.print(f"Failed to send UI progress (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    await self._handle_connection_error()
                    return False

        return False

    async def send_agent_status(self, agent_id: str, status: str, details: dict[str, Any] = None):
        """Send agent status updates."""
        if not self.is_connected or not self.ws or not self.ws.open:
            return

        try:
            status_message = {
                "agent_id": agent_id,
                "status": status,
                "details": details or {},
                "timestamp": asyncio.get_event_loop().time(),
                "capabilities": ["chat", "progress_tracking", "outline_visualization", "meta_tool_monitoring"]
            }

            message = WsMessage(event='agent_status_update', data=status_message)
            await self.ws.send(message.model_dump_json())

        except Exception as e:
            self.app.print(f"Failed to send agent status: {e}")
            await self._handle_connection_error()

    async def _send_error(self, request_id: str, error_message: str):
        """Send error message to server."""
        error_payload = ExecutionError(request_id=request_id, error=error_message)
        await self._send_message('execution_error', error_payload.model_dump())

    async def _check_connection_health(self) -> bool:
        """Check if the WebSocket connection is actually healthy."""
        if not self.ws:
            return False

        try:
            # Check basic connection state
            if self.ws.closed or not self.ws.open:
                return False

            # Try a quick ping to verify connectivity
            pong_waiter = await self.ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=3.0)
            return True

        except Exception as e:
            self.app.print(f"Connection health check failed: {e}")
            return False

    async def _send_message(self, event: str, data: dict, max_retries: int = 3):
        """Enhanced message sending with connection health verification."""
        for attempt in range(max_retries):
            # Check connection health before attempting to send
            if not await self._check_connection_health():
                self.app.print(f"Connection unhealthy for message '{event}' (attempt {attempt + 1})")

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    self.app.print(f"Cannot send message '{event}': Connection permanently failed")
                    asyncio.create_task(self._trigger_reconnect())
                    return False

            try:
                message = WsMessage(event=event, data=data)
                await self.ws.send(message.model_dump_json())
                return True

            except Exception as e:
                self.app.print(f"Send attempt {attempt + 1} failed for '{event}': {e}")

                # Check if this is a connection-related error
                error_str = str(e).lower()
                if any(err in error_str for err in ['connectionclosed', 'incomplete', 'connection', 'closed']):
                    self.app.print("Connection error detected - triggering reconnect")
                    asyncio.create_task(self._trigger_reconnect())
                    return False

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        return False
    async def _send_final_result_with_retry(self, request_id: str, final_result: Any, agent_id: str, session_id: str):
        """Send final result with robust retry logic."""
        final_event = ProgressEvent(
            event_type="execution_complete",
            node_name="RegistryClient",
            success=True,
            metadata={
                "result": final_result,
                "agent_id": agent_id,
                "session_id": session_id
            }
        )

        final_message = ExecutionResult(
            request_id=request_id,
            payload=final_event.to_dict(),
            is_final=True
        )

        max_retries = 10
        base_delay = 2

        for attempt in range(max_retries):
            try:
                if not self.is_connected or not self.ws or not self.ws.open:
                    self.app.print(f"⚠️  Connection lost - waiting for reconnection (attempt {attempt + 1})")
                    await asyncio.sleep(base_delay * (attempt + 1))
                    continue

                await self._send_message('execution_result', final_message.model_dump())
                self.app.print(f"✅ Final result sent successfully on attempt {attempt + 1}")
                return

            except Exception as e:
                delay = base_delay * (2 ** attempt)
                self.app.print(f"❌ Failed to send final result (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)

        self.app.print(f"❌ Failed to send final result after {max_retries} attempts")

    async def _send_error_with_retry(self, request_id: str, error_message: str):
        """Send error message with retry logic."""
        max_retries = 5

        for attempt in range(max_retries):
            try:
                if self.is_connected and self.ws and self.ws.open:
                    await self._send_error(request_id, error_message)
                    return
                else:
                    await asyncio.sleep(2 * (attempt + 1))
            except Exception as e:
                self.app.print(f"Error sending error message (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))

    async def _handle_connection_error(self):
        """Handle connection errors and cleanup."""
        self.is_connected = False
        if self.ws:
            with contextlib.suppress(builtins.BaseException):
                await self.ws.close()
            self.ws = None

    async def disconnect(self):
        """Enhanced disconnect with complete task cleanup."""
        self.app.print("Initiating clean shutdown...")
        self.is_connected = False
        self.should_reconnect = False

        # Cancel all background tasks
        tasks_to_cancel = []

        if self.connection_task and not self.connection_task.done():
            tasks_to_cancel.append(self.connection_task)

        if self.ping_task and not self.ping_task.done():
            tasks_to_cancel.append(self.ping_task)

        # Cancel message handler tasks
        for task in list(self.message_handler_tasks):
            if not task.done():
                tasks_to_cancel.append(task)

        # Cancel running executions
        for task in list(self.running_executions.values()):
            if not task.done():
                tasks_to_cancel.append(task)

        if tasks_to_cancel:
            self.app.print(f"Cancelling {len(tasks_to_cancel)} background tasks...")
            for task in tasks_to_cancel:
                task.cancel()

            # Wait for cancellation with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.app.print("Warning: Some tasks didn't cancel within timeout")

        # Close WebSocket connection
        if self.ws:
            with contextlib.suppress(Exception):
                await self.ws.close()
            self.ws = None

        # Cancel pending registrations
        for future in self.pending_registrations.values():
            if not future.done():
                future.cancel()
        self.pending_registrations.clear()

        # Clear state
        self.message_handler_tasks.clear()
        self.running_executions.clear()
        self.persistent_callbacks.clear()

        self.connection_task = None
        self.ping_task = None

        self.app.print("✅ Registry client shutdown completed")


# --- Module setup ---
Name = "registry"
registry_clients: dict[str, RegistryClient] = {}


def get_registry_client(app: App) -> RegistryClient:
    """Factory function to get a singleton RegistryClient instance."""
    app_id = app.id
    if app_id not in registry_clients:
        registry_clients[app_id] = RegistryClient(app)
    return registry_clients[app_id]


async def on_exit(app: App):
    client = get_registry_client(app)
    await client.disconnect()
