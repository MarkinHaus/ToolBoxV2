from dataclasses import asdict, is_dataclass
from enum import Enum
import asyncio
import json
from typing import Dict, Any, Optional, Callable, Awaitable
from toolboxv2 import get_app, App
from toolboxv2.mods.isaa.base.Agent.types import ProgressEvent
from .types import WsMessage, AgentRegistration, RunRequest, ExecutionResult, ExecutionError, AgentRegistered

# Use a more robust websocket client library if available, or fallback
try:
    import websockets.client as ws_client
    from websockets.exceptions import ConnectionClosed
except ImportError:
    ws_client = None
    ConnectionClosed = Exception


class RegistryClient:
    """Manages the client-side connection to the Registry Server."""

    def __init__(self, app: App):
        self.app = app
        self.ws: Optional[ws_client.WebSocketClientProtocol] = None
        self.connection_task: Optional[asyncio.Task] = None
        self.local_agents: Dict[str, Any] = {}
        self.registered_info: Dict[str, AgentRegistered] = {}
        self.custom_event_handlers: Dict[str, Callable[[Dict], Awaitable[None]]] = {}

        # Registration handling
        self.pending_registrations: Dict[str, asyncio.Future] = {}
        self.registration_counter = 0

        # Connection state
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    async def connect(self, server_url: str):
        """Connect to the registry server with retry logic."""
        if not ws_client:
            self.app.print("Websockets library not installed. Please run 'pip install websockets'")
            return

        if self.ws and self.ws.open:
            self.app.print("Already connected to the registry server.")
            return

        for attempt in range(self.max_reconnect_attempts):
            try:
                self.app.print(
                    f"Connecting to Registry Server at {server_url} (attempt {attempt + 1}/{self.max_reconnect_attempts})")
                self.ws = await ws_client.connect(server_url)
                self.is_connected = True
                self.reconnect_attempts = 0

                # Start listening for messages
                self.connection_task = asyncio.create_task(self._listen())
                self.app.print(f"Successfully connected to Registry Server at {server_url}")
                return

            except Exception as e:
                self.app.print(f"Connection attempt {attempt + 1} failed: {e}")
                self.ws = None
                self.is_connected = False

                if attempt < self.max_reconnect_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.app.print("Max reconnection attempts reached. Connection failed.")

    def on(self, event_name: str, handler: Callable[[Dict], Awaitable[None]]):
        """Register an async callback function to handle a custom event from the server."""
        self.app.print(f"Handler for custom event '{event_name}' registered.")
        self.custom_event_handlers[event_name] = handler

    async def send_custom_event(self, event_name: str, data: Dict[str, Any]):
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
        """Main message listening loop - handles ALL incoming messages."""
        self.app.print("Registry client is now listening for incoming requests...")

        try:
            while self.ws and self.ws.open and self.is_connected:
                try:
                    message_raw = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    self.app.print(f"Received message from server: {message_raw[:200]}...")

                    try:
                        message = WsMessage.model_validate_json(message_raw)
                        await self._handle_message(message)

                    except Exception as e:
                        self.app.print(f"Error processing message from server: {e} | Raw: {message_raw[:200]}")

                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.ws and self.ws.open:
                        await self.ws.ping()
                    continue

        except ConnectionClosed:
            self.app.print("Connection to Registry Server closed.")
        except Exception as e:
            self.app.print(f"Error in listen loop: {e}")
        finally:
            await self._handle_connection_error()

    async def _handle_message(self, message: WsMessage):
        """Handle incoming WebSocket messages."""
        try:
            if message.event == 'agent_registered':
                # Handle registration confirmation
                reg_info = AgentRegistered.model_validate(message.data)

                # Find pending registration by matching public_name
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
                run_data = RunRequest.model_validate(message.data)
                asyncio.create_task(self._handle_run_request(run_data))

            elif message.event in self.custom_event_handlers:
                self.app.print(f"Received custom event '{message.event}' from server.")
                handler = self.custom_event_handlers[message.event]
                asyncio.create_task(handler(message.data))

            else:
                self.app.print(f"Received unhandled event from server: '{message.event}'")

        except Exception as e:
            self.app.print(f"Error handling message: {e}")

    async def register(self, agent_instance: Any, public_name: str, description: Optional[str] = None) -> Optional[
        AgentRegistered]:
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

            except asyncio.TimeoutError:
                self.app.print("Timeout waiting for registration confirmation.")
                return None

        except Exception as e:
            self.app.print(f"Error during registration: {e}")
            return None
        finally:
            # Cleanup pending registration
            self.pending_registrations.pop(reg_id, None)

    async def _handle_run_request(self, run_request: RunRequest):
        """Handle incoming run requests from the server."""
        agent_id = run_request.public_agent_id
        agent = self.local_agents.get(agent_id)

        if not agent:
            await self._send_error(run_request.request_id, f"Agent with ID {agent_id} not found locally.")
            return

        # Create progress callback
        async def progress_callback(event: ProgressEvent):
            try:
                result = ExecutionResult(
                    request_id=run_request.request_id,
                    payload=event.to_dict(),
                    is_final=False
                )
                await self._send_message('execution_result', result.model_dump())
            except Exception as e:
                self.app.print(f"Failed to send progress event: {e}")

        # Store original callback
        original_callback = getattr(agent, 'progress_callback', None)

        try:
            # Set progress callback
            if hasattr(agent, 'set_progress_callback'):
                agent.set_progress_callback(progress_callback)
            elif hasattr(agent, 'progress_callback'):
                agent.progress_callback = progress_callback

            # Execute the agent
            final_result = await agent.a_run(
                query=run_request.query,
                session_id=run_request.session_id,
                **run_request.kwargs
            )

            # Send final result
            final_event = ProgressEvent(
                event_type="execution_complete",
                status="success",
                metadata={
                    "result": final_result,
                    "agent_id": agent_id,
                    "session_id": run_request.session_id
                }
            )

            final_message = ExecutionResult(
                request_id=run_request.request_id,
                payload=final_event.to_dict(),
                is_final=True
            )
            await self._send_message('execution_result', final_message.model_dump())

        except Exception as e:
            self.app.print(f"Agent execution failed for '{agent_id}': {e}")
            await self._send_error(run_request.request_id, str(e))
            import traceback
            traceback.print_exc()
        finally:
            # Restore original callback
            if hasattr(agent, 'set_progress_callback'):
                agent.set_progress_callback(original_callback)
            elif hasattr(agent, 'progress_callback'):
                agent.progress_callback = original_callback

    async def send_ui_progress(self, progress_data: Dict[str, Any], retry_count: int = 3):
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

    async def send_agent_status(self, agent_id: str, status: str, details: Dict[str, Any] = None):
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

    async def _send_message(self, event: str, data: dict):
        """Send a message to the server."""
        if not self.is_connected or not self.ws or not self.ws.open:
            self.app.print(f"Cannot send message '{event}': Not connected")
            return

        try:
            message = WsMessage(event=event, data=data)
            await self.ws.send(message.model_dump_json())
        except Exception as e:
            self.app.print(f"Failed to send message '{event}': {e}")
            await self._handle_connection_error()

    async def _handle_connection_error(self):
        """Handle connection errors and cleanup."""
        self.is_connected = False
        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None

    async def disconnect(self):
        """Disconnect from the server."""
        self.is_connected = False

        if self.connection_task:
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
            self.connection_task = None

        if self.ws:
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None

        # Cancel pending registrations
        for future in self.pending_registrations.values():
            if not future.done():
                future.cancel()
        self.pending_registrations.clear()

        self.app.print("Disconnected from Registry Server.")


# --- Module setup ---
Name = "registry"
registry_clients: Dict[str, RegistryClient] = {}


def get_registry_client(app: App) -> RegistryClient:
    """Factory function to get a singleton RegistryClient instance."""
    app_id = app.id
    if app_id not in registry_clients:
        registry_clients[app_id] = RegistryClient(app)
    return registry_clients[app_id]


async def on_exit(app: App):
    client = get_registry_client(app)
    await client.disconnect()
