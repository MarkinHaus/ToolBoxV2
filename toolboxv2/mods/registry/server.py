import asyncio
import json
import os
import secrets
from typing import Any

from toolboxv2 import App, RequestData, Result, get_app

from .types import (
    AgentRegistered,
    AgentRegistration,
    ExecutionError,
    ExecutionResult,
    RunRequest,
    WsMessage,
)

app_server = get_app("RegistryServer")
export = app_server.tb
Name = "registry"


class RegistryState:
    def __init__(self):
        self.client_agents: dict[str, list[str]] = {}
        self.agent_to_client: dict[str, str] = {}
        self.agent_details: dict[str, dict[str, Any]] = {}
        self.key_to_agent: dict[str, str] = {}
        self.pending_requests: dict[str, asyncio.Queue] = {}
        self.ui_clients: set[str] = set()
        self.recent_progress: dict[str, list] = {}


STATE = RegistryState()


async def on_connect(app: App, conn_id: str, session: dict):
    app.print(f"Registry client connected: {conn_id}")
    STATE.client_agents[conn_id] = []


async def on_message(app: App, conn_id: str, session: dict, payload: dict):
    """Enhanced message handler with proper error handling."""
    try:
        # Ensure payload is a dict
        if isinstance(payload, str):
            payload = json.loads(payload)

        message = WsMessage.model_validate(payload)
        app.print(f"Registry received event: {message.event} from {conn_id}")

        if message.event == 'register':
            await handle_registration(app, conn_id, session, message)

        elif message.event == 'ui_progress_update':
            await handle_ui_progress_update(app, message)

        elif message.event == 'execution_result':
            await handle_execution_result(app, message)

        elif message.event == 'execution_error':
            await handle_execution_error(app, message)

        elif message.event == 'agent_status_update':
            await handle_agent_status_update(app, message)

        else:
            app.print(f"Unhandled event '{message.event}' from client {conn_id}")

    except Exception as e:
        app.print(f"Error processing WebSocket message: {e}", error=True)


async def handle_registration(app: App, conn_id: str, session: dict, message: WsMessage):
    """Handle agent registration."""
    try:
        reg_data = AgentRegistration.model_validate(message.data)
        agent_id = f"agent_{secrets.token_urlsafe(16)}"
        api_key = f"tbk_{secrets.token_urlsafe(32)}"

        STATE.client_agents.setdefault(conn_id, []).append(agent_id)
        STATE.agent_to_client[agent_id] = conn_id
        STATE.key_to_agent[api_key] = agent_id
        STATE.agent_details[agent_id] = reg_data.model_dump()

        base_url = os.getenv("APP_BASE_URL", "http://localhost:8080") or session.get('host', 'localhost:8080')
        if base_url == "localhost":
            base_url = "localhost:8080"
            app.print("APP_BASE_URL is localhost. Using default port 8080.")
        public_url = f"{base_url}/api/registry/run?public_agent_id={agent_id}"

        if not public_url.startswith('http'):
            public_url = f"http://{public_url}"

        response = AgentRegistered(
            public_name=reg_data.public_name,
            public_agent_id=agent_id,
            public_api_key=api_key,
            public_url=public_url,
        )

        # Send registration confirmation
        response_message = WsMessage(event='agent_registered', data=response.model_dump())
        await app.ws_send(conn_id, response_message.model_dump())

        # Notify UI clients
        await broadcast_to_ui_clients(app, {
            "event": "agent_registered",
            "data": {
                "public_agent_id": agent_id,
                "public_name": reg_data.public_name,
                "description": reg_data.description,
                "status": "online"
            }
        })

        app.print(f"Agent '{reg_data.public_name}' registered with ID: {agent_id}")

    except Exception as e:
        app.print(f"Registration error: {e}", error=True)


async def handle_ui_progress_update(app: App, message: WsMessage):
    """Handle UI progress updates."""
    try:
        progress_data = message.data
        agent_id = progress_data.get('agent_id', 'unknown')

        # Store recent progress
        if agent_id not in STATE.recent_progress:
            STATE.recent_progress[agent_id] = []
        STATE.recent_progress[agent_id].append(progress_data)

        # Keep only last 50 events
        STATE.recent_progress[agent_id] = STATE.recent_progress[agent_id][-50:]

        # Broadcast to UI clients
        await broadcast_to_ui_clients(app, {
            "event": "live_progress_update",
            "data": progress_data
        })

    except Exception as e:
        app.print(f"UI progress update error: {e}", error=True)


async def handle_execution_result(app: App, message: WsMessage):
    """Handle execution results."""
    try:
        result = ExecutionResult.model_validate(message.data)

        if result.request_id in STATE.pending_requests:
            await STATE.pending_requests[result.request_id].put(result)

        # Broadcast to UI clients
        await broadcast_to_ui_clients(app, {
            'event': 'execution_progress',
            'data': {
                'request_id': result.request_id,
                'payload': result.payload,
                'is_final': result.is_final,
                'timestamp': asyncio.get_event_loop().time()
            }
        })

    except Exception as e:
        app.print(f"Execution result error: {e}", error=True)


async def handle_execution_error(app: App, message: WsMessage):
    """Handle execution errors."""
    try:
        error = ExecutionError.model_validate(message.data)

        if error.request_id in STATE.pending_requests:
            await STATE.pending_requests[error.request_id].put(error)

        await broadcast_to_ui_clients(app, {
            'event': 'execution_error',
            'data': {
                'request_id': error.request_id,
                'error': error.error,
                'timestamp': asyncio.get_event_loop().time()
            }
        })

    except Exception as e:
        app.print(f"Execution error handling error: {e}", error=True)


async def handle_agent_status_update(app: App, message: WsMessage):
    """Handle agent status updates."""
    try:
        status_data = message.data
        await broadcast_to_ui_clients(app, {
            'event': 'agent_status_update',
            'data': status_data
        })

    except Exception as e:
        app.print(f"Agent status update error: {e}", error=True)


async def broadcast_to_ui_clients(app: App, data: dict[str, Any]):
    """Broadcast updates to all connected UI clients."""
    if not STATE.ui_clients:
        app.print("No active UI clients to broadcast to")
        return

    app.print(f"Broadcasting to {len(STATE.ui_clients)} UI clients: {data.get('event', 'unknown')}")

    dead_clients = set()
    successful_broadcasts = 0

    for ui_conn_id in STATE.ui_clients.copy():
        try:
            await app.ws_send(ui_conn_id, data)
            successful_broadcasts += 1
        except Exception as e:
            app.print(f"Failed to broadcast to UI client {ui_conn_id}: {e}")
            dead_clients.add(ui_conn_id)

    # Clean up dead connections
    for dead_client in dead_clients:
        STATE.ui_clients.discard(dead_client)

    app.print(f"Broadcast completed: {successful_broadcasts} successful, {len(dead_clients)} failed")


async def on_disconnect(app: App, conn_id: str, session: dict = None):
    """Enhanced disconnect handler with comprehensive cleanup and UI notifications."""
    app.print(f"Registry client disconnected: {conn_id}")

    # Check if this is a UI client
    if conn_id in STATE.ui_clients:
        STATE.ui_clients.discard(conn_id)
        app.print(f"UI client {conn_id} removed from active clients")
        return

    # Handle agent client disconnection
    if conn_id in STATE.client_agents:
        agent_ids_to_cleanup = STATE.client_agents[conn_id].copy()

        for agent_id in agent_ids_to_cleanup:
            try:
                # Get agent details before removal for notification
                agent_details = STATE.agent_details.get(agent_id, {})
                agent_name = agent_details.get('public_name', 'Unknown')

                # Remove from all state dictionaries
                STATE.agent_to_client.pop(agent_id, None)
                STATE.agent_details.pop(agent_id, None)

                # Remove API key mapping
                key_to_remove = next((k for k, v in STATE.key_to_agent.items() if v == agent_id), None)
                if key_to_remove:
                    STATE.key_to_agent.pop(key_to_remove, None)

                # Clean up progress data
                STATE.recent_progress.pop(agent_id, None)

                # Clean up any pending requests for this agent by checking if queue exists and clearing it
                requests_to_cleanup = []
                for req_id in list(STATE.pending_requests.keys()):
                    try:
                        # Put error in queue to unblock any waiting requests
                        error_result = ExecutionError(
                            request_id=req_id,
                            error="Agent disconnected unexpectedly",
                            public_agent_id=agent_id
                        )
                        await STATE.pending_requests[req_id].put(error_result)
                        requests_to_cleanup.append(req_id)
                    except Exception as e:
                        app.print(f"Error cleaning up pending request {req_id}: {e}")

                # Remove cleaned up requests
                for req_id in requests_to_cleanup:
                    STATE.pending_requests.pop(req_id, None)

                # Notify UI clients about agent going offline (non-blocking)
                if agent_details:
                    asyncio.create_task(broadcast_to_ui_clients(app, {
                        "event": "agent_offline",
                        "data": {
                            "public_agent_id": agent_id,
                            "public_name": agent_name,
                            "status": "offline",
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    }))

                app.print(f"Agent '{agent_name}' (ID: {agent_id}) unregistered and cleaned up")

            except Exception as e:
                app.print(f"Error during agent cleanup for {agent_id}: {e}", error=True)

        # Remove the client connection entry
        STATE.client_agents.pop(conn_id, None)

        app.print(f"Client {conn_id} fully disconnected and cleaned up ({len(agent_ids_to_cleanup)} agents removed)")
    else:
        app.print(f"Unknown client {conn_id} disconnected (no agents to clean up)")

# UI WebSocket handlers
async def ui_on_connect(app: App, conn_id: str, session: dict):
    """UI Client connection."""
    app.print(f"UI Client connecting: {conn_id}")
    STATE.ui_clients.add(conn_id)
    app.print(f"UI Client connected: {conn_id} (Total: {len(STATE.ui_clients)})")

    # Send current agents list
    available_agents = []
    for agent_id, details in STATE.agent_details.items():
        if agent_id in STATE.agent_to_client:
            available_agents.append({
                "public_agent_id": agent_id,
                "public_name": details.get('public_name', 'Unknown'),
                "description": details.get('description', ''),
                "status": "online"
            })

    await app.ws_send(conn_id, {
        "event": "agents_list",
        "data": {"agents": available_agents}
    })


async def ui_on_message(app: App, conn_id: str, session: dict, payload: dict):
    """UI Client Message Handler."""
    try:
        # Ensure payload is a dict
        if isinstance(payload, str):
            payload = json.loads(payload)

        event = payload.get('event')
        data = payload.get('data', {})

        if event == 'subscribe_agent':
            agent_id = data.get('public_agent_id')
            if agent_id in STATE.agent_details:
                if agent_id in STATE.recent_progress:
                    for progress_event in STATE.recent_progress[agent_id][-10:]:
                        await app.ws_send(conn_id, {
                            "event": "historical_progress",
                            "data": progress_event
                        })

                await app.ws_send(conn_id, {
                    "event": "subscription_confirmed",
                    "data": {"public_agent_id": agent_id}
                })

        elif event == 'chat_message':
            agent_id = data.get('public_agent_id')
            message_text = data.get('message')
            session_id = data.get('session_id', f'ui_{conn_id}')
            api_key = data.get('api_key')

            if not api_key or STATE.key_to_agent.get(api_key) != agent_id:
                await app.ws_send(conn_id, {
                    "event": "error",
                    "data": {"error": "Invalid or missing API Key"}
                })
                return

            if agent_id in STATE.agent_to_client:
                agent_conn_id = STATE.agent_to_client[agent_id]
                request_id = f"ui_req_{secrets.token_urlsafe(16)}"

                run_request = RunRequest(
                    request_id=request_id,
                    public_agent_id=agent_id,
                    query=message_text,
                    session_id=session_id,
                    kwargs={}
                )

                response_queue = asyncio.Queue()
                STATE.pending_requests[request_id] = response_queue

                await app.ws_send(agent_conn_id, WsMessage(
                    event='run_request',
                    data=run_request.model_dump()
                ).model_dump())

                await app.ws_send(conn_id, {
                    "event": "message_acknowledged",
                    "data": {"request_id": request_id, "agent_id": agent_id}
                })

    except Exception as e:
        app.print(f"UI message handling error: {e}", error=True)
        await app.ws_send(conn_id, {
            "event": "error",
            "data": {"error": str(e)}
        })


async def ui_on_disconnect(app: App, conn_id: str, session: dict = None):
    """UI Client Disconnection."""
    app.print(f"UI Client disconnected: {conn_id}")
    STATE.ui_clients.discard(conn_id)


@export(mod_name=Name, websocket_handler="connect")
def register_ws_handlers(app: App):
    """Register WebSocket handlers for the registry."""
    return {
        "on_connect": on_connect,
        "on_message": on_message,
        "on_disconnect": on_disconnect,
    }


@export(mod_name=Name, websocket_handler="ui_connect")
def register_ui_ws_handlers(app: App):
    """Register UI-specific WebSocket handlers."""
    return {
        "on_connect": ui_on_connect,
        "on_message": ui_on_message,
        "on_disconnect": ui_on_disconnect,
    }


@export(mod_name=Name, api=True, version="1", request_as_kwarg=True, api_methods=['POST'])
async def run(app: App, public_agent_id: str, request: RequestData):
    """Public API endpoint to run agents."""
    if request is None:
        return Result.default_user_error(info="Failed to run agent: No request provided.")
    if not request.headers:
        return Result.default_user_error(info="Failed to run agent: No request headers provided.")

    auth_header = request.headers.authorization or request.headers.to_dict().get('authorization')

    if not auth_header or not auth_header.startswith('Bearer '):
        return Result.default_user_error("Authorization header missing or invalid.", exec_code=401)

    api_key = auth_header.split(' ')[1]

    if STATE.key_to_agent.get(api_key) != public_agent_id:
        return Result.default_user_error("Invalid API Key or Agent ID.", exec_code=403)

    conn_id = STATE.agent_to_client.get(public_agent_id)
    if not conn_id:
        return Result.default_internal_error("Agent is not currently connected/online.", exec_code=503)

    body = request.body
    request_id = f"req_{secrets.token_urlsafe(16)}"

    run_request = RunRequest(
        request_id=request_id,
        public_agent_id=public_agent_id,
        query=body.get('query', ''),
        session_id=body.get('session_id'),
        kwargs=body.get('kwargs', {})
    )

    response_queue = asyncio.Queue()
    STATE.pending_requests[request_id] = response_queue

    # Send run request to the client
    await app.ws_send(conn_id, WsMessage(event='run_request', data=run_request.model_dump()).model_dump())

    try:
        final_result = None
        while True:
            item = await asyncio.wait_for(response_queue.get(), timeout=120.0)

            if isinstance(item, ExecutionError):
                return Result.default_internal_error(
                    info=f"An error occurred during agent execution: {item.error}",
                    exec_code=500
                )

            if item.is_final:
                final_result = item.payload.get("details", {}).get("result")
                break

        return Result.json(data={"result": final_result})

    except TimeoutError:
        return Result.default_internal_error(
            info="The request timed out as the agent did not respond in time.",
            exec_code=504
        )
    finally:
        STATE.pending_requests.pop(request_id, None)


@export(mod_name=Name, api=True, version="1", api_methods=['GET'])
async def ui(app: App, public_agent_id: str = None):
    """Serve the interactive 3-panel agent UI."""
    # from ..isaa.ui import get_agent_ui_html
    # html_content = get_agent_ui_html()
    return Result.html(data="html_content", row=True)
