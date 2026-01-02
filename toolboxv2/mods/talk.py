# talk.py - Voice Assistant with Kernel Integration
# Version: 2.0.0 - Minimal Latency, Agent Selection, Embedded UI Components
"""
Talk Module - Voice Interface for AI Agents

Features:
- Kernel-based architecture with FlowAgent
- Agent selection based on user login
- Minimal end-to-end latency with parallel processing
- Auto-detection for speech start/end
- Mini-tools (delegate_to_agent, fetch_info) - non-blocking
- Custom iframe UI components managed by agent
- WebSocket-based real-time communication
"""

import asyncio
import base64
import json
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from toolboxv2 import App, MainTool, RequestData, Result, get_app
from toolboxv2.utils.extras.base_widget import get_current_user_from_request

# --- Constants ---
MOD_NAME = "talk"
VERSION = "2.0.0"
export = get_app(f"widgets.{MOD_NAME}").tb


# --- Enums ---
class TalkState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    DELEGATING = "delegating"


class UIComponentType(str, Enum):
    WIDGET = "widget"
    PANEL = "panel"
    OVERLAY = "overlay"
    NOTIFICATION = "notification"


# --- Models ---
class UIComponent(BaseModel):
    """Agent-managed UI component that can be embedded in Talk interface"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: UIComponentType = UIComponentType.WIDGET
    title: str = ""
    content_url: str = ""  # iframe src
    html_content: str = ""  # or inline HTML
    position: dict = Field(default_factory=lambda: {"x": 0, "y": 0})
    size: dict = Field(default_factory=lambda: {"width": 300, "height": 200})
    visible: bool = True
    pinned: bool = False  # User can pin to keep permanently
    created_at: float = Field(default_factory=time.time)
    metadata: dict = Field(default_factory=dict)


class TalkSession(BaseModel):
    """Voice conversation session with kernel integration"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    agent_name: str = "self"
    state: TalkState = TalkState.IDLE
    kernel: Any = None  # Kernel instance
    ui_components: dict[str, UIComponent] = Field(default_factory=dict)
    pending_delegations: dict[str, asyncio.Task] = Field(default_factory=dict, exclude=True)
    audio_buffer: list[bytes] = Field(default_factory=list, exclude=True)
    last_activity: float = Field(default_factory=time.time)
    settings: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class AgentInfo(BaseModel):
    """Information about available agent"""
    name: str
    display_name: str
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    avatar_url: str = ""
    is_default: bool = False


# --- WebSocket Output Router for Talk ---
class TalkOutputRouter:
    """Routes kernel output to WebSocket clients"""

    def __init__(self, app: App, session_id: str):
        self.app = app
        self.session_id = session_id
        self._ws_connections: set[str] = set()

    def add_connection(self, conn_id: str):
        self._ws_connections.add(conn_id)

    def remove_connection(self, conn_id: str):
        self._ws_connections.discard(conn_id)

    async def send_response(self, user_id: str, content: str, role: str = "assistant"):
        """Send agent response to all connected clients"""
        await self._broadcast({
            "type": "agent_response",
            "content": content,
            "role": role,
            "timestamp": time.time()
        })

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification"""
        await self._broadcast({
            "type": "notification",
            "content": content,
            "priority": priority,
            "metadata": metadata or {},
            "timestamp": time.time()
        })

    async def send_chunk(self, chunk: str):
        """Send streaming chunk"""
        await self._broadcast({
            "type": "chunk",
            "content": chunk,
            "timestamp": time.time()
        })

    async def send_state(self, state: TalkState):
        """Send state update"""
        await self._broadcast({
            "type": "state",
            "state": state.value,
            "timestamp": time.time()
        })

    async def send_audio(self, audio_data: bytes, format: str = "audio/mpeg"):
        """Send audio for playback"""
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await self._broadcast({
            "type": "audio",
            "content": audio_b64,
            "format": format,
            "timestamp": time.time()
        })

    async def send_ui_component(self, component: UIComponent):
        """Send UI component update"""
        await self._broadcast({
            "type": "ui_component",
            "component": component.model_dump(),
            "timestamp": time.time()
        })

    async def _broadcast(self, message: dict):
        """Broadcast to all WebSocket connections"""
        for conn_id in list(self._ws_connections):
            try:
                await self.app.ws_send(conn_id, message)
            except Exception as e:
                self.app.logger.warning(f"Failed to send to {conn_id}: {e}")
                self._ws_connections.discard(conn_id)


# --- Main Module Class ---
class Tools(MainTool):
    """Talk Module - Voice Interface with Kernel Integration"""

    def __init__(self, app: App):
        self.version = VERSION
        self.name = MOD_NAME
        self.color = "CYAN"
        self.sessions: dict[str, TalkSession] = {}
        self.routers: dict[str, TalkOutputRouter] = {}
        self.stt_func = None
        self.tts_func = None
        self.isaa_mod = None
        self.vad_model = None  # Voice Activity Detection
        super().__init__(
            load=self.on_start,
            v=VERSION,
            name=MOD_NAME,
            tool={},
            on_exit=self.on_exit
        )

    def on_start(self):
        """Initialize Talk module"""
        self.app.logger.info(f"Starting {self.name} v{self.version}...")

        # Get ISAA module
        self.isaa_mod = self.app.get_mod("isaa")
        if not self.isaa_mod:
            self.app.logger.error(f"{self.name}: ISAA module not found!")
            return

        # Initialize STT/TTS from AUDIO module
        from toolboxv2 import TBEF
        if hasattr(TBEF, "AUDIO") and self.app.get_mod("AUDIO"):
            self.stt_func = self.app.run_any(
                TBEF.AUDIO.STT_GENERATE,
                model="openai/whisper-small",
                row=True,
                device=0
            )
            self.tts_func = self.app.get_function(TBEF.AUDIO.SPEECH, state=False)[0]

            if self.stt_func and self.stt_func != "404":
                self.app.logger.info("Talk STT Online")
            else:
                self.stt_func = None

            if self.tts_func and self.tts_func != "404":
                self.app.logger.info("Talk TTS Online")
            else:
                self.tts_func = None

        # Register UI
        self.app.run_any(
            ("CloudM", "add_ui"),
            name=MOD_NAME,
            title="Voice Assistant",
            path=f"/api/{MOD_NAME}/ui",
            description="Voice interface with AI agents",
            auth=True
        )

        self.app.logger.info(f"{self.name} initialized successfully")

    def on_exit(self):
        """Cleanup"""
        for session in self.sessions.values():
            if session.kernel:
                asyncio.create_task(session.kernel.stop())
        self.app.logger.info(f"Closing {self.name}")

    async def _get_user_uid(self, request: RequestData) -> Optional[str]:
        """Get user ID from request"""
        user = await get_current_user_from_request(self.app, request)
        return user.uid if user and hasattr(user, 'uid') and user.uid else None

    async def _get_or_create_session(
        self,
        user_id: str,
        agent_name: str = "self"
    ) -> TalkSession:
        """Get existing session or create new one"""
        session_key = f"{user_id}:{agent_name}"

        if session_key in self.sessions:
            session = self.sessions[session_key]
            session.last_activity = time.time()
            return session

        # Create new session with kernel
        session = TalkSession(user_id=user_id, agent_name=agent_name)

        # Initialize kernel for this agent
        try:
            from toolboxv2.mods.isaa.kernel import Kernel
            agent = await self.isaa_mod.get_agent(agent_name)

            # Create output router
            router = TalkOutputRouter(self.app, session.session_id)
            self.routers[session.session_id] = router

            # Create kernel with router
            kernel = Kernel(agent=agent, output_router=router)
            await kernel.start()

            session.kernel = kernel
            self.sessions[session_key] = session

            self.app.logger.info(f"Created talk session for {user_id} with agent {agent_name}")

        except Exception as e:
            self.app.logger.error(f"Failed to create kernel: {e}")
            raise

        return session


# --- HTTP API Endpoints ---

@export(mod_name=MOD_NAME, api=True, name="agents", api_methods=['GET'], request_as_kwarg=True)
async def get_available_agents(self: Tools, request: RequestData) -> Result:
    """Get list of available agents for the user"""
    user_id = await self._get_user_uid(request)
    if not user_id:
        return Result.default_user_error(info="Authentication required", exec_code=401)

    # Get agents from ISAA config
    agents = []

    # Default agent
    agents.append(AgentInfo(
        name="self",
        display_name="Personal Assistant",
        description="Your default AI assistant",
        capabilities=["general", "coding", "research"],
        is_default=True
    ).model_dump())

    # Get other configured agents
    if self.isaa_mod and hasattr(self.isaa_mod, 'config'):
        agent_list = self.isaa_mod.config.get("agents-name-list", [])
        for agent_name in agent_list:
            if agent_name != "self":
                agents.append(AgentInfo(
                    name=agent_name,
                    display_name=agent_name.replace("_", " ").title(),
                    description=f"Agent: {agent_name}",
                    capabilities=[]
                ).model_dump())

    return Result.json(data={"agents": agents})


@export(mod_name=MOD_NAME, api=True, name="session", api_methods=['POST'], request_as_kwarg=True)
async def create_session(self: Tools, request: RequestData) -> Result:
    """Create or get talk session"""
    user_id = await self._get_user_uid(request)
    if not user_id:
        return Result.default_user_error(info="Authentication required", exec_code=401)

    body = request.body or {}
    agent_name = body.get("agent_name", "self")

    try:
        session = await self._get_or_create_session(user_id, agent_name)
        return Result.json(data={
            "session_id": session.session_id,
            "agent_name": session.agent_name,
            "state": session.state.value,
            "ui_components": [c.model_dump() for c in session.ui_components.values()]
        })
    except Exception as e:
        return Result.default_internal_error(info=str(e))


@export(mod_name=MOD_NAME, api=True, name="components", api_methods=['GET', 'POST', 'DELETE'], request_as_kwarg=True)
async def manage_ui_components(self: Tools, request: RequestData, session_id: str = None) -> Result:
    """Manage UI components for a session"""
    user_id = await self._get_user_uid(request)
    if not user_id:
        return Result.default_user_error(info="Authentication required", exec_code=401)

    # Find session
    session = None
    for s in self.sessions.values():
        if s.session_id == session_id and s.user_id == user_id:
            session = s
            break

    if not session:
        return Result.default_user_error(info="Session not found", exec_code=404)

    if request.method == "GET":
        return Result.json(data={
            "components": [c.model_dump() for c in session.ui_components.values()]
        })

    elif request.method == "POST":
        body = request.body or {}
        component = UIComponent(
            title=body.get("title", ""),
            type=UIComponentType(body.get("type", "widget")),
            content_url=body.get("content_url", ""),
            html_content=body.get("html_content", ""),
            position=body.get("position", {"x": 0, "y": 0}),
            size=body.get("size", {"width": 300, "height": 200}),
            pinned=body.get("pinned", False),
            metadata=body.get("metadata", {})
        )
        session.ui_components[component.id] = component

        # Notify clients
        router = self.routers.get(session.session_id)
        if router:
            await router.send_ui_component(component)

        return Result.json(data={"component": component.model_dump()})

    elif request.method == "DELETE":
        component_id = request.body.get("component_id") if request.body else None
        if component_id and component_id in session.ui_components:
            del session.ui_components[component_id]
            return Result.ok(data={"deleted": component_id})
        return Result.default_user_error(info="Component not found", exec_code=404)


# --- WebSocket Handler ---

@export(mod_name=MOD_NAME, websocket_handler="talk", request_as_kwarg=True)
def register_talk_websocket(app: App, request: RequestData = None):
    """WebSocket handler for Talk interface"""

    # Connection state
    connections: dict[str, dict] = {}

    async def on_connect(session: dict, conn_id: str = None, **kwargs):
        """Handle WebSocket connection"""
        conn_id = conn_id or session.get("connection_id", "unknown")
        app.logger.info(f"[Talk] WebSocket connected: {conn_id}")

        connections[conn_id] = {
            "session_id": None,
            "user_id": None,
            "state": TalkState.IDLE,
            "audio_buffer": [],
            "silence_frames": 0,
            "is_speaking": False
        }

        await app.ws_send(conn_id, {
            "type": "connected",
            "message": "Connected to Talk interface",
            "timestamp": time.time()
        })

        return {"accept": True}

    async def on_message(payload: dict, session: dict, conn_id: str = None, **kwargs):
        """Handle incoming WebSocket messages"""
        conn_id = conn_id or session.get("connection_id", "unknown")
        conn_state = connections.get(conn_id)

        if not conn_state:
            return

        msg_type = payload.get("type")
        tools_instance = app.get_mod(MOD_NAME)
        payload["user_id"] = "123"
        try:
            # === Session Management ===
            if msg_type == "init_session":
                user_id = payload.get("user_id")
                agent_name = payload.get("agent_name", "self")

                if not user_id:
                    await app.ws_send(conn_id, {"type": "error", "message": "user_id required"})
                    return

                # Create/get session
                talk_session = await tools_instance._get_or_create_session(user_id, agent_name)

                # Register this connection
                router = tools_instance.routers.get(talk_session.session_id)
                if router:
                    router.add_connection(conn_id)

                conn_state["session_id"] = talk_session.session_id
                conn_state["user_id"] = user_id

                await app.ws_send(conn_id, {
                    "type": "session_ready",
                    "session_id": talk_session.session_id,
                    "agent_name": talk_session.agent_name,
                    "state": talk_session.state.value,
                    "ui_components": [c.model_dump() for c in talk_session.ui_components.values()]
                })

            # === Audio Streaming ===
            elif msg_type == "audio_chunk":
                session_id = conn_state.get("session_id")
                if not session_id:
                    return

                audio_b64 = payload.get("audio")
                if not audio_b64:
                    return

                audio_data = base64.b64decode(audio_b64)
                conn_state["audio_buffer"].append(audio_data)

                # Voice Activity Detection (simplified)
                is_speech = len(audio_data) > 0 and max(audio_data) > 10

                if is_speech:
                    conn_state["silence_frames"] = 0
                    if not conn_state["is_speaking"]:
                        conn_state["is_speaking"] = True
                        await app.ws_send(conn_id, {"type": "vad", "speaking": True})
                else:
                    conn_state["silence_frames"] += 1
                    # End of speech after ~500ms silence (assuming 100ms chunks)
                    if conn_state["is_speaking"] and conn_state["silence_frames"] > 5:
                        conn_state["is_speaking"] = False
                        await app.ws_send(conn_id, {"type": "vad", "speaking": False})

                        # Auto-process accumulated audio
                        if conn_state["audio_buffer"]:
                            await process_audio_buffer(
                                tools_instance, app, conn_id, conn_state, session_id
                            )

            # === Manual Audio Submit ===
            elif msg_type == "audio_submit":
                session_id = conn_state.get("session_id")
                if not session_id:
                    return

                audio_b64 = payload.get("audio")
                if audio_b64:
                    audio_data = base64.b64decode(audio_b64)
                    conn_state["audio_buffer"] = [audio_data]

                await process_audio_buffer(
                    tools_instance, app, conn_id, conn_state, session_id
                )

            # === Text Input (bypass STT) ===
            elif msg_type == "text_input":
                session_id = conn_state.get("session_id")
                user_id = conn_state.get("user_id")
                text = payload.get("text", "").strip()

                if not session_id or not text:
                    return

                await process_user_input(
                    tools_instance, app, conn_id, session_id, user_id, text
                )

            # === Mini-Tools (Non-blocking) ===
            elif msg_type == "delegate":
                # Delegate to another agent without waiting
                session_id = conn_state.get("session_id")
                target_agent = payload.get("agent")
                task = payload.get("task")

                if session_id and target_agent and task:
                    asyncio.create_task(
                        delegate_to_agent(tools_instance, app, conn_id, target_agent, task)
                    )
                    await app.ws_send(conn_id, {
                        "type": "delegation_started",
                        "agent": target_agent,
                        "task": task
                    })

            elif msg_type == "fetch_info":
                # Quick info fetch without full agent call
                query = payload.get("query")
                if query:
                    asyncio.create_task(
                        fetch_info_quick(tools_instance, app, conn_id, query)
                    )

            # === UI Component Management ===
            elif msg_type == "pin_component":
                session_id = conn_state.get("session_id")
                component_id = payload.get("component_id")
                pinned = payload.get("pinned", True)

                for s in tools_instance.sessions.values():
                    if s.session_id == session_id:
                        if component_id in s.ui_components:
                            s.ui_components[component_id].pinned = pinned
                            await app.ws_send(conn_id, {
                                "type": "component_updated",
                                "component_id": component_id,
                                "pinned": pinned
                            })
                        break

            # === Settings ===
            elif msg_type == "update_settings":
                session_id = conn_state.get("session_id")
                settings = payload.get("settings", {})

                for s in tools_instance.sessions.values():
                    if s.session_id == session_id:
                        s.settings.update(settings)
                        await app.ws_send(conn_id, {
                            "type": "settings_updated",
                            "settings": s.settings
                        })
                        break

        except Exception as e:
            import traceback
            traceback.print_exc()
            app.logger.error(f"[Talk] WebSocket error: {e}")
            await app.ws_send(conn_id, {"type": "error", "message": str(e)})

    async def on_disconnect(session: dict, conn_id: str = None, **kwargs):
        """Handle WebSocket disconnection"""
        conn_id = conn_id or session.get("connection_id", "unknown")
        app.logger.info(f"[Talk] WebSocket disconnected: {conn_id}")

        conn_state = connections.pop(conn_id, None)
        if conn_state and conn_state.get("session_id"):
            # Remove from router
            tools_instance = app.get_mod(MOD_NAME)
            router = tools_instance.routers.get(conn_state["session_id"])
            if router:
                router.remove_connection(conn_id)

    return {
        "on_connect": on_connect,
        "on_message": on_message,
        "on_disconnect": on_disconnect
    }


# --- Audio Processing ---

async def process_audio_buffer(tools_instance: Tools, app: App, conn_id: str, conn_state: dict, session_id: str):
    """Process accumulated audio buffer"""
    audio_chunks = conn_state.get("audio_buffer", [])
    if not audio_chunks:
        return

    conn_state["audio_buffer"] = []
    conn_state["state"] = TalkState.PROCESSING

    await app.ws_send(conn_id, {"type": "state", "state": "processing"})

    # Combine audio chunks
    full_audio = b"".join(audio_chunks)

    # Transcribe
    if not tools_instance.stt_func:
        await app.ws_send(conn_id, {"type": "error", "message": "STT not available"})
        return

    try:
        result = tools_instance.stt_func(full_audio)
        text = result.get("text", "").strip()

        if not text:
            await app.ws_send(conn_id, {"type": "transcription", "text": "", "empty": True})
            conn_state["state"] = TalkState.IDLE
            await app.ws_send(conn_id, {"type": "state", "state": "idle"})
            return

        await app.ws_send(conn_id, {"type": "transcription", "text": text})

        # Process with agent
        user_id = conn_state.get("user_id")
        await process_user_input(tools_instance, app, conn_id, session_id, user_id, text)

    except Exception as e:
        app.logger.error(f"STT error: {e}")
        await app.ws_send(conn_id, {"type": "error", "message": f"Transcription failed: {e}"})


async def process_user_input(tools_instance: Tools, app: App, conn_id: str, session_id: str, user_id: str, text: str):
    """Process user input through kernel - minimal latency path"""

    # Find session
    session = None
    for s in tools_instance.sessions.values():
        if s.session_id == session_id:
            session = s
            break

    if not session or not session.kernel:
        await app.ws_send(conn_id, {"type": "error", "message": "Session not found"})
        return

    session.state = TalkState.PROCESSING
    await app.ws_send(conn_id, {"type": "state", "state": "processing"})

    try:
        # MINIMAL LATENCY PATH:
        # 1. Start TTS generation in parallel with response streaming
        # 2. Use fast model for immediate response
        # 3. Delegate complex tasks asynchronously

        agent = session.kernel.agent
        router = tools_instance.routers.get(session_id)

        # Stream response chunks
        full_response = ""
        async for chunk in agent.a_stream(text, session_id=user_id):
            full_response += chunk
            if router:
                await router.send_chunk(chunk)

        session.state = TalkState.SPEAKING
        await app.ws_send(conn_id, {"type": "state", "state": "speaking"})

        # Generate TTS
        if tools_instance.tts_func and full_response.strip():
            voice_settings = session.settings.get("voice", {})
            audio_data = tools_instance.tts_func(
                text=full_response,
                voice_index=voice_settings.get("voice_index", 0),
                provider=voice_settings.get("provider", "piper"),
                config={
                    "play_local": False,
                    "model_name": voice_settings.get("model_name", "ryan")
                },
                local=False,
                save=False
            )

            if audio_data and router:
                await router.send_audio(audio_data, "audio/mpeg")

        session.state = TalkState.IDLE
        await app.ws_send(conn_id, {"type": "state", "state": "idle"})

    except Exception as e:
        import traceback
        traceback.print_exc()
        session.state = TalkState.IDLE
        await app.ws_send(conn_id, {"type": "error", "message": str(e)})
        await app.ws_send(conn_id, {"type": "state", "state": "idle"})


# --- Mini-Tools (Non-blocking) ---

async def delegate_to_agent(tools_instance: Tools, app: App, conn_id: str, target_agent: str, task: str):
    """Delegate task to another agent without blocking"""
    try:
        agent = await tools_instance.isaa_mod.get_agent(target_agent)
        result = await agent.a_run(task, session_id=f"delegate_{conn_id}")

        await app.ws_send(conn_id, {
            "type": "delegation_complete",
            "agent": target_agent,
            "result": result
        })
    except Exception as e:
        await app.ws_send(conn_id, {
            "type": "delegation_error",
            "agent": target_agent,
            "error": str(e)
        })


async def fetch_info_quick(tools_instance: Tools, app: App, conn_id: str, query: str):
    """Quick info fetch without full agent reasoning"""
    try:
        # Use fast model directly for simple queries
        agent = await tools_instance.isaa_mod.get_agent("self")

        # Quick format call
        result = await agent.a_format_class(
            pydantic_model=None,
            prompt=f"Quick answer (1-2 sentences): {query}",
            auto_context=False,
            model_preference="fast"
        )

        await app.ws_send(conn_id, {
            "type": "quick_info",
            "query": query,
            "result": result if isinstance(result, str) else str(result)
        })
    except Exception as e:
        await app.ws_send(conn_id, {
            "type": "quick_info_error",
            "query": query,
            "error": str(e)
        })


# --- UI Endpoint ---

@export(mod_name=MOD_NAME, name="ui", api=True, api_methods=['GET'], request_as_kwarg=True)
def get_main_ui(self: Tools, request: RequestData) -> Result:
    """Serves the main Talk UI"""
    html_content = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Talk - Voice Assistant</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-glass: rgba(255,255,255,0.03);
            --text-primary: #ffffff;
            --text-secondary: rgba(255,255,255,0.6);
            --accent: #6366f1;
            --accent-glow: rgba(99,102,241,0.3);
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: rgba(255,255,255,0.08);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            z-index: 100;
        }

        .agent-selector {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .agent-selector select {
            background: var(--bg-glass);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-secondary);
        }

        .status-dot.connected { background: var(--success); }
        .status-dot.processing { background: var(--warning); animation: pulse 1s infinite; }
        .status-dot.error { background: var(--error); }

        /* Main Area */
        .main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            padding: 20px;
        }

        /* Visualizer Circle */
        .visualizer-container {
            position: relative;
            width: 280px;
            height: 280px;
        }

        .visualizer {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: var(--bg-glass);
            border: 2px solid var(--border);
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .visualizer.listening {
            border-color: var(--error);
            box-shadow: 0 0 30px rgba(239,68,68,0.3);
        }

        .visualizer.processing {
            border-color: var(--accent);
            animation: glow-pulse 2s infinite;
        }

        .visualizer.speaking {
            border-color: var(--success);
            box-shadow: 0 0 30px rgba(34,197,94,0.3);
        }

        @keyframes glow-pulse {
            0%, 100% { box-shadow: 0 0 20px var(--accent-glow); }
            50% { box-shadow: 0 0 40px var(--accent-glow); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Particles */
        .particle {
            position: absolute;
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            pointer-events: none;
            opacity: 0.6;
        }

        /* Response Text */
        .response-area {
            margin-top: 30px;
            max-width: 500px;
            text-align: center;
            min-height: 80px;
        }

        .response-text {
            font-size: 18px;
            line-height: 1.6;
            color: var(--text-primary);
        }

        .transcription {
            font-size: 14px;
            color: var(--text-secondary);
            font-style: italic;
            margin-bottom: 10px;
        }

        /* Controls */
        .controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }

        .mic-button {
            width: 72px;
            height: 72px;
            border-radius: 50%;
            border: none;
            background: var(--accent);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 20px var(--accent-glow);
            transition: all 0.2s ease;
        }

        .mic-button:hover { transform: scale(1.05); }
        .mic-button:active { transform: scale(0.95); }
        .mic-button:disabled { background: var(--text-secondary); cursor: not-allowed; }
        .mic-button .material-symbols-outlined { font-size: 32px; }

        .mic-button.recording {
            background: var(--error);
            animation: recording-pulse 1.5s infinite;
        }

        @keyframes recording-pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
            50% { box-shadow: 0 0 0 15px rgba(239,68,68,0); }
        }

        .secondary-btn {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: 1px solid var(--border);
            background: var(--bg-glass);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }

        .secondary-btn:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        /* UI Components Panel */
        .components-panel {
            position: fixed;
            right: 0;
            top: 60px;
            bottom: 0;
            width: 320px;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            transform: translateX(100%);
            transition: transform 0.3s ease;
            overflow-y: auto;
            z-index: 90;
        }

        .components-panel.open { transform: translateX(0); }

        .component-card {
            margin: 12px;
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        .component-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }

        .component-title {
            font-size: 13px;
            font-weight: 500;
        }

        .component-content {
            min-height: 150px;
        }

        .component-content iframe {
            width: 100%;
            height: 200px;
            border: none;
        }

        /* Voice Options */
        .voice-options {
            position: absolute;
            bottom: 20px;
            left: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .voice-select {
            background: var(--bg-glass);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
        }

        /* Toggle Button */
        .toggle-panel-btn {
            position: fixed;
            right: 20px;
            bottom: 20px;
            z-index: 100;
        }

        /* Text Input Mode */
        .text-input-container {
            display: none;
            width: 100%;
            max-width: 500px;
            margin-top: 20px;
        }

        .text-input-container.active { display: block; }

        .text-input {
            width: 100%;
            padding: 14px 16px;
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text-primary);
            font-size: 16px;
            resize: none;
        }

        .text-input:focus {
            outline: none;
            border-color: var(--accent);
        }

        /* Mobile Optimizations */
        @media (max-width: 640px) {
            .visualizer-container { width: 200px; height: 200px; }
            .components-panel { width: 100%; }
            .header { padding: 10px 15px; }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="agent-selector">
            <span class="material-symbols-outlined">smart_toy</span>
            <select id="agentSelect">
                <option value="self">Personal Assistant</option>
            </select>
        </div>
        <div class="status-indicator">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
    </header>

    <main class="main-area">
        <div class="visualizer-container">
            <div class="visualizer" id="visualizer"></div>
        </div>

        <div class="response-area">
            <p class="transcription" id="transcription"></p>
            <p class="response-text" id="responseText"></p>
        </div>

        <div class="controls">
            <button class="secondary-btn" id="textModeBtn" title="Text Input">
                <span class="material-symbols-outlined">keyboard</span>
            </button>
            <button class="mic-button" id="micButton" disabled>
                <span class="material-symbols-outlined">hourglass_empty</span>
            </button>
            <button class="secondary-btn" id="settingsBtn" title="Settings">
                <span class="material-symbols-outlined">settings</span>
            </button>
        </div>

        <div class="text-input-container" id="textInputContainer">
            <textarea class="text-input" id="textInput" placeholder="Type your message..." rows="2"></textarea>
        </div>

        <div class="voice-options">
            <label style="font-size:12px;color:var(--text-secondary)">Voice:</label>
            <select class="voice-select" id="voiceSelect">
                <option value='{"provider":"piper","model_name":"ryan","voice_index":0}'>Ryan (EN)</option>
                <option value='{"provider":"piper","model_name":"kathleen","voice_index":0}'>Kathleen (EN)</option>
                <option value='{"provider":"piper","model_name":"karlsson","voice_index":0}'>Karlsson (DE)</option>
            </select>
        </div>
    </main>

    <div class="components-panel" id="componentsPanel">
        <div id="componentsList"></div>
    </div>

    <button class="secondary-btn toggle-panel-btn" id="togglePanelBtn" title="UI Components">
        <span class="material-symbols-outlined">widgets</span>
    </button>

    <script>
    (function() {
        // State
        const state = {
            ws: null,
            sessionId: null,
            userId: null,
            isRecording: false,
            isProcessing: false,
            mediaRecorder: null,
            audioChunks: [],
            audioContext: null,
            analyser: null,
            particles: [],
            currentAudio: null,
            autoDetect: true,
            silenceTimeout: null,
            components: {}
        };

        // Elements
        const el = {
            visualizer: document.getElementById('visualizer'),
            micButton: document.getElementById('micButton'),
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            responseText: document.getElementById('responseText'),
            transcription: document.getElementById('transcription'),
            agentSelect: document.getElementById('agentSelect'),
            voiceSelect: document.getElementById('voiceSelect'),
            textInput: document.getElementById('textInput'),
            textInputContainer: document.getElementById('textInputContainer'),
            textModeBtn: document.getElementById('textModeBtn'),
            componentsPanel: document.getElementById('componentsPanel'),
            componentsList: document.getElementById('componentsList'),
            togglePanelBtn: document.getElementById('togglePanelBtn')
        };

        // Initialize
        async function init() {
            createParticles();
            animateParticles();
            await loadAgents();
            connectWebSocket();
            setupEventListeners();
        }

        // Load available agents
        async function loadAgents() {
            try {
                const response = await TB.api.request('talk', 'agents', {}, 'GET');
                if (response.error === 'none' && response.get()?.agents) {
                    const agents = response.get().agents;
                    el.agentSelect.innerHTML = agents.map(a =>
                        `<option value="${a.name}" ${a.is_default ? 'selected' : ''}>${a.display_name}</option>`
                    ).join('');
                }
            } catch (e) {
                console.error('Failed to load agents:', e);
            }
        }

        // WebSocket Connection
        function connectWebSocket() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${location.host}/ws/talk/talk`;

            state.ws = new WebSocket(wsUrl);

            state.ws.onopen = () => {
                setStatus('connected', 'Connected');
                // Get user ID from TB
                state.userId = TB.user?.get()?.uid || 'anonymous';
                // Initialize session
                state.ws.send(JSON.stringify({
                    type: 'init_session',
                    user_id: state.userId,
                    agent_name: el.agentSelect.value
                }));
            };

            state.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };

            state.ws.onerror = () => setStatus('error', 'Connection Error');
            state.ws.onclose = () => {
                setStatus('error', 'Disconnected');
                setTimeout(connectWebSocket, 3000);
            };
        }

        // Handle WebSocket Messages
        function handleMessage(msg) {
            switch (msg.type) {
                case 'session_ready':
                    state.sessionId = msg.session_id;
                    el.micButton.disabled = false;
                    el.micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>';
                    setStatus('connected', 'Ready');
                    // Load UI components
                    if (msg.ui_components) {
                        msg.ui_components.forEach(addUIComponent);
                    }
                    break;

                case 'state':
                    updateVisualizerState(msg.state);
                    break;

                case 'transcription':
                    el.transcription.textContent = msg.empty ? '' : `"${msg.text}"`;
                    break;

                case 'chunk':
                    el.responseText.textContent += msg.content;
                    break;

                case 'agent_response':
                    el.responseText.textContent = msg.content;
                    break;

                case 'audio':
                    playAudio(msg.content, msg.format);
                    break;

                case 'vad':
                    // Voice activity detection feedback
                    if (msg.speaking) {
                        el.visualizer.classList.add('listening');
                    }
                    break;

                case 'ui_component':
                    addUIComponent(msg.component);
                    break;

                case 'delegation_started':
                    el.responseText.textContent = `Delegating to ${msg.agent}...`;
                    break;

                case 'delegation_complete':
                    el.responseText.textContent = msg.result;
                    break;

                case 'quick_info':
                    // Show as notification or in response
                    el.responseText.textContent = msg.result;
                    break;

                case 'error':
                    console.error('Talk error:', msg.message);
                    el.responseText.textContent = msg.message;
                    setStatus('error', 'Error');
                    break;
            }
        }

        // Update visualizer based on state
        function updateVisualizerState(s) {
            el.visualizer.className = 'visualizer';
            if (s === 'listening') {
                el.visualizer.classList.add('listening');
                setStatus('connected', 'Listening...');
            } else if (s === 'processing') {
                el.visualizer.classList.add('processing');
                setStatus('processing', 'Processing...');
                el.micButton.disabled = true;
            } else if (s === 'speaking') {
                el.visualizer.classList.add('speaking');
                setStatus('connected', 'Speaking...');
            } else {
                setStatus('connected', 'Ready');
                el.micButton.disabled = false;
                el.micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>';
            }
        }

        // Set status
        function setStatus(type, text) {
            el.statusDot.className = 'status-dot ' + type;
            el.statusText.textContent = text;
        }

        // Create particles
        function createParticles(num = 40) {
            el.visualizer.innerHTML = '';
            state.particles = [];
            for (let i = 0; i < num; i++) {
                const p = document.createElement('div');
                p.className = 'particle';
                el.visualizer.appendChild(p);
                state.particles.push({
                    element: p,
                    angle: Math.random() * Math.PI * 2,
                    radius: 40 + Math.random() * 60,
                    speed: 0.01 + Math.random() * 0.02
                });
            }
        }

        // Animate particles
        function animateParticles() {
            let avg = 0;
            if (state.analyser) {
                const data = new Uint8Array(state.analyser.frequencyBinCount);
                state.analyser.getByteFrequencyData(data);
                avg = data.reduce((a, b) => a + b, 0) / data.length;
            }

            const cx = el.visualizer.offsetWidth / 2;
            const cy = el.visualizer.offsetHeight / 2;

            state.particles.forEach(p => {
                p.angle += p.speed;
                const scale = 1 + (avg / 150);
                const x = cx + Math.cos(p.angle) * p.radius * scale - 3;
                const y = cy + Math.sin(p.angle) * p.radius * scale - 3;
                p.element.style.transform = `translate(${x}px, ${y}px)`;
            });

            requestAnimationFrame(animateParticles);
        }

        // Toggle recording
        async function toggleRecording() {
            if (state.isProcessing) return;

            if (state.isRecording) {
                stopRecording();
            } else {
                await startRecording();
            }
        }

        // Start recording
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1 }
                });

                if (!state.audioContext) {
                    state.audioContext = new AudioContext();
                }

                const source = state.audioContext.createMediaStreamSource(stream);
                if (!state.analyser) {
                    state.analyser = state.audioContext.createAnalyser();
                    state.analyser.fftSize = 64;
                }
                source.connect(state.analyser);

                state.mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });

                state.audioChunks = [];
                state.mediaRecorder.ondataavailable = e => state.audioChunks.push(e.data);
                state.mediaRecorder.onstop = sendAudio;

                state.mediaRecorder.start();
                state.isRecording = true;

                el.micButton.classList.add('recording');
                el.micButton.innerHTML = '<span class="material-symbols-outlined">stop</span>';
                el.visualizer.classList.add('listening');
                setStatus('connected', 'Listening...');
                el.responseText.textContent = '';
                el.transcription.textContent = '';

            } catch (e) {
                console.error('Microphone access error:', e);
                setStatus('error', 'Microphone Error');
            }
        }

        // Stop recording
        function stopRecording() {
            if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
                state.mediaRecorder.stop();
            }
            state.isRecording = false;
            el.micButton.classList.remove('recording');
            el.micButton.innerHTML = '<span class="material-symbols-outlined">hourglass_top</span>';
            el.micButton.disabled = true;
        }

        // Send audio to server
        async function sendAudio() {
            if (state.audioChunks.length === 0) {
                el.micButton.disabled = false;
                el.micButton.innerHTML = '<span class="material-symbols-outlined">mic</span>';
                return;
            }

            const blob = new Blob(state.audioChunks, { type: 'audio/webm;codecs=opus' });
            const reader = new FileReader();

            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                state.ws.send(JSON.stringify({
                    type: 'audio_submit',
                    audio: base64
                }));
                state.isProcessing = true;
            };

            reader.readAsDataURL(blob);
        }

        // Play audio response
        async function playAudio(base64, format) {
            try {
                const blob = await (await fetch(`data:${format};base64,${base64}`)).blob();
                const url = URL.createObjectURL(blob);

                if (state.currentAudio) {
                    state.currentAudio.pause();
                }

                state.currentAudio = new Audio(url);

                if (!state.audioContext) {
                    state.audioContext = new AudioContext();
                }

                const source = state.audioContext.createMediaElementSource(state.currentAudio);
                if (!state.analyser) {
                    state.analyser = state.audioContext.createAnalyser();
                    state.analyser.fftSize = 64;
                }
                source.connect(state.analyser);
                state.analyser.connect(state.audioContext.destination);

                state.currentAudio.play();
                state.currentAudio.onended = () => {
                    URL.revokeObjectURL(url);
                    state.isProcessing = false;
                };

            } catch (e) {
                console.error('Audio playback error:', e);
                state.isProcessing = false;
            }
        }

        // Send text input
        function sendTextInput() {
            const text = el.textInput.value.trim();
            if (!text || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;

            state.ws.send(JSON.stringify({
                type: 'text_input',
                text: text
            }));

            el.textInput.value = '';
            el.transcription.textContent = `"${text}"`;
            el.responseText.textContent = '';
        }

        // Add UI Component
        function addUIComponent(component) {
            state.components[component.id] = component;
            renderComponents();
        }

        // Render UI Components
        function renderComponents() {
            el.componentsList.innerHTML = Object.values(state.components).map(c => `
                <div class="component-card" data-id="${c.id}">
                    <div class="component-header">
                        <span class="component-title">${c.title || 'Component'}</span>
                        <button class="secondary-btn" style="width:28px;height:28px" onclick="togglePin('${c.id}')">
                            <span class="material-symbols-outlined" style="font-size:16px">${c.pinned ? 'push_pin' : 'push_pin'}</span>
                        </button>
                    </div>
                    <div class="component-content">
                        ${c.content_url ? `<iframe src="${c.content_url}" sandbox="allow-scripts allow-same-origin"></iframe>` : c.html_content}
                    </div>
                </div>
            `).join('');
        }

        // Toggle pin component
        window.togglePin = function(componentId) {
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                const c = state.components[componentId];
                state.ws.send(JSON.stringify({
                    type: 'pin_component',
                    component_id: componentId,
                    pinned: !c?.pinned
                }));
            }
        };

        // Setup event listeners
        function setupEventListeners() {
            el.micButton.addEventListener('click', toggleRecording);

            el.textModeBtn.addEventListener('click', () => {
                el.textInputContainer.classList.toggle('active');
            });

            el.textInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendTextInput();
                }
            });

            el.agentSelect.addEventListener('change', () => {
                if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                    state.ws.send(JSON.stringify({
                        type: 'init_session',
                        user_id: state.userId,
                        agent_name: el.agentSelect.value
                    }));
                }
            });

            el.voiceSelect.addEventListener('change', () => {
                if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                    const voice = JSON.parse(el.voiceSelect.value);
                    state.ws.send(JSON.stringify({
                        type: 'update_settings',
                        settings: { voice }
                    }));
                }
            });

            el.togglePanelBtn.addEventListener('click', () => {
                el.componentsPanel.classList.toggle('open');
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.code === 'Space' && e.ctrlKey) {
                    e.preventDefault();
                    toggleRecording();
                }
            });
        }

        // Wait for TB.js and initialize
        if (window.TB?.events) {
            if (window.TB.config?.get('appRootId')) {
                init();
            } else {
                window.TB.events.on('tbjs:initialized', init, { once: true });
            }
        } else {
            document.addEventListener('tbjs:initialized', init, { once: true });
        }
    })();
    </script>
</body>
</html>"""
    return Result.html(data=html_content)
