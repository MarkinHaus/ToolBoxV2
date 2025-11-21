"""
ProA Kernel Web Interface
==========================

Production-ready WebSocket interface for the Enhanced ProA Kernel with:
- Auto-persistence (save/load on start/stop)
- WebSocket session management
- Broadcast support for multi-user
- Connection tracking
- Graceful shutdown
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from toolboxv2 import App, get_app, Result
from toolboxv2.mods.isaa.extras.terminal_progress import ProgressiveTreePrinter
from toolboxv2.mods.isaa.kernel.instace import Kernel
from toolboxv2.mods.isaa.kernel.types import Signal as KernelSignal, SignalType, KernelConfig, IOutputRouter


class WebSocketOutputRouter(IOutputRouter):
    """WebSocket-specific output router with session management"""

    def __init__(self, app: App, channel_id: str):
        self.app = app
        self.channel_id = channel_id
        self.connections: Dict[str, dict] = {}  # conn_id -> session info

    def register_connection(self, conn_id: str, session: dict):
        """Register a new WebSocket connection"""
        self.connections[conn_id] = {
            "session": session,
            "user_id": session.get("user_name", "Anonymous"),
            "connected_at": datetime.now().isoformat()
        }

    def unregister_connection(self, conn_id: str):
        """Unregister a WebSocket connection"""
        if conn_id in self.connections:
            del self.connections[conn_id]

    async def send_response(self, user_id: str, content: str, role: str = "assistant", metadata: dict = None):
        """Send agent response to specific user"""
        # Find connection for user
        conn_id = self._get_conn_id_for_user(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "agent_response",
                "data": {
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            })

    async def send_notification(self, user_id: str, content: str, priority: int = 5, metadata: dict = None):
        """Send notification to specific user"""
        conn_id = self._get_conn_id_for_user(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "notification",
                "data": {
                    "content": content,
                    "priority": priority,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            })

    async def send_error(self, user_id: str, error: str, metadata: dict = None):
        """Send error message to specific user"""
        conn_id = self._get_conn_id_for_user(user_id)
        if conn_id:
            await self.app.ws_send(conn_id, {
                "event": "error",
                "data": {
                    "error": error,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
            })

    async def broadcast(self, content: str, event_type: str = "broadcast", exclude_user: str = None):
        """Broadcast message to all connected users"""
        await self.app.ws_broadcast(
            channel_id=self.channel_id,
            payload={
                "event": event_type,
                "data": {
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
            },
            source_conn_id=self._get_conn_id_for_user(exclude_user) if exclude_user else None
        )

    def _get_conn_id_for_user(self, user_id: str) -> Optional[str]:
        """Get connection ID for user"""
        for conn_id, info in self.connections.items():
            if info["user_id"] == user_id:
                return conn_id
        return None


class WebKernel:
    """WebSocket-based ProA Kernel with auto-persistence"""

    def __init__(self, agent, app: App, channel_id: str = "kernel_chat", auto_save_interval: int = 300):
        """
        Initialize Web Kernel

        Args:
            agent: FlowAgent instance
            app: ToolBoxV2 App instance
            channel_id: WebSocket channel ID
            auto_save_interval: Auto-save interval in seconds (default: 5 minutes)
        """
        self.agent = agent
        self.app = app
        self.channel_id = channel_id
        self.auto_save_interval = auto_save_interval
        self.running = False
        self.save_path = self._get_save_path()

        # Initialize kernel with WebSocket output router
        config = KernelConfig(
            heartbeat_interval=30.0,
            idle_threshold=300.0,
            proactive_cooldown=60.0,
            max_proactive_per_hour=10
        )

        self.output_router = WebSocketOutputRouter(app, channel_id)
        self.kernel = Kernel(
            agent=agent,
            config=config,
            output_router=self.output_router
        )

        print(f"✓ Web Kernel initialized for channel: {channel_id}")

    def _get_save_path(self) -> Path:
        """Get save file path"""
        save_dir = Path(self.app.data_dir) / 'Agents' / 'kernel' / self.agent.amd.name / 'web'
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"web_kernel_{self.channel_id}.pkl"

    async def _auto_save_loop(self):
        """Auto-save kernel state periodically"""
        while self.running:
            await asyncio.sleep(self.auto_save_interval)
            if self.running:
                await self.kernel.save_to_file(str(self.save_path))
                print(f"💾 Auto-saved web kernel at {datetime.now().strftime('%H:%M:%S')}")

    async def start(self):
        """Start the Web kernel"""
        self.running = True

        # Load previous state if exists
        if self.save_path.exists():
            print("📂 Loading previous web session...")
            await self.kernel.load_from_file(str(self.save_path))

        # Start kernel
        await self.kernel.start()

        # Inject kernel prompt to agent
        self.kernel.inject_kernel_prompt_to_agent()

        # Start auto-save loop
        asyncio.create_task(self._auto_save_loop())

        print(f"✓ Web Kernel started on channel: {self.channel_id}")

    async def stop(self):
        """Stop the Web kernel"""
        if not self.running:
            return

        self.running = False
        print("💾 Saving web session...")

        # Save final state
        await self.kernel.save_to_file(str(self.save_path))

        # Stop kernel
        await self.kernel.stop()

        print("✓ Web Kernel stopped")

    async def handle_connect(self, conn_id: str, session: dict):
        """Handle new WebSocket connection"""
        user_id = session.get("user_name", "Anonymous")

        # Register connection
        self.output_router.register_connection(conn_id, session)

        # Send welcome message
        await self.app.ws_send(conn_id, {
            "event": "welcome",
            "data": {
                "message": f"Welcome to the ProA Kernel, {user_id}!",
                "kernel_status": self.kernel.to_dict()
            }
        })

        # Broadcast user joined
        await self.output_router.broadcast(
            f"👋 {user_id} joined the chat",
            event_type="user_joined",
            exclude_user=user_id
        )

        # Send signal to kernel
        signal = KernelSignal(
            type=SignalType.SYSTEM_EVENT,
            id="websocket",
            content=f"User {user_id} connected",
            metadata={"event": "user_connect", "conn_id": conn_id}
        )
        await self.kernel.process_signal(signal)

    async def handle_disconnect(self, conn_id: str, session: dict = None):
        """Handle WebSocket disconnection"""
        if session is None:
            session = {}

        user_id = session.get("user_name", "Anonymous")

        # Unregister connection
        self.output_router.unregister_connection(conn_id)

        # Broadcast user left
        await self.output_router.broadcast(
            f"😥 {user_id} left the chat",
            event_type="user_left"
        )

        # Send signal to kernel
        signal = KernelSignal(
            type=SignalType.SYSTEM_EVENT,
            id="websocket",
            content=f"User {user_id} disconnected",
            metadata={"event": "user_disconnect", "conn_id": conn_id}
        )
        await self.kernel.process_signal(signal)

    async def handle_message(self, conn_id: str, session: dict, payload: dict):
        """Handle incoming WebSocket message"""
        user_id = session.get("user_name", "Anonymous")
        message_text = payload.get("data", {}).get("message", "").strip()

        if not message_text:
            return

        # Send signal to kernel
        signal = KernelSignal(
            type=SignalType.USER_INPUT,
            id=user_id,
            content=message_text,
            metadata={"interface": "websocket", "conn_id": conn_id}
        )
        await self.kernel.process_signal(signal)


# ===== WEBSOCKET HANDLER REGISTRATION =====

Name = "isaa.KernelWeb"
version = "1.0.0"
app = get_app(Name)
export = app.tb

# Global kernel instance
_kernel_instance: Optional[WebKernel] = None


@export(mod_name=Name, version=version, api=True, name="ui", row=True)
def get_kernel_ui(app: App) -> Result:
    """
    Delivers the HTML UI for the ProA Kernel Web Interface.
    Uses app.web_context() to include necessary CSS and JS.
    """

    html_content = f"""
        {app.web_context()}
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .kernel-container {{
                width: 90%;
                max-width: 1200px;
                height: 85vh;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}

            .kernel-header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 30px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .kernel-header h1 {{
                margin: 0;
                font-size: 24px;
                font-weight: 600;
            }}

            .kernel-status {{
                display: flex;
                align-items: center;
                gap: 10px;
                font-size: 14px;
            }}

            .status-indicator {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #4ade80;
                animation: pulse 2s infinite;
            }}

            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}

            .kernel-messages {{
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }}

            .message {{
                max-width: 70%;
                padding: 12px 18px;
                border-radius: 18px;
                line-height: 1.5;
                animation: slideIn 0.3s ease-out;
            }}

            @keyframes slideIn {{
                from {{
                    opacity: 0;
                    transform: translateY(10px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .message.user {{
                align-self: flex-end;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}

            .message.assistant {{
                align-self: flex-start;
                background: #f3f4f6;
                color: #1f2937;
            }}

            .message.system {{
                align-self: center;
                background: #fef3c7;
                color: #92400e;
                font-size: 14px;
                max-width: 90%;
            }}

            .kernel-input {{
                padding: 20px;
                background: white;
                border-top: 1px solid #e5e7eb;
                display: flex;
                gap: 10px;
            }}

            .kernel-input input {{
                flex: 1;
                padding: 12px 18px;
                border: 2px solid #e5e7eb;
                border-radius: 25px;
                font-size: 15px;
                outline: none;
                transition: border-color 0.3s;
            }}

            .kernel-input input:focus {{
                border-color: #667eea;
            }}

            .kernel-input button {{
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }}

            .kernel-input button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }}

            .kernel-input button:active {{
                transform: translateY(0);
            }}
        </style>

        <div class="kernel-container">
            <div class="kernel-header">
                <h1>🤖 ProA Kernel Chat</h1>
                <div class="kernel-status">
                    <div class="status-indicator"></div>
                    <span id="status-text">Connected</span>
                </div>
            </div>

            <div class="kernel-messages" id="messages">
                <div class="message system">
                    Welcome to ProA Kernel! I'm an AI assistant with learning and memory capabilities.
                </div>
            </div>

            <div class="kernel-input">
                <input type="text" id="message-input" placeholder="Type your message..." />
                <button id="send-button">Send</button>
            </div>
        </div>

        <script unsave="true">
            TB.once(() => {{
                const messagesContainer = document.getElementById('messages');
                const messageInput = document.getElementById('message-input');
                const sendButton = document.getElementById('send-button');
                const statusText = document.getElementById('status-text');

                // WebSocket connection
                let ws = null;

                function connect() {{
                    ws = TB.ws.connect('{Name}/kernel_room');

                    ws.on('open', () => {{
                        statusText.textContent = 'Connected';
                        addMessage('system', 'Connected to ProA Kernel');
                    }});

                    ws.on('close', () => {{
                        statusText.textContent = 'Disconnected';
                        addMessage('system', 'Disconnected from server');
                    }});

                    ws.on('message', (data) => {{
                        if (data.event === 'agent_response') {{
                            addMessage('assistant', data.content);
                        }} else if (data.event === 'notification') {{
                            addMessage('system', data.content);
                        }} else if (data.event === 'user_joined') {{
                            addMessage('system', `${{data.user_id}} joined the chat`);
                        }} else if (data.event === 'user_left') {{
                            addMessage('system', `${{data.user_id}} left the chat`);
                        }}
                    }});
                }}

                function addMessage(role, content) {{
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${{role}}`;
                    messageDiv.textContent = content;
                    messagesContainer.appendChild(messageDiv);
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }}

                function sendMessage() {{
                    const message = messageInput.value.trim();
                    if (!message || !ws) return;

                    addMessage('user', message);
                    ws.send({{ content: message }});
                    messageInput.value = '';
                }}

                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keypress', (e) => {{
                    if (e.key === 'Enter') sendMessage();
                }});

                // Connect on load
                connect();
            }});
        </script>
    """

    return Result.html(data=html_content)


@export(mod_name=Name, version=version, initial=True)
def init_kernel_web(app: App):
    """Initialize the Web Kernel module"""
    app.run_any(("CloudM", "add_ui"),
                name=Name,
                title="ProA Kernel Chat",
                path=f"/api/{Name}/ui",
                description="AI-powered chat with ProA Kernel")
    return {"success": True, "info": "KernelWeb initialized"}


@export(mod_name=Name, version=version, websocket_handler="kernel_room")
def register_kernel_handlers(app: App) -> dict:
    """Register WebSocket handlers for kernel"""
    global _kernel_instance

    # Create kernel instance on first registration
    if _kernel_instance is None:
        # Get ISAA and create agent
        isaa = app.get_mod("isaa")
        builder = isaa.get_agent_builder("WebKernelAssistant")
        builder.with_system_message(
            "You are a helpful web assistant. Provide clear, engaging responses."
        )
        #builder.with_models(
        #    fast_llm_model="openrouter/anthropic/claude-3-haiku",
        #    complex_llm_model="openrouter/openai/gpt-4o"
        #)

        # Register and get agent (synchronous wrapper)
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(isaa.register_agent(builder))
        agent = loop.run_until_complete(isaa.get_agent("WebKernelAssistant"))
        agent.set_progress_callback(ProgressiveTreePrinter().progress_callback)
        # Create kernel
        _kernel_instance = WebKernel(agent, app, channel_id=f"{Name}/kernel_room")
        loop.run_until_complete(_kernel_instance.start())

    return {
        "on_connect": _kernel_instance.handle_connect,
        "on_message": _kernel_instance.handle_message,
        "on_disconnect": _kernel_instance.handle_disconnect
    }

