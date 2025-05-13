
import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from datetime import datetime

from fastapi import WebSocket
from starlette.responses import HTMLResponse

from toolboxv2 import TBEF, Singleton, get_app
from toolboxv2.utils.extras.base_widget import get_spec


@dataclass
class AgentState:
    is_running: bool = False
    current_task: asyncio.Task | None = None
    last_response: str = ""
    verbose_output: list[str] = None

    def to_dict(self):
        return {
            "is_running": self.is_running,
            "last_response": self.last_response,
            "verbose_output": self.verbose_output or []
        }


class IsaaWebSocketUI(metaclass=Singleton):
    def __init__(self, isaa_tool, name="IsaaWebSocket"):
        self.isaa = isaa_tool
        self.active_connections: dict[str, WebSocket] = {}
        self.message_history: list[dict] = []
        self.agent_states: dict[str, AgentState] = {}
        self.ping_interval = 60
        self.max_reconnect_attempts = 5

    def _setup_verbose_override(self, agent, f=None):
        original_print_verbose = agent.print_verbose

        def new_print_verbose(msg, *args, **kwargs):
            client_id = kwargs.get('client_id')
            if client_id and client_id in self.agent_states:
                if self.agent_states[client_id].verbose_output is None:
                    self.agent_states[client_id].verbose_output = []
                self.agent_states[client_id].verbose_output.append(msg)
            original_print_verbose(msg, *args, **kwargs)
            if f:
                f(msg)

        agent.print_verbose = new_print_verbose

    async def connect_websocket(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.agent_states[client_id] = AgentState()
        asyncio.create_task(self._keep_alive(client_id))

    async def _keep_alive(self, client_id: str):
        """Send periodic ping to keep connection alive"""
        while client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({"type": "ping"})
                await asyncio.sleep(self.ping_interval)
            except:
                await self.disconnect_websocket(client_id)
                break

    async def disconnect_websocket(self, client_id: str):
        if client_id in self.active_connections:
            with contextlib.suppress(Exception):
                await self.active_connections[client_id].close()
            del self.active_connections[client_id]
            if client_id in self.agent_states:
                del self.agent_states[client_id]

    async def _monitor_agent_task(self, task: asyncio.Task, client_id: str):
        """Monitor agent task and update state"""
        try:
            await task
        except Exception as e:
            await self._send_error(client_id, str(e))
        finally:
            if client_id in self.agent_states:
                self.agent_states[client_id].is_running = False
                await self._send_agent_state(client_id)

    async def stream_agent_response(self, message: str, client_id: str, agent_name: str | None = None):
        """Stream agent responses to the client with async task management"""
        if client_id not in self.agent_states:
            return

        state = self.agent_states[client_id]
        if state.is_running:
            await self._send_error(client_id, "An agent is already running")
            return

        async def run_agent():
            #try:
            state.is_running = True
            state.verbose_output = []
            await self._send_agent_state(client_id)

            def get_callback(agent_name_):
                def helper(response_chunk, *a, **k):
                    try:
                        return get_app('nice.get_callback.helper').run_a_from_sync(self._send_stream_update, *[client_id,
                                                                                     response_chunk,
                                                                                     agent_name_])
                    except Exception as e:
                        print(f"Agent {agent_name_} faint to report : {response_chunk}", e)
                        pass

                return helper

            self.isaa.default_setter = lambda x: x.set_verbose(True).set_post_callback(
                get_callback(x.amd_attributes['name'])).set_print_verbose(
                get_callback(x.amd_attributes['name'] + "-internal"))

            # self.isaa.run_callback = get_callback(agent_name)

            for agent, name in zip([self.isaa.get_agent(name_) for name_ in self.isaa.config['agents-name-list']],
                                   self.isaa.config['agents-name-list'], strict=False):
                agent.post_callback = get_callback(name)
                self._setup_verbose_override(agent, get_callback(name + "-internal"))

            response = await asyncio.to_thread(
                self.isaa.run_agent,
                agent_name,
                message,
                persist=True,
                verbose=True
            )
            # Save to network branch
            network = self.isaa.get_memory().cognitive_network.network
            branch_id = f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            network.data_holder.create_branch(network, branch_id)

            # Save chat history
            self.message_history.append({
                'timestamp': datetime.now().isoformat(),
                'role': 'agent' if agent_name else 'system',
                'content': response,
                'agent': agent_name,
                'branch_id': branch_id,
                'verbose_output': state.verbose_output
            })

            state.last_response = response
            await self._send_agent_state(client_id)

            #except Exception as e:
            #    await self._send_error(client_id, str(e))
            #finally:
            # state.is_running = False
            await self._send_agent_state(client_id)

        task = asyncio.create_task(run_agent())
        state.current_task = task
        await self._monitor_agent_task(task, client_id)

    async def _send_stream_update(self, client_id: str, content: str, agent_name: str | None = None):
        """Send streaming updates to connected client"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                'type': 'stream',
                'content': content,
                'agent': agent_name
            })

    async def _send_agent_state(self, client_id: str):
        """Send agent state update to client"""
        if client_id in self.active_connections and client_id in self.agent_states:
            await self.active_connections[client_id].send_json({
                'type': 'agent_state',
                'state': self.agent_states[client_id].to_dict()
            })

    async def _send_error(self, client_id: str, error: str):
        """Send error message to client"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                'type': 'error',
                'content': error
            })

    async def handle_websocket_chat(self, websocket: WebSocket):
        """Handle main chat WebSocket connection"""
        client_id = str(uuid.uuid4())
        await self.connect_websocket(websocket, client_id)

        try:
            while True:
                message = await websocket.receive_json()
                if message['type'] == 'message':
                    await self.stream_agent_response(
                        message['content'],
                        client_id,
                        message.get('agent')
                    )
                elif message['type'] == 'get_branches':
                    network = self.isaa.get_memory().cognitive_network.network
                    branches = network.data_holder.get_visualization_data().get('branches', [])
                    await websocket.send_json({
                        'type': 'branches',
                        'branches': branches
                    })
                elif message['type'] == 'switch_branch':
                    network = self.isaa.get_memory().cognitive_network.network
                    success = network.data_holder.switch_branch(
                        network,
                        message['branch_id']
                    )
                    await websocket.send_json({
                        'type': 'branch_switch',
                        'success': success
                    })

                elif message['type'] == 't2s':
                    audio_data: bytes = self.isaa.app.run_any(TBEF.AUDIO.SPEECH, text=message['content'], voice_index=0,
                                                              use_cache=False,
                                                              provider='piper',
                                                              config={'play_local': False,
                                                                      'model_name': message.get('model_name', 'ryan')},
                                                              local=False,
                                                              save=False)

                    if not audio_data:
                        return
                    await websocket.send_bytes(audio_data)
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await self.disconnect_websocket(client_id)

    def get_widget(self, **kwargs):
        """Generate the HTML widget"""
        template = """
        <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Chat Interface</title>
</head>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --primary: #4f46e5;
    --primary-light: #6366f1;
    --bg-dark: #0f172a;
    --panel-bg: #1e293b;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #334155;
    --danger: #ef4444;
    --success: #22c55e;
    --transition: all 0.3s ease;
    --sidebar-width: 45vw;
}

/* Base Styles */
body {
    margin: 0;
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg-dark);
    color: var(--text);
    line-height: 1.5;
    overflow-x: hidden;
}
/* Layout */
.main-layout {
    display: grid;
    grid-template-columns: 0.25fr 1fr 0.25fr;
    height: 100vh;
    transition: var(--transition);
    position: relative;
}

/* Panel Styles */
.sidebar, .right-sidebar {
    background: var(--panel-bg);
    border-right: 1px solid var(--border);
    padding: 1.5rem;
    overflow-y: auto;
    position: relative;
    transition: var(--transition);
}

.right-sidebar {
    border-left: 1px solid var(--border);
    border-right: none;
    width: var(--sidebar-width);
}

/* Panel Toggle States */
.main-layout.left-hidden .sidebar {
grid-template-columns: 0 1fr var(--sidebar-width);
    transform: translateX(-var(--sidebar-width));
    width: 0;
    padding: 0;
    opacity: 0;
}

.main-layout.right-hidden .right-sidebar {
grid-template-columns: var(--sidebar-width) 1fr 0;
    transform: translateX(var(--sidebar-width));
    width: 0;
    padding: 0;
    opacity: 0;
}

.main-layout.both-hidden .sidebar,
.main-layout.both-hidden .right-sidebar {
grid-template-columns: 0 1fr 0;
    width: 0;
    padding: 0;
    opacity: 0;
}

.main-layout.both-hidden .sidebar {
    transform: translateX(-var(--sidebar-width));
}

.main-layout.both-hidden .right-sidebar {
    transform: translateX(var(--sidebar-width));
}

/* Toggle Buttons */
.panel-toggle {
    position: fixed;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    color: var(--text);
    width: 24px;
    height: 60px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: var(--transition);
    z-index: 100;
}

.panel-toggle:hover {
    background: var(--primary);
    color: white;
}

.left-panel-toggle {
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    border-radius: 0 4px 4px 0;
}

.right-panel-toggle {
    right: 0;
    top: 50%;
    transform: translateY(-50%);
    border-radius: 4px 0 0 4px;
}

/* Hide all sidebar content when collapsed except toggle button */
.main-layout.left-hidden .sidebar > *:not(.panel-toggle),
.main-layout.right-hidden .right-sidebar > *:not(.panel-toggle) {
    display: none;
}

/* Media Queries */
@media (max-width: 1024px) {
    .main-layout {
        --sidebar-width: 65vw;
    }
}

@media (max-width: 768px) {
    .main-layout {
        grid-template-columns: 1fr;
    }

    .sidebar, .right-sidebar {
        position: fixed;
        top: 0;
        height: 100vh;
        z-index: 50;
    }

    .sidebar {
        left: 0;
        transform: translateX(-100%);
    }

    .right-sidebar {
        right: 0;
        transform: translateX(100%);
    }

    .main-layout:not(.left-hidden) .sidebar {
        transform: translateX(0);
    }

    .main-layout:not(.right-hidden) .right-sidebar {
        transform: translateX(0);
    }

    .chat-header {
        padding: 0.75rem;
    }

    #agentSelect {
        max-width: 150px;
    }
}

@media (max-width: 480px) {
    .message {
        flex-direction: column;
    }

    .chat-input {
        padding: 0.75rem;
    }

    #messageInput {
        font-size: 16px; /* Prevent zoom on mobile */
    }

    .chat-header {
        flex-direction: column;
        gap: 0.5rem;
    }

    #agentSelect {
        width: 100%;
        max-width: none;
    }

    .controls {
        display: flex;
        gap: 0.5rem;
        width: 100%;
    }

    .controls button {
        flex: 1;
        padding: 0.5rem;
    }
}

/* Main Container */
.main-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}

/* Chat Header */
.chat-header {
    padding: 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--panel-bg);
}

/* Chat Container */
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.message {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    padding: 1rem;
    border-radius: 8px;
    /*background: var(--panel-bg); */

    backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
    animation: messageAppear 0.5s ease-out;
}


@keyframes messageAppear {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.user-message {
    margin-left: auto;
    background: var(--primary);
}

.message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--primary-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
}

.message-content {
    flex: 1;
}

/* Input Area */
.chat-input {
    padding: 1rem;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 1rem;
    background: var(--panel-bg);
}

#messageInput {
    flex: 1;
    padding: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-dark);
    color: var(--text);
}

/* Buttons */
button {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 4px;
    background: var(--primary);
    color: white;
    cursor: pointer;
    transition: var(--transition);
}

button:hover {
    background: var(--primary-light);
}

/* Status Indicators */
.status-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding: 0.75rem;
    background: var(--bg-dark);
    border-radius: 4px;
}

.status-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    transition: var(--transition);
}

.status-idle { background: var(--primary); }
.status-active { background: var(--success); }
.status-error { background: var(--danger); }

/* Log Panel */
.log-panel {
    background: var(--bg-dark);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.75rem;
    height: 80vh;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.875rem;
}

.log-entry {
    padding: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.log-internal {
    color: var(--primary-light);
    font-style: italic;
}

.log-error {
    color: var(--danger);
}

/* File Upload */
.file-upload-zone {
    padding: 1rem;
    border: 2px dashed var(--border);
    border-radius: 4px;
    text-align: center;
    cursor: pointer;
    margin-bottom: 1rem;
}

.file-upload-zone.drag-over {
    border-color: var(--primary);
    background: var(--bg-dark);
}

.file-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem;
    background: var(--bg-dark);
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

/* Branch List */
.branch-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    background: var(--bg-dark);
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

.branch-item.active {
    border: 1px solid var(--primary);
}



@keyframes pulse {
    0% { box-shadow: 0 0 20px var(--primary-glow); }
    50% { box-shadow: 0 0 40px var(--primary-glow), 0 0 60px var(--accent-glow); }
    100% { box-shadow: 0 0 20px var(--primary-glow); }
}


/* Agent Select Styling */
#agentSelect {
    padding: 0.75rem 2rem 0.75rem 1rem;
    font-size: 0.875rem;
    color: var(--text);
    background-color: var(--bg-dark);
    border: 1px solid var(--border);
    border-radius: 4px;
    appearance: none;
    cursor: pointer;
    transition: var(--transition);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'%3E%3C/path%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.75rem center;
    background-size: 16px;
}

#agentSelect:hover {
    border-color: var(--primary);
}

#agentSelect:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 2px var(--primary-light);
}

#agentSelect option {
    background-color: var(--panel-bg);
    color: var(--text);
    padding: 0.5rem;
}
</style>
<body>
    <div class="main-layout both-hidden">
        <div class="sidebar">
            <h3>Branches</h3>

            <div class="file-upload-zone" id="fileUploadZone">
                <input type="file" id="fileInput" multiple>
                <p>Drag & drop files here or click to select</p>
            </div>
            <div class="file-list" id="fileList"></div>
            <div class="section-header">
                Branches
                <button id="new-branch-button">New Branch</button>
            </div>
            <div id="branchList" class="branch-list"></div>
            <div id="verboseOutput" class="verbose-output"></div>
          </div>
           <button class="panel-toggle left-panel-toggle" onclick="toggleLeftPanel()">◀</button>
        <div class="main-container">
            <div class="chat-header">
            <label for="agentSelect">Select Agent or System</label>
                <select id="agentSelect">
                    <option value="">ISAA system</option>
                    $agent_options
                </select>
            </div>
            <div id="chat" class="chat-container">
                <div class="messages-container"></div>
            </div>

            <div class="chat-input">
                    <input type="text" id="messageInput" placeholder="Type your message...">
                    <button id="sendButton">Send</button>
                </div>
        </div>
        <div class="right-sidebar">
            <div class="section-header">Agent Status</div>
            <div class="agent-status">
                <div class="status-indicator"></div>
                <span id="agentStatus">Idle</span>
            </div>

            <div id="logPanel" class="log-panel">
            <div class="section-header">Internal</div>
            </div>
        </div>
        <button class="panel-toggle right-panel-toggle" onclick="toggleRightPanel()">▶</button>
    </div>

    <script>

    function toggleLeftPanel() {
    const layout = document.querySelector('.main-layout');
    if (layout.classList.contains('right-hidden')) {
        layout.classList.replace('right-hidden', 'both-hidden');
    } else if (layout.classList.contains('both-hidden')) {
        layout.classList.remove('both-hidden');
        layout.classList.add('right-hidden');
    } else if (layout.classList.contains('left-hidden')) {
        layout.classList.remove('left-hidden');
    } else {
        layout.classList.add('left-hidden');
    }
}

function toggleRightPanel() {
    const layout = document.querySelector('.main-layout');
    if (layout.classList.contains('left-hidden')) {
        layout.classList.replace('left-hidden', 'both-hidden');
    } else if (layout.classList.contains('both-hidden')) {
        layout.classList.remove('both-hidden');
        layout.classList.add('left-hidden');
    } else if (layout.classList.contains('right-hidden')) {
        layout.classList.remove('right-hidden');
    } else {
        layout.classList.add('right-hidden');
    }
}
class StatusManager {
    constructor() {
        this.logPanel = document.getElementById('logPanel');
        this.statusIndicator = document.querySelector('.status-indicator');
    }

    updateStatus(state) {
        this.statusIndicator.className = 'status-indicator';
        this.statusIndicator.classList.add(`status-${state}`);
        this.addLog(`Status changed to: ${state}`);
    }

    addLog(message, type = 'info') {
        const entry = document.createElement('div');
        const time_ = document.createElement('div');
        entry.className = `log-entry log-${type}`;
        time_.textContent = `[${new Date().toLocaleTimeString()}]`;
        entry.innerHTML = marked.parse(message)
        entry.appendChild(time_);
        this.logPanel.appendChild(entry);
        this.logPanel.scrollTop = this.logPanel.scrollHeight;
    }
}

    class EnhancedChat {
    constructor() {
        this.ws = null;
        this.particles = [];

        this.setupWebSocket();
        this.setupUI();

        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.setupFileUpload();
        this.files = new Map();

        this.currentBranch = null;
    }

    setupFileUpload() {
        const zone = document.getElementById('fileUploadZone');
        const input = document.getElementById('fileInput');
        const fileList = document.getElementById('fileList');

        zone.addEventListener('click', () => input.click());

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });

        input.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
    }

    handleFiles(fileList) {
        Array.from(fileList).forEach(file => {
            const fileId = `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            this.files.set(fileId, file);

            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span>${file.name}</span>
                <div>
                    <button onclick="chat.uploadFile('${fileId}')">Upload</button>
                    <button onclick="chat.removeFile('${fileId}')">Remove</button>
                </div>
                <div class="file-progress"></div>
            `;

            document.getElementById('fileList').appendChild(fileItem);
        });
    }

    async uploadFile(fileId) {
        const file = this.files.get(fileId);
        if (!file) return;

        const fileItem = document.querySelector(`[data-file-id="${fileId}"]`);
        const progress = fileItem.querySelector('.file-progress');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();
            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    progress.style.width = percentComplete + '%';
                }
            };

            const response = await fetch('/api/isaa/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                this.addMessage('system', `File uploaded: ${file.name}`);
                this.removeFile(fileId);
            } else {
                throw new Error('Upload failed');
            }
        } catch (error) {
            this.addMessage('error', `Failed to upload ${file.name}: ${error.message}`);
            progress.style.backgroundColor = '#ef4444';
        }
    }

    removeFile(fileId) {
        this.files.delete(fileId);
        const fileItem = document.querySelector(`[data-file-id="${fileId}"]`);
        if (fileItem) {
            fileItem.remove();
        }
    }


    setupWebSocket() {
        if (this.ws) {
            this.ws.close();
        }

        this.ws = new WebSocket(`wss://${window.location.host}/api/isaa/chat_websocket`);
        this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));
        this.ws.onclose = () => this.handleDisconnect();
        this.ws.onerror = (error) => this.handleError(error);

        this.ws.onopen = () => {
            this.reconnectAttempts = 0;
        };
    }

    handleDisconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => this.setupWebSocket(), 5000);
        } else {
            alert('Failed to reconnect. Please refresh the page.');
        }
    }

    handleError(error) {
        console.error('WebSocket error:', error);
    }


    setupUI() {
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        document.getElementById('sendButton').onclick = () => this.sendMessage();
        document.getElementById('new-branch-button').onclick = () => this.createNewBranch();
    }

    handleMessage(data) {
        const statusManager = new StatusManager();

        console.log(data, data.type);
        switch (data.type) {
            case 'stream':
                if (data.agent.endsWith('-internal')) {
                    statusManager.addLog(data.agent+'\\n'+data.content, 'internal');
                } else {
                    this.addMessage('agent', data.content, data.agent);
                    statusManager.updateStatus('responding');
                }
                break;
            case 'error':
                this.addMessage('error', data.content);
                statusManager.updateStatus('error');
                statusManager.addLog(data.content, 'error');
                break;
            case 'agent_state':
                this.updateAgentState(data.state);
                break;
            case 'branches':
                this.updateBranchList(data.branches);
                break;
            case 'ping':
                this.ws.send(JSON.stringify({ type: 'pong' }));
                break;
        }
    }

    updateAgentState(state) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.getElementById('agentStatus');
        const verboseOutput = document.getElementById('verboseOutput');

        if (state.is_running) {
            statusIndicator.classList.add('running');
            statusIndicator.classList.remove('idle');
            statusText.textContent = 'Running';
        } else {
            statusIndicator.classList.add('idle');
            statusIndicator.classList.remove('running');
            statusText.textContent = 'Idle';
        }

        if (state.verbose_output) {
            verboseOutput.textContent = state.verbose_output.join('\\n');
        }

    }

    updateBranchList(branches) {
        const branchList = document.getElementById('branchList');
        branchList.innerHTML = '';

        branches.forEach(branch => {
            const branchDiv = document.createElement('div');
            branchDiv.className = `branch-item ${branch.id === this.currentBranch ? 'active' : ''}`;

            branchDiv.innerHTML = `
                <span>${branch.id}</span>
                <div class="branch-actions">
                    <button onclick="chat.switchBranch('${branch.id}')">Switch</button>
                    <button onclick="chat.restoreChat('${branch.id}')">Restore</button>
                </div>
            `;

            branchList.appendChild(branchDiv);
        });

        document.getElementById('currentBranch').textContent = `Current Branch: ${this.currentBranch || 'None'}`;
    }

    switchBranch(branchId) {
        this.currentBranch = branchId;
        this.ws.send(JSON.stringify({
            type: 'switch_branch',
            branch_id: branchId
        }));
        this.updateBranchList([{ id: branchId }]);
    }

    restoreChat(branchId) {
        this.ws.send(JSON.stringify({
            type: 'restore_chat',
            branch_id: branchId
        }));
    }

    createNewBranch() {
        const branchId = `branch-${Date.now()}`;
        this.ws.send(JSON.stringify({
            type: 'create_branch',
            branch_id: branchId
        }));
    }

    addToLogs(message) {
        const logPanel = document.getElementById('logPanel');
        const logEntry = document.createElement('div');
        logEntry.textContent = `${new Date().toISOString()} - ${message}`;
        logPanel.appendChild(logEntry);
        logPanel.scrollTop = logPanel.scrollHeight;
    }

    sendMessage() {
        const input = document.getElementById('messageInput');
        const agentSelect = document.getElementById('agentSelect');

        if (!input.value) return;

        this.ws.send(JSON.stringify({
            type: 'message',
            content: input.value,
            agent: agentSelect.value
        }));

        this.addMessage('user', input.value);
        input.value = '';
    }

    addMessage(role, content, agent = null) {
        if (agent && agent.endsWith('-internal')) {
            return;
        }
        const chat = document.getElementById('chat');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'A';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (agent) {
            contentDiv.innerHTML = `<strong>${agent}:</strong> ${marked.parse(content)}`;
        } else {
            contentDiv.innerHTML =  marked.parse(content);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        chat.appendChild(messageDiv);
        chat.scrollTop = chat.scrollHeight;
    }
}

window.onload = () => new EnhancedChat();
    </script>
</body>
</html>
        """

        # Generate agent options HTML
        agent_options = ''.join([
            f'<option value="{agent}">{agent}</option>'
            for agent in self.isaa.config.get('agents-name-list', [])
        ])

        return template.replace("$agent_options", agent_options)


export = get_app('nice.export').tb

from fastapi import Request


# Register the WebSocket endpoint
@export(mod_name="isaa", request_as_kwarg=True, level=1, api=True,
        name="handle_websocket_audio", row=True)
async def chat_websocket(websocket: WebSocket, spec: str = "main"):
    IsaaWebSocketUI(get_app('chat.websocket').get_mod("isaa", spec=spec))
    # await chat_widget.handle_websocket_audio(websocket)


# Register the WebSocket endpoint
@export(mod_name="isaa", request_as_kwarg=True, level=1, api=True,
        name="chat_websocket", row=True)
async def chat_websocket(websocket: WebSocket, spec: str = "main"):
    chat_widget = IsaaWebSocketUI(get_app('chat.websocket').get_mod("isaa", spec=spec))
    await chat_widget.handle_websocket_chat(websocket)


@export(mod_name="isaa", request_as_kwarg=True, level=1, api=True,
        name="main_web_isaa_entry", row=True)
async def main_web_isaa_entry(request: Request or None = None):
    if request is None:
        return
    spec = get_spec(request).get(default="main-demo")
    chat_widget = IsaaWebSocketUI(get_app('chat.main_web_isaa_entry').get_mod("isaa", spec=spec))
    content = chat_widget.get_widget()
    content = content.replace("/handle_websocket_audio", "/handle_websocket_audio?spec=" + spec)
    content = content.replace("/chat_websocket", "/chat_websocket?spec=" + spec)
    return HTMLResponse(content=content)


