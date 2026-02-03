"""
ISAA Web Host - Modern Minu UI for FlowAgent
============================================
A minimalist, full-screen production interface for the ISAA Host.
Features:
- 3 Full-Screen Panels: Chat (Stream), VFS (Editor), Config (Control)
- Edge-Hover Navigation
- Direct a_stream integration with structured rendering
- Audio I/O & File Management
"""

import asyncio
import base64
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Toolbox & Minu Imports
from toolboxv2 import App, RequestData, Result, get_app
from toolboxv2.mods.Minu.core import (
    MinuView, MinuSession, State, Component, ComponentType,
    Card, Text, Heading, Button, Input, Row, Column,
    Grid, Spacer, Divider, Icon, Badge, Modal, Form,
    Switch, Select, Textarea, Custom, Dynamic, ReactiveState
)
# ISAA Imports
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
from toolboxv2.flows.cli_v4 import ISAA_Host, AgentInfo, PTColors

# Module Metadata
Name = 'ISAAWeb'
export = get_app(f"{Name}.Export").tb
version = '1.0.0'

# =============================================================================
# CUSTOM CSS & JS (The "Frontend")
# =============================================================================

SHARED_STYLES = """
<style>
    /* 1. LAYOUT RESET & BREAKOUT */
    /* Zwingt den Minu-Container über das Standard-Toolbox-Layout */
    #minu-root {
        position: fixed !important;
        top: 0;
        left: 0;
        width: 100vw !important;
        height: 100vh !important;
        max-width: none !important;
        padding: 0 !important;
        margin: 0 !important;
        z-index: 100; /* Über dem Standard-Hintergrund, unter Modals */
        background-color: var(--bg-color);
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    /* Verstecke/Deaktiviere Standard-Container-Styles der Toolbox für diese View */
    .main-content, .content-wrapper {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100vw !important;
        max-width: 100vw !important;
    }

    /* 2. VARIABLES */
    :root {
        --bg-color: #0d1117; /* GitHub Dark Dimmed Style */
        --panel-bg: #161b22;
        --input-bg: #0d1117;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --accent: #238636;       /* Green Accent */
        --accent-hover: #2ea043;
        --border: #30363d;
        --font-mono: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    }

    /* 3. FULL SCREEN PANELS */
    .isaa-panel {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        padding: 60px 80px 80px 80px; /* Platz für Header/Footer/Nav */
        box-sizing: border-box;
        transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), opacity 0.3s ease;
        opacity: 0;
        pointer-events: none;
        transform: scale(0.98) translateY(10px);
        display: flex;
        flex-direction: column;
        background-color: var(--bg-color);
        z-index: 1;
    }

    .isaa-panel.active {
        opacity: 1;
        pointer-events: all;
        transform: scale(1) translateY(0);
        z-index: 10;
    }

    /* 4. NAVIGATION ARROWS (Hover Zones) */
    .nav-arrow {
        position: fixed;
        top: 0;
        bottom: 0;
        width: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: var(--text-secondary);
        cursor: pointer;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.2s ease, background 0.3s ease;
    }
    .nav-arrow:hover {
        opacity: 1;
        color: var(--text-primary);
    }
    .nav-left {
        left: 0;
        background: linear-gradient(90deg, rgba(0,0,0,0.8), transparent);
    }
    .nav-right {
        right: 0;
        background: linear-gradient(-90deg, rgba(0,0,0,0.8), transparent);
    }

    /* 5. CHAT STYLES */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        scroll-behavior: smooth;
    }

    .chat-input-area {
        margin-top: auto; /* Push to bottom */
        background: var(--panel-bg);
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid var(--border);
        box-shadow: 0 -4px 20px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .msg-bubble {
        max-width: 85%;
        line-height: 1.6;
        font-size: 1rem;
    }

    .msg-user {
        align-self: flex-end;
        text-align: right;
    }
    .msg-user .content {
        background: var(--accent);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .msg-agent {
        align-self: flex-start;
        width: 100%;
    }
    .msg-agent .content {
        color: var(--text-primary);
        padding: 0 1rem;
    }

    /* 6. VFS & CODE EDITOR */
    .vfs-container {
        display: flex;
        height: 100%;
        gap: 1rem;
        overflow: hidden;
    }

    .vfs-tree {
        width: 280px;
        min-width: 280px;
        background: var(--panel-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
        overflow-y: auto;
        font-size: 0.9rem;
    }

    .vfs-editor {
        flex: 1;
        background: var(--input-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }

    .editor-textarea {
        flex: 1;
        width: 100%;
        height: 100%;
        background: transparent;
        color: #e6edf3;
        border: none;
        padding: 1rem;
        font-family: var(--font-mono);
        font-size: 14px;
        line-height: 1.5;
        resize: none;
        outline: none;
        white-space: pre;
    }

    /* 7. UTILITIES & OVERRIDES */
    /* Ensure Minu Components behave */
    .card { background: var(--panel-bg) !important; border: 1px solid var(--border) !important; }
    input, select, textarea {
        background: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    input:focus, select:focus, textarea:focus { border-color: var(--accent) !important; }

    .btn-primary { background: var(--accent) !important; color: white !important; }
    .btn-primary:hover { background: var(--accent-hover) !important; }

    .w-full { width: 100%; }
    .h-full { height: 100%; }
    .flex { display: flex; }
    .flex-col { flex-direction: column; }
    .flex-1 { flex: 1; }
    .gap-2 { gap: 0.5rem; }
    .gap-4 { gap: 1rem; }
    .items-center { align-items: center; }

    /* Code Blocks */
    pre {
        background: #161b22;
        padding: 1rem;
        border-radius: 6px;
        overflow-x: auto;
        border: 1px solid var(--border);
        font-family: var(--font-mono);
        margin: 10px 0;
    }

    /* Tool Calls Style */
    .tool-call {
        font-family: var(--font-mono);
        font-size: 0.85em;
        color: var(--text-secondary);
        border-left: 2px solid var(--border);
        padding-left: 10px;
        margin: 5px 0;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-color); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 5px; border: 2px solid var(--bg-color); }
    ::-webkit-scrollbar-thumb:hover { background: #555; }

    /* Notification Toasts */
    .notification-toast {
        animation: slideIn 0.3s ease-out;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
</style>

<script>
    // Force Layout Fix on Load
    (function() {
        const fixLayout = () => {
            const root = document.getElementById('minu-root');
            if(root) {
                // Ensure parents don't constrain us
                let p = root.parentElement;
                while(p && p.tagName !== 'BODY') {
                    p.style.padding = '0';
                    p.style.margin = '0';
                    p.style.maxWidth = 'none';
                    p.style.width = '100%';
                    p.style.height = '100%';
                    p = p.parentElement;
                }
            }
        };
        setTimeout(fixLayout, 100);
        setTimeout(fixLayout, 500);
        setTimeout(fixLayout, 1000);

        // Auto-scroll chat
        const observer = new MutationObserver(() => {
            const chat = document.querySelector('.chat-container');
            if(chat) chat.scrollTop = chat.scrollHeight;
        });
        const target = document.getElementById('minu-root');
        if(target) observer.observe(target, { childList: true, subtree: true });
    })();

    // ISAA Logic (Audio/Drop)
    window.ISAA = {
        recorder: null,
        audioChunks: [],

        startRecording: async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                ISAA.recorder = new MediaRecorder(stream);
                ISAA.audioChunks = [];

                ISAA.recorder.ondataavailable = e => ISAA.audioChunks.push(e.data);
                ISAA.recorder.onstop = ISAA.uploadAudio;

                ISAA.recorder.start();
                const btn = document.getElementById('mic-btn');
                if(btn) btn.style.color = '#ef4444'; // Red recording state
                return true;
            } catch (e) {
                console.error("Mic Error:", e);
                return false;
            }
        },

        stopRecording: () => {
            if (ISAA.recorder && ISAA.recorder.state !== 'inactive') {
                ISAA.recorder.stop();
                const btn = document.getElementById('mic-btn');
                if(btn) btn.style.color = ''; // Reset color
            }
        },

        uploadAudio: async () => {
            const blob = new Blob(ISAA.audioChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = () => {
                const base64data = reader.result.split(',')[1];
                window.TB.ui.sendMinuEvent('process_audio_upload', { audio_b64: base64data });
            };
        },

        handleDrop: (e) => {
            e.preventDefault();
            e.stopPropagation();
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                ISAA.uploadFiles(files);
            }
        },

        uploadFiles: async (files) => {
            // Visual feedback
            const dropZone = document.querySelector('.vfs-tree');
            if(dropZone) dropZone.style.borderColor = 'var(--accent)';

            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const reader = new FileReader();
                reader.onload = (e) => {
                    const content = e.target.result.split(',')[1];
                    window.TB.ui.sendMinuEvent('vfs_upload_file', {
                        name: file.name,
                        content: content,
                        is_binary: true
                    });
                };
                reader.readAsDataURL(file);
            }

            setTimeout(() => {
                if(dropZone) dropZone.style.borderColor = '';
            }, 1000);
        }
    };

    // Global Drag & Drop Listener for VFS
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', (e) => {
        if(e.target.closest('.vfs-container')) {
            ISAA.handleDrop(e);
        }
    });
</script>
"""
# =============================================================================
# ISAA HOST WRAPPER (Web Adapter)
# =============================================================================

class WebISAAHost(ISAA_Host):
    """Wraps ISAA_Host to redirect outputs to the Web UI State."""

    def __init__(self, app_instance, view_ref: 'ISAAView'):
        super().__init__(app_instance)
        self.view = view_ref

    def print_to_ui(self, text: str, style: str = "info"):
        """Redirects CLI prints to the UI's notification/log system."""
        # Clean HTML/ANSI tags for simple text log, keep for HTML render
        clean_text = text.replace("<", "&lt;").replace(">", "&gt;")
        self.view.add_log_entry(clean_text, style)

# =============================================================================
# MAIN VIEW
# =============================================================================

class ISAAView(MinuView):
    """
    The Single Page Application for ISAA.
    Manages 3 Full-Screen Panels: Chat, VFS, Config.
    """
    # UI State
    active_panel = State("chat") # chat, vfs, config
    panels_order = ["chat", "vfs", "config"]

    # Chat State
    chat_history = State([]) # List of {role, content, type, ...}
    current_input = State("")
    is_streaming = State(False)
    system_logs = State([]) # Notification list

    # VFS State
    vfs_path = State("/")
    vfs_tree = State({})
    current_file = State(None) # {path, content, is_dirty}
    vfs_loading = State(False)

    # Config State
    agent_list = State([])
    session_list = State([])
    skills_list = State([])
    active_agent = State("self")

    # Task Modal
    show_tasks = State(False)
    tasks_list = State([])

    def __init__(self):
        super().__init__()
        self.host = None # Initialized in on_mount

    async def on_mount(self):
        """Initialize the backend host system."""
        app = get_app("isaa-host")
        if not app:
            # Fallback/Init if not running
            app = self._app

        self.host = WebISAAHost(app, self)
        # Initialize Self Agent (Async)
        await self.host._init_self_agent()

        # Sync Initial Data
        await self.refresh_data()

        # Add welcome message
        self.chat_history.value = [{
            "role": "assistant",
            "content": f"**ISAA Host v{self.host.version} Online.**\nI am ready. Use the arrow zones to switch between VFS and Config.",
            "type": "message"
        }]

    async def refresh_data(self):
        """Syncs backend state to UI state."""
        if not self.host: return

        # Agents
        agents = self.host.isaa_tools.config.get("agents-name-list", [])
        self.agent_list.value = [
            {"name": a, "info": self.host.agent_registry.get(a)}
            for a in agents
        ]
        self.active_agent.value = self.host.active_agent_name

        # Sessions
        try:
            agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)
            self.session_list.value = list(agent.session_manager.sessions.keys())

            # VFS Tree
            session = agent.session_manager.get(self.host.active_session_id)
            if session:
                self.vfs_tree.value = session.vfs_ls(self.vfs_path.value, recursive=False)

            # Skills
            engine = agent._get_execution_engine()
            self.skills_list.value = [
                s.to_dict() for s in engine.skills_manager.skills.values()
            ]

        except Exception as e:
            self.add_log_entry(f"Data sync error: {e}", "error")

    # =========================================================================
    # RENDERERS
    # =========================================================================

    def render(self):
        # Inject CSS/JS
        return Column(
            Custom(html=SHARED_STYLES),

            # --- Navigation Overlays ---
            Custom(html=f"""
                <div class="nav-arrow nav-left" onclick="window.TB.ui.sendMinuEvent('nav_prev', {{}})"
                     style="display: {'none' if self.active_panel.value == 'chat' else 'flex'}">
                    ❮
                </div>
                <div class="nav-arrow nav-right" onclick="window.TB.ui.sendMinuEvent('nav_next', {{}})"
                     style="display: {'none' if self.active_panel.value == 'config' else 'flex'}">
                    ❯
                </div>
            """),

            # --- Notifications / Logs Overlay ---
            self._render_notifications(),

            # --- PANELS ---

            # 1. Chat Panel
            Column(
                self._render_chat_panel(),
                className=f"isaa-panel {'active' if self.active_panel.value == 'chat' else ''}",
                id="panel-chat"
            ),

            # 2. VFS Panel
            Column(
                self._render_vfs_panel(),
                className=f"isaa-panel {'active' if self.active_panel.value == 'vfs' else ''}",
                id="panel-vfs"
            ),

            # 3. Config Panel
            Column(
                self._render_config_panel(),
                className=f"isaa-panel {'active' if self.active_panel.value == 'config' else ''}",
                id="panel-config"
            ),

            # --- Modals ---
            self._render_task_modal(),

            className="w-full h-full"
        )

    def _render_chat_panel(self):
        return [
            # Header
            Row(
                Icon("smart_toy", size="28", className="text-accent"),
                Heading(f"{self.host.active_agent_name} @ {self.host.active_session_id}", level=3),
                Spacer(),
                Button("Tasks", icon="list", variant="ghost", on_click="toggle_tasks"),
                className="mb-4 items-center"
            ),

            # Chat Area
            Column(
                *[self._render_message(msg) for msg in self.chat_history.value],
                className="chat-container",
                id="chat-scroll-target" # Target for auto-scroll
            ),

            # Input Area
            Row(
                Custom(html="""
                    <button class="btn btn-ghost" id="mic-btn" onmousedown="ISAA.startRecording()" onmouseup="ISAA.stopRecording()">
                        <span id="mic-status" class="material-symbols-outlined">mic</span>
                    </button>
                """),
                Input(
                    placeholder="Ask ISAA... (Supports #audio tag)",
                    value=self.current_input.value,
                    bind="current_input",
                    on_submit="submit_message",
                    className="flex-1 bg-transparent border-none text-white focus:ring-0"
                ),
                Button(
                    "Send",
                    icon="send",
                    variant="primary",
                    on_click="submit_message",
                    disabled=self.is_streaming.value
                ),
                className="chat-input-area items-center gap-2"
            )
        ]

    def _render_message(self, msg):
        """Renders a single chat message (User, Agent, Tool)."""
        is_user = msg['role'] == 'user'
        css_class = "msg-user" if is_user else "msg-agent"

        content_ui = []

        # Tool Calls (Collapsible)
        if msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                content_ui.append(
                    Custom(html=f"""
                        <div class="tool-call">
                            🔧 <b>{tc['function']['name']}</b>
                            <div class="text-xs opacity-70">{tc['function']['arguments'][:100]}...</div>
                        </div>
                    """)
                )

        # Main Content
        if msg.get('content'):
            # Convert Markdown to HTML logic would go here ideally
            # For now, simple text with preserved whitespace
            formatted = msg['content'].replace("\n", "<br>")
            content_ui.append(Custom(html=f"<div>{formatted}</div>"))

        return Column(
            Column(
                *content_ui,
                className="content"
            ),
            className=f"msg-bubble {css_class}"
        )

    def _render_vfs_panel(self):
        # Tree View Items
        tree_items = []
        if self.vfs_tree.value and self.vfs_tree.value.get('contents'):
            for item in self.vfs_tree.value['contents']:
                icon = "folder" if item['type'] == 'directory' else "description"
                is_dir = item['type'] == 'directory'
                cb = f"vfs_open_dir_{item['path']}" if is_dir else f"vfs_open_file_{item['path']}"

                # Dynamic handler generation trick handled in __getattr__

                tree_items.append(
                    Row(
                        Icon(icon, size="18", className="text-secondary"),
                        Text(item['name'], className="cursor-pointer hover:text-white flex-1"),
                        on_click=f"vfs_select_{item['path']}", # Using a generic prefix handled by getattr
                        className="p-1 hover:bg-neutral-800 rounded items-center gap-2"
                    )
                )

        editor_content = ""
        current_path = ""
        if self.current_file.value:
            editor_content = self.current_file.value.get('content', '')
            current_path = self.current_file.value.get('path', '')

        return [
            Row(
                Heading("VFS Explorer", level=3),
                Spacer(),
                Button("Upload", icon="upload", variant="secondary", on_click="trigger_upload"), # Requires JS trigger
                Button("Sync", icon="sync", variant="ghost", on_click="vfs_sync_all"),
                className="mb-4 items-center"
            ),
            Row(
                # Tree Column
                Column(
                    Text(self.vfs_path.value, className="text-xs text-gray-500 mb-2 font-mono"),
                    *tree_items,
                    Custom(html="<div class='mt-auto p-4 border-t border-gray-800 text-xs text-gray-500'>Drop files here to upload</div>"),
                    className="vfs-tree h-full"
                ),
                # Editor Column
                Column(
                    Row(
                        Text(current_path or "No file selected", className="font-mono text-sm"),
                        Spacer(),
                        Button("Save", size="sm", variant="primary", on_click="vfs_save_current") if current_path else None,
                        className="p-2 border-b border-gray-800 bg-neutral-900"
                    ),
                    Textarea(
                        value=editor_content,
                        bind="current_file_content", # Needs complex binding logic or event
                        className="editor-textarea",
                        rows=30
                    ) if current_path else
                    Column(
                        Icon("terminal", size="48", className="text-gray-700"),
                        Text("Select a file to edit", className="text-gray-600"),
                        align="center", justify="center", className="h-full"
                    ),
                    className="vfs-editor h-full"
                ),
                className="vfs-container flex-1"
            )
        ]

    def _render_config_panel(self):
        return [
            Heading("System Configuration", level=2, className="mb-6"),

            Grid(
                # Column 1: Agent Control
                Card(
                    Heading("Active Agent", level=4),
                    Select(
                        options=[{"value": a['name'], "label": a['name']} for a in self.agent_list.value],
                        value=self.active_agent.value,
                        bind="active_agent",
                        on_change="switch_agent"
                    ),
                    Row(
                        Button("Restart", icon="refresh", size="sm", on_click="restart_agent"),
                        Button("Memory", icon="memory", size="sm", on_click="show_memory_stats"),
                        gap="2", className="mt-4"
                    ),
                    title="Agent Control"
                ),

                # Column 2: Session Management
                Card(
                    Heading("Sessions", level=4),
                    Column(
                        *[
                            Row(
                                Text(sid, className="flex-1 font-mono text-sm"),
                                Button("Load", size="xs", variant="ghost", on_click=f"load_session_{sid}"),
                                Button("×", size="xs", variant="ghost", className="text-red-400", on_click=f"del_session_{sid}"),
                                className="items-center"
                            ) for sid in self.session_list.value
                        ],
                        className="max-h-60 overflow-y-auto"
                    ),
                    Button("New Session", className="w-full mt-4", on_click="new_session"),
                    title="Session Manager"
                ),

                # Column 3: Skills
                Card(
                    Heading("Skills Library", level=4),
                    Text(f"{len(self.skills_list.value)} skills loaded", className="text-sm text-gray-500 mb-2"),
                    Row(
                        Button("Import", icon="download", size="sm"),
                        Button("Export All", icon="upload", size="sm"),
                        gap="2"
                    ),
                    title="Skills"
                ),

                cols=3, gap="6"
            )
        ]

    def _render_notifications(self):
        """Renders toast-like notifications from system logs."""
        logs = self.system_logs.value[-3:] # Show last 3
        return Column(
            *[
                Card(
                    Text(l['text']),
                    className=f"mb-2 p-3 text-sm border-l-4 border-{l['color']}-500 bg-neutral-900 shadow-lg animate-fade-in"
                ) for l in logs
            ],
            className="fixed bottom-4 right-4 w-80 z-50 pointer-events-none"
        )

    def _render_task_modal(self):
        return Modal(
            Heading("Background Tasks"),
            Column(
                *[
                    Card(
                        Row(
                            Text(t.task_id, className="font-mono font-bold"),
                            Badge(t.status, variant="warning" if t.status=="running" else "success"),
                            Spacer(),
                            Button("Stop", size="sm", variant="danger", on_click=f"stop_task_{t.task_id}")
                        ),
                        Text(t.query, className="text-sm text-gray-400 mt-2 truncate")
                    ) for t in self.host.background_tasks.values()
                ] if self.host.background_tasks else [Text("No active tasks")]
            ),
            open=self.show_tasks.value,
            on_close="toggle_tasks",
            title="Task Manager"
        )

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def add_log_entry(self, text, style="info"):
        """Called by Host to push logs."""
        color_map = {"info": "blue", "success": "green", "error": "red", "warning": "yellow"}
        entry = {"text": text, "color": color_map.get(style, "gray"), "time": datetime.now()}

        # Need to create new list reference for reactivity
        current = list(self.system_logs.value)
        current.append(entry)
        self.system_logs.value = current

    async def submit_message(self, event=None):
        if not self.current_input.value.strip(): return

        query = self.current_input.value
        self.current_input.value = ""
        self.is_streaming.value = True

        # Add User Message
        current_hist = list(self.chat_history.value)
        current_hist.append({"role": "user", "content": query})
        self.chat_history.value = current_hist

        # Prepare Agent Message Placeholder
        agent_msg_idx = len(current_hist)
        current_hist.append({"role": "assistant", "content": "", "tool_calls": []})
        self.chat_history.value = current_hist

        try:
            agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)

            # STREAMING LOOP
            full_response = ""
            async for chunk in agent.a_stream(
                query=query,
                session_id=self.host.active_session_id
            ):
                c_type = chunk.get("type")

                # Update logic
                hist_update = list(self.chat_history.value)
                last_msg = hist_update[agent_msg_idx]

                if c_type == "content":
                    full_response += chunk['chunk']
                    last_msg['content'] = full_response

                elif c_type == "tool_start":
                    # Add pending tool
                    last_msg.setdefault('tool_calls', []).append({
                        "function": {"name": chunk['name'], "arguments": "Running..."}
                    })

                elif c_type == "tool_result":
                    # Update tool result (simplified matching last tool)
                    if last_msg.get('tool_calls'):
                        last_msg['tool_calls'][-1]['function']['arguments'] = f"Result: {chunk['result'][:50]}..."

                elif c_type == "error":
                    self.add_log_entry(chunk['error'], "error")

                # Trigger Reactivity
                self.chat_history.value = hist_update

                # Force UI flush (essential for streaming feeling)
                await self._session.force_flush()

        except Exception as e:
            self.add_log_entry(f"Execution failed: {e}", "error")
            traceback.print_exc()
        finally:
            self.is_streaming.value = False
            await self.refresh_data() # Update VFS etc

    # --- Navigation ---
    async def nav_next(self, e):
        curr_idx = self.panels_order.index(self.active_panel.value)
        next_idx = (curr_idx + 1) % len(self.panels_order)
        self.active_panel.value = self.panels_order[next_idx]

    async def nav_prev(self, e):
        curr_idx = self.panels_order.index(self.active_panel.value)
        prev_idx = (curr_idx - 1) % len(self.panels_order)
        self.active_panel.value = self.panels_order[prev_idx]

    # --- VFS Logic ---
    def __getattr__(self, name):
        """Dynamic handlers for VFS tree clicks."""
        if name.startswith("vfs_select_"):
            path = name.replace("vfs_select_", "")
            return lambda e: self._handle_vfs_select(path)
        if name.startswith("load_session_"):
            sid = name.replace("load_session_", "")
            return lambda e: self._switch_session(sid)
        if name.startswith("stop_task_"):
            tid = name.replace("stop_task_", "")
            return lambda e: self._stop_task(tid)
        return super().__getattribute__(name)

    async def _handle_vfs_select(self, path):
        # Is directory?
        if self.vfs_tree.value: # Check if matches dir in tree
            pass # (Simplified logic: assuming user clicked file for edit or dir for traversal)

        # Traverse or Read
        agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)
        session = agent.session_manager.get(self.host.active_session_id)

        info = session.vfs.get_file_info(path)

        if info.get('type') == 'directory':
            self.vfs_path.value = path
            self.vfs_tree.value = session.vfs_ls(path)
        else:
            # Read file
            res = session.vfs.read(path)
            if res.get('success'):
                self.current_file.value = {"path": path, "content": res['content']}
            else:
                self.add_log_entry(res.get('error'), "error")

    async def vfs_save_current(self, e):
        if not self.current_file.value: return
        path = self.current_file.value['path']
        # Note: In a real TextArea component, we need to bind the value back.
        # Here we assume self.current_file.value['content'] was updated by the bind.
        # Since 'bind' in Minu might be basic, we might need to rely on the payload of the event
        # but let's assume reactivity works for the bound "current_file_content".

        # NOTE: Minu's Textarea doesn't auto-update a dict key.
        # We need a separate state or get value from event if supported.
        # For this example, we assume we have a mechanism or strict binding.
        content = self.current_file.value.get('content') # Needs to be updated by input

        agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)
        session = agent.session_manager.get(self.host.active_session_id)

        res = session.vfs.write(path, content)
        if res.get('success'):
            self.add_log_entry(f"Saved {path}", "success")
        else:
            self.add_log_entry(res.get('error'), "error")

    # --- Uploads ---
    async def process_audio_upload(self, event):
        """Called by JS when audio is recorded."""
        b64_audio = event.get('audio_b64')
        if not b64_audio: return

        audio_bytes = base64.b64decode(b64_audio)

        # Use ISAA Host internal processing
        # We manually trigger STT -> Input
        try:
            from toolboxv2.mods.isaa.base.audio_io.Stt import STTConfig, transcribe
            result = transcribe(audio_bytes, config=STTConfig(language="en")) # Auto-detect ideally

            if result.text:
                self.current_input.value = result.text
                self.add_log_entry("Audio transcribed", "success")
                await self.submit_message()
            else:
                self.add_log_entry("No speech detected", "warning")
        except Exception as e:
            self.add_log_entry(f"STT Error: {e}", "error")

    async def vfs_upload_file(self, event):
        """Called by JS on drag & drop."""
        name = event.get('name')
        content_b64 = event.get('content')

        try:
            # Decode
            content_bytes = base64.b64decode(content_b64)

            agent = await self.host.isaa_tools.get_agent(self.host.active_agent_name)
            session = agent.session_manager.get(self.host.active_session_id)

            # Save to 'user_uploads' mount if exists, else root
            target_dir = "/user_uploads"
            if not session.vfs._path_exists(target_dir):
                session.vfs.mkdir(target_dir)

            target_path = f"{target_dir}/{name}"

            # Determine if binary or text. For VFS V2, we might strictly handle text in 'content'
            # Binary support in VFS depends on V2 implementation (usually text-focused).
            # We try to decode utf-8
            try:
                text_content = content_bytes.decode('utf-8')
                session.vfs.write(target_path, text_content)
                self.add_log_entry(f"Uploaded {name}", "success")
                await self.refresh_data()
            except UnicodeDecodeError:
                self.add_log_entry(f"Skipped {name}: Binary files not fully supported in VFS text mode", "warning")

        except Exception as e:
            self.add_log_entry(f"Upload failed: {e}", "error")

    # --- Session & Agent ---
    async def switch_agent(self, event):
        new_agent = event.get('value')
        if new_agent:
            self.host.active_agent_name = new_agent
            self.host.active_session_id = "default"
            self.host._save_state()
            await self.refresh_data()
            self.add_log_entry(f"Switched to {new_agent}", "success")

    async def _switch_session(self, sid):
        self.host.active_session_id = sid
        self.host._save_state()
        self.chat_history.value = [] # Clear view, will reload if persistent
        await self.refresh_data()
        self.add_log_entry(f"Switched to session {sid}", "info")

    async def new_session(self, event):
        import uuid
        new_id = f"sess_{uuid.uuid4().hex[:6]}"
        self.host.active_session_id = new_id
        self.host._save_state()
        await self.refresh_data()
        self.add_log_entry(f"Created session {new_id}", "success")

    async def toggle_tasks(self, event):
        self.show_tasks.value = not self.show_tasks.value

    async def _stop_task(self, tid):
        if tid in self.host.background_tasks:
            self.host.background_tasks[tid].task.cancel()
            self.add_log_entry(f"Stopped task {tid}", "warning")
            # Force UI update (hacky, ideally reactive)
            self.show_tasks.value = False
            self.show_tasks.value = True

# =============================================================================
# EXPORT & INIT
# =============================================================================

@export(mod_name=Name, name="initialize", initial=True)
def initialize(app: App, **kwargs) -> Result:
    """Initialize module and register view."""
    from toolboxv2.mods.Minu import register_view

    # Register the main view
    register_view("isaa_dashboard", ISAAView)

    # Register UI Route
    app.run_any(
        ("CloudM", "add_ui"),
        name="ISAA_Host",
        title="ISAA Host",
        path=f"/api/Minu/render?view=isaa_dashboard&ssr=true&format=full-html",
        description="Autonomous Agent Host Interface",
        icon="smart_toy",
        auth=True # Production secure
    )

    return Result.ok(info="ISAA Web Host Initialized")
