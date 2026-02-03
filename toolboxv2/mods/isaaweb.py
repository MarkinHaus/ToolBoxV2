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
import json

# ISAA & System Imports
try:
    from toolboxv2.mods.isaa.base.Agent.flow_agent import FlowAgent
    from toolboxv2.mods.isaa.base.Agent.vfs_v2 import VirtualFileSystemV2, FileBackingType
    from toolboxv2.mods.isaa.base.audio_io.audioIo import AudioStreamPlayer
    # Versuche Audio-Input (Server-Side)
    import sounddevice as sd
    import numpy as np
except ImportError:
    pass  # Graceful degradation if deps missing

# Toolbox & Minu Imports
from toolboxv2 import App, RequestData, Result, get_app
from toolboxv2.mods.Minu.core import (
    MinuView, MinuSession, State, Component, ComponentType,
    Card, Text, Heading, Button, Input, Row, Column,
    Grid, Spacer, Divider, Icon, Badge, Modal, Form,
    Switch, Select, Textarea, Custom, Dynamic, ReactiveState, MinuJSONEncoder, ComponentStyle
)
# ISAA Imports
from toolboxv2.mods.isaa.base.Agent.vfs_v2 import FileBackingType
import toolboxv2.flows.cli_v4 as cli_mod

# Module Metadata
Name = 'ISAAWeb'
export = get_app(f"{Name}.Export").tb
version = '1.0.0'

"""
ISAA HQ - Cyberpunk Minimalist Web UI
=====================================
A full-screen, sliding-panel interface for ISAA Agents.
Integrates Chat (Stream), VFS (Editor), and System Config.

Design: GitHub Dark Dimmed / Cyberpunk
Stack: Minu, FlowAgent, ToolBoxV2
"""
# ============================================================================
#  STYLES & THEME (Cyberpunk / GitHub Dark Dimmed)
# ============================================================================

GLOBAL_CSS = """
<style>
    :root {
        --bg-dark: #0d1117;
        --bg-panel: rgba(22, 27, 34, 0.85);
        --border-color: #30363d;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --accent-green: #238636;
        --accent-hover: #2ea043;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        --glass-blur: blur(12px);
    }

    body, html {
        margin: 0; padding: 0; overflow: hidden;
        background-color: var(--bg-dark);
        color: var(--text-primary);
        font-family: system-ui, -apple-system, sans-serif;
    }

    #minu-root {
    padding: 0 !important;
    max-width: none !important;
    margin: 0 !important;
    height: 100vh;
    width: 100vw;
}


    /* --- LAYOUT & SLIDER --- */
   isaa-viewport, .isaa-slider, .isaa-panel {
        gap: 0 !important;
    }

    .isaa-viewport {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: 100;
        background: #0d1117; /* Fallback */
    }

    .isaa-slider {
        display: flex !important; /* Sicherstellen, dass es Row bleibt */
        flex-direction: row !important;
        width: 300vw !important;
        height: 100% !important;
    }

    .isaa-panel {
        width: 100vw !important;
        height: 100% !important;
        flex-shrink: 0 !important; /* Verhindert das Zusammenquetschen der Panels */
        display: flex !important;
        justify-content: center;
        align-items: center;
    }

    .panel-content {
        width: 95% !important;
        max-width: 1400px !important;
        height: 90vh !important;
        background: var(--bg-panel);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        backdrop-filter: var(--glass-blur);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }

    /* --- NAVIGATION ZONES --- */
    .nav-zone {
        position: fixed; top: 0; bottom: 0; width: 60px;
        z-index: 200; display: flex; align-items: center; justify-content: center;
        opacity: 0; transition: opacity 0.3s, background 0.3s;
        cursor: pointer;
    }
    .nav-zone:hover { opacity: 1; background: linear-gradient(90deg, rgba(0,0,0,0.8), transparent); }
    .nav-zone.left { left: 0; }
    .nav-zone.right { right: 0; background: linear-gradient(-90deg, rgba(0,0,0,0.8), transparent); }
    .nav-arrow { font-size: 2rem; color: var(--text-secondary); }

    /* --- CHAT STYLES --- */
    .chat-scroll-area {
        flex: 1; overflow-y: auto; padding: 1.5rem;
        display: flex; flex-direction: column; gap: 1rem;
    }
    .msg-row { display: flex; gap: 1rem; max-width: 100%; }
    .msg-row.user { justify-content: flex-end; }
    .msg-bubble {
        padding: 0.75rem 1rem; border-radius: 8px;
        max-width: 80%; line-height: 1.5;
        font-size: 0.95rem;
    }
    .msg-bubble.user { background: var(--accent-green); color: white; border-radius: 12px 12px 0 12px; }
    .msg-bubble.ai { background: #21262d; border: 1px solid var(--border-color); border-radius: 12px 12px 12px 0; }
    .msg-bubble pre { background: #0d1117; padding: 0.5rem; border-radius: 4px; overflow-x: auto; font-family: var(--font-mono); font-size: 0.85rem; }

    .tool-log {
        font-family: var(--font-mono); font-size: 0.8rem;
        background: #0d1117; border-left: 2px solid var(--text-secondary);
        padding: 0.5rem; margin-top: 0.5rem; color: var(--text-secondary);
    }

    /* --- EDITOR / VFS --- */
    .vfs-container { display: flex; height: 100%; }
    .vfs-tree { width: 250px; border-right: 1px solid var(--border-color); overflow-y: auto; padding: 1rem; background: rgba(0,0,0,0.2); }
    .vfs-editor { flex: 1; display: flex; flex-direction: column; background: #0d1117; }
    .editor-textarea {
        flex: 1; width: 100%; border: none; outline: none;
        background: transparent; color: var(--text-primary);
        padding: 1rem; font-family: var(--font-mono); line-height: 1.6; resize: none;
    }

    .tree-item {
        padding: 4px 8px; cursor: pointer; border-radius: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        display: flex; align-items: center; gap: 6px; font-size: 0.9rem;
    }
    .tree-item:hover { background: rgba(255,255,255,0.05); }
    .tree-item.active { background: rgba(35, 134, 54, 0.2); color: #fff; }

    /* --- COMPONENTS --- */
    .isaa-btn {
        background: #21262d; border: 1px solid var(--border-color); color: var(--text-primary);
        padding: 6px 12px; border-radius: 6px; cursor: pointer; transition: all 0.2s;
        font-size: 0.9rem; display: flex; align-items: center; gap: 6px;
    }
    .isaa-btn:hover { background: #30363d; border-color: #8b949e; }
    .isaa-btn.primary { background: var(--accent-green); border-color: rgba(255,255,255,0.1); color: white; }
    .isaa-btn.primary:hover { background: var(--accent-hover); }

    .isaa-input {
        background: #0d1117; border: 1px solid var(--border-color); color: var(--text-primary);
        padding: 8px 12px; border-radius: 6px; width: 100%; outline: none;
    }
    .isaa-input:focus { border-color: var(--text-secondary); }

    /* --- UTILS --- */
    .mono { font-family: var(--font-mono); }
    .text-sm { font-size: 0.85rem; }
    .text-muted { color: var(--text-secondary); }
    .flex-center { display: flex; align-items: center; justify-content: center; }
    .w-full { width: 100%; }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
"""


# ============================================================================
#  HELPER COMPONENTS
# ============================================================================

def IconBtn(icon_name, on_click=None, variant="default", tooltip="", style=None, **props):
    if style is None:
        style = {}
    return Button(
        "", icon=icon_name, on_click=on_click,
        className=f"isaa-btn {variant}",
        style={"padding": "6px", "justifyContent": "center", **style},
        **props
    )


def CyberPanel(children, active_index, panel_index):
    """Wrapper for one of the 3 main panels"""
    # Logic: if active_index == panel_index, it's visible.
    # But we rely on the slider logic for visibility.
    return Custom(
        html=f"""<div class="isaa-panel">
            <div class="panel-content">
                <!-- Content Injection Point -->
                {children}
            </div>
        </div>"""
    )


# ============================================================================
#  MAIN VIEW CLASS
# ============================================================================

# ============================================================================
#  MAIN VIEW CLASS - IMPROVED
# ============================================================================

class IsaaHqView(MinuView):
    # --- STATE ---
    panel_index = State(0)  # 0=Chat, 1=VFS, 2=Config

    # Chat State
    messages = State([])
    input_text = State("")
    is_streaming = State(False)

    # VFS State
    vfs_tree = State({})
    editor_content = State("")
    editor_path = State(None)
    editor_dirty = State(False)

    # System State
    agent_name = State("self")
    agent_status = State("Idle")
    available_agents = State([])
    session_id = State("default")
    new_agent_name = State("")

    def __init__(self):
        super().__init__()
        self.app = get_app()
        self.isaa = self.app.get_mod("isaa")

    async def on_mount(self):
        """Initial data load"""
        await self._refresh_agent_list()
        await self._refresh_vfs_tree()
        if not self.messages.value:
            self.messages.value = [{
                "role": "assistant",
                "content": f"System online. Connected to **{self.agent_name.value}**."
            }]

    # ========================================================================
    #  RENDER LOGIC
    # ========================================================================

    def render(self):
        # 1. CSS Injection
        css_injection = Custom(html=GLOBAL_CSS)

        # 2. Slider Rendering Logic (wird durch Dynamic getriggert)
        def render_slider():
            # Berechne Transform basierend auf panel_index
            offset = self.panel_index.value * -100
            transform_str = f"translateX({offset}vw)"

            return Row(
                # Panel 0
                Column(
                    self._render_chat_header(),
                    self._render_chat_area(),
                    self._render_chat_input(),
                    className="isaa-panel panel-content",
                    gap="0", align="stretch"  # WICHTIG: Stretch damit Breite genutzt wird
                ),
                # Panel 1
                Column(
                    self._render_vfs_panel(),
                    className="isaa-panel panel-content",
                    gap="0", align="stretch"
                ),
                # Panel 2
                Column(
                    self._render_config_panel(),
                    className="isaa-panel panel-content",
                    gap="0", align="stretch"
                ),
                className="isaa-slider",
                gap="0",  # Keine Lücken zwischen den Panels!
                style=ComponentStyle.from_str(
                    f"transform: {transform_str}; display: flex; width: 300vw; transition: transform 0.5s ease-in-out;"
                )
            )

        # Der Haupt-Viewport
        viewport = Component(
            type=ComponentType.COLUMN,
            className="isaa-viewport",
            props={"gap":0},
            children=[
                Dynamic(render_fn=render_slider, bind=self.panel_index),
                self._render_nav_overlays()
            ]
        )

        # Ganz oben: Ein Container ohne Abstände
        return Column(
            css_injection,
            viewport,
            gap="0",
            align="stretch",
            style=ComponentStyle.from_str("height: 100vh; width: 100vw; overflow: hidden;")
        )

    def _render_nav_overlays(self):
        """Erzeugt die klickbaren Zonen links/rechts vom Viewport"""
        zones = []

        if self.panel_index.value > 0:
            zones.append(Button(
                "‹", on_click="nav_left",
                className="nav-zone left"
            ))

        if self.panel_index.value < 2:
            zones.append(Button(
                "›", on_click="nav_right",
                className="nav-zone right"
            ))

        return Row(*zones, className="nav-overlay-container", gap="0")

    # --- CHAT RENDERING ---

    def _render_chat_header(self):
        status_color = "var(--accent-green)" if self.agent_status.value == "Active" else "var(--text-secondary)"
        return Row(
            Icon("smart_toy", size="24", className="text-muted"),
            Text(self.agent_name.value, className="font-bold text-lg"),
            Badge(self.agent_status.value, className="ml-2", background =status_color),
            Spacer(),
            IconBtn("folder", on_click="goto_vfs", tooltip="Go to Files"),
            IconBtn("settings", on_click="goto_config", tooltip="Settings"),
            className="p-4 border-b border-gray-700 bg-gray-900"
        )

    def _render_chat_area(self):
        msg_components = []
        for msg in self.messages.value:
            is_user = msg["role"] == "user"
            bubble_class = "msg-bubble user" if is_user else "msg-bubble ai"

            content_parts = [
                Custom(html=f"<div class='{bubble_class}'>{self._format_text(msg['content'])}</div>")
            ]

            # Tools loggen
            if not is_user and "tools" in msg:
                for tool in msg["tools"]:
                    content_parts.append(
                        Custom(html=f"""
                        <details class="tool-log">
                            <summary>🔧 {tool['name']}</summary>
                            <pre style="font-size: 10px;">{tool.get('args', '')}</pre>
                            <div class="tool-result">{tool.get('result', '...')}</div>
                        </details>
                        """)
                    )

            msg_components.append(
                Row(
                    Column(*content_parts, align="end" if is_user else "start"),
                    justify="end" if is_user else "start",
                    className="msg-row w-full"
                )
            )

        return Column(*msg_components, className="chat-scroll-area")

    def _render_chat_input(self):
        return Row(
            Button("", icon="mic", on_click="toggle_recording", className="btn-icon"),
            Input(
                value=self.input_text.value,
                bind="input_text",
                on_submit="send_message",
                placeholder="Frag ISAA...",
                className="flex-1"
            ),
            Button("", icon="send", on_click="send_message", variant="primary", disabled=self.is_streaming.value),
            className="p-4 bg-gray-900 gap-2 border-t border-gray-700"
        )

    # --- VFS RENDERING ---

    def _render_vfs_panel(self):
        # Generiere Baum-Items
        tree_items = self._build_tree_recursive(self.vfs_tree.value)

        return Column(
            # VFS Header
            Row(
                Button("", icon="chat", on_click="goto_chat", variant="ghost"),
                Text("Virtual File System", className="font-bold"),
                Spacer(),
                Button("Save", on_click="save_file", variant="primary", disabled=not self.editor_dirty.value),
                Button("", icon="sync", on_click="sync_vfs"),
                className="p-3 border-b border-gray-700"
            ),
            Row(
                # Sidebar
                Column(*tree_items, className="vfs-tree-container", style={"width": "250px"}),
                # Editor Area
                Column(
                    Text(self.editor_path.value or "Keine Datei gewählt", className="p-2 text-xs bg-black"),
                    Textarea(
                        value=self.editor_content.value,
                        bind="editor_content",
                        on_change="on_editor_change",
                        className="editor-textarea flex-1"
                    ),
                    className="flex-1"
                ),
                className="flex-1 overflow-hidden",
                gap="0"
            ),
            className="h-full"
        )

    def _build_tree_recursive(self, tree, path_prefix=""):
        items = []
        if not isinstance(tree, dict): return []

        for key, val in sorted(tree.items(), key=lambda x: (not isinstance(x[1], dict), x[0])):
            full_path = f"{path_prefix}/{key}".replace("//", "/")

            if isinstance(val, dict):
                # Ordner
                items.append(
                    Button(f"📂 {key}", on_click=f"toggle_folder_{full_path.replace('/', '_')}",
                           className="tree-item-dir")
                )
                # Kinder einrücken
                sub_items = self._build_tree_recursive(val, full_path)
                if sub_items:
                    items.append(Column(*sub_items, style={"padding-left": "15px"}, gap="1"))
            else:
                # Datei
                active_class = "active-file" if self.editor_path.value == full_path else ""
                # Trick: Wir nutzen einen Button als Baum-Item für saubere Events
                items.append(
                    Button(
                        f"📄 {key}",
                        on_click=f"select_file_{full_path.replace('/', '_')}",
                        className=f"tree-item-file {active_class}"
                    )
                )
        return items

    def _render_tree_items(self, tree, path_prefix=""):
        items = []
        if not tree:
            return []

        # Sort folders first
        try:
            sorted_keys = sorted(tree.keys(), key=lambda k: (not isinstance(tree[k], dict), k))
        except AttributeError:
            return []  # Fallback falls tree kein dict ist

        for key in sorted_keys:
            val = tree[key]
            full_path = f"{path_prefix}/{key}".replace("//", "/")

            if isinstance(val, dict):
                # Folder
                items.append(
                    Custom(html=f"""
                    <div class="tree-item" onclick="window.minu.emit('{self._view_id}', 'toggle_folder', {{path: '{full_path}'}})">
                        📂 {key}
                    </div>
                    """)
                )
                # Recursion
                sub_items = self._render_tree_items(val, full_path)
                if sub_items:
                    # Hier auch wichtig: *sub_items entpacken, da Column args erwartet
                    items.append(Column(*sub_items, style={"paddingLeft": "12px"}))
            else:
                # File
                is_active = "active" if self.editor_path.value == full_path else ""
                items.append(
                    Custom(html=f"""
                    <div class="tree-item {is_active}" onclick="window.minu.emit('{self._view_id}', 'open_file', {{path: '{full_path}'}})">
                        📄 {key}
                    </div>
                    """)
                )
        return items

    # ========================================================================
    #  HANDLERS
    # ========================================================================

    async def nav_left(self, _):
        self.panel_index.value = max(0, self.panel_index.value - 1)

    async def nav_right(self, _):
        self.panel_index.value = min(2, self.panel_index.value + 1)

    async def on_editor_change(self, e):
        # e.value kommt vom Textarea binding
        self.editor_dirty.value = True


    # --- Chat ---

    async def send_message(self, _):
        txt = self.input_text.value.strip()
        if not txt or self.is_streaming.value: return

        # 1. User Nachricht adden
        new_msgs = list(self.messages.value)
        new_msgs.append({"role": "user", "content": txt})

        # 2. AI Placeholder adden
        new_msgs.append({"role": "assistant", "content": "Thinking...", "tools": []})
        self.messages.value = new_msgs
        self.input_text.value = ""
        self.is_streaming.value = True

        try:
            agent = await self.isaa.get_agent(self.agent_name.value)
            full_response = ""

            async for chunk in agent.a_stream(query=txt, session_id=self.session_id.value):
                if chunk.get("type") == "content":
                    full_response += chunk["chunk"]
                    # Update nur die letzte Nachricht
                    self.messages.value[-1]["content"] = full_response
                    self.messages.update_hash()  # Force update für Liste
        except Exception as e:
            self.messages.value[-1]["content"] = f"Error: {str(e)}"
        finally:
            self.is_streaming.value = False

    # --- FALLBACK HANDLER FÜR DYNAMISCHE VFS PFADE ---
    def __getattr__(self, name):
        """Fängt dynamische Pfad-Klicks aus dem VFS Baum ab"""
        if name.startswith("select_file_"):
            path = name.replace("select_file_", "").replace("_", "/")
            return lambda e: self._handle_open_file(path)
        return super().__getattribute__(name)

    async def _handle_open_file(self, path):
        self.editor_path.value = path
        # Hier käme die Logik zum Laden der Datei via ISAA VFS
        self.editor_content.value = f"Lade {path}..."
        self.editor_dirty.value = False

    def _format_text(self, text):
        import html
        return html.escape(text).replace("\n", "<br>").replace("**", "<b>")

    # (Andere Handler wie sync_vfs, switch_agent etc. hier ...)


    def _render_config_panel(self):
        return Column(
            Row(IconBtn("arrow_forward", on_click="goto_vfs"), Text("System Config"),
                className="p-3 border-b border-gray-700"),

            # Agent Control
            Card(
                Heading("Agent Control", level=3),
                Row(
                    Select(
                        options=[{"value": a, "label": a} for a in self.available_agents.value],
                        value=self.agent_name.value,
                        bind="agent_name",
                        label="Active Agent"
                    ),
                    Button("Switch", on_click="switch_agent"),
                    align="end"
                ),
                Row(
                    Input(placeholder="New Agent Name", bind="new_agent_name"),
                    Button("Spawn", on_click="spawn_agent", variant="primary"),
                    align="end", className="mt-2"
                ),
                className="mb-4 bg-gray-800 p-4 rounded"
            ),

            # Session Config
            Card(
                Heading("Session", level=3),
                Row(
                    Input(value=self.session_id.value, bind="session_id", label="Session ID"),
                    Button("Load", on_click="load_session"),
                    Button("Clear History", on_click="clear_history", variant="danger"),
                    align="end"
                ),
                className="mb-4 bg-gray-800 p-4 rounded"
            ),

            # Skills
            Card(
                Heading("Skills", level=3),
                Button("Scan/Import Skills", on_click="scan_skills"),
                Custom(
                    html="<div class='text-sm text-muted mt-2'>Skills are managed automatically via the Agent's skill manager.</div>"),
                className="bg-gray-800 p-4 rounded"
            ),

            className="p-4"
        )

    async def goto_chat(self, e):
        self.panel_index.value = 0

    async def goto_vfs(self, e):
        self.panel_index.value = 1

    async def goto_config(self, e):
        self.panel_index.value = 2


    # --- VFS ---
    async def _refresh_vfs_tree(self):
        try:
            agent = await self.isaa.get_agent(self.agent_name.value)
            session = await agent.session_manager.get_or_create(self.session_id.value)

            # Get flat file list and convert to tree
            files_dict = session.vfs.list_files()  # Returns dict with metadata

            tree = {}
            for file_info in files_dict.get("files", []):
                path_parts = file_info["path"].strip("/").split("/")
                current = tree
                for part in path_parts[:-1]:
                    current = current.setdefault(part, {})
                current[path_parts[-1]] = "file"  # Marker

            self.vfs_tree.value = tree
        except Exception:
            self.vfs_tree.value = {}

    async def open_file(self, event):
        path = event.get("path")
        if not path: return

        try:
            agent = await self.isaa.get_agent(self.agent_name.value)
            session = agent.session_manager.get(self.session_id.value)
            content = session.vfs.read(path)

            if isinstance(content, dict) and content.get("success"):
                self.editor_content.value = content["content"]
            else:
                self.editor_content.value = "// Error reading file"

            self.editor_path.value = path
            self.editor_dirty.value = False
        except Exception:
            pass

    async def update_editor(self, event):
        self.editor_content.value = event.get("value", "")
        self.editor_dirty.value = True

    async def save_file(self, e):
        if not self.editor_path.value: return
        try:
            agent = await self.isaa.get_agent(self.agent_name.value)
            session = agent.session_manager.get(self.session_id.value)
            session.vfs.write(self.editor_path.value, self.editor_content.value)
            self.editor_dirty.value = False
            # Trigger toast or status update here
        except Exception:
            pass

    # --- Config / Agents ---
    async def _refresh_agent_list(self):
        # This assumes isaa config structure
        try:
            agents = self.isaa.config.get("agents-name-list", ["self"])
            self.available_agents.value = agents
        except:
            self.available_agents.value = ["self"]

    async def switch_agent(self, e):
        # Just updates state, logic uses state
        await self._refresh_vfs_tree()
        self.messages.value = []  # Clear chat on switch

    async def spawn_agent(self, e):
        # Logic to call builder
        pass

    # --- Audio ---
    async def toggle_recording(self, e):
        if self.is_recording.value:
            self.is_recording.value = False
            # Stop recording logic (Server side)
            # In a web context, this button would typically trigger client-side JS to stop sending blobs.
            # Here we assume local-host CLI-like behavior as per prompt (F4 logic).
            pass
        else:
            self.is_recording.value = True
            # Start recording logic
            pass


# ============================================================================
#  REGISTRATION
# ============================================================================


@export(mod_name=Name, name="initialize", initial=True)
def initialize(app: App, **kwargs) -> Result:
    """Initialize module and register view."""
    from toolboxv2.mods.Minu import register_view

    # Register the main view
    register_view("isaa_ui", IsaaHqView)

    # Register UI Route
    app.run_any(
        ("CloudM", "add_ui"),
        name="ISAA HQ",
        title="ISAA HQ",
        path="/api/Minu/render?view=isaa_ui&ssr=true",
        icon="smart_toy",
        description="Cyberpunk Agent Control"
    )
    return Result.ok()

