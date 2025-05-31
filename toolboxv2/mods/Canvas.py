# toolboxv2/mods/idea_enhancer/module.py
# (No significant changes needed in the Python part for the requested frontend features)
# The existing IdeaSessionData model can store the canvas_app_state which might include
# new fields like currentMode or toolDefaults if you decide to persist them.
# The primary changes are in the HTML and JavaScript.

import asyncio
import json
import os
import uuid
from typing import Dict, Optional, List, Any, Union

from pydantic import BaseModel, Field as PydanticField

from toolboxv2 import App, Result, RequestData, get_app, MainTool

# --- Module Definition ---
MOD_NAME = Name = "Canvas"  # Renamed slightly for clarity if this is a new version
VERSION = "0.1.0"  # Incremented version
export = get_app(f"widgets.{MOD_NAME}").tb

# --- Constants ---
SESSION_DATA_PREFIX = "enhancedcanvas_session"  # Adjusted prefix if it's a distinct app
SESSION_LIST_KEY_SUFFIX = "_list"


# --- Pydantic Models for Canvas Elements and Session Data ---

class CanvasElement(BaseModel):
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4().hex[:12]))
    type: str  # "pen", "rectangle", "ellipse", "text", "image"

    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    strokeColor: str = "#000000"
    strokeWidth: float = 2
    angle: float = 0.0

    points: Optional[List[List[float]]] = None  # For pen: [[x, y, pressure], ...]

    fillStyle: Optional[str] = "hachure"
    roughness: Optional[float] = 1
    fill: Optional[str] = None

    text: Optional[str] = None
    fontSize: int = 16
    fontFamily: str = "Arial"
    textAlign: str = "left"

    src: Optional[str] = None
    opacity: float = 1.0

    # For selection state, not persisted, but useful for client-side logic if sent back for debug
    # isSelected: Optional[bool] = False

    model_config = {"extra": "allow"}


class IdeaSessionData(BaseModel):
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Canvas"
    canvas_elements: List[CanvasElement] = []
    canvas_app_state: Dict[str, Any] = {
        "viewBackgroundColor": "#ffffff",
        "currentTool": "pen",  # This might be 'select' tool in the new version
        "currentMode": "draw",  # New: 'draw' or 'select'
        "strokeColor": "#000000",
        "fillColor": "#cccccc",
        "strokeWidth": 2,
        "fontFamily": "Arial",
        "fontSize": 16,
        "zoom": 1.0,
        "scrollX": 0,  # old name, now offsetX
        "scrollY": 0,  # old name, now offsetY
        "offsetX": 0,  # Consistent naming with JS
        "offsetY": 0,
        "toolDefaults": {  # New: For default tool settings
            "pen": {"strokeColor": "#000000", "strokeWidth": 2},
            "rectangle": {"strokeColor": "#000000", "fillColor": "#cccccc", "strokeWidth": 2, "fillStyle": "solid"},
            "ellipse": {"strokeColor": "#000000", "fillColor": "#cccccc", "strokeWidth": 2, "fillStyle": "hachure"},
            "text": {"strokeColor": "#000000", "fontSize": 16, "fontFamily": "Arial"},
            # image doesn't have defaults in the same way
        }
    }
    text_notes: str = ""
    last_modified: float = PydanticField(default_factory=lambda: float(uuid.uuid4().int & (1 << 32) - 1))


class Tools(MainTool):
    def __init__(self, app: App):
        self.name = MOD_NAME
        self.version = VERSION
        self.color = "GREEN"  # Changed color for distinction
        self.tools = {"name": MOD_NAME, "Version": self.show_version}
        MainTool.__init__(self, load=self.on_start, v=self.version, tool=self.tools, name=self.name, color=self.color)
        self.db_mod = None
        self.app.logger.info(f"{self.name} v{self.version} instance created.")

    def on_start(self):
        self.app.logger.info(f"Initializing {self.name} v{self.version}...")
        try:
            self.db_mod = self.app.get_mod("DB")
            if not self.db_mod:
                self.app.logger.error(f"{self.name}: DB module not found. Session persistence will not work.")
        except Exception as e:
            self.app.logger.error(f"Error during {self.name} on_start: {e}", exc_info=True)
            return

        self.app.run_any(
            ("CloudM", "add_ui"),
            name=f"{MOD_NAME}UI_v010",  # Updated UI name
            title=f"Enhanced Canvas Studio v{VERSION}",
            path=f"/api/{MOD_NAME}/ui",
            description="Interactive Canvas with draw/move modes and enhanced configuration.",
            auth=True
        )
        self.app.logger.info(f"{self.name} UI (v{VERSION}) registered with CloudM.")
        self.app.logger.info(f"{self.name} (v{VERSION}) initialized successfully.")

    def show_version(self):
        self.app.logger.info(f"{self.name} Version: {self.version}")
        return self.version

    async def _get_user_specific_db_key(self, request: RequestData, base_key: str) -> Optional[str]:
        user = await self.app.run_any(("UserM", "get_user_from_request"), request=request)
        if user and hasattr(user, 'uid') and user.uid:
            return f"{base_key}_{user.uid}"
        if request and request.session and request.session.get('SiID'):
            return f"{base_key}_{request.session.get('SiID')}"
        self.app.logger.warning(f"Could not get UID for user to form DB key. Request session: {request.session}")
        return None

    @export(mod_name=MOD_NAME, api=True, version=VERSION, name="ui", api_methods=['GET'])
    async def get_main_ui(self, request: Optional[RequestData] = None) -> Result:
        # The HTML template will be named differently to reflect the new version
        html_content = ENHANCED_CANVAS_HTML_TEMPLATE_V0_1_0
        return Result.html(data=self.app.web_context() + html_content)

    # Save, List, Load, Export session methods remain largely the same.
    # Ensure IdeaSessionData is correctly (de)serialized.
    # For brevity, I'm not re-listing them fully, but they would use the updated
    # MOD_NAME, VERSION, SESSION_DATA_PREFIX, and IdeaSessionData.
    # Make sure that when saving, the full canvas_app_state (including new fields) is saved.
    # When loading, ensure these new fields are properly restored or defaulted.

    @export(mod_name=MOD_NAME, api=True, version=VERSION, name="save_session", api_methods=['POST'],
            request_as_kwarg=True)
    async def save_session(self, request: RequestData, data: Union[Dict[str, Any], IdeaSessionData]) -> Result:
        if not self.db_mod:
            return Result.custom_error(info="Database module not available for saving.", exec_code=503)

        user_db_key_base = await self._get_user_specific_db_key(request, SESSION_DATA_PREFIX)
        if not user_db_key_base:
            return Result.default_user_error(info="User authentication required to save session.", exec_code=401)

        try:
            # Allow partial updates to canvas_app_state if only that is sent for some reason
            # or ensure the client always sends the full structure.
            # For simplicity, assume client sends full structure or IdeaSessionData correctly populates defaults.
            session_data_obj = IdeaSessionData(**data) if isinstance(data, dict) else data
        except Exception as e:
            self.app.logger.error(f"Invalid session data for save: {e}. Data: {data}", exc_info=True)
            return Result.default_user_error(info=f"Invalid session data: {e}", exec_code=400)

        session_data_obj.last_modified = float(uuid.uuid4().int & (1 << 32) - 1)  # Or time.time()
        session_db_key = f"{user_db_key_base}_{session_data_obj.id}"

        db_op_result = self.db_mod.set(session_db_key, session_data_obj.model_dump_json())
        if asyncio.iscoroutine(db_op_result): await db_op_result

        session_list_key = f"{user_db_key_base}{SESSION_LIST_KEY_SUFFIX}"
        list_res_obj = self.db_mod.get(session_list_key)
        if asyncio.iscoroutine(list_res_obj): list_res_obj = await list_res_obj

        user_sessions = []
        if list_res_obj and not list_res_obj.is_error() and list_res_obj.is_data():
            try:
                list_content = list_res_obj.get()
                json_str_to_load = ""
                if isinstance(list_content, list) and len(list_content) > 0:
                    json_str_to_load = list_content[0]
                elif isinstance(list_content, str):
                    json_str_to_load = list_content

                if json_str_to_load: user_sessions = json.loads(json_str_to_load)
                if not isinstance(user_sessions, list): user_sessions = []
            except (json.JSONDecodeError, TypeError):
                user_sessions = []

        found_in_list = False
        for i, sess_meta in enumerate(user_sessions):
            if sess_meta.get("id") == session_data_obj.id:
                user_sessions[i] = {"id": session_data_obj.id, "name": session_data_obj.name,
                                    "last_modified": session_data_obj.last_modified}
                found_in_list = True;
                break
        if not found_in_list:
            user_sessions.append({"id": session_data_obj.id, "name": session_data_obj.name,
                                  "last_modified": session_data_obj.last_modified})

        list_set_res = self.db_mod.set(session_list_key, json.dumps(user_sessions))
        if asyncio.iscoroutine(list_set_res): await list_set_res

        return Result.ok(info="Session saved successfully.",
                         data={"id": session_data_obj.id, "last_modified": session_data_obj.last_modified})

    @export(mod_name=MOD_NAME, api=True, version=VERSION, name="list_sessions", api_methods=['GET'],
            request_as_kwarg=True)
    async def list_sessions(self, request: RequestData) -> Result:
        # This function should remain largely the same
        if not self.db_mod:
            return Result.custom_error(info="Database module not available.", exec_code=503)
        user_db_key_base = await self._get_user_specific_db_key(request, SESSION_DATA_PREFIX)
        if not user_db_key_base:
            return Result.default_user_error(info="User authentication required.", exec_code=401)

        session_list_key = f"{user_db_key_base}{SESSION_LIST_KEY_SUFFIX}"
        list_res_obj = self.db_mod.get(session_list_key)
        if asyncio.iscoroutine(list_res_obj): list_res_obj = await list_res_obj

        user_sessions = []
        if list_res_obj and not list_res_obj.is_error() and list_res_obj.is_data():
            try:
                list_content = list_res_obj.get()  # Get can return list or string based on DB adapter
                json_str_to_load = list_content[0] if isinstance(list_content,
                                                                 list) and list_content else list_content if isinstance(
                    list_content, str) else "[]"
                user_sessions = json.loads(json_str_to_load)
                if not isinstance(user_sessions, list): user_sessions = []
            except (json.JSONDecodeError, TypeError) as e:
                self.app.logger.warning(f"Error decoding session list for {user_db_key_base}: {e}")
                user_sessions = []

        user_sessions.sort(key=lambda x: x.get("last_modified", 0), reverse=True)
        return Result.json(data=user_sessions)

    @export(mod_name=MOD_NAME, api=True, version=VERSION, name="load_session", api_methods=['GET'],
            request_as_kwarg=True)
    async def load_session(self, request: RequestData, session_id: str) -> Result:
        # This function should remain largely the same
        if not self.db_mod:
            return Result.custom_error(info="Database module not available.", exec_code=503)
        user_db_key_base = await self._get_user_specific_db_key(request, SESSION_DATA_PREFIX)
        if not user_db_key_base:
            return Result.default_user_error(info="User authentication required.", exec_code=401)

        session_db_key = f"{user_db_key_base}_{session_id}"
        get_res_obj = self.db_mod.get(session_db_key)
        if asyncio.iscoroutine(get_res_obj): get_res_obj = await get_res_obj

        if get_res_obj and not get_res_obj.is_error() and get_res_obj.is_data():
            try:
                get_content = get_res_obj.get()
                session_data_str = get_content[0] if isinstance(get_content,
                                                                list) and get_content else get_content if isinstance(
                    get_content, str) else "{}"

                if session_data_str and session_data_str != "{}":
                    session_data_dict = json.loads(session_data_str)
                    # Ensure defaults are applied for new fields if loading old data
                    merged_app_state = {**IdeaSessionData().model_fields['canvas_app_state'].default,
                                        **session_data_dict.get("canvas_app_state", {})}
                    session_data_dict["canvas_app_state"] = merged_app_state

                    session_data = IdeaSessionData(**session_data_dict)
                    return Result.json(data=session_data.model_dump())
                else:
                    return Result.default_user_error(info="Session data is empty.", exec_code=404)
            except (json.JSONDecodeError, TypeError, Exception) as e:
                self.app.logger.error(f"Error loading or parsing session {session_id}: {e}", exc_info=True)
                return Result.custom_error(info=f"Failed to load session data: {e}", exec_code=500)
        else:
            return Result.default_user_error(info="Session not found.", exec_code=404)

    @export(mod_name=MOD_NAME, api=True, version=VERSION, name="export_canvas_json", api_methods=['POST'],
            request_as_kwarg=True)
    async def export_canvas_json(self, request: RequestData, data: Dict[str, Any]) -> Result:
        # This function remains largely the same
        try:
            session_data_to_export = IdeaSessionData(**data)
            filename = f"{session_data_to_export.name.replace(' ', '_') or 'canvas_export'}.json"
            return Result.json(data=session_data_to_export.model_dump(), download_name=filename)
        except Exception as e:
            self.app.logger.error(f"Error preparing canvas JSON export: {e}", exc_info=True)
            return Result.default_user_error(info=f"Invalid data for JSON export: {e}", exec_code=400)


# --- HTML Template for v0.1.0 ---
# (This will be a new variable name and contain the updated HTML)
ENHANCED_CANVAS_HTML_TEMPLATE_V0_1_0 = """
<title>Enhanced Canvas Studio v0.1.0</title>
<!-- Rough.js and Perfect Freehand will be loaded via CDN in the script module -->
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    .studio-container { display: flex; flex-direction: column; height: 100%; }
    .toolbar {
        padding: 6px 10px; background-color: var(--tb-bg-secondary, #f0f0f0);
        border-bottom: 1px solid var(--tb-border-color, #ddd);
        display: flex; gap: 6px; align-items: center; flex-wrap: wrap; flex-shrink: 0;
    }
    .dark .toolbar { background-color: var(--tb-bg-secondary-dark, #2d3748); border-bottom-color: var(--tb-border-color-dark, #4a5562); }
    .toolbar .tb-btn, .toolbar .tb-input, .toolbar input[type='color'], .toolbar label { margin-bottom: 2px; }

    .main-layout { display: flex; flex-grow: 1; overflow: hidden; }
    .canvas-panel { flex-grow: 1; display: flex; justify-content: center; align-items: center; background-color: var(--tb-neutral-200, #e5e7eb); overflow: auto; position: relative; }
    .dark .canvas-panel { background-color: var(--tb-neutral-800, #1f2937); }

    #mainCanvas {
        background-color: var(--canvas-bg, #ffffff);
        cursor: crosshair; box-shadow: 0 0 10px rgba(0,0,0,0.1);

    }
    #textNotesArea { flex-grow: 1; padding: 10px; font-family: monospace; font-size: 0.9rem; border: none; outline: none; resize: none; background-color: transparent; color: inherit; line-height: 1.5; }
    .dark #textNotesArea { background-color: var(--tb-input-bg-dark, #22273869); }

    .toolbar-group { display: flex; align-items: center; gap: 5px; padding: 2px 6px; margin-right: 6px; border-right: 1px solid var(--tb-border-color-light, #e0e0e0); }
    .dark .toolbar-group { border-right-color: var(--tb-border-color-dark, #374151); }
    .toolbar-group:last-child { border-right: none; margin-right: 0; }
    .toolbar label { font-size: 0.75rem; margin-right: 3px; color: var(--tb-text-secondary); }
    .dark .toolbar label { color: var(--tb-text-secondary-dark); }
    input[type="color"] { width: 28px; height: 28px; border: 1px solid var(--tb-border-color-light); padding: 2px; border-radius: 4px; cursor: pointer; background-color: transparent; }
    input[type="file"] { display: none; }

    #textInputOverlay {
        position: absolute; border: 1px dashed var(--tb-primary-500, #007bff); background: rgba(255, 255, 255, 0.95);
        padding: 8px; font-family: Arial; font-size: 16px; line-height: 1.3; white-space: pre-wrap; word-wrap: break-word;
        z-index: 1000; min-width: 80px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); border-radius: 4px; outline: none;
    }
    .dark #textInputOverlay { background: rgba(40, 42, 54, 0.95); color: #f8f8f2; border-color: var(--tb-primary-400, #3b82f6); }

    .toolbar .tb-btn.active {
        background-color: var(--tb-primary-500, #3b82f6) !important; color: white !important;
    }
    .dark .toolbar .tb-btn.active {
        background-color: var(--tb-primary-400, #60a5fa) !important; color: var(--tb-neutral-900, #171717) !important;
    }

    /* Styles for Settings Modal */
    .settings-modal-content { padding: 10px; max-height: 70vh; overflow-y: auto; }
    .settings-modal-content h4 { margin-top: 15px; margin-bottom: 5px; font-size: 0.9rem; font-weight: bold; }
    .settings-modal-content .tool-config-group { margin-bottom: 10px; padding: 8px; border: 1px solid var(--tb-border-color-light); border-radius: 4px; }
    .dark .settings-modal-content .tool-config-group { border-color: var(--tb-border-color-dark); }
    .settings-modal-content label { display: inline-block; min-width: 90px; margin-bottom: 5px; font-size:0.8rem; }
    .settings-modal-content input[type="color"] { vertical-align: middle; }
    .settings-modal-content input[type="number"], .settings-modal-content select {
        padding: 4px 6px; font-size: 0.8rem;
        /* Ensure tb-input styles are applied if available, or define basic ones */
        border: 1px solid var(--tb-input-border, #ccc);
        border-radius: 4px;
        background-color: var(--tb-input-bg, #fff);
        color: var(--tb-input-text, #000);
    }
    .dark .settings-modal-content input[type="number"], .dark .settings-modal-content select {
        border-color: var(--tb-input-border-dark, #555);
        background-color: var(--tb-input-bg-dark, #333);
        color: var(--tb-input-text-dark, #fff);
    }


</style>

<div id="studioAppContainerV010" class="studio-container tb-bg-primary dark:tb-bg-primary-dark tb-text-primary dark:tb-text-primary-dark">
    <div class="toolbar">
        <div class="toolbar-group">
            <input type="text" id="canvasNameInput" placeholder="Canvas Name..." class="tb-input tb-input-sm" style="width: 140px;">
            <button id="newSessionBtn" title="New Canvas" class="tb-btn tb-btn-neutral tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">add_circle</span></button>
        </div>
        <div class="toolbar-group"> <!-- Mode Switch -->
            <button id="modeDrawBtn" title="Draw Mode (D)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">draw</span></button>
            <button id="modeSelectBtn" title="Select/Move Mode (V)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">pan_tool</span></button> <!-- Or 'near_me' for selection arrow -->
        </div>
        <div id="drawToolsGroup" class="toolbar-group"> <!-- Drawing Tools (shown in Draw mode) -->
            <button id="toolPenBtn" title="Pen (P)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">edit</span></button>
            <button id="toolRectBtn" title="Rectangle (R)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">rectangle</span></button>
            <button id="toolEllipseBtn" title="Ellipse (O)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">circle</span></button>
            <button id="toolTextBtn" title="Text (T)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">title</span></button>
            <button id="toolImageBtn" title="Image" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">image</span></button>
        </div>
        <div id="commonToolsGroup" class="toolbar-group"> <!-- Common properties -->
            <label for="strokeColorPicker" title="Stroke Color">S:</label><input type="color" id="strokeColorPicker" value="#000000">
            <label for="fillColorPicker" title="Fill Color (for shapes)">F:</label><input type="color" id="fillColorPicker" value="#cccccc">
            <label for="bgColorPicker" title="Canvas Background Color">BG:</label><input type="color" id="bgColorPicker" value="#ffffff">
            <label for="strokeWidthInput" title="Stroke Width">W:</label><input type="number" id="strokeWidthInput" value="2" min="1" max="100" class="tb-input tb-input-xs" style="width: 50px;">
        </div>
         <div class="toolbar-group">
            <button id="undoBtn" title="Undo (Ctrl+Z)" class="tb-btn tb-btn-neutral tb-btn-sm tb-btn-icon" disabled><span class="material-symbols-outlined">undo</span></button>
            <button id="redoBtn" title="Redo (Ctrl+Y)" class="tb-btn tb-btn-neutral tb-btn-sm tb-btn-icon" disabled><span class="material-symbols-outlined">redo</span></button>
        </div>
        <div class="toolbar-group" style="margin-left: auto;"> <!-- Align to right -->
            <button id="settingsBtn" title="Settings" class="tb-btn tb-btn-neutral tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">settings</span></button>
            <button id="saveSessionBtn" title="Save Session" class="tb-btn tb-btn-primary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">save</span></button>
            <button id="loadSessionBtn" title="Load Session" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">folder_open</span></button>
            <button id="exportJsonBtn" title="Export JSON" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">file_download</span></label>
            <label for="importJsonInput" title="Import JSON" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">file_upload</span></label>
            <input type="file" id="importJsonInput" accept=".json">
            <div id="darkModeToggleContainer" style="display: inline-flex; align-items: center;"></div>
        </div>
    </div>
    <div class="main-layout">
        <div class="canvas-panel">
            <canvas id="mainCanvas" style="height: 100vh; width: 100vw;"></canvas>
            <textarea id="textInputOverlay" style="display:none;"></textarea>
        </div>
        <div class="notes-panel none">
            <h3>Notes</h3>
            <textarea id="textNotesArea" placeholder="Type your notes here..."></textarea>
        </div>
    </div>
</div>

<!-- Settings Modal HTML (initially hidden) -->
<div id="settingsModal" class="tb-modal" style="display:none;">
    <div class="tb-modal-dialog tb-modal-lg">
        <div class="tb-modal-content">
            <div class="tb-modal-header">
                <h5 class="tb-modal-title">Default Tool Settings</h5>
                <button type="button" class="tb-btn-close" data-dismiss="modal" aria-label="Close" onclick="TB.ui.Modal.getById('settingsModal').close()"></button>
            </div>
            <div class="tb-modal-body settings-modal-content">
                <!-- Pen Defaults -->
                <div class="tool-config-group">
                    <h4>Pen Tool</h4>
                    <label for="defaultPenStrokeColor">Stroke Color:</label>
                    <input type="color" id="defaultPenStrokeColor" data-tool="pen" data-prop="strokeColor"><br>
                    <label for="defaultPenStrokeWidth">Stroke Width:</label>
                    <input type="number" id="defaultPenStrokeWidth" data-tool="pen" data-prop="strokeWidth" min="1" max="100" class="tb-input tb-input-xs" style="width: 60px;">
                </div>
                <!-- Rectangle Defaults -->
                <div class="tool-config-group">
                    <h4>Rectangle Tool</h4>
                    <label for="defaultRectStrokeColor">Stroke Color:</label>
                    <input type="color" id="defaultRectStrokeColor" data-tool="rectangle" data-prop="strokeColor"><br>
                    <label for="defaultRectFillColor">Fill Color:</label>
                    <input type="color" id="defaultRectFillColor" data-tool="rectangle" data-prop="fillColor"><br>
                    <label for="defaultRectStrokeWidth">Stroke Width:</label>
                    <input type="number" id="defaultRectStrokeWidth" data-tool="rectangle" data-prop="strokeWidth" min="1" max="100" class="tb-input tb-input-xs" style="width: 60px;"><br>
                    <label for="defaultRectFillStyle">Fill Style:</label>
                    <select id="defaultRectFillStyle" data-tool="rectangle" data-prop="fillStyle">
                        <option value="hachure">Hachure</option>
                        <option value="solid">Solid</option>
                        <option value="zigzag">Zigzag</option>
                        <option value="cross-hatch">Cross-Hatch</option>
                        <option value="dots">Dots</option>
                        <option value="dashed">Dashed</option>
                        <option value="zigzag-line">Zigzag Line</option>
                    </select>
                </div>
                <!-- Ellipse Defaults -->
                <div class="tool-config-group">
                    <h4>Ellipse Tool</h4>
                    <label for="defaultEllipseStrokeColor">Stroke Color:</label>
                    <input type="color" id="defaultEllipseStrokeColor" data-tool="ellipse" data-prop="strokeColor"><br>
                    <label for="defaultEllipseFillColor">Fill Color:</label>
                    <input type="color" id="defaultEllipseFillColor" data-tool="ellipse" data-prop="fillColor"><br>
                    <label for="defaultEllipseStrokeWidth">Stroke Width:</label>
                    <input type="number" id="defaultEllipseStrokeWidth" data-tool="ellipse" data-prop="strokeWidth" min="1" max="100" class="tb-input tb-input-xs" style="width: 60px;"><br>
                    <label for="defaultEllipseFillStyle">Fill Style:</label>
                    <select id="defaultEllipseFillStyle" data-tool="ellipse" data-prop="fillStyle">
                        <option value="hachure">Hachure</option>
                        <option value="solid">Solid</option>
                        <option value="zigzag">Zigzag</option>
                        <!-- Add more RoughJS fill styles as needed -->
                    </select>
                </div>
                <!-- Text Defaults -->
                <div class="tool-config-group">
                    <h4>Text Tool</h4>
                    <label for="defaultTextColor">Text Color:</label>
                    <input type="color" id="defaultTextColor" data-tool="text" data-prop="strokeColor"><br>
                    <label for="defaultTextFontSize">Font Size:</label>
                    <input type="number" id="defaultTextFontSize" data-tool="text" data-prop="fontSize" min="8" max="120" class="tb-input tb-input-xs" style="width: 60px;"><br>
                    <label for="defaultTextFontFamily">Font Family:</label>
                    <input type="text" id="defaultTextFontFamily" data-tool="text" data-prop="fontFamily" class="tb-input tb-input-sm" style="width: 120px;">
                </div>
            </div>
            <div class="tb-modal-footer">
                <button type="button" class="tb-btn tb-btn-secondary" onclick="TB.ui.Modal.getById('settingsModal').close()">Close</button>
                <button type="button" class="tb-btn tb-btn-primary" id="saveSettingsBtn">Save Defaults</button>
            </div>
        </div>
    </div>
</div>


<script src="https://cdn.jsdelivr.net/npm/roughjs@4.6.6/bundled/rough.min.js"></script>
<script type="module" defer>
  import { getStroke } from 'https://cdn.jsdelivr.net/npm/perfect-freehand@1.2.2/dist/esm/index.mjs';

  window.getStroke = getStroke; // Make globally reachable
  // rough is already global via its UMD bundle

    if (!window.TB) {
        console.error("TB (ToolBox Client-Side Library) is not loaded. Canvas UI cannot function.");
        // Display error in UI
    }

    // --- Global State & Configuration ---
    let currentSessionId = null;
    let currentCanvasName = "Untitled Canvas"; // Renamed from ideaName for clarity
    let canvasElements = [];
    let textNotesContent = "";

    // Default canvas app state - this will be merged with loaded session data
    const DEFAULT_CANVAS_APP_STATE = {
        viewBackgroundColor: "#ffffff",
        currentMode: "draw", // 'draw' or 'select'
        currentTool: "pen",  // Active tool in 'draw' mode
        strokeColor: "#000000",
        fillColor: "transparent", // Default to transparent for shapes initially
        strokeWidth: 2,
        fontFamily: "Arial",
        fontSize: 16,
        zoom: 1.0,
        offsetX: 0,
        offsetY: 0,
        toolDefaults: {
            pen: { strokeColor: "#000000", strokeWidth: 2, opacity: 1.0 },
            rectangle: { strokeColor: "#000000", fillColor: "#cccccc", strokeWidth: 2, fillStyle: "solid", roughness: 1, opacity: 1.0 },
            ellipse: { strokeColor: "#000000", fillColor: "#dddddd", strokeWidth: 2, fillStyle: "hachure", roughness: 1, opacity: 1.0 },
            text: { strokeColor: "#000000", fontSize: 16, fontFamily: "Arial", textAlign: "left", opacity: 1.0 },
            image: { opacity: 1.0 } // No color/stroke defaults needed
        }
    };
    let canvasAppState = JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE)); // Deep copy

    // Canvas, context, and drawing related
    let canvas, ctx, roughCanvasInstance;
    let isDrawing = false; // For drawing new shapes/pen strokes
    let currentPenStroke = null;
    let startDragX, startDragY; // World coordinates for drawing shapes

    // Panning state
    let isPanning = false;
    let panStartViewX, panStartViewY; // View coordinates for panning calculation

    // Selection and Moving state
    let selectedElement = null; // The actual element object
    let isDraggingSelection = false;
    let selectionDragStartWorldX, selectionDragStartWorldY; // Mouse down point in world for dragging
    let selectedElementOriginalX, selectedElementOriginalY; // Original pos of element for drag delta

    // Undo/Redo
    let historyStack = [];
    let redoStack = [];
    const MAX_HISTORY_SIZE = 50;

    // Text input
    let textInputOverlayEl, currentTextElementData;
    let activeToolButtons = {};
    let activeModeButtons = {};

    function initializeCanvasStudio() {
        TB.logger.info("Canvas Studio v0.1.0: Initializing...");

        if (TB.ui && TB.ui.DarkModeToggle && document.getElementById('darkModeToggleContainer')) {
            new TB.ui.DarkModeToggle({ target: document.getElementById('darkModeToggleContainer') });
        }

        // DOM Elements
        const canvasNameInputEl = document.getElementById('canvasNameInput');
        const newSessionBtnEl = document.getElementById('newSessionBtn');
        const saveSessionBtnEl = document.getElementById('saveSessionBtn');
        const loadSessionBtnEl = document.getElementById('loadSessionBtn');
        const exportJsonBtnEl = document.getElementById('exportJsonBtn');
        const importJsonInputEl = document.getElementById('importJsonInput');
        const textNotesAreaEl = document.getElementById('textNotesArea');
        const settingsBtnEl = document.getElementById('settingsBtn');
        const saveSettingsBtnEl = document.getElementById('saveSettingsBtn');

        canvas = document.getElementById('mainCanvas');
        ctx = canvas.getContext('2d');
        roughCanvasInstance = rough.canvas(canvas);
        textInputOverlayEl = document.getElementById('textInputOverlay');

        activeModeButtons = {
            draw: document.getElementById('modeDrawBtn'),
            select: document.getElementById('modeSelectBtn'),
        };
        activeToolButtons = {
            pen: document.getElementById('toolPenBtn'),
            rectangle: document.getElementById('toolRectBtn'),
            ellipse: document.getElementById('toolEllipseBtn'),
            text: document.getElementById('toolTextBtn'),
            image: document.getElementById('toolImageBtn'),
            // 'pan' tool was removed, panning is via Ctrl/Meta key or select mode + drag empty space
        };

        const strokeColorPickerEl = document.getElementById('strokeColorPicker');
        const fillColorPickerEl = document.getElementById('fillColorPicker');
        const bgColorPickerEl = document.getElementById('bgColorPicker');
        const strokeWidthInputEl = document.getElementById('strokeWidthInput');
        const undoBtnEl = document.getElementById('undoBtn');
        const redoBtnEl = document.getElementById('redoBtn');

        const canvasPanel = document.querySelector('.canvas-panel');

        function resizeCanvas() {
            const dpr = window.devicePixelRatio || 1;
            const panelRect = canvasPanel.getBoundingClientRect();

            canvas.width = panelRect.width * dpr;
            canvas.height = panelRect.height * dpr;
            canvas.style.width = panelRect.width + 'px';
            canvas.style.height = panelRect.height + 'px';

            ctx.resetTransform(); // Clear previous transforms
            ctx.scale(dpr, dpr); // Apply DPR scaling FIRST
            // The main renderCanvas function will apply zoom and pan transforms on top of this
            renderCanvas();
        }
        window.addEventListener('resize', resizeCanvas);
        setTimeout(resizeCanvas, 50); // Initial resize

        // --- Event Listeners ---
        canvasNameInputEl.addEventListener('input', () => currentCanvasName = canvasNameInputEl.value);
        newSessionBtnEl.addEventListener('click', startNewSession);
        saveSessionBtnEl.addEventListener('click', handleSaveSession);
        loadSessionBtnEl.addEventListener('click', handleLoadSession); // Uses TB.ui.Modal
        exportJsonBtnEl.addEventListener('click', handleExportJSON);
        importJsonInputEl.addEventListener('change', handleImportJSON);
        textNotesAreaEl.addEventListener('input', () => {
            textNotesContent = textNotesAreaEl.value;
            // Consider debouncing save or indicate unsaved changes here
        });
        settingsBtnEl.addEventListener('click', openSettingsModal);
        saveSettingsBtnEl.addEventListener('click', saveToolDefaults);


        Object.entries(activeModeButtons).forEach(([modeName, btn]) => {
            btn.addEventListener('click', () => setActiveMode(modeName));
        });
        Object.entries(activeToolButtons).forEach(([toolName, btn]) => {
            btn.addEventListener('click', () => setActiveTool(toolName));
        });

        strokeColorPickerEl.addEventListener('input', (e) => { canvasAppState.strokeColor = e.target.value; if(selectedElement) { selectedElement.strokeColor = e.target.value; pushToHistory(); renderCanvas();} });
        fillColorPickerEl.addEventListener('input', (e) => { canvasAppState.fillColor = e.target.value; if(selectedElement && (selectedElement.type === 'rectangle' || selectedElement.type === 'ellipse')) { selectedElement.fill = e.target.value; pushToHistory(); renderCanvas(); }});
        bgColorPickerEl.addEventListener('input', (e) => {
            canvasAppState.viewBackgroundColor = e.target.value;
            renderCanvas();
        });
        strokeWidthInputEl.addEventListener('input', (e) => { canvasAppState.strokeWidth = parseInt(e.target.value, 10); if(selectedElement) { selectedElement.strokeWidth = canvasAppState.strokeWidth; pushToHistory(); renderCanvas(); }});

        undoBtnEl.addEventListener('click', undo);
        redoBtnEl.addEventListener('click', redo);

        document.addEventListener('keydown', handleGlobalKeyDown);

        // Canvas Mouse Events
        canvas.addEventListener('mousedown', handleCanvasMouseDown);
        canvas.addEventListener('mousemove', handleCanvasMouseMove);
        canvas.addEventListener('mouseup', handleCanvasMouseUp);
        canvas.addEventListener('mouseleave', handleCanvasMouseLeave); // Important for finishing drags/pans
        canvas.addEventListener('wheel', handleCanvasWheel, { passive: false });

        // Canvas Touch Events
        canvas.addEventListener('touchstart', handleCanvasTouchStart, { passive: false });
        canvas.addEventListener('touchmove', handleCanvasTouchMove, { passive: false });
        canvas.addEventListener('touchend', handleCanvasTouchEnd);
        canvas.addEventListener('touchcancel', handleCanvasTouchEnd); // Treat cancel like end

        textInputOverlayEl.addEventListener('blur', finalizeTextInput);
        textInputOverlayEl.addEventListener('keydown', handleTextInputKeyDown);

        TB.events.on('theme:changed', (themeData) => {
            const isDark = themeData.mode === 'dark';
            const lightBg = DEFAULT_CANVAS_APP_STATE.viewBackgroundColor; // e.g. #ffffff
            const darkBg = canvasAppState.toolDefaults?.viewBackgroundColorDark || '#1e1e1e'; // A default dark bg

            if (canvasAppState.viewBackgroundColor === lightBg && isDark) {
                 bgColorPickerEl.value = darkBg;
            } else if (canvasAppState.viewBackgroundColor === darkBg && !isDark) {
                 bgColorPickerEl.value = lightBg;
            }
            bgColorPickerEl.dispatchEvent(new Event('input'));
        });

        startNewSession(); // Initialize with default state
        TB.logger.info("Canvas Studio v0.1.0: Initialized.");
    }

    // --- Global Keydown Handler ---
    function handleGlobalKeyDown(e) {
        if (document.activeElement === textInputOverlayEl ||
            document.activeElement === textNotesAreaEl ||
            document.activeElement === canvasNameInputEl ||
            TB.ui.Modal.isAnyModalOpen()) { // Don't process shortcuts if typing in inputs or modal is open
            if (e.key === 'Escape' && document.activeElement === textInputOverlayEl) textInputOverlayEl.blur();
            return;
        }

        if (e.ctrlKey || e.metaKey) { // Ctrl/Cmd shortcuts
            switch (e.key.toLowerCase()) {
                case 'z': e.preventDefault(); undo(); break;
                case 'y': e.preventDefault(); redo(); break;
                case 's': e.preventDefault(); handleSaveSession(); break;
                // Add more like copy/paste for selected elements later
            }
        } else { // Single key shortcuts
            switch(e.key.toLowerCase()) {
                case 'd': setActiveMode('draw'); break;
                case 'v': case 's': setActiveMode('select'); break; // V for select/move (common in vector apps), S for Select
                case 'p': if(canvasAppState.currentMode === 'draw') setActiveTool('pen'); break;
                case 'r': if(canvasAppState.currentMode === 'draw') setActiveTool('rectangle'); break;
                case 'o': if(canvasAppState.currentMode === 'draw') setActiveTool('ellipse'); break;
                case 't': if(canvasAppState.currentMode === 'draw') setActiveTool('text'); break;
                // 'h' for pan is now implicit with Ctrl/Meta + drag or spacebar + drag (if implemented)
                case 'delete': case 'backspace':
                    if (canvasAppState.currentMode === 'select' && selectedElement) {
                        e.preventDefault();
                        deleteSelectedElement();
                    }
                    break;
            }
        }
    }

    // --- History Management (Undo/Redo) ---
    function pushToHistory(actionName = "unknown") { // Optional action name for debugging
        const serializableElements = canvasElements.map(el => {
            const { imgObject, ...rest } = el; // Always strip out non-serializable imgObject
            return rest;
        });
        historyStack.push(JSON.stringify(serializableElements));
        if (historyStack.length > MAX_HISTORY_SIZE) {
            historyStack.shift();
        }
        redoStack = [];
        updateUndoRedoButtons();
        // TB.logger.debug(`History: ${actionName}, Stack: ${historyStack.length}, Redo: ${redoStack.length}`);
    }

    async function restoreElementsFromHistory(elementsData) {
        // This is used by undo/redo AND by loading sessions.
        // It needs to correctly re-create image objects.
        const newElements = [];
        for (const elData of elementsData) {
            const newEl = { ...elData }; // Copy element data
            if (newEl.type === 'image' && newEl.src) {
                try {
                    newEl.imgObject = await loadImageAsync(newEl.src);
                } catch (err) {
                    TB.logger.error("Failed to reload image during history restore/load:", newEl.src, err);
                    newEl.imgObject = null; // Or a placeholder
                }
            }
            newElements.push(newEl);
        }
        canvasElements = newElements;
        selectedElement = null; // Deselect after history change or load
        renderCanvas();
    }

    async function undo() {
        if (historyStack.length <= 1 && canvasElements.length === 0) return;

        if (historyStack.length > 1) {
            redoStack.push(historyStack.pop());
            const prevState = JSON.parse(historyStack[historyStack.length - 1]);
            await restoreElementsFromHistory(prevState);
        } else if (historyStack.length === 1 && canvasElements.length > 0) {
            redoStack.push(historyStack.pop());
            canvasElements = [];
            historyStack.push(JSON.stringify([])); // Push the empty state as current
            await restoreElementsFromHistory([]); // Will call renderCanvas
        }
        updateUndoRedoButtons();
    }

    async function redo() {
        if (redoStack.length === 0) return;
        const nextStateJson = redoStack.pop();
        historyStack.push(nextStateJson);
        const nextState = JSON.parse(nextStateJson);
        await restoreElementsFromHistory(nextState);
        updateUndoRedoButtons();
    }

    function updateUndoRedoButtons() {
        document.getElementById('undoBtn').disabled = historyStack.length <= 1;
        document.getElementById('redoBtn').disabled = redoStack.length === 0;
    }

    // --- Mode and Tool Activation ---
    function setActiveMode(modeName) {
        if (canvasAppState.currentMode === modeName) return;
        finalizeTextInput();
        selectedElement = null; // Deselect when changing modes

        canvasAppState.currentMode = modeName;
        TB.logger.info(`Mode changed to: ${modeName}`);

        for (const [name, btn] of Object.entries(activeModeButtons)) {
            btn.classList.toggle('active', name === modeName);
            btn.classList.toggle('tb-btn-primary', name === modeName);
            btn.classList.toggle('tb-btn-secondary', name !== modeName);
        }
        document.getElementById('drawToolsGroup').style.display = (modeName === 'draw') ? 'flex' : 'none';

        if (modeName === 'select') {
            canvas.style.cursor = 'default'; // Or 'grab' if you want to indicate panning possibility
            setActiveTool(null); // No drawing tool active in select mode
        } else { // 'draw' mode
            setActiveTool(canvasAppState.currentTool || 'pen'); // Reactivate last drawing tool
        }
        renderCanvas(); // To clear selection highlights if any
    }

    function setActiveTool(toolName) {
        // If toolName is null, it means no specific drawing tool is active (e.g., in select mode)
        if (toolName === null) {
             Object.values(activeToolButtons).forEach(btn => {
                btn.classList.remove('active', 'tb-btn-primary');
                btn.classList.add('tb-btn-secondary');
            });
            canvas.style.cursor = (canvasAppState.currentMode === 'select') ? 'default' : 'crosshair'; // Default for select, crosshair if somehow draw mode without tool
            canvasAppState.currentTool = null;
            return;
        }

        if (canvasAppState.currentMode !== 'draw') {
            setActiveMode('draw'); // Switch to draw mode if a tool is selected
        }
        finalizeTextInput();
        canvasAppState.currentTool = toolName;

        for (const [name, btn] of Object.entries(activeToolButtons)) {
            const isActive = name === toolName;
            btn.classList.toggle('active', isActive);
            btn.classList.toggle('tb-btn-primary', isActive);
            btn.classList.toggle('tb-btn-secondary', !isActive);
        }

        // Apply default settings for this tool to the main controls
        const defaults = canvasAppState.toolDefaults[toolName];
        if (defaults) {
            document.getElementById('strokeColorPicker').value = defaults.strokeColor || canvasAppState.strokeColor;
            canvasAppState.strokeColor = defaults.strokeColor || canvasAppState.strokeColor;

            // Fill color is relevant for shapes
            if (toolName === 'rectangle' || toolName === 'ellipse') {
                document.getElementById('fillColorPicker').value = defaults.fillColor || canvasAppState.fillColor;
                canvasAppState.fillColor = defaults.fillColor || canvasAppState.fillColor;
                document.getElementById('fillColorPicker').parentElement.style.display = '';
            } else {
                 document.getElementById('fillColorPicker').parentElement.style.display = 'none';
            }

            document.getElementById('strokeWidthInput').value = defaults.strokeWidth || canvasAppState.strokeWidth;
            canvasAppState.strokeWidth = defaults.strokeWidth || canvasAppState.strokeWidth;

            if (toolName === 'text') {
                canvasAppState.fontFamily = defaults.fontFamily || canvasAppState.fontFamily;
                canvasAppState.fontSize = defaults.fontSize || canvasAppState.fontSize;
            }
        }


        if (toolName === 'text') canvas.style.cursor = 'text';
        else if (toolName === 'pan') canvas.style.cursor = 'grab'; // Kept for consistency, though pan is now implicit
        else canvas.style.cursor = 'crosshair';
    }

    // --- Coordinate Transformation ---
    function getCanvasCoordinates(eventOrTouch) {
        const rect = canvas.getBoundingClientRect();
        let clientX, clientY;

        if (eventOrTouch.clientX !== undefined) { // MouseEvent
            clientX = eventOrTouch.clientX;
            clientY = eventOrTouch.clientY;
        } else { // Touch object
            clientX = eventOrTouch.clientX;
            clientY = eventOrTouch.clientY;
        }

        const viewX = clientX - rect.left;
        const viewY = clientY - rect.top;

        const worldX = (viewX - canvasAppState.offsetX) / canvasAppState.zoom;
        const worldY = (viewY - canvasAppState.offsetY) / canvasAppState.zoom;
        return { x: worldX, y: worldY, viewX: viewX, viewY: viewY };
    }


    // --- Mouse Event Handlers ---
    function handleCanvasMouseDown(e) {
        if (e.button !== 0) return; // Only main (left) button
        e.preventDefault();
        finalizeTextInput();

        const { x: worldX, y: worldY, viewX, viewY } = getCanvasCoordinates(e);

        // Universal Panning: Ctrl/Meta + Drag, or Middle Mouse Button
        // Or Spacebar + Drag (needs keydown/up for spacebar state)
        if (e.ctrlKey || e.metaKey || e.button === 1 /* Middle Mouse */) {
            isPanning = true;
            panStartViewX = viewX; // Panning is based on view coordinate delta
            panStartViewY = viewY;
            canvas.style.cursor = 'grabbing';
            return;
        }

        if (canvasAppState.currentMode === 'select') {
            const clickedElement = getElementAtPosition(worldX, worldY);
            if (clickedElement) {
                selectedElement = clickedElement;
                isDraggingSelection = true;
                selectionDragStartWorldX = worldX;
                selectionDragStartWorldY = worldY;
                selectedElementOriginalX = selectedElement.x; // For all types, x,y is top-left or start
                selectedElementOriginalY = selectedElement.y;
                // If it's a path, we'd need to store original points or use a different drag mechanism
                if (selectedElement.type === 'pen') { // Store initial points for pen stroke dragging
                    selectedElement.originalPoints = JSON.parse(JSON.stringify(selectedElement.points));
                }

                // Update color pickers to reflect selected element's properties
                document.getElementById('strokeColorPicker').value = selectedElement.strokeColor || canvasAppState.strokeColor;
                if(selectedElement.type === 'rectangle' || selectedElement.type === 'ellipse'){
                    document.getElementById('fillColorPicker').value = selectedElement.fill || canvasAppState.fillColor;
                }
                document.getElementById('strokeWidthInput').value = selectedElement.strokeWidth || canvasAppState.strokeWidth;

            } else {
                selectedElement = null;
                // Allow pan by dragging empty space in select mode
                isPanning = true;
                panStartViewX = viewX;
                panStartViewY = viewY;
                canvas.style.cursor = 'grabbing';
            }
            renderCanvas(); // To show selection
        } else { // Draw mode
            isDrawing = true;
            startDragX = worldX;
            startDragY = worldY;

            if (canvasAppState.currentTool === 'pen') {
                currentPenStroke = {
                    id: TB.utils.uniqueId('pen_'), type: 'pen',
                    points: [[worldX, worldY, e.pressure || 0.5]],
                    strokeColor: canvasAppState.strokeColor, strokeWidth: canvasAppState.strokeWidth,
                    opacity: canvasAppState.toolDefaults.pen.opacity || 1.0, angle: 0
                };
            } else if (canvasAppState.currentTool === 'text') {
                isDrawing = false; // Text placement doesn't involve dragging to draw
                showTextInputOverlay(worldX, worldY);
            }
            // Other tools like rect/ellipse start drawing on mousemove
        }
    }

    function handleCanvasMouseMove(e) {
        e.preventDefault();
        const { x: worldX, y: worldY, viewX, viewY } = getCanvasCoordinates(e);

        if (isPanning) {
            const dxView = viewX - panStartViewX; // Delta in view coordinates
            const dyView = viewY - panStartViewY;
            canvasAppState.offsetX += dxView;
            canvasAppState.offsetY += dyView;
            panStartViewX = viewX;
            panStartViewY = viewY;
            renderCanvas();
            return;
        }

        if (canvasAppState.currentMode === 'select') {
            if (isDraggingSelection && selectedElement) {
                const deltaX = worldX - selectionDragStartWorldX;
                const deltaY = worldY - selectionDragStartWorldY;

                if (selectedElement.type === 'pen') {
                    // Translate all points of the pen stroke
                    selectedElement.points = selectedElement.originalPoints.map(p => [
                        p[0] + deltaX,
                        p[1] + deltaY,
                        p[2] // pressure remains the same
                    ]);
                } else {
                    // For other shapes, just move their top-left (x,y)
                    selectedElement.x = selectedElementOriginalX + deltaX;
                    selectedElement.y = selectedElementOriginalY + deltaY;
                }
                renderCanvas();
            } else {
                // Hover effect or change cursor over elements could be done here
                const hoveredElement = getElementAtPosition(worldX, worldY);
                canvas.style.cursor = hoveredElement ? 'move' : 'default';
            }
        } else { // Draw mode
            if (!isDrawing) return;

            if (canvasAppState.currentTool === 'pen' && currentPenStroke) {
                currentPenStroke.points.push([worldX, worldY, e.pressure || 0.5]);
                // For pen, preview is drawn by adding to currentPenStroke and re-rendering everything
                // then drawing the "live" part. The provided code was a bit complex for live pen.
                // Simplification: render all committed, then draw current stroke on top as temp.
                renderCanvas(); // Render committed elements
                drawTemporaryPenStroke(currentPenStroke); // Draw the current stroke live
            } else if (['rectangle', 'ellipse'].includes(canvasAppState.currentTool)) {
                renderCanvas(); // Redraw existing elements
                const tempShape = {
                    type: canvasAppState.currentTool,
                    x: Math.min(startDragX, worldX),
                    y: Math.min(startDragY, worldY),
                    width: Math.abs(worldX - startDragX),
                    height: Math.abs(worldY - startDragY),
                    strokeColor: canvasAppState.strokeColor,
                    fill: canvasAppState.fillColor,
                    strokeWidth: canvasAppState.strokeWidth,
                    opacity: 0.6, // Preview opacity
                    fillStyle: canvasAppState.toolDefaults[canvasAppState.currentTool]?.fillStyle || (canvasAppState.currentTool === 'rectangle' ? 'solid' : 'hachure'),
                    roughness: canvasAppState.toolDefaults[canvasAppState.currentTool]?.roughness || 1,
                    angle: 0
                };
                // This temporary shape is drawn using world coordinates.
                // The canvas context is already transformed in renderCanvas/drawElementOnCanvas.
                drawElementOnCanvas(tempShape);
            }
        }
    }
    function drawTemporaryPenStroke(strokeData) {
        if (!strokeData || strokeData.points.length < 1) return;
        ctx.save();
        // Ensure context is transformed for world coordinates (already done by renderCanvas if called before)
        // If not, apply transforms here:
        // ctx.translate(canvasAppState.offsetX, canvasAppState.offsetY);
        // ctx.scale(canvasAppState.zoom, canvasAppState.zoom);

        if (getStroke && typeof getStroke === 'function') {
            const strokeOptions = {
                size: strokeData.strokeWidth, // perfect-freehand size is in world units if context is scaled
                thinning: 0.6, smoothing: 0.5, streamline: 0.5,
                last: false, // false for live drawing
            };
            const strokePathPoints = getStroke(strokeData.points, strokeOptions);
            const pathData = getSvgPathFromStroke(strokePathPoints);
            const path2d = new Path2D(pathData);
            ctx.fillStyle = strokeData.strokeColor;
            ctx.globalAlpha = strokeData.opacity || 1.0;
            ctx.fill(path2d);
        } else { // Fallback simple line preview
            ctx.beginPath();
            ctx.moveTo(strokeData.points[0][0], strokeData.points[0][1]);
            for (let i = 1; i < strokeData.points.length; i++) {
                ctx.lineTo(strokeData.points[i][0], strokeData.points[i][1]);
            }
            ctx.strokeStyle = strokeData.strokeColor;
            ctx.lineWidth = strokeData.strokeWidth;
            ctx.globalAlpha = strokeData.opacity || 1.0;
            ctx.stroke();
        }
        ctx.restore();
    }


    function handleCanvasMouseUp(e) {
        if (isPanning) {
            isPanning = false;
            // Restore cursor based on mode/tool
            if (canvasAppState.currentMode === 'select') {
                 canvas.style.cursor = selectedElement ? 'move' : 'default';
            } else {
                 setActiveTool(canvasAppState.currentTool); // This will set cursor
            }
            return;
        }

        if (canvasAppState.currentMode === 'select') {
            if (isDraggingSelection && selectedElement) {
                // Finalize position - already done in mousemove
                // If it was a pen stroke, clean up temporary data
                if (selectedElement.type === 'pen') {
                    delete selectedElement.originalPoints;
                }
                pushToHistory("Move Element");
            }
            isDraggingSelection = false;
            // Don't deselect here, click on empty space to deselect (handled in mousedown)
        } else { // Draw mode
            if (!isDrawing) return;
            isDrawing = false;
            const { x: worldX, y: worldY } = getCanvasCoordinates(e);

            if (canvasAppState.currentTool === 'pen' && currentPenStroke) {
                if (currentPenStroke.points.length > 1) {
                    // Smooth the final stroke if desired
                    const finalStrokeOptions = { size: currentPenStroke.strokeWidth, thinning: 0.6, smoothing: 0.5, streamline: 0.5, last: true };
                    // currentPenStroke.points = getStroke(currentPenStroke.points, finalStrokeOptions); // This changes structure, careful
                    // For now, just use raw points.
                    canvasElements.push(currentPenStroke);
                    pushToHistory("Draw Pen");
                }
                currentPenStroke = null;
            } else if (['rectangle', 'ellipse'].includes(canvasAppState.currentTool)) {
                const width = Math.abs(worldX - startDragX);
                const height = Math.abs(worldY - startDragY);
                if (width > 5 && height > 5) { // Minimum size to add shape
                    const toolDefaults = canvasAppState.toolDefaults[canvasAppState.currentTool] || {};
                    const newElement = {
                        id: TB.utils.uniqueId(`${canvasAppState.currentTool}_`), type: canvasAppState.currentTool,
                        x: Math.min(startDragX, worldX), y: Math.min(startDragY, worldY),
                        width: width, height: height,
                        strokeColor: canvasAppState.strokeColor,
                        fill: canvasAppState.fillColor,
                        strokeWidth: canvasAppState.strokeWidth,
                        opacity: toolDefaults.opacity || 1.0,
                        fillStyle: toolDefaults.fillStyle || (canvasAppState.currentTool === 'rectangle' ? 'solid' : 'hachure'),
                        roughness: toolDefaults.roughness || 1,
                        angle: 0
                    };
                    canvasElements.push(newElement);
                    pushToHistory(`Draw ${canvasAppState.currentTool}`);
                }
            }
        }
        renderCanvas(); // Final render after operation
    }

    function handleCanvasMouseLeave(e) {
        // If actively drawing or panning, treat mouse leave as a mouse up to finalize action
        if (isDrawing || isPanning || isDraggingSelection) {
            handleCanvasMouseUp(e); // Pass the event for coordinates if needed, though often not relevant for leave
        }
        isDrawing = false;
        isPanning = false;
        isDraggingSelection = false;
        // Restore cursor if needed
        if (canvasAppState.currentMode === 'draw' && canvasAppState.currentTool) {
            setActiveTool(canvasAppState.currentTool); // Resets cursor
        } else {
            canvas.style.cursor = 'default';
        }
    }

    function handleCanvasWheel(e) {
        e.preventDefault(); // Prevent page scroll

        // Get mouse position in world coordinates *before* zoom change.
        // This is the point we want to zoom towards/away from.
        const { x: mouseWorldX, y: mouseWorldY } = getCanvasCoordinates(e);

        const zoomIntensity = 0.1;
        const direction = e.deltaY < 0 ? 1 : -1; // 1 for zoom in, -1 for zoom out
        const oldZoom = canvasAppState.zoom;
        const newZoom = Math.max(0.05, Math.min(20, oldZoom * (1 + direction * zoomIntensity)));

        // Calculate new offsets to keep mouseWorldX, mouseWorldY at the same view position
        // mouseWorldX = (viewX - newOffsetX) / newZoom  => newOffsetX = viewX - mouseWorldX * newZoom
        // viewX = mouseWorldX * oldZoom + oldOffsetX (current viewX of the mouse pointer)
        // newOffsetX = (mouseWorldX * oldZoom + oldOffsetX) - mouseWorldX * newZoom
        // newOffsetX = oldOffsetX + mouseWorldX * (oldZoom - newZoom)

        canvasAppState.offsetX = canvasAppState.offsetX + mouseWorldX * (oldZoom - newZoom);
        canvasAppState.offsetY = canvasAppState.offsetY + mouseWorldY * (oldZoom - newZoom);
        canvasAppState.zoom = newZoom;

        // Update text input overlay if visible due to zoom
        if (textInputOverlayEl.style.display !== 'none' && currentTextElementData) {
            const viewX = currentTextElementData.startX * canvasAppState.zoom + canvasAppState.offsetX;
            const viewY = currentTextElementData.startY * canvasAppState.zoom + canvasAppState.offsetY;
            const panelRect = document.querySelector('.canvas-panel').getBoundingClientRect();
            textInputOverlayEl.style.left = `${viewX + panelRect.left}px`;
            textInputOverlayEl.style.top =  `${viewY + panelRect.top}px`;
            textInputOverlayEl.style.fontSize = `${currentTextElementData.fontSize * canvasAppState.zoom}px`;
        }

        renderCanvas();
    }

    // --- Touch Event Handlers (basic mirror of mouse, single touch) ---
    let lastTouch = null; // For calculating movement delta for panning in touch

    function handleCanvasTouchStart(e) {
        if (e.touches.length > 1) { // Handle multi-touch for pinch-zoom later if needed
            isPanning = false; isDrawing = false; isDraggingSelection = false; // Stop other ops
            return;
        }
        e.preventDefault();
        const touch = e.touches[0];
        lastTouch = { clientX: touch.clientX, clientY: touch.clientY }; // For panning
        // Simulate mouse down with the first touch
        handleCanvasMouseDown({ ...touch, button: 0, preventDefault: () => {} }); // Spread touch and add button property
    }

    function handleCanvasTouchMove(e) {
        if (e.touches.length > 1) {
             // Basic Pinch Zoom (DEBUG THIS AREA CAREFULLY)
            // This is a simplified version and might need refinement
            // if (e.touches.length === 2 && !isDrawing && !isDraggingSelection) {
            //     handlePinchZoom(e); // Implement handlePinchZoom
            //     lastTouch = null; // Reset for next single touch
            //     return;
            // }
            return; // For now, ignore multi-touch move after start
        }
        e.preventDefault();
        const touch = e.touches[0];

        // If panning was initiated by touch (e.g. drag empty space in select mode, or Ctrl+Touch),
        // we need to use the delta from lastTouch to update panStartViewX/Y for handleCanvasMouseMove's pan logic.
        if (isPanning && lastTouch) {
            // This part is tricky because handleCanvasMouseMove expects panStartViewX to be set.
            // We're essentially simulating the mouse's clientX/Y behavior.
            // The pan logic in handleCanvasMouseMove uses clientX/Y of the event.
            // So, we pass the touch event directly.
        }
        handleCanvasMouseMove({ ...touch, preventDefault: () => {} }); // Spread touch
        if(lastTouch) { // Update lastTouch for next move delta if panning.
            lastTouch = { clientX: touch.clientX, clientY: touch.clientY };
        }
    }

    function handleCanvasTouchEnd(e) {
        // If e.touches.length > 0, it means some fingers are still on screen (e.g., pinch ended)
        // For simplicity, we treat any touchend as finishing the current operation.
        e.preventDefault();
        // Use changedTouches to get the touch that was lifted
        const touch = e.changedTouches[0] || lastTouch || { clientX:0, clientY:0 }; // Fallback if changedTouches is empty
        handleCanvasMouseUp({ ...touch, button: 0, preventDefault: () => {} });
        lastTouch = null;
    }


    // --- Element Selection & Manipulation ---
    function getElementAtPosition(worldX, worldY) {
        // Iterate in reverse to select top-most element
        for (let i = canvasElements.length - 1; i >= 0; i--) {
            const el = canvasElements[i];
            if (isPointInsideElement(el, worldX, worldY)) {
                return el;
            }
        }
        return null;
    }

    function isPointInsideElement(element, worldX, worldY) {
        // DEBUG: This function is critical for selection. Add detailed checks.
        // Consider a small tolerance for selection, especially for lines/paths.
        const tolerance = 5 / canvasAppState.zoom; // 5px tolerance in view space, converted to world space

        // Apply inverse rotation if element is rotated (more complex, for simplicity assume no rotation for hit test first)
        // Or, rotate the point: P' = R(-angle) * (P - Center) + Center

        switch (element.type) {
            case 'rectangle':
            case 'image': // Images are also rectangular
                return worldX >= element.x - tolerance &&
                       worldX <= element.x + element.width + tolerance &&
                       worldY >= element.y - tolerance &&
                       worldY <= element.y + element.height + tolerance;
            case 'ellipse':
                // Check point in ellipse equation: ((x-h)^2 / a^2) + ((y-k)^2 / b^2) <= 1
                const cx = element.x + element.width / 2;
                const cy = element.y + element.height / 2;
                const rx = element.width / 2 + tolerance;
                const ry = element.height / 2 + tolerance;
                if (rx <= 0 || ry <= 0) return false;
                const term1 = Math.pow((worldX - cx) / rx, 2);
                const term2 = Math.pow((worldY - cy) / ry, 2);
                return term1 + term2 <= 1;
            case 'text':
                // Approximate bounding box for text. ctx.measureText is tricky with multiline & zoom.
                // For simplicity, using a rough estimate based on fontSize and text length.
                // A more accurate way involves rendering text to a hidden canvas or better metrics.
                if (!element.text) return false;
                const lines = element.text.split('\\n');
                const charWidthApprox = (element.fontSize || 16) * 0.6; // Very rough estimate
                const estWidth = Math.max(...lines.map(l => l.length)) * charWidthApprox + tolerance;
                const estHeight = lines.length * (element.fontSize || 16) * 1.2 + tolerance; // 1.2 for line height
                return worldX >= element.x - tolerance &&
                       worldX <= element.x + estWidth &&
                       worldY >= element.y - tolerance && // y is typically baseline start
                       worldY <= element.y + estHeight;
            case 'pen':
                // Bounding box check first for performance
                if (!element.points || element.points.length === 0) return false;
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                element.points.forEach(p => {
                    minX = Math.min(minX, p[0]);
                    minY = Math.min(minY, p[1]);
                    maxX = Math.max(maxX, p[0]);
                    maxY = Math.max(maxY, p[1]);
                });
                if (worldX < minX - tolerance || worldX > maxX + tolerance || worldY < minY - tolerance || worldY > maxY + tolerance) {
                    return false; // Outside bounding box
                }
                // More precise: Check distance to line segments or polygon defined by perfect-freehand outline
                // For now, bounding box is a coarse approximation.
                // A simple improvement: check distance to each segment.
                for (let i = 0; i < element.points.length - 1; i++) {
                    if (isPointNearLine(worldX, worldY, element.points[i], element.points[i+1], (element.strokeWidth / 2) + tolerance)) {
                        return true;
                    }
                }
                return false; // If only one point or no segment close enough
            default:
                return false;
        }
    }
    function isPointNearLine(px, py, startPt, endPt, maxDistance) {
        const [x1, y1] = startPt;
        const [x2, y2] = endPt;
        const L2 = Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2);
        if (L2 === 0) return Math.hypot(px - x1, py - y1) <= maxDistance; // Point, not line
        let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / L2;
        t = Math.max(0, Math.min(1, t)); // Clamp t to the segment
        const closestX = x1 + t * (x2 - x1);
        const closestY = y1 + t * (y2 - y1);
        return Math.hypot(px - closestX, py - closestY) <= maxDistance;
    }


    function deleteSelectedElement() {
        if (!selectedElement) return;
        canvasElements = canvasElements.filter(el => el.id !== selectedElement.id);
        selectedElement = null;
        pushToHistory("Delete Element");
        renderCanvas();
    }


    // --- Text Element Specific ---
    function showTextInputOverlay(worldX, worldY) {
        finalizeTextInput();
        const toolDefaults = canvasAppState.toolDefaults.text || {};
        currentTextElementData = {
            type: 'text', text: '',
            strokeColor: canvasAppState.strokeColor, // Use current, or default from tool
            fontSize: canvasAppState.fontSize || toolDefaults.fontSize,
            fontFamily: canvasAppState.fontFamily || toolDefaults.fontFamily,
            textAlign: toolDefaults.textAlign || 'left',
            opacity: toolDefaults.opacity || 1.0,
            angle: 0,
            // Store world coordinates where text input was initiated
            startX: worldX,
            startY: worldY
        };

        const viewX = worldX * canvasAppState.zoom + canvasAppState.offsetX;
        const viewY = worldY * canvasAppState.zoom + canvasAppState.offsetY;
        const panelRect = document.querySelector('.canvas-panel').getBoundingClientRect();

        textInputOverlayEl.style.left = `${viewX + panelRect.left}px`;
        textInputOverlayEl.style.top =  `${viewY + panelRect.top}px`;
        textInputOverlayEl.style.fontFamily = currentTextElementData.fontFamily;
        textInputOverlayEl.style.fontSize = `${currentTextElementData.fontSize * canvasAppState.zoom}px`; // Scale font for input field
        textInputOverlayEl.style.color = currentTextElementData.strokeColor;
        textInputOverlayEl.value = '';
        textInputOverlayEl.style.display = 'block';
        textInputOverlayEl.style.minWidth = '50px'; // ensure it's visible
        textInputOverlayEl.style.minHeight = `${currentTextElementData.fontSize * canvasAppState.zoom * 1.2}px`;
        textInputOverlayEl.focus();
    }

    function finalizeTextInput() {
        if (textInputOverlayEl.style.display === 'none' || !currentTextElementData) return;

        const text = textInputOverlayEl.value;
        if (text.trim()) { // Only add if text is not just whitespace
            const newElement = {
                id: TB.utils.uniqueId('text_'),
                ...currentTextElementData, // Includes type, color, font, etc.
                text: text,
                x: currentTextElementData.startX, // Use stored world coordinates
                y: currentTextElementData.startY,
                // width/height can be estimated later if needed for bounding boxes
            };
            delete newElement.startX; // Clean up temporary properties
            delete newElement.startY;
            canvasElements.push(newElement);
            pushToHistory("Add Text");
        }
        textInputOverlayEl.style.display = 'none';
        textInputOverlayEl.value = '';
        currentTextElementData = null;
        renderCanvas();
    }

    function handleTextInputKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            finalizeTextInput();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            textInputOverlayEl.style.display = 'none';
            textInputOverlayEl.value = '';
            currentTextElementData = null;
        }
        // Auto-resize textarea (simple version)
        setTimeout(() => {
            textInputOverlayEl.style.height = 'auto';
            textInputOverlayEl.style.width = 'auto';
            textInputOverlayEl.style.height = `${Math.max(textInputOverlayEl.scrollHeight, currentTextElementData.fontSize * canvasAppState.zoom * 1.2)}px`;
            textInputOverlayEl.style.width = `${Math.max(textInputOverlayEl.scrollWidth, 50)}px`;
        }, 0);
    }

    // --- Image Element Specific ---
    async function handleAddImageByURL() {
        if (canvasAppState.currentMode !== 'draw') setActiveMode('draw');
        finalizeTextInput();
        const imageUrl = await TB.ui.Modal.prompt({title: "Add Image by URL", placeholder: "Enter image URL", useTextArea: false});
        if (!imageUrl || !imageUrl.trim()) return;

        const loaderId = TB.ui.Loader.show("Loading image...");
        try {
            const imgObject = await loadImageAsync(imageUrl.trim());
            const aspectRatio = imgObject.width / imgObject.height;
            const defaultWidth = 200 / canvasAppState.zoom; // Target 200px width in current view
            const defaultHeight = defaultWidth / aspectRatio;

            // Center in current view (world coordinates)
            const viewCenterX = (canvas.width / (window.devicePixelRatio || 1)) / 2;
            const viewCenterY = (canvas.height / (window.devicePixelRatio || 1)) / 2;
            const worldCenterX = (viewCenterX - canvasAppState.offsetX) / canvasAppState.zoom;
            const worldCenterY = (viewCenterY - canvasAppState.offsetY) / canvasAppState.zoom;

            const imageElement = {
                id: TB.utils.uniqueId('image_'), type: 'image', src: imageUrl.trim(),
                x: worldCenterX - defaultWidth / 2, y: worldCenterY - defaultHeight / 2,
                width: defaultWidth, height: defaultHeight,
                imgObject: imgObject, // Store loaded Image object
                opacity: canvasAppState.toolDefaults.image?.opacity || 1.0, angle: 0
            };
            canvasElements.push(imageElement);
            pushToHistory("Add Image");
            renderCanvas();
            TB.ui.Toast.showSuccess("Image added.");
        } catch (err) {
            TB.logger.error("Failed to load image from URL:", imageUrl, err);
            TB.ui.Toast.showError("Could not load image from URL.");
        } finally {
            TB.ui.Loader.hide(loaderId);
        }
    }
    document.getElementById('toolImageBtn').addEventListener('click', handleAddImageByURL);
    // Utility to load an image
    function loadImageAsync(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "Anonymous"; // Attempt to avoid CORS issues for images from URLs
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = src;
        });
    }

    // --- Main Canvas Rendering Logic ---
    function renderCanvas() {
        const dpr = window.devicePixelRatio || 1;
        ctx.save();

        // Clear canvas with background color (respecting DPR)
        ctx.fillStyle = canvasAppState.viewBackgroundColor;
        ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr); // Fill using device-independent pixels

        // Apply pan and zoom AFTER DPR scaling is set up
        // The order is: scale(dpr), then translate(offsetX, offsetY), then scale(zoom)
        // However, offsetX/Y are already in view pixels.
        // The getCanvasCoordinates converts view to world. Drawing happens in world.
        // So, the context needs to be set up to map world to view.
        ctx.translate(canvasAppState.offsetX, canvasAppState.offsetY);
        ctx.scale(canvasAppState.zoom, canvasAppState.zoom);

        // Render all stored elements
        canvasElements.forEach(el => drawElementOnCanvas(el));

        // Draw selection highlight if an element is selected in 'select' mode
        if (canvasAppState.currentMode === 'select' && selectedElement) {
            drawSelectionHighlight(selectedElement);
        }

        ctx.restore(); // Restore to state before pan/zoom/DPR scaling for next frame
        updateUndoRedoButtons();
    }

    function drawElementOnCanvas(el) {
        // Assumes ctx is already transformed (pan, zoom) to draw in world coordinates.
        ctx.save();
        ctx.globalAlpha = el.opacity === undefined ? 1.0 : el.opacity;

        if (el.angle) { // Basic rotation handling
            const centerX = el.x + (el.width || 0) / 2;
            const centerY = el.y + (el.height || 0) / 2;
            ctx.translate(centerX, centerY);
            ctx.rotate(el.angle * Math.PI / 180);
            ctx.translate(-centerX, -centerY);
        }

        const stroke = el.strokeColor || canvasAppState.strokeColor;
        const fill = el.fill; // Can be 'transparent' or a color
        const strokeWidth = el.strokeWidth || canvasAppState.strokeWidth;

        // DEBUG: Check coordinates and dimensions here if displacement occurs.
        // console.log(`Drawing ${el.type} at x:${el.x}, y:${el.y}, w:${el.width}, h:${el.height} with zoom ${canvasAppState.zoom}`);

        switch (el.type) {
            case 'pen':
                if (el.points && el.points.length > 0) {
                    if (getStroke && typeof getStroke === 'function') {
                        const strokeOptions = {
                            size: strokeWidth, // perfect-freehand size is in world units when ctx is scaled
                            thinning: 0.6, smoothing: 0.5, streamline: 0.5,
                            last: true, // For finalized strokes
                        };
                        // Ensure points are in correct format for getStroke [[x,y,pressure],...]
                        const strokePathPoints = getStroke(el.points.map(p => [p[0], p[1], p[2] || 0.5]), strokeOptions);
                        const pathData = getSvgPathFromStroke(strokePathPoints);
                        const path2d = new Path2D(pathData);
                        ctx.fillStyle = stroke; // Perfect Freehand creates filled paths
                        ctx.fill(path2d);
                    } else { // Fallback if getStroke not available
                        ctx.beginPath();
                        ctx.moveTo(el.points[0][0], el.points[0][1]);
                        for (let i = 1; i < el.points.length; i++) {
                            ctx.lineTo(el.points[i][0], el.points[i][1]);
                        }
                        ctx.strokeStyle = stroke;
                        ctx.lineWidth = strokeWidth;
                        ctx.stroke();
                    }
                }
                break;
            case 'rectangle':
                if (roughCanvasInstance) {
                    roughCanvasInstance.rectangle(el.x, el.y, el.width, el.height, {
                        stroke: stroke,
                        fill: (fill && fill !== 'transparent') ? fill : undefined,
                        strokeWidth: strokeWidth,
                        fillStyle: el.fillStyle || 'solid',
                        roughness: el.roughness === undefined ? 1 : el.roughness,
                    });
                }
                break;
            case 'ellipse':
                if (roughCanvasInstance) {
                    roughCanvasInstance.ellipse(el.x + el.width / 2, el.y + el.height / 2, el.width, el.height, {
                        stroke: stroke,
                        fill: (fill && fill !== 'transparent') ? fill : undefined,
                        strokeWidth: strokeWidth,
                        fillStyle: el.fillStyle || 'hachure',
                        roughness: el.roughness === undefined ? 1 : el.roughness,
                    });
                }
                break;
            case 'text':
                ctx.fillStyle = stroke;
                ctx.font = `${el.fontSize || canvasAppState.fontSize}px ${el.fontFamily || canvasAppState.fontFamily}`;
                ctx.textAlign = el.textAlign || 'left';
                const lines = (el.text || "").split('\\n');
                const lineHeight = (el.fontSize || canvasAppState.fontSize) * 1.2;
                // Adjust Y for baseline. ForfillText's y is baseline.
                const textRenderYOffset = (el.fontSize || canvasAppState.fontSize) * 0.85; // Approx to align top of text with y
                lines.forEach((line, index) => {
                    ctx.fillText(line, el.x, el.y + textRenderYOffset + (index * lineHeight));
                });
                break;
            case 'image':
                if (el.imgObject && el.imgObject.complete) {
                    try {
                         ctx.drawImage(el.imgObject, el.x, el.y, el.width, el.height);
                    } catch (e) { // Handles tainted canvas error for cross-origin images if not properly handled by server/image
                        TB.logger.warn("Error drawing image (possibly tainted canvas):", el.src, e);
                        ctx.strokeStyle = 'red'; ctx.lineWidth = 1;
                        ctx.strokeRect(el.x, el.y, el.width, el.height);
                        ctx.fillText("Image Error", el.x + 5, el.y + 15);
                    }
                } else if (el.src && !el.imgObject) { // Lazy load if imgObject missing
                    loadImageAsync(el.src).then(img => {
                        el.imgObject = img;
                        renderCanvas(); // Redraw once loaded
                    }).catch(err => {
                        TB.logger.error("Failed to lazy-load image for drawing:", el.src, err);
                        el.imgObject = null; // Prevent repeated attempts
                        renderCanvas(); // Re-render to potentially show placeholder
                    });
                    // Draw placeholder while loading
                    ctx.strokeStyle = 'gray'; ctx.lineWidth = 1;
                    ctx.strokeRect(el.x, el.y, el.width, el.height);
                    ctx.fillText("Loading...", el.x + 5, el.y + 15);
                }
                break;
        }
        ctx.restore();
    }

    function drawSelectionHighlight(element) {
        if (!element) return;
        // Get bounding box of the element in world coordinates.
        // This needs to be accurate for each element type.
        let bbox = getElementBoundingBox(element);
        if (!bbox) return;

        ctx.save();
        // No need to re-apply pan/zoom, context is already transformed.
        ctx.strokeStyle = 'rgba(0, 123, 255, 0.8)'; // Blue selection color
        ctx.lineWidth = 1.5 / canvasAppState.zoom; // Keep line width visually constant
        ctx.setLineDash([6 / canvasAppState.zoom, 3 / canvasAppState.zoom]); // Dashed line, scaled with zoom

        // Apply element's rotation to the bounding box highlight
        if (element.angle) {
            const centerX = element.x + (element.width || 0) / 2; // Use element's center for bbox rotation
            const centerY = element.y + (element.height || 0) / 2;
            ctx.translate(centerX, centerY);
            ctx.rotate(element.angle * Math.PI / 180);
            ctx.translate(-centerX, -centerY);
        }

        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
        ctx.setLineDash([]);
        ctx.restore();

        // TODO: Add resize/rotate handles if needed
    }

    function getElementBoundingBox(element) {
        // Returns {x, y, width, height} in world coordinates
        // This is simplified. For pen strokes, it should be the tightest bounding box.
        // For rotated elements, this bbox is pre-rotation. The highlight drawing handles rotation.
        switch (element.type) {
            case 'rectangle':
            case 'ellipse': // Ellipse uses its defined x,y,width,height as bbox
            case 'image':
                return { x: element.x, y: element.y, width: element.width, height: element.height };
            case 'text':
                // Crude bbox for text, similar to hit test
                if (!element.text) return null;
                const lines = element.text.split('\\n');
                const charWidthApprox = (element.fontSize || 16) * 0.6;
                const estWidth = Math.max(...lines.map(l => l.length)) * charWidthApprox;
                const estHeight = lines.length * (element.fontSize || 16) * 1.2;
                return { x: element.x, y: element.y, width: estWidth, height: estHeight };
            case 'pen':
                if (!element.points || element.points.length === 0) return null;
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                element.points.forEach(p => {
                    minX = Math.min(minX, p[0]);
                    minY = Math.min(minY, p[1]);
                    maxX = Math.max(maxX, p[0]);
                    maxY = Math.max(maxY, p[1]);
                });
                const buffer = (element.strokeWidth || 2) / 2; // Add half stroke width as buffer
                return { x: minX - buffer, y: minY - buffer, width: (maxX - minX) + 2 * buffer, height: (maxY - minY) + 2 * buffer };
            default:
                return null;
        }
    }


    // --- Perfect Freehand Helpers (from docs, adapted) ---
    function getSvgPathFromStroke(strokePoints) {
        if (!strokePoints || strokePoints.length === 0) return '';
        const d = strokePoints.reduce(
            (acc, [x0, y0], i, arr) => {
                const [x1, y1] = arr[(i + 1) % arr.length];
                acc.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2);
                return acc;
            },
            ['M', ...strokePoints[0], 'Q'] // Start with M command, first point, then Q
        );
        d.push('Z'); // Close the path for a filled shape
        return d.join(' ');
    }
    // drawPath was integrated into drawElementOnCanvas for 'pen'

    // --- Settings Modal & Tool Defaults ---
    function openSettingsModal() {
        // Populate modal with current defaults from canvasAppState.toolDefaults
        for (const tool in canvasAppState.toolDefaults) {
            for (const prop in canvasAppState.toolDefaults[tool]) {
                const inputId = `default${tool.charAt(0).toUpperCase() + tool.slice(1)}${prop.charAt(0).toUpperCase() + prop.slice(1)}`;
                const inputEl = document.getElementById(inputId);
                if (inputEl) {
                    if (inputEl.type === 'color' || inputEl.type === 'text' || inputEl.tagName === 'SELECT') {
                        inputEl.value = canvasAppState.toolDefaults[tool][prop];
                    } else if (inputEl.type === 'number') {
                        inputEl.value = parseFloat(canvasAppState.toolDefaults[tool][prop]);
                    }
                } else {
                    // TB.logger.warn(`Settings input not found: ${inputId} for tool ${tool}, prop ${prop}`);
                }
            }
        }
        TB.ui.Modal.show({ id: 'settingsModal', target: '#settingsModal' });
    }

    function saveToolDefaults() {
        // Read values from modal inputs and save to canvasAppState.toolDefaults
        const inputs = document.querySelectorAll('#settingsModal .tool-config-group [data-tool]');
        inputs.forEach(input => {
            const tool = input.dataset.tool;
            const prop = input.dataset.prop;
            if (tool && prop) {
                if (!canvasAppState.toolDefaults[tool]) canvasAppState.toolDefaults[tool] = {};
                if (input.type === 'number') {
                    canvasAppState.toolDefaults[tool][prop] = parseFloat(input.value);
                } else {
                    canvasAppState.toolDefaults[tool][prop] = input.value;
                }
            }
        });
        TB.ui.Toast.showSuccess("Default settings saved.");
        TB.ui.Modal.getById('settingsModal').close();
        // If current tool's defaults changed, re-apply them
        if (canvasAppState.currentTool && canvasAppState.currentMode === 'draw') {
            setActiveTool(canvasAppState.currentTool);
        }
        // Consider persisting these defaults to localStorage or backend user profile
        // For now, they are part of the session data if saved.
    }


    // --- Session Management (New, Load, Save, Import, Export) ---
    function startNewSession(showToast = true) {
        currentSessionId = null;
        currentCanvasName = "Untitled Canvas";
        document.getElementById('canvasNameInput').value = currentCanvasName;
        canvasElements = [];
        selectedElement = null;
        textNotesContent = "";
        document.getElementById('textNotesArea').value = "";

        // Reset app state to a deep copy of defaults
        canvasAppState = JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE));

        // Update UI controls to reflect new state
        const currentTheme = TB.ui.theme.getCurrentMode ? TB.ui.theme.getCurrentMode() : 'light';
        canvasAppState.viewBackgroundColor = currentTheme === 'dark' ? (canvasAppState.toolDefaults?.viewBackgroundColorDark || '#1e1e1e') : DEFAULT_CANVAS_APP_STATE.viewBackgroundColor;
        document.getElementById('bgColorPicker').value = canvasAppState.viewBackgroundColor;

        // Set mode to draw and pen tool by default
        setActiveMode('draw'); // This will also call setActiveTool
        setActiveTool(canvasAppState.currentTool || 'pen'); // Ensure tool defaults are applied

        historyStack = [JSON.stringify([])]; // Initial empty state for history
        redoStack = [];
        renderCanvas(); // This also calls updateUndoRedoButtons
        if (showToast) TB.ui.Toast.showInfo("New canvas started.");
    }

    async function handleSaveSession() {
        if (!currentCanvasName.trim()) {
            TB.ui.Toast.showWarning("Please enter a canvas name.");
            document.getElementById('canvasNameInput').focus();
            return;
        }
        const loaderId = TB.ui.Loader.show("Saving session...");
        const serializableElements = canvasElements.map(el => {
            const { imgObject, ...rest } = el; return rest; // Strip non-serializable imgObject
        });
        const sessionData = {
            id: currentSessionId || TB.utils.uniqueId('canvas-session-'),
            name: currentCanvasName,
            canvas_elements: serializableElements,
            canvas_app_state: canvasAppState, // Includes modes, tool defaults, zoom/pan etc.
            text_notes: textNotesContent,
            // last_modified is set by server
        };
        currentSessionId = sessionData.id;

        try {
            const response = await TB.api.request(MOD_NAME, 'save_session', sessionData, 'POST');
            if (response.isSuccess()) {
                TB.ui.Toast.showSuccess("Session saved!");
                if (response.data && response.data.last_modified) {
                    // Update internal last_modified if server sends it back, for consistency
                    canvasAppState.last_modified = response.data.last_modified;
                }
            } else {
                TB.ui.Toast.showError(`Save error: ${response.info?.message || response.info?.help_text || 'Unknown error'}`);
            }
        } catch (err) {
            TB.logger.error("Save Session Error:", err);
            TB.ui.Toast.showError("Failed to save session due to a client-side error.");
        } finally {
            TB.ui.Loader.hide(loaderId);
        }
    }
    async function handleLoadSession() {
        const loaderId = TB.ui.Loader.show("Fetching session list...");
        try {
            const response = await TB.api.request(MOD_NAME, 'list_sessions', null, 'GET');
            TB.ui.Loader.hide(loaderId);

            if (response.isSuccess() && response.get()) {
                const sessions = response.get();
                if (sessions.length === 0) {
                    TB.ui.Toast.showInfo("No saved sessions found."); return;
                }
                // Sort sessions by last_modified descending if not already sorted
                sessions.sort((a, b) => (b.last_modified || 0) - (a.last_modified || 0));

                let modalContent = '<select id="sessionSelectModal" class="tb-input tb-w-full">';
                sessions.forEach(s => {
                    const dateStr = s.last_modified ? new Date(s.last_modified / 100000).toLocaleString() : 'N/A';
                    modalContent += `<option value="${s.id}">${TB.utils.escapeHtml(s.name)} (Saved: ${dateStr})</option>`;
                });
                modalContent += '</select>';

                TB.ui.Modal.show({
                    title: "Load Session", content: modalContent,
                    buttons: [
                        { text: "Cancel", action: modal => modal.close(), variant: 'secondary' },
                        { text: "Load", variant: 'primary', action: async modal => {
                            const selectedId = document.getElementById('sessionSelectModal').value;
                            modal.close();
                            if (selectedId) await actuallyLoadSessionData(selectedId);
                        }}
                    ]
                });
            } else {
                TB.ui.Toast.showError(`Error listing sessions: ${response.info?.message || 'Unknown error'}`);
            }
        } catch (err) {
            TB.ui.Loader.hide(loaderId); TB.logger.error("List Sessions Error:", err);
            TB.ui.Toast.showError("Failed to list sessions.");
        }
    }

    async function actuallyLoadSessionData(sessionId) {
        const loaderId = TB.ui.Loader.show("Loading session data...");
        try {
            const response = await TB.api.request(MOD_NAME, `load_session?session_id=${sessionId}`, null, 'GET');
            if (response.isSuccess() && response.get()) {
                const data = response.get();

                // Start with a fresh default state and merge loaded data
                // This helps if old sessions are missing new appState fields
                startNewSession(false); // false to suppress "New canvas started" toast

                currentSessionId = data.id;
                currentCanvasName = data.name;
                textNotesContent = data.text_notes || "";

                // Carefully merge canvas_app_state
                canvasAppState = {
                    ...JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE)), // Start with fresh defaults
                    ...(data.canvas_app_state || {}) // Overlay loaded state
                };
                 // Ensure toolDefaults are fully populated even if partially saved
                canvasAppState.toolDefaults = {
                    ...JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE.toolDefaults)),
                    ...(data.canvas_app_state?.toolDefaults || {})
                };


                document.getElementById('canvasNameInput').value = currentCanvasName;
                document.getElementById('textNotesArea').value = textNotesContent;

                // Update UI controls from the potentially merged canvasAppState
                document.getElementById('bgColorPicker').value = canvasAppState.viewBackgroundColor;
                document.getElementById('strokeColorPicker').value = canvasAppState.strokeColor;
                document.getElementById('fillColorPicker').value = canvasAppState.fillColor;
                document.getElementById('strokeWidthInput').value = canvasAppState.strokeWidth;

                setActiveMode(canvasAppState.currentMode || 'draw');
                setActiveTool(canvasAppState.currentTool || 'pen');

                // Restore elements (this also calls renderCanvas)
                await restoreElementsFromHistory(data.canvas_elements || []);
                // Set history for loaded data
                historyStack = [JSON.stringify(data.canvas_elements || [])];
                redoStack = [];
                updateUndoRedoButtons();

                TB.ui.Toast.showSuccess(`Session '${currentCanvasName}' loaded.`);
            } else {
                TB.ui.Toast.showError(`Error loading session: ${response.info?.message || 'Unknown error'}`);
            }
        } catch (err) {
            TB.logger.error("Load Session Data Error:", err);
            TB.ui.Toast.showError("Failed to process loaded session data.");
        } finally {
            TB.ui.Loader.hide(loaderId);
        }
    }

    function handleExportJSON() {
        finalizeTextInput(); // Ensure any pending text is captured
        const serializableElements = canvasElements.map(el => {
            const { imgObject, ...rest } = el; return rest;
        });
        const dataToExport = {
            id: currentSessionId || TB.utils.uniqueId('canvas-export-'),
            name: currentCanvasName,
            version: VERSION, // Add app version to export
            canvas_elements: serializableElements,
            canvas_app_state: canvasAppState,
            text_notes: textNotesContent,
            exported_at: new Date().toISOString()
        };
        const jsonString = JSON.stringify(dataToExport, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const safeFilename = currentCanvasName.replace(/[^a-z0-9_.-]/gi, '_').substring(0,50) || 'canvas_export';
        TB.utils.downloadBlob(blob, `${safeFilename}.json`);
        TB.ui.Toast.showSuccess("Canvas exported as JSON.");
    }

    function handleImportJSON(event) {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                const importedData = JSON.parse(e.target.result);
                if (importedData.canvas_elements && importedData.canvas_app_state) {
                    // Similar to loading a session:
                    startNewSession(false); // Reset to defaults first

                    currentSessionId = importedData.id || TB.utils.uniqueId('canvas-imported-');
                    currentCanvasName = importedData.name || "Imported Canvas";
                    textNotesContent = importedData.text_notes || "";

                    canvasAppState = {
                         ...JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE)),
                        ...(importedData.canvas_app_state || {})
                    };
                    canvasAppState.toolDefaults = {
                        ...JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE.toolDefaults)),
                        ...(importedData.canvas_app_state?.toolDefaults || {})
                    };


                    document.getElementById('canvasNameInput').value = currentCanvasName;
                    document.getElementById('textNotesArea').value = textNotesContent;
                    document.getElementById('bgColorPicker').value = canvasAppState.viewBackgroundColor;
                    document.getElementById('strokeColorPicker').value = canvasAppState.strokeColor;
                    document.getElementById('fillColorPicker').value = canvasAppState.fillColor;
                    document.getElementById('strokeWidthInput').value = canvasAppState.strokeWidth;

                    setActiveMode(canvasAppState.currentMode || 'draw');
                    setActiveTool(canvasAppState.currentTool || 'pen');

                    await restoreElementsFromHistory(importedData.canvas_elements || []);
                    historyStack = [JSON.stringify(importedData.canvas_elements || [])];
                    redoStack = [];
                    updateUndoRedoButtons();

                    TB.ui.Toast.showSuccess("Canvas imported successfully.");
                } else {
                    TB.ui.Toast.showError("Invalid JSON format. Missing canvas_elements or canvas_app_state.");
                }
            } catch (err) {
                TB.logger.error("Import JSON error:", err);
                TB.ui.Toast.showError("Failed to parse or process JSON file.");
            }
        };
        reader.readAsText(file);
        event.target.value = null; // Reset file input
    }

    // Initialize after TB.js is ready
    if (window.TB?.ready) { // TB.ready is a promise in modern tb.js
        window.TB.ready.then(initializeCanvasStudio);
    } else if (window.TB?.events?.on) { // Fallback for older tb.js or if TB.ready isn't there
         if (window.TB.config?.get('appRootId')) {
            initializeCanvasStudio();
        } else {
            window.TB.events.on('tbjs:initialized', initializeCanvasStudio, { once: true });
        }
    } else { // Absolute fallback
        document.addEventListener('tbjs:initialized', initializeCanvasStudio, { once: true });
         // Or window.addEventListener('load', ...) if tbjs:initialized is not guaranteed
    }
</script>
"""
