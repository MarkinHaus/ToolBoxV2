# toolboxv2/mods/Canvas.py
import asyncio
import base64
import contextlib
import json
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
try:
    from datetime import UTC, datetime, timedelta
except ImportError:
    from datetime import datetime, timedelta, timezone
    UTC = timezone.utc
from typing import Any

import markdown2
from pydantic import BaseModel
from pydantic import Field as PydanticField

from toolboxv2 import App, MainTool, RequestData, Result, get_app
from toolboxv2.utils.extras.base_widget import get_user_from_request

# --- Module Definition ---
MOD_NAME = Name = "Canvas"  # Renamed slightly for clarity if this is a new version
VERSION = "0.1.0"
export = get_app(f"widgets.{MOD_NAME}").tb

# --- Constants ---
SESSION_DATA_PREFIX = "enhancedcanvas_session"
SESSION_LIST_KEY_SUFFIX = "_list"
CANVAS_INTERNAL_BROADCAST_HANDLER_EVENT_PREFIX = "canvas_internal_sse_dispatcher_"


# --- Pydantic Models for Canvas Elements and Session Data ---

class CanvasElement(BaseModel):
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4().hex[:12]))
    type: str  # "pen", "rectangle", "ellipse", "text", "image"

    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    strokeColor: str = "#000000"
    strokeWidth: float = 2
    angle: float = 0.0

    points: list[list[float]] | None = None  # For pen: [[x, y, pressure], ...]

    fillStyle: str | None = "hachure"
    roughness: float | None = 1
    fill: str | None = None
    seed: int | None = None

    text: str | None = None
    fontSize: int = 16
    fontFamily: str = "Arial"
    textAlign: str = "left"

    src: str | None = None
    opacity: float = 1.0

    # For selection state, not persisted, but useful for client-side logic if sent back for debug
    # isSelected: Optional[bool] = False

    model_config = {"extra": "allow"}


class IdeaSessionData(BaseModel):
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Canvas"
    canvas_elements: list[CanvasElement] = []
    canvas_app_state: dict[str, Any] = {
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
        },
        "elementPresets": []
    }
    text_notes: str = ""
    last_modified: float = PydanticField(default_factory=lambda: float(uuid.uuid4().int & (1 << 32) - 1))


class Tools(MainTool):  # Removed EventManager for simplicity, as it was causing the issue. Direct SSE is better here.
    def __init__(self, app: App):
        self.name = MOD_NAME
        self.version = VERSION
        self.color = "GREEN"
        self.tools_dict = {"name": MOD_NAME, "Version": self.show_version}

        # Canvas specific state
        self.live_canvas_sessions: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self.active_user_previews: dict[str, dict[str, Any]] = defaultdict(dict)
        self.previews_lock = asyncio.Lock()

        MainTool.__init__(self, load=on_start, v=self.version, tool=self.tools_dict, name=self.name,
                          color=self.color, app=app)
        self.app.logger.info(f"Canvas Tools (v{self.version}) initialized for app {self.app.id}.")

    @property
    def db_mod(self):
        db = self.app.get_mod("DB", spec=Name)
        if db.mode.value != "CLUSTER_BLOB":
            db.edit_cli("CB")
        return db

    def _broadcast_to_canvas_listeners(self, canvas_id: str, event_type: str, data: dict[str, Any],
                                       originator_user_id: str | None = None):
        """
        Creates a broadcast coroutine and submits it to the app's dedicated
        async manager to be run in the background.
        This is now a non-blocking fire-and-forget operation.
        """

        async def broadcast_coro():
            if canvas_id not in self.live_canvas_sessions:
                return

            message_obj = {
                "event": event_type,
                "data": json.dumps({
                    "canvas_id": canvas_id,
                    "originator_user_id": originator_user_id,
                    **data
                })
            }

            listeners = list(self.live_canvas_sessions.get(canvas_id, []))

            for q in listeners:
                try:
                    # Non-blocking put. If the queue is full, the client is lagging,
                    # and it's better to drop a message than to block the server.
                    q.put_nowait(message_obj)
                except asyncio.QueueFull:
                    self.app.logger.warning(
                        f"SSE queue full for canvas {canvas_id}. Message '{event_type}' dropped for one client.")
                except Exception as e:
                    self.app.logger.error(f"Error putting message on SSE queue: {e}")

        # Use the app's robust background runner to execute immediately and not block the caller.
        self.app.run_bg_task(broadcast_coro)

    def show_version(self):
        self.app.logger.info(f"{self.name} Version: {self.version}")
        return self.version

    async def _get_user_specific_db_key(self, request: RequestData, base_key: str) -> str | None:
        # This logic is correct and can remain as is.

        user = await get_user_from_request(self.app, request)
        if user and user.uid:
            return f"{base_key}_{user.uid}"
        self.print("ok")
        # Fallback for public/guest access if you want to support it
        return f"{base_key}_public"


@export(mod_name=MOD_NAME, api=False, version=VERSION, name="on_start", initial=True)
async def on_start(self):  # Renamed from on_start to avoid conflict if MainTool calls it `load`
    self.app.logger.info(
        f"Initializing {self.name} v{self.version}")
    # UI Registration
    try:
        self.app.run_any(
            ("CloudM", "add_ui"),
            name=f"{MOD_NAME}UI_v{VERSION.replace('.', '_')}",
            title=f"Enhanced Canvas Studio v{VERSION}",
            path=f"/api/{MOD_NAME}/ui",
            description="Interactive Canvas with draw/move modes and enhanced configuration.",
            auth=True
        )
        self.app.logger.info(f"{self.name} UI (v{VERSION}) registered with CloudM.")
    except Exception as e:
        self.app.logger.error(f"Error registering UI for {self.name}: {e}", exc_info=True)

    self.app.logger.info(f"{self.name} (v{VERSION}) initialized successfully.")


# In FileWidget.py

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="markdown_to_svg", api_methods=['POST'],
        request_as_kwarg=True)
async def markdown_to_svg(self, request: RequestData, markdown_text: str = "", width: int = 400,
                          font_family: str = "sans-serif", font_size: int = 14,
                          bg_color: str = "#ffffff", text_color: str = "#000000") -> Result:
    """
    Converts a string of Markdown text into an SVG image.
    The SVG is returned as a base64 encoded data URL.
    This version uses a viewBox for better scalability and multi-line handling.
    """
    if request is None:
        return Result.default_user_error("Request data is missing.", 400)
    if not markdown_text and request.data:
        markdown_text = request.data.get("markdown_text", "")

    if not markdown_text:
        return Result.default_user_error("markdown_text cannot be empty.")

    try:
        # Convert Markdown to HTML
        html_content = markdown2.markdown(markdown_text, extras=["fenced-code-blocks", "tables", "strike"])

        # --- FIX for Multi-line text ---
        # The key is to NOT set a fixed height on the SVG itself, but to use a viewBox.
        # The client will determine the final rendered size.
        # The width of the div inside the foreignObject controls the line wrapping.

        # We still need a rough height for the viewBox.
        # Estimate height: (number of lines * line-height) + padding
        # A simple line-height estimate is font_size * 1.6
        line_height_estimate = font_size * 1.6
        num_lines_estimate = len(html_content.split('\n')) + html_content.count('<br') + html_content.count(
            '<p>') + html_content.count('<li>')
        estimated_height = (num_lines_estimate * line_height_estimate) + 40  # 20px top/bottom padding

        svg_template = f"""
        <svg viewBox="0 0 {width} {int(estimated_height)}" xmlns="http://www.w3.org/2000/svg">
            <foreignObject x="0" y="0" width="{width}" height="{int(estimated_height)}">
                <div xmlns="http://www.w3.org/1999/xhtml">
                    <style>
                        div {{
                            font-family: {font_family};
                            font-size: {font_size}px;
                            color: {text_color};
                            background-color: {bg_color};
                            padding: 10px;
                            border-radius: 5px;
                            line-height: 1.6;
                            width: {width - 20}px; /* Width minus padding */
                            word-wrap: break-word;
                            height: 100%;
                            overflow-y: auto; /* Allow scrolling if content overflows estimate */
                        }}
                        h1, h2, h3 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 1em; }}
                        pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 4px; overflow-x: auto; }}
                        code {{ font-family: monospace; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; }}
                        th {{ background-color: #f2f2f2; }}
                        blockquote {{ border-left: 4px solid #ccc; padding-left: 10px; color: #555; margin-left: 0; }}
                    </style>
                    {html_content}
                </div>
            </foreignObject>
        </svg>
        """

        svg_base64 = base64.b64encode(svg_template.encode('utf-8')).decode('utf-8')
        data_url = f"data:image/svg+xml;base64,{svg_base64}"

        # --- FIX for Editability ---
        # Return the original markdown text along with the SVG
        return Result.ok(data={"svg_data_url": data_url, "original_markdown": markdown_text})

    except Exception as e:
        self.app.logger.error(f"Error converting Markdown to SVG: {e}", exc_info=True)
        return Result.default_internal_error("Failed to convert Markdown to SVG.")

@export(mod_name=MOD_NAME, api=True, version=VERSION, name="ui", api_methods=['GET'])
async def get_main_ui(self, **kwargs) -> Result:
    # The HTML template will be named differently to reflect the new version
    html_content = ENHANCED_CANVAS_HTML_TEMPLATE_V0_1_0
    return Result.html(data=self.app.web_context() + html_content)


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="save_session", api_methods=['POST'], request_as_kwarg=True)
async def save_session(app: App, request: RequestData, data: dict[str, Any] | IdeaSessionData) -> Result:
    """
    Saves the entire state of a canvas session to the database.
    This is typically triggered by a user's explicit "Save" action.
    """
    if not data:
        return Result.default_user_error("Request data is missing.", 400)
    if request is None:
        return Result.default_user_error("Request data is missing.", 400)
    canvas_tool = app.get_mod(MOD_NAME)
    if not canvas_tool or not canvas_tool.db_mod:
        app.logger.error("Save failed: Canvas module or DB not available.")
        return Result.custom_error(info="Database module not available.", exec_code=503)

    user_db_key_base = await canvas_tool._get_user_specific_db_key(request, SESSION_DATA_PREFIX)
    if not user_db_key_base:
        return Result.default_user_error(info="User authentication required to save.", exec_code=401)

    try:
        # Validate the incoming data against the Pydantic model
        session_data_obj = IdeaSessionData(**data) if isinstance(data, dict) else data
    except Exception as e:
        app.logger.error(f"Invalid session data for save: {e}. Data: {str(data)[:500]}", exc_info=True)
        return Result.default_user_error(info=f"Invalid session data format: {e}", exec_code=400)

    # Update timestamp and construct the main session key
    if session_data_obj:
        session_data_obj.last_modified = datetime.now(UTC).timestamp()
    session_db_key = f"{user_db_key_base}_{session_data_obj.id}"

    # Save the full session object to the database
    canvas_tool.db_mod.set(session_db_key, session_data_obj.model_dump_json(exclude_none=True))
    app.logger.info(f"Saved session data for C:{session_data_obj.id}")

    # --- Update the session list metadata ---
    session_list_key = f"{user_db_key_base}{SESSION_LIST_KEY_SUFFIX}"
    try:
        list_res_obj = canvas_tool.db_mod.get(session_list_key)
        user_sessions = []
        if list_res_obj and not list_res_obj.is_error() and list_res_obj.get():
            list_content = list_res_obj.get()[0] if isinstance(list_res_obj.get(), list) else list_res_obj.get()
            user_sessions = json.loads(list_content)

        # Find and update the existing entry, or add a new one
        session_metadata = {
            "id": session_data_obj.id,
            "name": session_data_obj.name,
            "last_modified": session_data_obj.last_modified
        }
        found_in_list = False
        for i, sess_meta in enumerate(user_sessions):
            if sess_meta.get("id") == session_data_obj.id:
                user_sessions[i] = session_metadata
                found_in_list = True
                break
        if not found_in_list:
            user_sessions.append(session_metadata)

        canvas_tool.db_mod.set(session_list_key, json.dumps(user_sessions))
        app.logger.info(f"Updated session list for user key ending in ...{user_db_key_base[-12:]}")

    except Exception as e:
        app.logger.error(f"Failed to update session list for C:{session_data_obj.id}. Error: {e}", exc_info=True)
        # Non-fatal error; the main data was saved. We can continue.

    return Result.ok(
        info="Session saved successfully.",
        data={"id": session_data_obj.id, "last_modified": session_data_obj.last_modified}
    )


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

    user_sessions = []
    if list_res_obj and not list_res_obj.is_error() and list_res_obj.is_data():
        try:
            list_content = list_res_obj.get()  # Get can return list or string based on DB adapter
            json_str_to_load = list_content[0] if isinstance(list_content,
                                                             list) and list_content else list_content if isinstance(list_content, str | bytes) else "[]"
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

    if get_res_obj and get_res_obj.is_error():
        return Result.default_user_error(info="Session not found.", exec_code=404)

    get_res_obj = get_res_obj.get()

    if isinstance(get_res_obj, list):
        get_res_obj = get_res_obj[0]

    try:
        session_data_str = get_res_obj
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


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="export_canvas_json", api_methods=['POST'],
        request_as_kwarg=True)
async def export_canvas_json(self, request: RequestData, data: dict[str, Any]) -> Result:
    # This function remains largely the same
    try:
        session_data_to_export = IdeaSessionData(**data)
        filename = f"{session_data_to_export.name.replace(' ', '_') or 'canvas_export'}.json"
        return Result.file(data=session_data_to_export.model_dump(), filename=filename)
    except Exception as e:
        self.app.logger.error(f"Error preparing canvas JSON export: {e}", exc_info=True)
        return Result.default_user_error(info=f"Invalid data for JSON export: {e}", exec_code=400)


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="open_canvas_stream", api_methods=['GET'],
        request_as_kwarg=True)
async def stream_canvas_updates_sse(app: App, request: RequestData, canvas_id: str,
                                    client_id: str | None = None) -> Result:
    canvas_tool = app.get_mod(MOD_NAME)
    if not canvas_id:
        async def _error_gen(): yield {'event': 'error', 'data': json.dumps({'message': 'canvas_id is required'})}

        return Result.sse(stream_generator=_error_gen())

    session_client_id = client_id or str(uuid.uuid4().hex[:12])
    sse_client_queue = asyncio.Queue(maxsize=100)

    # Register the queue with the canvas session
    canvas_tool.live_canvas_sessions[canvas_id].append(sse_client_queue)
    app.logger.info(
        f"SSE: Client {session_client_id} connected to C:{canvas_id}. Listeners: {len(canvas_tool.live_canvas_sessions[canvas_id])}")

    async def event_generator() -> AsyncGenerator[dict[str, Any], None]:
        # Send initial data right away
        try:
            # Send a connection confirmation event
            await sse_client_queue.put({"event": "stream_connected", "data": json.dumps(
                {"message": "Connected", "canvasId": canvas_id, "clientId": session_client_id})})

            # Send the current state of other users' previews to the new client
            async with canvas_tool.previews_lock:
                if canvas_id in canvas_tool.active_user_previews:
                    for uid, pdata in canvas_tool.active_user_previews[canvas_id].items():
                        if uid != session_client_id:
                            await sse_client_queue.put({"event": "user_preview_update", "data": json.dumps(
                                {"canvas_id": canvas_id, "user_id": uid, "preview_data": pdata})})
        except Exception as e:
            app.logger.error(f"SSE: Error sending initial data: {e}")

        # Main event consumption loop
        try:
            p = .25
            while True:
                if p > 90: p = 30
                try:
                    # MODIFICATION: Wait for a message, but with a timeout shorter than the server's.
                    # This prevents the server from closing the connection due to inactivity.
                    await asyncio.sleep(0.1)
                    message = await asyncio.wait_for(sse_client_queue.get(), timeout=p)  # 60s is safer than 90s
                    p = .25
                    yield message
                    sse_client_queue.task_done()
                    print(p, session_client_id)
                except TimeoutError:
                    p += 0.01
                    # MODIFICATION: If no message arrives, send a ping to keep the connection alive.
                    yield {"event": "ping", "data": json.dumps({"timestamp": datetime.now(UTC).isoformat()})}
                    await asyncio.sleep(0.75)
        except asyncio.CancelledError:
            app.logger.info(f"SSE: Client {session_client_id} disconnected (stream cancelled).")
        finally:
            # The cleanup logic itself is fine. It will run when the generator exits.
            # Define the cleanup as a coroutine
            async def cleanup_coro():
                app.logger.info(f"SSE: Cleaning up for client {session_client_id}, C:{canvas_id}.")
                if canvas_id in canvas_tool.live_canvas_sessions:
                    with contextlib.suppress(ValueError):
                        canvas_tool.live_canvas_sessions[canvas_id].remove(sse_client_queue)
                    if not canvas_tool.live_canvas_sessions[canvas_id]:
                        del canvas_tool.live_canvas_sessions[canvas_id]

                preview_cleared = False
                async with canvas_tool.previews_lock:
                    if canvas_id in canvas_tool.active_user_previews and session_client_id in \
                        canvas_tool.active_user_previews[canvas_id]:
                        del canvas_tool.active_user_previews[canvas_id][session_client_id]
                        preview_cleared = True

                if preview_cleared:
                    # This broadcast needs to happen in the background without blocking cleanup
                    canvas_tool._broadcast_to_canvas_listeners(  # MODIFICATION: Calling the non-blocking version
                        canvas_id=canvas_id, event_type="clear_user_preview",
                        data={"user_id": session_client_id}, originator_user_id=session_client_id
                    )

            # Submit the cleanup coro to the non-blocking background runner
            app.run_bg_task(cleanup_coro)

    return Result.sse(stream_generator=event_generator())


@export(mod_name=MOD_NAME, api=True, version=VERSION, name="send_canvas_action", api_methods=['POST'],
        request_as_kwarg=True)
async def handle_send_canvas_action(app: App, request: RequestData, data: dict[str, Any]):
    """
    Handles incremental, real-time actions from clients (e.g., adding an element).
    It persists the change to the database and then broadcasts it to all live listeners.
    """
    canvas_tool = app.get_mod(MOD_NAME)
    if not canvas_tool or not canvas_tool.db_mod:
        return Result.default_internal_error("Canvas module or DB not loaded.")

    if not data:
        return Result.default_user_error("Request data is missing.", 400)

    canvas_id = data.get("canvas_id")
    action_type = data.get("action_type")
    action_payload = data.get("payload")
    user_id = data.get("user_id")

    if not all([canvas_id, action_type, user_id]) or action_payload is None:
        return Result.default_user_error("Request missing required fields.", 400)

    # --- Flow 1: Ephemeral 'preview' actions that DO NOT get persisted ---
    if action_type in ["preview_update", "preview_clear"]:
        sse_event_type = "user_preview_update" if action_type == "preview_update" else "clear_user_preview"
        sse_data = {"user_id": user_id}

        async with canvas_tool.previews_lock:
            if action_type == "preview_update":
                canvas_tool.active_user_previews[canvas_id][user_id] = action_payload
                sse_data["preview_data"] = action_payload
            elif user_id in canvas_tool.active_user_previews.get(canvas_id, {}):
                del canvas_tool.active_user_previews[canvas_id][user_id]

        # MODIFICATION: Call the non-blocking broadcast method. This returns immediately.
        canvas_tool._broadcast_to_canvas_listeners(
            canvas_id=canvas_id, event_type=sse_event_type,
            data=sse_data, originator_user_id=user_id
        )
        return Result.ok(info=f"'{action_type}' broadcasted.")

    # --- Flow 2: Persistent actions that modify the canvas state ---
    if action_type not in ["element_add", "element_update", "element_remove"]:
        return Result.default_user_error(f"Unknown persistent action_type: {action_type}", 400)

    # Load the full, current session state from the database
    user_db_key_base = await canvas_tool._get_user_specific_db_key(request, SESSION_DATA_PREFIX)
    session_db_key = f"{user_db_key_base}_{canvas_id}"
    try:
        db_result = canvas_tool.db_mod.get(session_db_key)
        if not db_result or db_result.is_error() or not db_result.get():
            return Result.default_user_error("Canvas session not found in database.", 404)

        session_data_str = db_result.get()[0] if isinstance(db_result.get(), list) else db_result.get()
        session_data = IdeaSessionData.model_validate_json(session_data_str)
    except Exception as e:
        app.logger.error(f"DB Load/Parse failed for C:{canvas_id}. Error: {e}", exc_info=True)
        return Result.default_internal_error("Could not load canvas data to apply changes.")

    # Apply the action to the in-memory Pydantic object
    if action_type == "element_add":
        session_data.canvas_elements.append(CanvasElement(**action_payload))
    elif action_type == "element_update":
        element_id = action_payload.get("id")
        for i, el in enumerate(session_data.canvas_elements):
            if el.id == element_id:
                session_data.canvas_elements[i] = el.model_copy(update=action_payload)
                break
    elif action_type == "element_remove":
        ids_to_remove = set(action_payload.get("ids", [action_payload.get("id")]))
        session_data.canvas_elements = [el for el in session_data.canvas_elements if el.id not in ids_to_remove]

    # Save the modified object back to the database
    session_data.last_modified = datetime.now(UTC).timestamp()
    canvas_tool.db_mod.set(session_db_key, session_data.model_dump_json(exclude_none=True))

    # Broadcast the successful, persisted action to all connected clients
    # MODIFICATION: Call the non-blocking broadcast method.
    canvas_tool._broadcast_to_canvas_listeners(
        canvas_id=canvas_id,
        event_type="canvas_elements_changed",
        data={"action": action_type, "element": action_payload},
        originator_user_id=user_id
    )

    # Clear the temporary preview of the user who made the change
    async with canvas_tool.previews_lock:
        if user_id in canvas_tool.active_user_previews.get(canvas_id, {}):
            del canvas_tool.active_user_previews[canvas_id][user_id]

    # MODIFICATION: Call the non-blocking broadcast method.
    canvas_tool._broadcast_to_canvas_listeners(
        canvas_id=canvas_id, event_type="clear_user_preview",
        data={"user_id": user_id}, originator_user_id=user_id
    )

    return Result.ok(info=f"Action '{action_type}' persisted and broadcast.")

ENHANCED_CANVAS_HTML_TEMPLATE_V0_1_0 = """
<title>Enhanced Canvas Studio v0.1.0</title>
<!-- Rough.js and Perfect Freehand will be loaded via CDN in the script module -->
<style>
    /*body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    */
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
<div id="presetManagementModal" class="tb-modal" style="display:none;">
    <div class="tb-modal-dialog tb-modal-lg">
        <div class="tb-modal-content">
            <div class="tb-modal-header">
                <h5 class="tb-modal-title">Manage Element Presets</h5>
                <button type="button" class="tb-btn-close" data-dismiss="modal" aria-label="Close" onclick="closePresetModal()"></button> <!-- Updated close method -->
            </div>
            <div class="tb-modal-body" style="max-height: 70vh; overflow-y: auto; padding: 15px;">
                <div class="tb-mb-3">
                    <button id="addNewPresetBtn" class="tb-btn tb-btn-success tb-btn-sm">
                        <span class="material-symbols-outlined tb-mr-1" style="font-size: 1em; vertical-align: middle;">add_circle</span>Add New Preset
                    </button>
                </div>
                <div id="presetListContainer" class="tb-mb-3">
                    <!-- Presets will be dynamically listed here, e.g.:
                    <div class="preset-item" data-preset-id="xyz">
                        <span>Preset Name (Tool Type)</span>
                        <div>
                            <button class="apply-preset-btn tb-btn tb-btn-xs tb-btn-primary">Apply</button>
                            <button class="edit-preset-btn tb-btn tb-btn-xs tb-btn-secondary">Edit</button>
                            <button class="delete-preset-btn tb-btn tb-btn-xs tb-btn-danger">Delete</button>
                        </div>
                    </div>
                    -->
                </div>
                <hr>
                <div id="presetEditFormContainer" style="display:none;">
                    <h4 id="presetFormTitle">Add New Preset</h4>
                    <input type="hidden" id="presetEditId">
                    <div class="tb-form-group tb-mb-2">
                        <label for="presetNameInput" class="tb-form-label tb-form-label-sm">Preset Name:</label>
                        <input type="text" id="presetNameInput" class="tb-input tb-input-sm">
                    </div>
                    <div class="tb-form-group tb-mb-2">
                        <label for="presetToolTypeSelect" class="tb-form-label tb-form-label-sm">For Tool Type:</label>
                        <select id="presetToolTypeSelect" class="tb-input tb-input-sm">
                            <option value="pen">Pen</option>
                            <option value="rectangle">Rectangle</option>
                            <option value="ellipse">Ellipse</option>
                            <option value="text">Text</option>
                        </select>
                    </div>
                    <div id="presetPropertiesFields" class="tb-mb-2">
                        <!-- Property fields (strokeColor, strokeWidth, etc.) will be dynamically added here based on tool type -->
                        <!-- Example for pen:
                        <div class="tb-form-group tb-mb-1">
                            <label class="tb-form-label tb-form-label-xs">Stroke Color:</label> <input type="color" data-prop="strokeColor">
                        </div>
                        <div class="tb-form-group tb-mb-1">
                            <label class="tb-form-label tb-form-label-xs">Stroke Width:</label> <input type="number" data-prop="strokeWidth" min="1" class="tb-input tb-input-xs">
                        </div>
                        -->
                    </div>
                    <button id="savePresetChangesBtn" class="tb-btn tb-btn-primary tb-btn-sm">Save Preset</button>
                    <button type="button" id="cancelPresetEditBtn" class="tb-btn tb-btn-secondary tb-btn-sm">Cancel</button>
                </div>
            </div>
            <div class="tb-modal-footer">
                <button type="button" class="tb-btn tb-btn-secondary" onclick="closePresetModal()">Close</button> <!-- Updated close method -->
            </div>
        </div>
    </div>
</div>
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
            <button id="toolEraserBtn" title="Eraser (E)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">ink_eraser</span></button>
            <button id="toolRectBtn" title="Rectangle (R)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">rectangle</span></button>
            <button id="toolEllipseBtn" title="Ellipse (O)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">circle</span></button>
            <button id="toolTextBtn" title="Text (T)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">title</span></button>
            <button id="toolImageBtn" title="Image" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">image</span></button>
            <label for="fileWidgetUploadInput" title="Upload File (Image, PDF)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon">
                <span class="material-symbols-outlined">upload_file</span>
            </label>
            <input type="file" id="fileWidgetUploadInput" accept="image/*,application/pdf" style="display: none;">
            <button id="toolMarkdownBtn" title="Add Markdown Text (M)" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">markdown</span></button>
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
            <button id="exportJsonBtn" title="Export JSON" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">file_download</span></button>
            <label for="importJsonInput" title="Import JSON" class="tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon"><span class="material-symbols-outlined">file_upload</span></label>
            <input type="file" id="importJsonInput" accept=".json" style="display: none;">
            <div id="darkModeToggleContainer" style="display: inline-flex; align-items: center;"></div>
            <button id="shareCanvasBtn"> </button>
        </div>
    </div>
    <div class="main-layout">
        <div class="canvas-panel">
            <canvas id="mainCanvas" style="height: 80vh; width: 100vw;"></canvas>
            <textarea id="textInputOverlay" style="display:none;"></textarea>
        </div>
        <div class="notes-panel none">
            <h3>Notes</h3>
            <textarea id="textNotesArea" placeholder="Type your notes here..."></textarea>
        </div>
    </div>
</div>
<div id="selectionContextToolbar" class="toolbar-group" style="position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: var(--tb-bg-secondary, #fff); padding: 5px 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); z-index: 100; display: none;">
    <!-- PDF Controls -->
    <div id="pdfControls" style="display: none; align-items: center; gap: 5px;">
        <button id="pdfPrevPageBtn" class="tb-btn tb-btn-sm tb-btn-icon" title="Previous Page"><span class="material-symbols-outlined">arrow_back_ios</span></button>
        <span id="pdfPageInfo">Page 1 / 1</span>
        <button id="pdfNextPageBtn" class="tb-btn tb-btn-sm tb-btn-icon" title="Next Page"><span class="material-symbols-outlined">arrow_forward_ios</span></button>
    </div>
    <!-- Scaling Controls -->
    <div id="scaleControls" style="display: flex; align-items: center; gap: 8px; margin-left: 15px;">
        <label for="scaleSlider" class="tb-text-xs">Scale:</label>
        <input type="range" id="scaleSlider" min="0.1" max="3" step="0.05" value="1" style="width: 120px;">
        <span id="scaleValue" class="tb-text-xs">100%</span>
    </div>
</div>
<!-- Settings Modal HTML (initially hidden) -->
<div id="settingsModal" class="tb-modal" style="display:none; width: 100%;">
    <div class="tb-modal-dialog tb-modal-lg">
        <div class="tb-modal-content">
            <div class="tb-modal-header">
                <h5 class="tb-modal-title">Default Tool Settings</h5>
                <button type="button" class="tb-btn-close none" data-dismiss="modal" aria-label="Close" onclick="closeSettingsModal()"></button>
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
                <button type="button" class="tb-btn tb-btn-secondary" onclick="closeSettingsModal()">Close</button>
                <button type="button" class="tb-btn tb-btn-primary" id="saveSettingsBtn">Save Defaults</button>
            </div>
        </div>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/roughjs@4.6.6/bundled/rough.min.js"></script>
<script type="module" defer>
//pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`;
  import { getStroke } from 'https://cdn.jsdelivr.net/npm/perfect-freehand@1.2.2/dist/esm/index.mjs';

  window.getStroke = getStroke; // Make globally reachable
  // rough is already global via its UMD bundle

    if (!window.TB) {
        console.error("TB (ToolBox Client-Side Library) is not loaded. Canvas UI cannot function.");
    }

    // --- Global State & Configuration ---
    let currentSessionId = null;
    let currentCanvasName = "Untitled Canvas";
    let canvasElements = [];
    let textNotesContent = "";
    let settingsModalInstance = null;
    let presetManagementModalInstance = null;

    let sseConnection = null;
    let sseCanvasId = null; // Track which canvas the SSE is for
    let localClientId = null; // Unique ID for this browser session/user

    // For remote user previews
    // Structure: { "remote_user_id_1": element_preview_data_1, "remote_user_id_2": element_preview_data_2 }
    let remoteUserPreviews = {};
    let sendPreviewDataThrottled; // For throttling preview updates


    const DEFAULT_CANVAS_APP_STATE = {
        viewBackgroundColor: "#ffffff",
        currentMode: "draw",
        currentTool: "pen",
        strokeColor: "#000000",
        fillColor: "transparent",
        strokeWidth: 2,
        fontFamily: "Arial",
        fontSize: 16,
        zoom: 1.0,
        offsetX: 0,
        offsetY: 0,
        toolDefaults: {
            pen: { strokeColor: "#000000", strokeWidth: 2, opacity: 1.0, previewOpacity: 0.8 },
            rectangle: { strokeColor: "#000000", fillColor: "#cccccc", strokeWidth: 2, fillStyle: "solid", roughness: 1, opacity: 1.0, previewOpacity: 0.6, seed: 0  },
            ellipse: { strokeColor: "#000000", fillColor: "#dddddd", strokeWidth: 2, fillStyle: "hachure", roughness: 1, opacity: 1.0, previewOpacity: 0.6, seed: 0  },
            text: { strokeColor: "#000000", fontSize: 16, fontFamily: "Arial", textAlign: "left", opacity: 1.0 },
            image: { opacity: 1.0 }
        },
        elementPresets: [
            { id: TB.utils.uniqueId('preset_'), name: "Fine Red Pen", toolType: "pen", properties: { strokeColor: "#FF0000", strokeWidth: 1, opacity: 1.0 } },
            { id: TB.utils.uniqueId('preset_'), name: "Blue Dashed Box", toolType: "rectangle", properties: { strokeColor: "#0000FF", strokeWidth: 2, fillStyle: "dashed", fillColor: "transparent", roughness: 0.5, opacity: 1.0 } }
        ]
    };
    let canvasAppState = JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE));

    // Canvas, context, and drawing related
    let canvas, ctx, roughCanvasInstance;
    let isDrawing = false;
    let currentPenStroke = null;
    let startDragX, startDragY;

    // Panning state
    let isPanning = false;
    let panStartViewX, panStartViewY;

    let erasedThisStroke = new Set(); // For eraser tool
    let isMarqueeSelecting = false;   // For multi-select tool
    let marqueeRect = { x: 0, y: 0, width: 0, height: 0 };

    // Selection and Moving state
    let selectedElements = [];
    let isDraggingSelection = false;
    let selectionDragStartWorldX, selectionDragStartWorldY;
    // selectedElementOriginalX, selectedElementOriginalY removed, use per-element originalDragX/Y

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

        localClientId = localStorage.getItem('canvasClientId');
        if (!localClientId) {
            localClientId = TB.utils.uniqueId('client_');
            localStorage.setItem('canvasClientId', localClientId);
        }
        TB.logger.info(`Canvas Client ID: ${localClientId}`);

        // Initialize throttled function for sending previews
        sendPreviewDataThrottled = TB.utils.throttle(sendCurrentPreviewDataToServer, 100);

        const canvasNameInputEl = document.getElementById('canvasNameInput');
        const newSessionBtnEl = document.getElementById('newSessionBtn');
        const saveSessionBtnEl = document.getElementById('saveSessionBtn');
        const loadSessionBtnEl = document.getElementById('loadSessionBtn');
        const exportJsonBtnEl = document.getElementById('exportJsonBtn');
        const importJsonInputEl = document.getElementById('importJsonInput');
        const textNotesAreaEl = document.getElementById('textNotesArea');
        const settingsBtnEl = document.getElementById('settingsBtn');
        const saveSettingsBtnEl = document.getElementById('saveSettingsBtn');
        const managePresetsBtnEl = document.getElementById('managePresetsBtn');

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
            eraser: document.getElementById('toolEraserBtn'),
             // image tool is handled separately now
        };

        const fileWidgetUploadInputEl = document.getElementById('fileWidgetUploadInput');
        fileWidgetUploadInputEl.addEventListener('change', (e) => handleFileUpload(e.target.files));

         const markdownBtnEl = document.getElementById('toolMarkdownBtn');
        markdownBtnEl.addEventListener('click', handleAddMarkdown);

        // NEW: Add event listeners for the new contextual controls
        const pdfPrevPageBtn = document.getElementById('pdfPrevPageBtn');
        const pdfNextPageBtn = document.getElementById('pdfNextPageBtn');
        const scaleSlider = document.getElementById('scaleSlider');
        pdfPrevPageBtn.addEventListener('click', () => changeSelectedPdfPage(-1));
        pdfNextPageBtn.addEventListener('click', () => changeSelectedPdfPage(1));
        scaleSlider.addEventListener('input', (e) => scaleSelectedElements(parseFloat(e.target.value)));

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
            ctx.resetTransform();
            ctx.scale(dpr, dpr);
            renderCanvas();
        }
        window.addEventListener('resize', resizeCanvas);
        setTimeout(resizeCanvas, 50);

        canvasNameInputEl.addEventListener('input', () => currentCanvasName = canvasNameInputEl.value);
        newSessionBtnEl.addEventListener('click', startNewSession);
        saveSessionBtnEl.addEventListener('click', handleSaveSession);
        loadSessionBtnEl.addEventListener('click', handleLoadSession);
        exportJsonBtnEl.addEventListener('click', handleExportJSON);
        importJsonInputEl.addEventListener('change', handleImportJSON);
        textNotesAreaEl.addEventListener('input', () => {
            textNotesContent = textNotesAreaEl.value;
        });
        settingsBtnEl.addEventListener('click', openSettingsModal);
        saveSettingsBtnEl.addEventListener('click', saveToolDefaults);
        if(managePresetsBtnEl) managePresetsBtnEl.addEventListener('click', openPresetManagementModal);


        Object.entries(activeModeButtons).forEach(([modeName, btn]) => {
            btn.addEventListener('click', () => setActiveMode(modeName));
        });
        Object.entries(activeToolButtons).forEach(([toolName, btn]) => {
            btn.addEventListener('click', () => setActiveTool(toolName));
        });
        const toolImageBtnEl = document.getElementById('toolImageBtn');

        if (toolImageBtnEl) {
            toolImageBtnEl.addEventListener('click', handleAddImageByURL);
        }

        strokeColorPickerEl.addEventListener('input', (e) => {
            const newColor = e.target.value;
            canvasAppState.strokeColor = newColor;
            if (selectedElements.length > 0) {
                let changed = false;
                selectedElements.forEach(selEl => {
                    if (selEl.strokeColor !== newColor) { selEl.strokeColor = newColor; changed = true; }
                });
                if (changed) { pushToHistory("Change Stroke Color"); renderCanvas(); }
            }
        });
        fillColorPickerEl.addEventListener('input', (e) => {
            const newFill = e.target.value;
            canvasAppState.fillColor = newFill;
            if (selectedElements.length > 0) {
                let changed = false;
                selectedElements.forEach(selEl => {
                    if (selEl.type === 'rectangle' || selEl.type === 'ellipse') {
                        if (selEl.fill !== newFill) { selEl.fill = newFill; changed = true; }
                    }
                });
                if (changed) { pushToHistory("Change Fill Color"); renderCanvas(); }
            }
        });
         strokeWidthInputEl.addEventListener('input', (e) => { // Combined original and new logic
            const newWidth = parseInt(e.target.value, 10);
            canvasAppState.strokeWidth = newWidth;
            if (selectedElements.length > 0) { // For multi-select
                let changed = false;
                selectedElements.forEach(selEl => {
                    if (selEl.strokeWidth !== newWidth) { selEl.strokeWidth = newWidth; changed = true; }
                });
                if (changed) { pushToHistory("Change Stroke Width Multiple"); renderCanvas(); }
            }
            // Kept old single selectedElement logic in case it's still used somewhere, but should be covered by above.
            // if(selectedElement && selectedElement.strokeWidth !== newWidth) { selectedElement.strokeWidth = newWidth; pushToHistory("Change Stroke Width Single"); renderCanvas(); }
        });
        bgColorPickerEl.addEventListener('input', (e) => {
            canvasAppState.viewBackgroundColor = e.target.value;
            renderCanvas();
        });

        undoBtnEl.addEventListener('click', undo);
        redoBtnEl.addEventListener('click', redo);
        document.addEventListener('keydown', handleGlobalKeyDown);

        if (TB.ui && TB.ui.Modal && document.getElementById('presetManagementModal')) {
            presetManagementModalInstance = TB.ui.Modal.getById('presetManagementModal') || new TB.ui.Modal({
                id: 'presetManagementModal', target: '#presetManagementModal',
            });
        } else {
            console.error("Preset management modal HTML or TB.ui.Modal not found.");
        }
        if (TB.ui && TB.ui.Modal && document.getElementById('settingsModal')) {
            settingsModalInstance = TB.ui.Modal.getById('settingsModal') || new TB.ui.Modal({
                id: 'settingsModal', target: '#settingsModal',
            });
        } else {
            console.error("Settings modal HTML or TB.ui.Modal not found.");
        }

        document.getElementById('addNewPresetBtn')?.addEventListener('click', showPresetEditForm);
        document.getElementById('cancelPresetEditBtn')?.addEventListener('click', hidePresetEditForm);
        document.getElementById('savePresetChangesBtn')?.addEventListener('click', saveCurrentPresetFromForm);
        document.getElementById('presetToolTypeSelect')?.addEventListener('change', () => populatePresetPropertyFields()); // Pass no args to use selected value

        const shareBtnEl = document.getElementById('shareCanvasBtn');

        shareBtnEl.title = 'Share & Collaborate';
        shareBtnEl.className = 'tb-btn tb-btn-secondary tb-btn-sm tb-btn-icon';
        shareBtnEl.innerHTML = '<span class="material-symbols-outlined">share</span>';
        shareBtnEl.addEventListener('click', handleShareCanvas);

        canvas.addEventListener('dblclick', handleCanvasDoubleClick);
        canvas.addEventListener('mousedown', handleCanvasMouseDown);
        canvas.addEventListener('mousemove', handleCanvasMouseMove);
        canvas.addEventListener('mouseup', handleCanvasMouseUp);
        canvas.addEventListener('mouseleave', handleCanvasMouseLeave);
        canvas.addEventListener('wheel', handleCanvasWheel, { passive: false });
        canvas.addEventListener('touchstart', handleCanvasTouchStart, { passive: false });
        canvas.addEventListener('touchmove', handleCanvasTouchMove, { passive: false });
        canvas.addEventListener('touchend', handleCanvasTouchEnd);
        canvas.addEventListener('touchcancel', handleCanvasTouchEnd);
        textInputOverlayEl.addEventListener('blur', finalizeTextInput);
        textInputOverlayEl.addEventListener('keydown', handleTextInputKeyDown);

        TB.events.on('theme:changed', (themeData) => {
            const isDark = themeData.mode === 'dark';
            const lightBg = DEFAULT_CANVAS_APP_STATE.viewBackgroundColor;
            const darkBg = canvasAppState.toolDefaults?.viewBackgroundColorDark || '#1e1e1e';
            if (canvasAppState.viewBackgroundColor === lightBg && isDark) {
                 bgColorPickerEl.value = darkBg;
            } else if (canvasAppState.viewBackgroundColor === darkBg && !isDark) {
                 bgColorPickerEl.value = lightBg;
            }
            bgColorPickerEl.dispatchEvent(new Event('input'));
        });

        startNewSession();
        TB.logger.info("Canvas Studio v0.1.0: Initialized.");
    }

    // --- Global Keydown Handler ---
    function handleGlobalKeyDown(e) {
        if (document.activeElement === textInputOverlayEl ||
            // document.activeElement === textNotesAreaEl ||
            document.activeElement === document.getElementById('canvasNameInput') ||
            (presetManagementModalInstance && presetManagementModalInstance.isOpen && presetManagementModalInstance.isVisible) ||
            (settingsModalInstance && settingsModalInstance.isOpen && settingsModalInstance.isVisible)) {
            if (e.key === 'Escape' && document.activeElement === textInputOverlayEl) textInputOverlayEl.blur();
            return;
        }

        if (e.ctrlKey || e.metaKey) {
            switch (e.key.toLowerCase()) {
                case 'z': e.preventDefault(); undo(); break;
                case 'y': e.preventDefault(); redo(); break;
                case 's': e.preventDefault(); handleSaveSession(); break;
            }
        } else {
            switch(e.key.toLowerCase()) {
                case 'd': setActiveMode('draw'); break;
                case 'v': case 's': setActiveMode('select'); break;
                case 'p': if(canvasAppState.currentMode === 'draw') setActiveTool('pen'); break;
                case 'r': if(canvasAppState.currentMode === 'draw') setActiveTool('rectangle'); break;
                case 'o': if(canvasAppState.currentMode === 'draw') setActiveTool('ellipse'); break;
                case 't': if(canvasAppState.currentMode === 'draw') setActiveTool('text'); break;
                case 'delete': case 'backspace':
                    if (canvasAppState.currentMode === 'select' && selectedElements.length > 0) {
                        e.preventDefault();
                        deleteSelectedElements();
                    }
                    break;
            }
        }
    }

    function handleCanvasDoubleClick(e) {
    const coords = getCanvasCoordinates(e);
    if (coords.error) return;
    const { x: worldX, y: worldY } = coords;

    const clickedElement = getElementAtPosition(worldX, worldY);

    if (clickedElement && clickedElement.isMarkdown) {
        // If we double-clicked a Markdown element, open the editor
        // We need to select it first so the editor knows which element to update
        selectedElements = [clickedElement];
        renderCanvas(); // Show selection highlight
        handleAddMarkdown(); // This will now pre-fill the modal
    }
}

    // --- History Management (Undo/Redo) ---
    function pushToHistory(actionName = "unknown") {
        const serializableElements = canvasElements.map(el => {
            const { imgObject, ...rest } = el;
            return rest;
        });
        historyStack.push(JSON.stringify(serializableElements));
        if (historyStack.length > MAX_HISTORY_SIZE) {
            historyStack.shift();
        }
        redoStack = [];
        updateUndoRedoButtons();
    }

    async function restoreElementsFromHistory(elementsData) {
        const newElements = [];
        for (const elData of elementsData) {
            const newEl = { ...elData };
            if (newEl.type === 'image' && newEl.src) {
                try {
                    newEl.imgObject = await loadImageAsync(newEl.src);
                } catch (err) {
                    TB.logger.error("Failed to reload image during history restore/load:", newEl.src, err);
                    newEl.imgObject = null;
                }
            }
            newElements.push(newEl);
        }
        canvasElements = newElements;
        selectedElements = []; // Deselect after history change or load
        renderCanvas();
    }

    async function undo() {
        if (historyStack.length <= 1 && canvasElements.length === 0) return;

        if (historyStack.length > 1) {
            redoStack.push(historyStack.pop());
            const prevState = JSON.parse(historyStack[historyStack.length - 1]);
            await restoreElementsFromHistory(prevState);
        } else if (historyStack.length === 1 && canvasElements.length > 0) {
            // This means we are at the initial state (which might have elements if loaded)
            // and want to undo to an empty canvas.
            redoStack.push(historyStack.pop()); // Save the current state (which has elements)
            canvasElements = []; // Make canvas empty
            historyStack.push(JSON.stringify([])); // Push the new empty state as "current" for history
            await restoreElementsFromHistory([]); // This will call renderCanvas
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
        selectedElements = []; // Deselect when changing modes

        canvasAppState.currentMode = modeName;
        TB.logger.info(`Mode changed to: ${modeName}`);

        for (const [name, btn] of Object.entries(activeModeButtons)) {
            btn.classList.toggle('active', name === modeName);
            btn.classList.toggle('tb-btn-primary', name === modeName);
            btn.classList.toggle('tb-btn-secondary', name !== modeName);
        }
        document.getElementById('drawToolsGroup').style.display = (modeName === 'draw') ? 'flex' : 'none';
        const fillColorPickerParent = document.getElementById('fillColorPicker').parentElement;


        if (modeName === 'select') {
            canvas.style.cursor = 'default';
            setActiveTool(null);
            // In select mode, fill picker should generally be hidden unless a shape that uses it is selected
             if (fillColorPickerParent) fillColorPickerParent.style.display = 'none';

        } else { // 'draw' mode
            setActiveTool(canvasAppState.currentTool || 'pen');
        }
        renderCanvas();
    }

    function setActiveTool(toolName) {
        if (toolName === null) { // Deactivating draw tool (e.g., going to select mode)
             Object.values(activeToolButtons).forEach(btn => {
                btn.classList.remove('active', 'tb-btn-primary');
                btn.classList.add('tb-btn-secondary');
            });
            canvas.style.cursor = (canvasAppState.currentMode === 'select') ? 'default' : 'crosshair';
            canvasAppState.currentTool = null;
            // Hide fill color picker if no shape tool is active implicitly
            const fillColorPickerParent = document.getElementById('fillColorPicker').parentElement;
            if (fillColorPickerParent) fillColorPickerParent.style.display = 'none';
            return;
        }

        if (canvasAppState.currentMode !== 'draw') {
            setActiveMode('draw');
        }
        finalizeTextInput();
        canvasAppState.currentTool = toolName;

        for (const [name, btn] of Object.entries(activeToolButtons)) {
            const isActive = name === toolName;
            btn.classList.toggle('active', isActive);
            btn.classList.toggle('tb-btn-primary', isActive);
            btn.classList.toggle('tb-btn-secondary', !isActive);
        }

        const defaults = canvasAppState.toolDefaults[toolName];
        const fillColorPickerParent = document.getElementById('fillColorPicker').parentElement;

        if (defaults) {
            document.getElementById('strokeColorPicker').value = defaults.strokeColor || canvasAppState.strokeColor;
            canvasAppState.strokeColor = defaults.strokeColor || canvasAppState.strokeColor;

            if (toolName === 'rectangle' || toolName === 'ellipse') {
                document.getElementById('fillColorPicker').value = defaults.fillColor || canvasAppState.fillColor;
                canvasAppState.fillColor = defaults.fillColor || canvasAppState.fillColor;
                if(fillColorPickerParent) fillColorPickerParent.style.display = '';
            } else {
                 if(fillColorPickerParent) fillColorPickerParent.style.display = 'none';
            }

            document.getElementById('strokeWidthInput').value = defaults.strokeWidth || canvasAppState.strokeWidth;
            canvasAppState.strokeWidth = defaults.strokeWidth || canvasAppState.strokeWidth;

            if (toolName === 'text') {
                canvasAppState.fontFamily = defaults.fontFamily || canvasAppState.fontFamily;
                canvasAppState.fontSize = defaults.fontSize || canvasAppState.fontSize;
            }
        } else { // No defaults for this tool (e.g. 'image' or custom), hide fill picker
            if(fillColorPickerParent) fillColorPickerParent.style.display = 'none';
        }


        if (toolName === 'text') canvas.style.cursor = 'text';
        else if (toolName === 'pan') canvas.style.cursor = 'grab';
        else canvas.style.cursor = 'crosshair';
    }

    // --- Coordinate Transformation ---
    function getCanvasCoordinates(eventOrTouch) {
        const rect = canvas.getBoundingClientRect();
        let clientX, clientY;

        if (eventOrTouch.clientX !== undefined && eventOrTouch.clientY !== undefined) {
            clientX = eventOrTouch.clientX;
            clientY = eventOrTouch.clientY;
        } else if (eventOrTouch.touches && eventOrTouch.touches.length > 0) {
            clientX = eventOrTouch.touches[0].clientX;
            clientY = eventOrTouch.touches[0].clientY;
        } else if (eventOrTouch.changedTouches && eventOrTouch.changedTouches.length > 0) {
            clientX = eventOrTouch.changedTouches[0].clientX;
            clientY = eventOrTouch.changedTouches[0].clientY;
        } else {
            console.error("getCanvasCoordinates: Could not determine clientX/Y from event:", eventOrTouch);
            return { x: 0, y: 0, viewX: 0, viewY: 0, error: true }; // Indicate error
        }

        const viewX = clientX - rect.left;
        const viewY = clientY - rect.top;

        const zoom = canvasAppState.zoom || 1.0;
        const worldX = (viewX - canvasAppState.offsetX) / zoom;
        const worldY = (viewY - canvasAppState.offsetY) / zoom;
        return { x: worldX, y: worldY, viewX: viewX, viewY: viewY };
    }


    // --- Mouse Event Handlers ---
    function handleCanvasMouseDown(e) {
    if (e.button !== 0) return;
    e.preventDefault();
    finalizeTextInput();

    const coords = getCanvasCoordinates(e);
    if (coords.error) return;
    const { x: worldX, y: worldY, viewX, viewY } = coords;

    if ((e.ctrlKey || e.metaKey || e.button === 1) && canvasAppState.currentMode !== 'draw') {
        isPanning = true; panStartViewX = viewX; panStartViewY = viewY;
        canvas.style.cursor = 'grabbing'; return;
    }

    if (canvasAppState.currentMode === 'select') {
        const clickedElement = getElementAtPosition(worldX, worldY);
        if (clickedElement) {
            const isAlreadySelected = selectedElements.find(el => el.id === clickedElement.id);
            if (e.shiftKey) {
                if (isAlreadySelected) selectedElements = selectedElements.filter(el => el.id !== clickedElement.id);
                else selectedElements.push(clickedElement);
            } else {
                 if (!isAlreadySelected) selectedElements = [clickedElement];
            }
            isDraggingSelection = true;
            selectionDragStartWorldX = worldX;
            selectionDragStartWorldY = worldY;
            selectedElements.forEach(selEl => {
                selEl.originalDragX = selEl.x;
                selEl.originalDragY = selEl.y;
                if (selEl.type === 'pen') selEl.originalPoints = JSON.parse(JSON.stringify(selEl.points));
            });
        } else {
            if (!e.shiftKey) selectedElements = [];
            isMarqueeSelecting = true;
            marqueeRect = { startX: worldX, startY: worldY, endX: worldX, endY: worldY };
        }
        renderCanvas();
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
            if (currentPenStroke) sendPreviewDataThrottled(currentPenStroke);
        } else if (canvasAppState.currentTool === 'eraser') {
            erasedThisStroke.clear();
            const elementsUnderCursor = getElementAtPosition(worldX, worldY, true);
            if (elementsUnderCursor.length > 0) {
                elementsUnderCursor.forEach(el => erasedThisStroke.add(el.id));
                renderCanvas();
            }
        } else if (canvasAppState.currentTool === 'text') {
            isDrawing = false;
            showTextInputOverlay(worldX, worldY);
        }
    }
}

function handleCanvasMouseMove(e) {
    e.preventDefault();
    const coords = getCanvasCoordinates(e);
    if (coords.error) return;
    const { x: worldX, y: worldY, viewX, viewY } = coords;

    if (isPanning) {
        canvasAppState.offsetX += viewX - panStartViewX;
        canvasAppState.offsetY += viewY - panStartViewY;
        panStartViewX = viewX; panStartViewY = viewY;
        renderCanvas(); return;
    }

    if (canvasAppState.currentMode === 'select') {
        if (isDraggingSelection) {
            const deltaX = worldX - selectionDragStartWorldX;
            const deltaY = worldY - selectionDragStartWorldY;
            selectedElements.forEach(selEl => {
                if (selEl.type === 'pen') {
                   if (selEl.originalPoints) selEl.points = selEl.originalPoints.map(p => [ p[0] + deltaX, p[1] + deltaY, p[2] ]);
                } else {
                   if (selEl.originalDragX !== undefined) selEl.x = selEl.originalDragX + deltaX;
                   if (selEl.originalDragY !== undefined) selEl.y = selEl.originalDragY + deltaY;
                }
            });
            const previewDragData = selectedElements.map(el => ({...el, originalDragX: undefined, originalDragY: undefined, originalPoints: undefined}));
            sendPreviewDataThrottled({ type: "group_drag", elements: previewDragData });
            renderCanvas();
        } else if (isMarqueeSelecting) {
            marqueeRect.endX = worldX; marqueeRect.endY = worldY;
            renderCanvas();
        } else {
            canvas.style.cursor = getElementAtPosition(worldX, worldY) ? 'move' : 'default';
        }
    } else { // Draw mode
        if (!isDrawing) return;
        if (canvasAppState.currentTool === 'pen' && currentPenStroke) {
            currentPenStroke.points.push([worldX, worldY, e.pressure || 0.5]);
            renderCanvas(); // Redraws everything
            ctx.save();
            ctx.translate(canvasAppState.offsetX, canvasAppState.offsetY);
            ctx.scale(canvasAppState.zoom, canvasAppState.zoom);
            drawTemporaryPenStroke(currentPenStroke); // Draw preview on top
            ctx.restore();
            sendPreviewDataThrottled(currentPenStroke);
        } else if (canvasAppState.currentTool === 'eraser') {
            const elementsUnderCursor = getElementAtPosition(worldX, worldY, true);
            let needsRender = false;
            elementsUnderCursor.forEach(el => {
                if (!erasedThisStroke.has(el.id)) { erasedThisStroke.add(el.id); needsRender = true; }
            });
            renderCanvas(); // Redraws ghosted elements
            ctx.save();
            ctx.translate(canvasAppState.offsetX, canvasAppState.offsetY);
            ctx.scale(canvasAppState.zoom, canvasAppState.zoom);
            ctx.beginPath();
            ctx.arc(worldX, worldY, (canvasAppState.strokeWidth / 2) + 2, 0, Math.PI * 2);
            ctx.strokeStyle = '#888'; ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 1.5 / canvasAppState.zoom;
            ctx.fill(); ctx.stroke();
            ctx.restore();
        } else if (['rectangle', 'ellipse'].includes(canvasAppState.currentTool)) {
            renderCanvas();
            ctx.save();
            ctx.translate(canvasAppState.offsetX, canvasAppState.offsetY);
            ctx.scale(canvasAppState.zoom, canvasAppState.zoom);
            const tempShape = {
                type: canvasAppState.currentTool,
                x: Math.min(startDragX, worldX), y: Math.min(startDragY, worldY),
                width: Math.abs(worldX - startDragX), height: Math.abs(worldY - startDragY),
                strokeColor: canvasAppState.strokeColor, fill: canvasAppState.fillColor, strokeWidth: canvasAppState.strokeWidth,
                opacity: canvasAppState.toolDefaults[canvasAppState.currentTool]?.previewOpacity || 0.6,
                fillStyle: canvasAppState.toolDefaults[canvasAppState.currentTool]?.fillStyle,
                roughness: canvasAppState.toolDefaults[canvasAppState.currentTool]?.roughness,
                seed: Math.floor(Math.random() * 2**31)
            };
            drawElementOnCanvas(tempShape);
            ctx.restore();
            sendPreviewDataThrottled(tempShape);
        }
    }
}

function handleCanvasMouseUp(e) {
    const coords = getCanvasCoordinates(e);
    if (isPanning) {
        isPanning = false;
        setActiveTool(canvasAppState.currentTool); // Resets cursor
        return;
    }
    if (isMarqueeSelecting) {
        const finalRect = {
            x: Math.min(marqueeRect.startX, marqueeRect.endX), y: Math.min(marqueeRect.startY, marqueeRect.endY),
            width: Math.abs(marqueeRect.endX - marqueeRect.startX), height: Math.abs(marqueeRect.endY - marqueeRect.startY)
        };
        const elementsInRect = getElementsInRect(finalRect);
        if (e.shiftKey) {
            elementsInRect.forEach(el => {
                const index = selectedElements.findIndex(sel => sel.id === el.id);
                if (index > -1) selectedElements.splice(index, 1);
                else selectedElements.push(el);
            });
        } else {
            selectedElements = elementsInRect;
        }
    }
    if (isDraggingSelection) {
        pushToHistory("Move Elements");
        selectedElements.forEach(selEl => {
            const { originalDragX, originalDragY, originalPoints, ...elementToSend } = selEl;
            sendActionToServer("element_update", elementToSend);
        });
        clearOwnPreviewOnServer();
    }
    if (isDrawing) {
        if (canvasAppState.currentTool === 'eraser' && erasedThisStroke.size > 0) {
            const idsToErase = Array.from(erasedThisStroke);
            canvasElements = canvasElements.filter(el => !idsToErase.includes(el.id));
            pushToHistory("Erase Elements");
            sendActionToServer("element_remove", { ids: idsToErase });
            clearOwnPreviewOnServer();
        } else if (canvasAppState.currentTool === 'pen' && currentPenStroke && currentPenStroke.points.length > 2) {
            canvasElements.push(currentPenStroke);
            pushToHistory("Draw Pen");
            sendActionToServer("element_add", currentPenStroke);
            clearOwnPreviewOnServer();
        } else if (['rectangle', 'ellipse'].includes(canvasAppState.currentTool)) {
            const width = Math.abs(coords.x - startDragX);
            const height = Math.abs(coords.y - startDragY);
            if (width > 2 && height > 2) {
                const defaults = canvasAppState.toolDefaults[canvasAppState.currentTool];
                const newElement = {
                    id: TB.utils.uniqueId(`${canvasAppState.currentTool}_`), type: canvasAppState.currentTool,
                    x: Math.min(startDragX, coords.x), y: Math.min(startDragY, coords.y),
                    width, height, strokeColor: canvasAppState.strokeColor, fill: canvasAppState.fillColor,
                    strokeWidth: canvasAppState.strokeWidth, opacity: defaults.opacity, fillStyle: defaults.fillStyle,
                    roughness: defaults.roughness, seed: Math.floor(Math.random() * 2**31)
                };
                canvasElements.push(newElement);
                pushToHistory(`Draw ${canvasAppState.currentTool}`);
                sendActionToServer("element_add", newElement);
                clearOwnPreviewOnServer();
            }
        }
    }

    // Reset all action states
    isDrawing = isPanning = isDraggingSelection = isMarqueeSelecting = false;
    currentPenStroke = null;
    erasedThisStroke.clear();
    renderCanvas();
}

     function checkAndJoinCollabSessionFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        const collabSessionId = urlParams.get('collab_session_id');
        if (collabSessionId) {
            TB.logger.info(`Found collab_session_id in URL: ${collabSessionId}. Attempting to load and join.`);
            // Load the session data first (this sets currentSessionId)
            // then connect to the stream.
            // This assumes collabSessionId is the same as a normal sessionId.
            // If they are different, the logic to load/join would need adjustment.
            actuallyLoadSessionData(collabSessionId).then(() => {
                if (currentSessionId === collabSessionId) { // Successfully loaded
                    connectToCanvasStream(collabSessionId);
                     // Remove from URL to prevent re-joining on refresh if user navigates away and back
                    // window.history.replaceState({}, document.title, window.location.pathname);
                } else {
                    TB.ui.Toast.showError(`Could not load shared canvas: ${collabSessionId}`);
                }
            }).catch(err => {
                 TB.ui.Toast.showError(`Error loading shared canvas: ${collabSessionId}`);
                 TB.logger.error("Error in actuallyLoadSessionData for collab join:", err);
            });
        }
    }

    function getElementsInRect(selectionRect) {
    return canvasElements.filter(el => {
        const bbox = getElementBoundingBox(el);
        // Standard Axis-Aligned Bounding Box (AABB) intersection test
        return bbox &&
            bbox.x < selectionRect.x + selectionRect.width &&
            bbox.x + bbox.width > selectionRect.x &&
            bbox.y < selectionRect.y + selectionRect.height &&
            bbox.y + bbox.height > selectionRect.y;
    });
}
    function drawTemporaryPenStroke(strokeData) {
        if (!strokeData || strokeData.points.length < 1) return;
        // ctx is already transformed by the caller (handleCanvasMouseMove)
        // ctx.save(); // Not needed if caller saves/restores and only this is drawn

        if (getStroke && typeof getStroke === 'function') {
            const strokeOptions = {
                size: strokeData.strokeWidth,
                thinning: 0.6, smoothing: 0.5, streamline: 0.5,
                last: false,
            };
            const strokePathPoints = getStroke(strokeData.points, strokeOptions);
            const pathData = getSvgPathFromStroke(strokePathPoints);
            const path2d = new Path2D(pathData);
            ctx.fillStyle = strokeData.strokeColor;
            ctx.globalAlpha = strokeData.opacity || canvasAppState.toolDefaults.pen.previewOpacity || 0.8; // Use specific preview opacity
            ctx.fill(path2d);
        } else {
            ctx.beginPath();
            ctx.moveTo(strokeData.points[0][0], strokeData.points[0][1]);
            for (let i = 1; i < strokeData.points.length; i++) {
                ctx.lineTo(strokeData.points[i][0], strokeData.points[i][1]);
            }
            ctx.strokeStyle = strokeData.strokeColor;
            ctx.lineWidth = strokeData.strokeWidth;
            ctx.globalAlpha = strokeData.opacity || canvasAppState.toolDefaults.pen.previewOpacity || 0.8;
            ctx.stroke();
        }
        // ctx.restore(); // Not needed if caller saves/restores
    }


    function handleCanvasMouseLeave(e) {
        if (isDrawing || isPanning || isDraggingSelection) {
            handleCanvasMouseUp(e);
        }
        // Reset states more defensively
        isDrawing = false;
        isPanning = false;
        isDraggingSelection = false;
        currentPenStroke = null; // Ensure pen stroke is cleared

        if (canvasAppState.currentMode === 'draw' && canvasAppState.currentTool) {
            setActiveTool(canvasAppState.currentTool);
        } else {
            canvas.style.cursor = 'default';
        }
    }

    function handleCanvasWheel(e) {
        if (canvasAppState.currentMode === 'draw') {
            e.preventDefault();
            return;
        }
        e.preventDefault();

        const coords = getCanvasCoordinates(e);
         if (coords.error) return;
        const { x: mouseWorldX, y: mouseWorldY } = coords;


        const zoomIntensity = 0.1;
        const direction = e.deltaY < 0 ? 1 : -1;
        const oldZoom = canvasAppState.zoom;
        const newZoom = Math.max(0.05, Math.min(20, oldZoom * (1 + direction * zoomIntensity)));

        canvasAppState.offsetX = canvasAppState.offsetX + mouseWorldX * (oldZoom - newZoom);
        canvasAppState.offsetY = canvasAppState.offsetY + mouseWorldY * (oldZoom - newZoom);
        canvasAppState.zoom = newZoom;

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

    let lastTouch = null;

    function handleCanvasTouchStart(e) {
        if (e.touches.length > 1) {
            isPanning = false; isDrawing = false; isDraggingSelection = false;
            return;
        }
        e.preventDefault();
        const touch = e.touches[0];

        const currentTime = new Date().getTime();
        const tapLength = currentTime - lastTap.time;

        const coords = getCanvasCoordinates(touch);
        if (coords.error) return;

        // Check if this tap is close in time and space to the last one
        if (tapLength < 300 && tapLength > 0 && Math.abs(coords.viewX - lastTap.x) < 30 && Math.abs(coords.viewY - lastTap.y) < 30) {

            // It's a double tap, trigger the edit logic
            handleCanvasDoubleClick({
                clientX: touch.clientX,
                clientY: touch.clientY,
                preventDefault: () => {} // Mock event object
            });

            // Reset lastTap to prevent a third tap from also triggering
            lastTap = { time: 0, x: 0, y: 0 };
            e.stopPropagation(); // Stop the event from proceeding to single-tap logic
        return; // Exit here to not process as a single tap
        }

        lastTouch = { clientX: touch.clientX, clientY: touch.clientY };
        handleCanvasMouseDown({ button: 0, clientX: touch.clientX, clientY: touch.clientY, preventDefault: () => {}, target: e.target });
    }

    function handleCanvasTouchMove(e) {
        if (e.touches.length > 1) {
            return;
        }
        e.preventDefault();
        const touch = e.touches[0];
        handleCanvasMouseMove({ clientX: touch.clientX, clientY: touch.clientY, preventDefault: () => {}, target: e.target });
        if(lastTouch && isPanning) { // Update lastTouch for panning delta calculation if panning
             // The panning logic in handleCanvasMouseMove implicitly uses its current event's coords
             // and panStartViewX/Y set in mousedown. So this update of lastTouch is for other potential uses.
        }
        // lastTouch = { clientX: touch.clientX, clientY: touch.clientY }; // update last touch for next delta - moved to handleCanvasMouseMove if needed.
    }

    function handleCanvasTouchEnd(e) {
        e.preventDefault();
        const touch = e.changedTouches[0] || lastTouch || { clientX:0, clientY:0, button: 0, preventDefault: () => {}, target: e.target };
        handleCanvasMouseUp({ button: 0, clientX: touch.clientX, clientY: touch.clientY, preventDefault: () => {}, target: e.target });
        lastTouch = null;
    }

    function getElementAtPosition(worldX, worldY, getAll = false) {
    const foundElements = [];
    // Iterate from top-most to bottom-most
    for (let i = canvasElements.length - 1; i >= 0; i--) {
        const el = canvasElements[i];
        if (isPointInsideElement(el, worldX, worldY)) {
            if (!getAll) return el; // Return the first one found (top-most)
            foundElements.push(el);
        }
    }
    return getAll ? foundElements : null;
}

    function isPointInsideElement(element, worldX, worldY) {
        const tolerance = Math.max(2, 5 / canvasAppState.zoom); // Ensure minimum tolerance

        switch (element.type) {
            case 'rectangle':
            case 'image':
                return worldX >= element.x - tolerance &&
                       worldX <= element.x + element.width + tolerance &&
                       worldY >= element.y - tolerance &&
                       worldY <= element.y + element.height + tolerance;
            case 'ellipse':
                const cx = element.x + element.width / 2;
                const cy = element.y + element.height / 2;
                const rx = element.width / 2 + tolerance;
                const ry = element.height / 2 + tolerance;
                if (rx <= 0 || ry <= 0) return false;
                const term1 = Math.pow((worldX - cx) / rx, 2);
                const term2 = Math.pow((worldY - cy) / ry, 2);
                return term1 + term2 <= 1;
            case 'text':
                if (!element.text) return false;
                // This is a very rough approximation. For better accuracy, measure text on a hidden canvas or use DOM.
                // For now, let's assume ctx is already set up with transformations for font metrics.
                ctx.save();
                ctx.font = `${element.fontSize || canvasAppState.fontSize}px ${element.fontFamily || canvasAppState.fontFamily}`;
                const metrics = ctx.measureText(element.text.split('\\n')[0]); // Measure first line for width approx
                const lines = element.text.split('\\n');
                const estHeight = lines.length * (element.fontSize || 16) * 1.2 + tolerance; // 1.2 for line height
                // This estimated width is still problematic for multiline text.
                // Use a bounding box that covers the rough area.
                let textWidth = 0;
                lines.forEach(line => {
                    textWidth = Math.max(textWidth, ctx.measureText(line).width);
                });
                ctx.restore();

                return worldX >= element.x - tolerance &&
                       worldX <= element.x + textWidth + tolerance &&
                       worldY >= element.y - tolerance &&
                       worldY <= element.y + estHeight; // y is typically baseline or top-left start
            case 'pen':
                if (!element.points || element.points.length === 0) return false;
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                element.points.forEach(p => {
                    minX = Math.min(minX, p[0]);
                    minY = Math.min(minY, p[1]);
                    maxX = Math.max(maxX, p[0]);
                    maxY = Math.max(maxY, p[1]);
                });
                 const penTolerance = (element.strokeWidth || canvasAppState.strokeWidth) / 2 + tolerance;
                if (worldX < minX - penTolerance || worldX > maxX + penTolerance || worldY < minY - penTolerance || worldY > maxY + penTolerance) {
                    return false;
                }
                for (let i = 0; i < element.points.length - 1; i++) {
                    if (isPointNearLine(worldX, worldY, element.points[i], element.points[i+1], penTolerance)) {
                        return true;
                    }
                }
                 // Check last point if only one point or if near endpoint
                if (element.points.length === 1) {
                    return Math.hypot(worldX - element.points[0][0], worldY - element.points[0][1]) <= penTolerance;
                }

                return false;
            default:
                return false;
        }
    }
    function isPointNearLine(px, py, startPt, endPt, maxDistance) {
        const [x1, y1] = startPt;
        const [x2, y2] = endPt;
        const L2 = Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2);
        if (L2 === 0) return Math.hypot(px - x1, py - y1) <= maxDistance;
        let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / L2;
        t = Math.max(0, Math.min(1, t));
        const closestX = x1 + t * (x2 - x1);
        const closestY = y1 + t * (y2 - y1);
        return Math.hypot(px - closestX, py - closestY) <= maxDistance;
    }

     function deleteSelectedElements() {
    if (selectedElements.length === 0) return;

    // Capture payload for each element to be deleted *before* modifying selections or canvasElements

    let idsToDelete = new Set(selectedElements.map(el => el.id));
    canvasElements = canvasElements.filter(el => !idsToDelete.has(el.id));
    selectedElements = []; // Now it's safe to clear selection

    pushToHistory("Delete Elements");
    renderCanvas();
    idsToDelete = selectedElements.map(el => el.id);
    // Send action for each deleted element
    sendActionToServer("element_remove", { ids: idsToDelete });
    clearOwnPreviewOnServer(); // Clear any lingering preview for this client
}

    function showTextInputOverlay(worldX, worldY) {
        finalizeTextInput();
        const toolDefaults = canvasAppState.toolDefaults.text || {};
        currentTextElementData = {
            type: 'text', text: '',
            strokeColor: canvasAppState.strokeColor,
            fontSize: canvasAppState.fontSize || toolDefaults.fontSize,
            fontFamily: canvasAppState.fontFamily || toolDefaults.fontFamily,
            textAlign: toolDefaults.textAlign || 'left',
            opacity: toolDefaults.opacity || 1.0,
            angle: 0,
            startX: worldX,
            startY: worldY
        };

        const viewX = worldX * canvasAppState.zoom + canvasAppState.offsetX;
        const viewY = worldY * canvasAppState.zoom + canvasAppState.offsetY;
        const panelRect = document.querySelector('.canvas-panel').getBoundingClientRect();

        textInputOverlayEl.style.left = `${viewX + panelRect.left}px`;
        textInputOverlayEl.style.top =  `${viewY + panelRect.top}px`;
        textInputOverlayEl.style.fontFamily = currentTextElementData.fontFamily;
        textInputOverlayEl.style.fontSize = `${currentTextElementData.fontSize * canvasAppState.zoom}px`;
        textInputOverlayEl.style.color = currentTextElementData.strokeColor;
        textInputOverlayEl.value = '';
        textInputOverlayEl.style.display = 'block';
        textInputOverlayEl.style.minWidth = '50px';
        textInputOverlayEl.style.minHeight = `${currentTextElementData.fontSize * canvasAppState.zoom * 1.2}px`;
        textInputOverlayEl.focus();
    }

    function finalizeTextInput() {
        if (textInputOverlayEl.style.display === 'none' || !currentTextElementData) return;

        const text = textInputOverlayEl.value;
        if (text.trim()) {
            const newElement = {
                id: TB.utils.uniqueId('text_'),
                ...currentTextElementData,
                text: text,
                x: currentTextElementData.startX,
                y: currentTextElementData.startY,
            };
            delete newElement.startX;
            delete newElement.startY;
            canvasElements.push(newElement);
            pushToHistory("Add Text");
            sendActionToServer("element_add", newElement);
            clearOwnPreviewOnServer();
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
             renderCanvas(); // Render to remove any lingering visual artifacts
        }

        setTimeout(() => {
            if (textInputOverlayEl.style.display !== 'none' && currentTextElementData) { // Check if still active
                textInputOverlayEl.style.height = 'auto';
                textInputOverlayEl.style.width = 'auto'; // Allow it to shrink if text is deleted
                // Ensure a minimum height based on font size, and use scrollHeight for content
                const minHeight = currentTextElementData.fontSize * canvasAppState.zoom * 1.2;
                textInputOverlayEl.style.height = `${Math.max(textInputOverlayEl.scrollHeight, minHeight)}px`;
                textInputOverlayEl.style.width = `${Math.max(textInputOverlayEl.scrollWidth, 50)}px`;
            }
        }, 0);
    }

    async function handleAddImageByURL() {
        // Ensure drawing mode is active if necessary, or simply allow adding images anytime
        // if (canvasAppState.currentMode !== 'draw') setActiveMode('draw');
        finalizeTextInput(); // Finalize any pending text input

        const imageUrl = await TB.ui.Modal.prompt({title: "Add Image by URL", placeholder: "Enter image URL", useTextArea: false});
        if (!imageUrl || !imageUrl.trim()) return;

        const loaderId = TB.ui.Loader.show("Loading image...");
        try {
            const imgObject = await loadImageAsync(imageUrl.trim());
            const aspectRatio = imgObject.width / imgObject.height;
            const defaultWidth = 200 / canvasAppState.zoom; // Adjust size based on current zoom
            const defaultHeight = defaultWidth / aspectRatio;

            // Calculate center of the current view in world coordinates
            const viewCenterX = (canvas.width / (window.devicePixelRatio || 1)) / 2;
            const viewCenterY = (canvas.height / (window.devicePixelRatio || 1)) / 2;
            const worldCenterX = (viewCenterX - canvasAppState.offsetX) / canvasAppState.zoom;
            const worldCenterY = (viewCenterY - canvasAppState.offsetY) / canvasAppState.zoom;

            const imageElement = {
                id: TB.utils.uniqueId('image_'), type: 'image', src: imageUrl.trim(),
                x: worldCenterX - defaultWidth / 2, y: worldCenterY - defaultHeight / 2,
                width: defaultWidth, height: defaultHeight,
                imgObject: imgObject, // Keep locally for rendering
                opacity: canvasAppState.toolDefaults.image?.opacity || 1.0, angle: 0
            };
            canvasElements.push(imageElement);
            pushToHistory("Add Image");
            renderCanvas();

            // Broadcast the new image element (without imgObject)
            const { imgObject: _, ...elementToSend } = imageElement; // Destructure to exclude imgObject
            sendActionToServer("element_add", elementToSend);
            // No preview phase for image adding, so clearOwnPreviewOnServer might not be strictly needed,
            // but good practice if any generic preview state could be active.
            clearOwnPreviewOnServer();

            TB.ui.Toast.showSuccess("Image added.");
        } catch (err) {
            TB.logger.error("Failed to load image from URL:", imageUrl, err);
            TB.ui.Toast.showError("Could not load image from URL.");
        } finally {
            TB.ui.Loader.hide(loaderId);
        }
    }

    function loadImageAsync(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => resolve(img);
            img.onerror = (err) => reject(err);
            img.src = src;
        });
    }

    function getUniqueClientId() {
        if (!localClientId) {
            localClientId = localStorage.getItem('canvasClientId');
            if (!localClientId) {
                localClientId = TB.utils.uniqueId('client_');
                localStorage.setItem('canvasClientId', localClientId);
            }
        }
        return localClientId;
    }


    async function handleShareCanvas() {
        if (!currentSessionId) {
            await handleSaveSession(); // Ensure session is saved and has an ID
            if (!currentSessionId) {
                TB.ui.Toast.showError("Could not save session to get a sharable ID.");
                return;
            }
        }

        // For simplicity, just use currentSessionId as the collaboration ID
        // In a real app, you might generate a separate "share ID" or have explicit join flows.
        const collabId = currentSessionId;

        const shareUrl = `${window.location.origin}${window.location.pathname}?collab_session_id=${collabId}`;

        try {
            await navigator.clipboard.writeText(shareUrl);
            TB.ui.Toast.showSuccess("Share link copied to clipboard!");
        } catch (err) {
            TB.logger.error("Failed to copy share link:", err);
            TB.ui.Modal.alert({title: "Share Link", message: `Share this link: ${shareUrl}`});
        }

        // Start or ensure SSE connection for this canvas_id
        connectToCanvasStream(collabId);
    }

    function connectToCanvasStream(canvasIdToStream) {
        if (sseConnection && sseCanvasId === canvasIdToStream) {
            TB.logger.info(`SSE: Already connected to stream for canvas: ${canvasIdToStream}`);
            return;
        }
        disconnectFromCanvasStream(); // Disconnect any existing stream

        if (!canvasIdToStream) {
            TB.logger.error("SSE: Cannot connect, canvas_id is missing.");
            return;
        }

        sseCanvasId = canvasIdToStream;
        // Corrected SSE Path: Use /api/ModuleName/endpoint_name
        const ssePath = `/sse/Canvas/open_canvas_stream?canvas_id=${sseCanvasId}&client_id=${getUniqueClientId()}`;
        TB.logger.info(`SSE: Attempting to connect to ${ssePath}`);

        remoteUserPreviews = {}; // Clear old remote previews when connecting to a new stream

        sseConnection = TB.sse.connect(ssePath, {
            onOpen: (event) => {
                TB.logger.info(`SSE: Connection opened to ${ssePath}`, event);
                TB.ui.Toast.showSuccess("Live collaboration active!", { duration: 2000 });
            },
            onError: (error) => {
                TB.logger.error(`SSE: Connection error with ${ssePath}`, error);
                TB.ui.Toast.showError("Live collaboration connection failed.", { duration: 3000 });
                sseConnection = null; sseCanvasId = null; remoteUserPreviews = {}; renderCanvas();
            },
            listeners: {
                'stream_connected': (payload, event) => {
                    TB.logger.info('SSE Event (stream_connected):', payload);
                    // Client ID specific logic can go here if needed from payload.clientId
                },
                'canvas_elements_changed': (payload, event) => {
                    TB.logger.info('SSE Event (canvas_elements_changed):', payload);
                    if (payload.originator_user_id === getUniqueClientId()) {
                        return;
                    }
                    if (payload.canvas_id !== sseCanvasId) {
                        TB.logger.warn(`SSE: Received elements_changed for wrong canvas. Expected ${sseCanvasId}, got ${payload.canvas_id}`);
                        return;
                    }
                    handleIncomingCanvasAction(payload.action, payload.element);
                },
                'user_preview_update': (payload, event) => {
                    if (payload.user_id === getUniqueClientId()) {
                        return;
                    }
                     if (payload.canvas_id !== sseCanvasId) {
                        return;
                    }
                    remoteUserPreviews[payload.user_id] = payload.preview_data;
                    renderCanvas();
                },
                'clear_user_preview': (payload, event) => {
                     if (payload.user_id === getUniqueClientId()) {
                        return;
                    }
                     if (payload.canvas_id !== sseCanvasId) return;

                    if (remoteUserPreviews[payload.user_id]) {
                        delete remoteUserPreviews[payload.user_id];
                        renderCanvas();
                    }
                },
                'ping': (payload, event) => {
                    // TB.logger.debug("SSE Ping received:", payload.timestamp);
                },
                'error': (payload, event) => {
                    TB.logger.error('SSE Event (server error):', payload);
                    TB.ui.Toast.showError(`Collaboration Error: ${payload.message || 'Unknown stream error'}`);
                    if (payload.message && (payload.message.includes("canvas_id is required") || payload.message.toLowerCase().includes("not found"))){
                        disconnectFromCanvasStream();
                    }
                }
            }
        });
    }

    function disconnectFromCanvasStream() {
        if (sseConnection && sseCanvasId) {
            TB.logger.info(`SSE: Disconnecting from canvas stream: ${sseCanvasId}`);
            // Corrected SSE Path for disconnect
            const ssePath = `/sse/Canvas/open_canvas_stream?canvas_id=${sseCanvasId}&client_id=${getUniqueClientId()}`;
            TB.sse.disconnect(ssePath);
        }
        sseConnection = null;
        sseCanvasId = null;
        remoteUserPreviews = {};
        renderCanvas(); // Clear any remote previews
    }

    // Function to send client's actions to the server for broadcast
    async function sendActionToServer(actionType, payloadData) {
        if (!sseCanvasId || !sseConnection) { // Only send if in a collaborative session
            // TB.logger.debug("Not in collaborative session, action not sent to server:", actionType);
            return;
        }

        const actionPayload = {
            canvas_id: sseCanvasId,
            user_id: getUniqueClientId(),
            action_type: actionType,
            payload: payloadData
        };

        try {
            // Using a fire-and-forget approach for most actions, assuming server handles broadcast.
            // For critical actions, you might want to await and handle response.
            TB.api.request('Canvas', 'send_canvas_action', actionPayload, 'POST').catch(err => {
                TB.logger.error("Error sending canvas action to server:", err, actionPayload);
                // Potentially show a non-blocking error to the user
            });
        } catch (err) {
            TB.logger.error("Exception sending canvas action:", err);
        }
    }

    // Wrapper to send preview data (throttled)
    function sendCurrentPreviewDataToServer(previewElementData) {
        if (!isDrawing && !isDraggingSelection) return; // Only send if actively drawing/dragging
        if (previewElementData) {
             // TB.logger.debug("Throttled: Sending preview data for client:", getUniqueClientId());
            sendActionToServer("preview_update", previewElementData);
        }
    }

    // Call this when a local drawing/drag operation finishes to clear its preview on other clients
    function clearOwnPreviewOnServer() {
        if (sseCanvasId && sseConnection) {
            // TB.logger.debug("Clearing own preview on server for client:", getUniqueClientId());
            sendActionToServer("preview_clear", {}); // Payload can be empty or {action: "cleared"}
        }
    }


    // Handle incoming actions from other users
    function handleIncomingCanvasAction(action, element) {
        TB.logger.info(`Handling incoming action: ${action}`, element);
        let elementExists = false;
        let existingElementIndex = -1;
        if (element && element.id) {
             existingElementIndex = canvasElements.findIndex(el => el.id === element.id);
             elementExists = existingElementIndex !== -1;
        }

        switch (action) {
            case 'element_add':
                if (!elementExists && element && element.id) { // Ensure it's not a duplicate
                    // If it's an image, we need to load it
                    if (element.type === 'image' && element.src && !element.imgObject) {
                        loadImageAsync(element.src)
                            .then(img => {
                                element.imgObject = img;
                                canvasElements.push(element);
                                renderCanvas();
                            })
                            .catch(err => {
                                TB.logger.error("Failed to load incoming image element:", err, element);
                                // Add placeholder or skip
                            });
                    } else {
                        canvasElements.push(element);
                    }
                } else if (!element || !element.id) {
                     TB.logger.warn("Incoming 'element_add' without valid element or ID.", element);
                }
                break;
            case 'element_update':
                if (elementExists && element && element.id) {
                    // Preserve imgObject if it exists locally and not in payload
                    const localImgObject = canvasElements[existingElementIndex].imgObject;
                    canvasElements[existingElementIndex] = {...canvasElements[existingElementIndex], ...element};
                    if (element.type === 'image' && localImgObject && !element.imgObject) {
                        canvasElements[existingElementIndex].imgObject = localImgObject;
                    }
                    // If it's an image and src changed OR imgObject is missing, reload
                    else if (element.type === 'image' && element.src &&
                             (!canvasElements[existingElementIndex].imgObject || canvasElements[existingElementIndex].imgObject.src !== element.src)) {
                        loadImageAsync(element.src)
                            .then(img => {
                                canvasElements[existingElementIndex].imgObject = img;
                                renderCanvas(); // Render after image loaded
                            })
                            .catch(err => TB.logger.error("Failed to load updated image element:", err, element));
                    }
                } else if (element && element.id) { // Element not found locally, treat as add
                    TB.logger.warn("Incoming 'element_update' for non-existent element. Treating as 'add'.", element);
                     if (element.type === 'image' && element.src && !element.imgObject) {
                        loadImageAsync(element.src).then(img => { element.imgObject = img; canvasElements.push(element); renderCanvas(); });
                    } else { canvasElements.push(element); }
                }
                break;
            case 'element_remove':
                // Handles both single element deletion (payload has .id)
                // and multi-element eraser (payload has .ids array)
                const idsToRemove = new Set(element.ids || (element.id ? [element.id] : []));

                if (idsToRemove.size > 0) {
                    canvasElements = canvasElements.filter(el => !idsToRemove.has(el.id));
                    // Also ensure these elements are removed from the local selection if they happen to be selected
                    selectedElements = selectedElements.filter(el => !idsToRemove.has(el.id));
                } else {
                    TB.logger.warn("Incoming 'element_remove' with no valid 'id' or 'ids' property.", element);
                }
                break;
            default:
                TB.logger.warn("Unknown canvas action received:", action);
                return; // Don't render if unknown
        }
        renderCanvas(); // Re-render after applying changes from others
    }

    function renderCanvas() {
    const dpr = window.devicePixelRatio || 1;
    ctx.save();
    ctx.fillStyle = canvasAppState.viewBackgroundColor;
    ctx.fillRect(0, 0, canvas.width / dpr, canvas.height / dpr);
    ctx.translate(canvasAppState.offsetX, canvasAppState.offsetY);
    ctx.scale(canvasAppState.zoom, canvasAppState.zoom);

    canvasElements.forEach(el => drawElementOnCanvas(el));

    if (sseConnection) {
        for (const userId in remoteUserPreviews) {
            const previewData = remoteUserPreviews[userId];
            if (previewData && previewData.type) {
                // Handle a group of dragged elements
                if (previewData.type === 'group_drag' && Array.isArray(previewData.elements)) {
                     previewData.elements.forEach(pEl => drawElementOnCanvas({...pEl, opacity: 0.5}));
                } else { // Handle single element preview
                     drawElementOnCanvas({...previewData, opacity: 0.5});
                }
            }
        }
    }

    if (canvasAppState.currentMode === 'select' && selectedElements.length > 0) {
        selectedElements.forEach(selEl => drawSelectionHighlight(selEl));
        updateContextualToolbar(selectedElements);
    }else {
        // NEW: Hide toolbar if no selection
        updateContextualToolbar([]);
    }

    if (isMarqueeSelecting) {
        const x = Math.min(marqueeRect.startX, marqueeRect.endX);
        const y = Math.min(marqueeRect.startY, marqueeRect.endY);
        const width = Math.abs(marqueeRect.endX - marqueeRect.startX);
        const height = Math.abs(marqueeRect.endY - marqueeRect.startY);
        ctx.fillStyle = 'rgba(0, 123, 255, 0.1)';
        ctx.strokeStyle = 'rgba(0, 123, 255, 0.6)';
        ctx.lineWidth = 1 / canvasAppState.zoom;
        ctx.fillRect(x, y, width, height);
        ctx.strokeRect(x, y, width, height);
    }

    ctx.restore();
    updateUndoRedoButtons();
}

    function drawElementOnCanvas(el) {
    const isGhosted = erasedThisStroke.has(el.id);
    ctx.save();
    // Use the element's own opacity, but reduce it if ghosted
    let effectiveOpacity = el.opacity === undefined ? 1.0 : el.opacity;
    if (isGhosted) {
        effectiveOpacity *= 0.2;
    }
    ctx.globalAlpha = effectiveOpacity;

    if (el.angle) {
        const centerX = el.x + (el.width || 0) / 2;
        const centerY = el.y + (el.height || 0) / 2;
        ctx.translate(centerX, centerY);
        ctx.rotate(el.angle * Math.PI / 180);
        ctx.translate(-centerX, -centerY);
    }

    const stroke = el.strokeColor || canvasAppState.strokeColor;
    const fill = el.fill;
    const strokeWidth = el.strokeWidth || canvasAppState.strokeWidth;

        switch (el.type) {
            case 'pen':
                if (el.points && el.points.length > 0) {
                    if (getStroke && typeof getStroke === 'function') {
                        const strokeOptions = {
                            size: strokeWidth,
                            thinning: 0.6, smoothing: 0.5, streamline: 0.5,
                            last: true,
                        };
                        const strokePathPoints = getStroke(el.points.map(p => [p[0], p[1], p[2] || 0.5]), strokeOptions);
                        const pathData = getSvgPathFromStroke(strokePathPoints);
                        const path2d = new Path2D(pathData);
                        ctx.fillStyle = stroke;
                        ctx.fill(path2d);
                    } else {
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
                        seed: el.seed
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
                        seed: el.seed
                    });
                }
                break;
            case 'text':
                ctx.fillStyle = stroke;
                ctx.font = `${el.fontSize || canvasAppState.fontSize}px ${el.fontFamily || canvasAppState.fontFamily}`;
                ctx.textAlign = el.textAlign || 'left';
                const lines = (el.text || "").split('\\n'); // Changed from \\n to \\n for actual newlines
                const lineHeight = (el.fontSize || canvasAppState.fontSize) * 1.2;
                const textRenderYOffset = (el.fontSize || canvasAppState.fontSize) * 0.85;
                lines.forEach((line, index) => {
                    ctx.fillText(line, el.x, el.y + textRenderYOffset + (index * lineHeight));
                });
                break;
            case 'image':
                if (el.imgObject && el.imgObject.complete) {
                    try {
                         ctx.drawImage(el.imgObject, el.x, el.y, el.width, el.height);
                    } catch (e) {
                        TB.logger.warn("Error drawing image (possibly tainted canvas):", el.src, e);
                        ctx.strokeStyle = 'red'; ctx.lineWidth = 1;
                        ctx.strokeRect(el.x, el.y, el.width, el.height);
                        ctx.fillText("Image Error", el.x + 5, el.y + 15);
                    }
                } else if (el.src && !el.imgObject) {
                    if (el.isPdf) {
                    // For incoming PDF elements from other clients
                    pdfjsLib.getDocument(el.src).promise.then(async (pdfDoc) => {
                        el.pdfDoc = pdfDoc;
                        const page = await pdfDoc.getPage(el.pdfPageNum || 1);
                        const viewport = page.getViewport({ scale: 1.5 });
                        el.imgObject = await renderPdfPageToImage(page, viewport);
                        renderCanvas();
                    });
                    } else {
                        loadImageAsync(el.src).then(img => {
                            el.imgObject = img;
                            renderCanvas();
                        }).catch(err => {
                            TB.logger.error("Failed to lazy-load image for drawing:", el.src, err);
                            el.imgObject = null;
                            renderCanvas();
                        });
                    }
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
        let bbox = getElementBoundingBox(element);
        if (!bbox) return;

        ctx.save();
        ctx.strokeStyle = 'rgba(0, 123, 255, 0.8)';
        ctx.lineWidth = Math.max(0.5, 1.5 / canvasAppState.zoom); // Ensure minimum visible line width
        ctx.setLineDash([Math.max(2, 6 / canvasAppState.zoom), Math.max(1, 3 / canvasAppState.zoom)]);

        if (element.angle) {
            const centerX = bbox.x + bbox.width / 2; // Use BBOX center for rotation of highlight
            const centerY = bbox.y + bbox.height / 2;
            ctx.translate(centerX, centerY);
            ctx.rotate(element.angle * Math.PI / 180);
            ctx.translate(-centerX, -centerY);
        }
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
        ctx.setLineDash([]);
        ctx.restore();
    }

    function getElementBoundingBox(element) {
        if (!element) return null; // Added guard
        switch (element.type) {
            case 'rectangle':
            case 'ellipse':
            case 'image':
                return { x: element.x, y: element.y, width: element.width, height: element.height };
            case 'text':
                if (!element.text) return { x: element.x, y: element.y, width: 0, height: 0 }; // Default for empty text
                ctx.save(); // Save context before changing font
                ctx.font = `${element.fontSize || canvasAppState.fontSize}px ${element.fontFamily || canvasAppState.fontFamily}`;
                const lines = element.text.split('\\n');
                let maxWidth = 0;
                lines.forEach(line => {
                    maxWidth = Math.max(maxWidth, ctx.measureText(line).width);
                });
                const estHeight = lines.length * (element.fontSize || 16) * 1.2;
                ctx.restore(); // Restore context
                return { x: element.x, y: element.y, width: maxWidth, height: estHeight };
            case 'pen':
                if (!element.points || element.points.length === 0) return { x:0, y:0, width:0, height:0}; // Default for no points
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                element.points.forEach(p => {
                    minX = Math.min(minX, p[0]);
                    minY = Math.min(minY, p[1]);
                    maxX = Math.max(maxX, p[0]);
                    maxY = Math.max(maxY, p[1]);
                });
                const buffer = (element.strokeWidth || 2) / 2;
                return { x: minX - buffer, y: minY - buffer, width: (maxX - minX) + 2 * buffer, height: (maxY - minY) + 2 * buffer };
            default:
                return null;
        }
    }

    function getSvgPathFromStroke(strokePoints) {
        if (!strokePoints || strokePoints.length === 0) return '';
        const d = strokePoints.reduce(
            (acc, [x0, y0], i, arr) => {
                const [x1, y1] = arr[(i + 1) % arr.length];
                acc.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2);
                return acc;
            },
            ['M', strokePoints[0][0], strokePoints[0][1], 'Q'] // Corrected initial M command
        );
        d.push('Z');
        return d.join(' ');
    }

    function openSettingsModal() {
            if (!settingsModalInstance) {
                TB.ui.Toast.showError("Settings modal not initialized.");
                console.error("Attempted to open settings modal, but instance is null.");
                return;
            }
            for (const tool in canvasAppState.toolDefaults) {
                for (const prop in canvasAppState.toolDefaults[tool]) {
                    if (prop === 'seed' || prop === 'previewOpacity') continue;
                    const inputId = `default${tool.charAt(0).toUpperCase() + tool.slice(1)}${prop.charAt(0).toUpperCase() + prop.slice(1)}`;
                    const inputEl = document.getElementById(inputId);
                    if (inputEl) {
                        if (inputEl.type === 'color' || inputEl.type === 'text' || inputEl.tagName === 'SELECT') {
                            inputEl.value = canvasAppState.toolDefaults[tool][prop];
                        } else if (inputEl.type === 'number') {
                            inputEl.value = parseFloat(canvasAppState.toolDefaults[tool][prop]);
                        }
                    }
                }
            }
            settingsModalInstance.show();
    }

     function closeSettingsModal() {
        if (settingsModalInstance) {
            settingsModalInstance.close();
        }
    }

    function saveToolDefaults() {
        const inputs = document.querySelectorAll('#settingsModal .tool-config-group [data-tool]');
        inputs.forEach(input => {
            const tool = input.dataset.tool;
            const prop = input.dataset.prop;
            if (tool && prop && canvasAppState.toolDefaults[tool]) { // Added check for canvasAppState.toolDefaults[tool]
                if (input.type === 'number') {
                    canvasAppState.toolDefaults[tool][prop] = parseFloat(input.value);
                } else {
                    canvasAppState.toolDefaults[tool][prop] = input.value;
                }
            }
        });
        TB.ui.Toast.showSuccess("Default settings saved.");
        closeSettingsModal();
        if (canvasAppState.currentTool && canvasAppState.currentMode === 'draw') {
            setActiveTool(canvasAppState.currentTool);
        }
    }

    function updateContextualToolbar(selected) {
    const toolbar = document.getElementById('selectionContextToolbar');
    const pdfControls = document.getElementById('pdfControls');
    const scaleControls = document.getElementById('scaleControls');

    if (selected.length !== 1) { // Only show for single selection
        toolbar.style.display = 'none';
        return;
    }

    const el = selected[0];
    toolbar.style.display = 'flex';

    // PDF controls visibility
    if (el.isPdf) {
        pdfControls.style.display = 'flex';
        document.getElementById('pdfPageInfo').textContent = `Page ${el.pdfPageNum} / ${el.pdfTotalPages}`;
        document.getElementById('pdfPrevPageBtn').disabled = el.pdfPageNum <= 1;
        document.getElementById('pdfNextPageBtn').disabled = el.pdfPageNum >= el.pdfTotalPages;
    } else {
        pdfControls.style.display = 'none';
    }

    // Scale controls
    const currentScale = el.width / el.originalWidth;
    document.getElementById('scaleSlider').value = currentScale;
    document.getElementById('scaleValue').textContent = `${Math.round(currentScale * 100)}%`;
}

async function changeSelectedPdfPage(direction) {
    if (selectedElements.length !== 1 || !selectedElements[0].isPdf) return;

    const pdfElement = selectedElements[0];
    const newPageNum = pdfElement.pdfPageNum + direction;

    if (newPageNum < 1 || newPageNum > pdfElement.pdfTotalPages) return;

    const loaderId = TB.ui.Loader.show("Loading PDF page...");
    try {
        if (!pdfElement.pdfDoc) { // Lazy load the PDF document proxy if not present
             pdfElement.pdfDoc = await pdfjsLib.getDocument(pdfElement.src).promise;
        }
        const page = await pdfElement.pdfDoc.getPage(newPageNum);
        const viewport = page.getViewport({ scale: 1.5 }); // Use same scale for consistency

        pdfElement.pdfPageNum = newPageNum;
        pdfElement.imgObject = await renderPdfPageToImage(page, viewport);

        // Adjust size while maintaining aspect ratio
        const currentScale = pdfElement.width / pdfElement.originalWidth;
        pdfElement.width = viewport.width * currentScale;
        pdfElement.height = viewport.height * currentScale;
        pdfElement.originalWidth = viewport.width;
        pdfElement.originalHeight = viewport.height;

        pushToHistory("Change PDF Page");
        renderCanvas();
        const { imgObject, pdfDoc, ...elementToSend } = pdfElement;
        sendActionToServer("element_update", elementToSend);

    } catch (error) {
        TB.ui.Toast.showError("Failed to change PDF page: " + error.message);
    } finally {
        TB.ui.Loader.hide(loaderId);
    }
}

function scaleSelectedElements(newScale) {
    if (selectedElements.length === 0) return;

    selectedElements.forEach(el => {
        if (el.originalWidth && el.originalHeight) {
            const centerX = el.x + el.width / 2;
            const centerY = el.y + el.height / 2;

            el.width = el.originalWidth * newScale;
            el.height = el.originalHeight * newScale;

            // Keep center point fixed
            el.x = centerX - el.width / 2;
            el.y = centerY - el.height / 2;
        }
    });

    // Use a throttled update to avoid spamming history/network
    TB.utils.throttle(() => {
        pushToHistory("Scale Elements");
        selectedElements.forEach(el => {
             const { imgObject, pdfDoc, ...elementToSend } = el;
             sendActionToServer("element_update", elementToSend);
        });
    }, 250)(); // Throttle to 4 times a second max

    renderCanvas();
}

    function startNewSession(showToast = true) {
        currentSessionId = null;
        currentCanvasName = "Untitled Canvas";
        document.getElementById('canvasNameInput').value = currentCanvasName;
        canvasElements = [];
        selectedElements = []; // Reset multi-selection
        textNotesContent = "";
        document.getElementById('textNotesArea').value = "";
        canvasAppState = JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE));
        const currentTheme = (TB.ui && TB.ui.theme && TB.ui.theme.getCurrentMode) ? TB.ui.theme.getCurrentMode() : 'light';
        canvasAppState.viewBackgroundColor = currentTheme === 'dark' ? (canvasAppState.toolDefaults?.viewBackgroundColorDark || '#333333') : DEFAULT_CANVAS_APP_STATE.viewBackgroundColor; // Darker default
        document.getElementById('bgColorPicker').value = canvasAppState.viewBackgroundColor;
        setActiveMode('draw');
        setActiveTool(canvasAppState.currentTool || 'pen');
        historyStack = [JSON.stringify([])];
        redoStack = [];
        renderCanvas();
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
            const { imgObject, originalDragX, originalDragY, originalPoints, ...rest } = el; // Strip non-serializable/temp properties
            return rest;
        });
        const sessionData = {
            id: currentSessionId || TB.utils.uniqueId('canvas-session-'),
            name: currentCanvasName,
            canvas_elements: serializableElements,
            canvas_app_state: canvasAppState,
            text_notes: textNotesContent,
        };
        currentSessionId = sessionData.id;
        try {
            const response = await TB.api.request('Canvas', 'save_session', sessionData, 'POST');
            if (response.error === window.TB.ToolBoxError.none) {
                TB.ui.Toast.showSuccess("Session saved!");
                if (response.get() && response.get().last_modified) {
                    // canvasAppState.last_modified = response.get().last_modified; // Server sets this
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
            const response = await TB.api.request('Canvas', 'list_sessions', null, 'GET');
            TB.ui.Loader.hide(loaderId);
            if (response.error === window.TB.ToolBoxError.none && response.get()) {
                const sessions = response.get();
                if (sessions.length === 0) {
                    TB.ui.Toast.showInfo("No saved sessions found."); return;
                }
                sessions.sort((a, b) => (b.last_modified || 0) - (a.last_modified || 0));
                let modalContent = '<p class="tb-text-sm tb-mb-2">Select a session to load:</p><select id="sessionSelectModal" class="tb-input tb-w-full">';
                sessions.forEach(s => {
                    const dateStr = s.last_modified ? new Date(s.last_modified).toLocaleString() : 'N/A'; // Assuming last_modified is a standard timestamp
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
            const response = await TB.api.request('Canvas', `load_session?session_id=${sessionId}`, null, 'GET');
            if (response.error === TB.ToolBoxError.none && response.get()) {
                const data = response.get();
                startNewSession(false);
                currentSessionId = data.id;
                currentCanvasName = data.name;
                textNotesContent = data.text_notes || "";
                canvasAppState = {
                    ...JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE)),
                    ...(data.canvas_app_state || {})
                };
                canvasAppState.toolDefaults = {
                    ...JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE.toolDefaults)),
                    ...(data.canvas_app_state?.toolDefaults || {})
                };
                 canvasAppState.elementPresets = data.canvas_app_state?.elementPresets || JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE.elementPresets));


                document.getElementById('canvasNameInput').value = currentCanvasName;
                document.getElementById('textNotesArea').value = textNotesContent;
                document.getElementById('bgColorPicker').value = canvasAppState.viewBackgroundColor;
                document.getElementById('strokeColorPicker').value = canvasAppState.strokeColor;
                document.getElementById('fillColorPicker').value = canvasAppState.fillColor;
                document.getElementById('strokeWidthInput').value = canvasAppState.strokeWidth;

                setActiveMode(canvasAppState.currentMode || 'draw');
                setActiveTool(canvasAppState.currentTool || 'pen');
                await restoreElementsFromHistory(data.canvas_elements || []);
                historyStack = [JSON.stringify(data.canvas_elements || [])];
                redoStack = [];
                updateUndoRedoButtons();
                TB.ui.Toast.showSuccess(`Session '${TB.utils.escapeHtml(currentCanvasName)}' loaded.`);
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
        finalizeTextInput();
        const serializableElements = canvasElements.map(el => {
            const { imgObject, originalDragX, originalDragY, originalPoints, ...rest } = el;
            return rest;
        });
        const dataToExport = {
            id: currentSessionId || TB.utils.uniqueId('canvas-export-'),
            name: currentCanvasName,
            version: '0.1.0',
            canvas_elements: serializableElements,
            canvas_app_state: canvasAppState,
            text_notes: textNotesContent,
            exported_at: new Date().toISOString()
        };
        const jsonString = JSON.stringify(dataToExport, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        const safeFilename = (currentCanvasName.replace(/[^a-z0-9_.-]/gi, '_') || 'canvas_export').substring(0,50);
        TB.utils.downloadBlob(`${safeFilename}.json`, blob);
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
                    startNewSession(false);
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
                    canvasAppState.elementPresets = importedData.canvas_app_state?.elementPresets || JSON.parse(JSON.stringify(DEFAULT_CANVAS_APP_STATE.elementPresets));

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
        event.target.value = null;
    }

    function openPresetManagementModal() {
        if (!presetManagementModalInstance) {
            TB.ui.Toast.showError("Preset modal not initialized.");
            return;
        }
        loadAndDisplayPresets();
        hidePresetEditForm();
        presetManagementModalInstance.show();
    }

    function closePresetModal() {
        if (presetManagementModalInstance) {
            presetManagementModalInstance.close();
        }
    }

    function loadAndDisplayPresets() {
        const container = document.getElementById('presetListContainer');
        if (!container) return;
        container.innerHTML = '';

        const presets = canvasAppState.elementPresets || [];
        if (presets.length === 0) {
            container.innerHTML = '<p class="tb-text-secondary tb-text-sm">No presets defined yet.</p>';
            return;
        }

        presets.forEach(preset => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'preset-item tb-p-2 tb-border tb-rounded tb-mb-2 tb-flex tb-justify-between tb-items-center dark:tb-border-neutral-700';
            itemDiv.dataset.presetId = preset.id;
            itemDiv.innerHTML = `
                <span class="tb-font-medium">${TB.utils.escapeHtml(preset.name)} <span class="tb-text-xs tb-text-secondary">(${preset.toolType})</span></span>
                <div>
                    <button class="apply-preset-btn tb-btn tb-btn-xs tb-btn-primary tb-mr-1" title="Apply Preset"><span class="material-symbols-outlined" style="font-size:1em; vertical-align:middle;">check_circle</span></button>
                    <button class="edit-preset-btn tb-btn tb-btn-xs tb-btn-secondary tb-mr-1" title="Edit Preset"><span class="material-symbols-outlined" style="font-size:1em; vertical-align:middle;">edit_note</span></button>
                    <button class="delete-preset-btn tb-btn tb-btn-xs tb-btn-danger" title="Delete Preset"><span class="material-symbols-outlined" style="font-size:1em; vertical-align:middle;">delete_forever</span></button>
                </div>
            `;
            itemDiv.querySelector('.apply-preset-btn').addEventListener('click', () => applyPreset(preset.id));
            itemDiv.querySelector('.edit-preset-btn').addEventListener('click', () => showPresetEditForm(preset.id));
            itemDiv.querySelector('.delete-preset-btn').addEventListener('click', () => deletePreset(preset.id));
            container.appendChild(itemDiv);
        });
    }

    function showPresetEditForm(presetIdToEdit = null) {
        const formContainer = document.getElementById('presetEditFormContainer');
        const formTitle = document.getElementById('presetFormTitle');
        const presetIdInput = document.getElementById('presetEditId');
        const presetNameInput = document.getElementById('presetNameInput');
        const presetToolTypeSelect = document.getElementById('presetToolTypeSelect');

        if (!formContainer || !formTitle || !presetIdInput || !presetNameInput || !presetToolTypeSelect) {
            console.error("Preset edit form elements not found."); return;
        }

        let propertiesToPopulate = {};
        if (presetIdToEdit) {
            const preset = canvasAppState.elementPresets.find(p => p.id === presetIdToEdit);
            if (!preset) {
                TB.ui.Toast.showError("Preset not found for editing.");
                return;
            }
            formTitle.textContent = "Edit Preset";
            presetIdInput.value = preset.id;
            presetNameInput.value = preset.name;
            presetToolTypeSelect.value = preset.toolType;
            propertiesToPopulate = preset.properties;
        } else {
            formTitle.textContent = "Add New Preset";
            presetIdInput.value = "";
            presetNameInput.value = "";
            presetToolTypeSelect.value = "pen";
            // For new preset, propertiesToPopulate will be empty, so defaults from config will be used
        }
        populatePresetPropertyFields(propertiesToPopulate); // Pass properties or empty object
        presetToolTypeSelect.disabled = !!presetIdToEdit;
        formContainer.style.display = 'block';
    }

    function hidePresetEditForm() {
        const formContainer = document.getElementById('presetEditFormContainer');
        if (formContainer) {
            formContainer.style.display = 'none';
        }
        const presetToolTypeSelect = document.getElementById('presetToolTypeSelect');
        if (presetToolTypeSelect) {
            presetToolTypeSelect.disabled = false;
        }
    }

    function getPresetPropertyFieldsConfig() {
        return {
            pen: [
                { name: "strokeColor", label: "Stroke Color", type: "color", default: "#000000" },
                { name: "strokeWidth", label: "Stroke Width", type: "number", min: 1, default: 2 },
                { name: "opacity", label: "Opacity", type: "number", min:0, max:1, step: 0.1, default: 1.0 }
            ],
            rectangle: [
                { name: "strokeColor", label: "Stroke Color", type: "color", default: "#000000" },
                { name: "strokeWidth", label: "Stroke Width", type: "number", min: 1, default: 2 },
                { name: "fillColor", label: "Fill Color", type: "color", default: "#cccccc" },
                { name: "fillStyle", label: "Fill Style", type: "select", options: ["solid", "hachure", "zigzag", "cross-hatch", "dots", "dashed", "zigzag-line"], default: "solid" },
                { name: "roughness", label: "Roughness", type: "number", min:0, max:3, step: 0.1, default: 1 },
                { name: "opacity", label: "Opacity", type: "number", min:0, max:1, step: 0.1, default: 1.0 }
            ],
            ellipse: [
                { name: "strokeColor", label: "Stroke Color", type: "color", default: "#000000" },
                { name: "strokeWidth", label: "Stroke Width", type: "number", min: 1, default: 2 },
                { name: "fillColor", label: "Fill Color", type: "color", default: "#dddddd" },
                { name: "fillStyle", label: "Fill Style", type: "select", options: ["hachure", "solid", "zigzag", "cross-hatch", "dots"], default: "hachure" },
                { name: "roughness", label: "Roughness", type: "number", min:0, max:3, step: 0.1, default: 1 },
                { name: "opacity", label: "Opacity", type: "number", min:0, max:1, step: 0.1, default: 1.0 }
            ],
            text: [
                { name: "strokeColor", label: "Text Color", type: "color", default: "#000000" },
                { name: "fontSize", label: "Font Size", type: "number", min: 8, default: 16 },
                { name: "fontFamily", label: "Font Family", type: "text", default: "Arial" },
                { name: "opacity", label: "Opacity", type: "number", min:0, max:1, step: 0.1, default: 1.0 }
            ]
        };
    }

    function populatePresetPropertyFields(currentValues = {}) {
        const toolType = document.getElementById('presetToolTypeSelect').value;
        const container = document.getElementById('presetPropertiesFields');
        if (!container) { console.error("Preset properties container not found."); return; }
        container.innerHTML = '';

        const config = getPresetPropertyFieldsConfig()[toolType];
        if (!config) { console.warn(`No preset field config for tool type: ${toolType}`); return; }

        config.forEach(field => {
            const group = document.createElement('div');
            group.className = 'tb-form-group tb-flex tb-items-center tb-mb-1';
            const label = document.createElement('label');
            label.className = 'tb-form-label tb-form-label-xs tb-mr-2';
            label.style.minWidth = '90px';
            label.textContent = field.label + ':';
            label.htmlFor = `presetProp_${field.name}`;
            group.appendChild(label);
            let input;
            if (field.type === 'select') {
                input = document.createElement('select');
                input.className = 'tb-input tb-input-xs dark:tb-input-dark';
                field.options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt.charAt(0).toUpperCase() + opt.slice(1);
                    input.appendChild(option);
                });
            } else {
                input = document.createElement('input');
                input.type = field.type;
                input.className = 'tb-input tb-input-xs dark:tb-input-dark';
                if (field.type === 'number') {
                    if(field.min !== undefined) input.min = field.min;
                    if(field.max !== undefined) input.max = field.max;
                    if(field.step !== undefined) input.step = field.step;
                }
                if (field.type === 'color') {
                    input.style.padding = '1px';
                    input.style.height = '24px';
                } else if (field.type !== 'checkbox') { // Checkbox doesn't need width styling
                     input.style.width = '120px';
                }
            }
            input.id = `presetProp_${field.name}`;
            input.dataset.prop = field.name;
            input.value = currentValues[field.name] !== undefined ? currentValues[field.name] : field.default;
            group.appendChild(input);
            container.appendChild(group);
        });
    }

    function saveCurrentPresetFromForm() {
        const presetId = document.getElementById('presetEditId').value;
        const presetName = document.getElementById('presetNameInput').value.trim();
        const toolType = document.getElementById('presetToolTypeSelect').value;

        if (!presetName) {
            TB.ui.Toast.showWarning("Preset name cannot be empty.");
            return;
        }

        const properties = {};
        const propInputs = document.querySelectorAll('#presetPropertiesFields [data-prop]');
        propInputs.forEach(input => {
            const propName = input.dataset.prop;
            properties[propName] = input.type === 'number' ? parseFloat(input.value) : (input.type === 'checkbox' ? input.checked : input.value);
        });

        if (presetId) {
            const index = canvasAppState.elementPresets.findIndex(p => p.id === presetId);
            if (index > -1) {
                canvasAppState.elementPresets[index].name = presetName;
                canvasAppState.elementPresets[index].properties = properties;
                TB.ui.Toast.showSuccess("Preset updated.");
            }
        } else {
            const newPreset = {
                id: TB.utils.uniqueId('preset_'),
                name: presetName,
                toolType: toolType,
                properties: properties
            };
            canvasAppState.elementPresets.push(newPreset);
            TB.ui.Toast.showSuccess("Preset added.");
        }
        loadAndDisplayPresets();
        hidePresetEditForm();
    }

    function deletePreset(presetId) {
        TB.ui.Modal.confirm({
            title: "Delete Preset",
            message: "Are you sure you want to delete this preset?",
            confirmButtonText: "Delete",
            confirmAction: () => {
                canvasAppState.elementPresets = canvasAppState.elementPresets.filter(p => p.id !== presetId);
                loadAndDisplayPresets();
                TB.ui.Toast.showInfo("Preset deleted.");
            }
        });
    }

    function applyPreset(presetId) {
        const preset = canvasAppState.elementPresets.find(p => p.id === presetId);
        if (!preset) {
            TB.ui.Toast.showError("Preset not found.");
            return;
        }
        setActiveMode('draw');
        setActiveTool(preset.toolType);
        const propsToUpdate = preset.properties;
        for (const prop in propsToUpdate) {
            if (prop === 'strokeColor') {
                document.getElementById('strokeColorPicker').value = propsToUpdate[prop];
                canvasAppState.strokeColor = propsToUpdate[prop];
            } else if (prop === 'fillColor' && (preset.toolType === 'rectangle' || preset.toolType === 'ellipse')) {
                document.getElementById('fillColorPicker').value = propsToUpdate[prop];
                canvasAppState.fillColor = propsToUpdate[prop];
            } else if (prop === 'strokeWidth') {
                document.getElementById('strokeWidthInput').value = propsToUpdate[prop];
                canvasAppState.strokeWidth = parseInt(propsToUpdate[prop], 10);
            } else if (prop === 'fontSize' && preset.toolType === 'text') {
                canvasAppState.fontSize = parseInt(propsToUpdate[prop], 10);
            } else if (prop === 'fontFamily' && preset.toolType === 'text') {
                canvasAppState.fontFamily = propsToUpdate[prop];
            }
            // Update current app state directly, not just tool defaults, for immediate effect
            canvasAppState[prop] = propsToUpdate[prop];

            // Optionally, also update the specific tool's defaults for persistence if that's desired
            // if (canvasAppState.toolDefaults[preset.toolType]) {
            //      canvasAppState.toolDefaults[preset.toolType][prop] = propsToUpdate[prop];
            // }
        }
        // Re-apply tool settings from main controls now that canvasAppState has preset values
        setActiveTool(preset.toolType);

        TB.ui.Toast.showSuccess(`Preset "${TB.utils.escapeHtml(preset.name)}" applied.`);
        if (presetManagementModalInstance && presetManagementModalInstance.isOpen) {
            presetManagementModalInstance.close();
        }
    }

    async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0]; // Handle one file at a time for simplicity

    // Use the existing uploader from FileWidget's UI logic
    const loaderId = TB.ui.Loader.show(`Uploading ${file.name}...`);

    const chunkSize = 1 * 1024 * 1024; // 1MB, must match FileWidget
    const totalChunks = Math.ceil(file.size / chunkSize);

    for (let i = 0; i < totalChunks; i++) {
        const chunk = file.slice(i * chunkSize, (i + 1) * chunkSize);
        const formData = new FormData();
        formData.append('file', chunk, file.name);
        formData.append('fileName', file.name);
        formData.append('chunkIndex', i.toString());
        formData.append('totalChunks', totalChunks.toString());

        try {
            // Call the FileWidget upload endpoint
            const response = await TB.api.request('FileWidget', 'upload', formData, 'POST');
            if (response && response.error && response.error !== "none") {
                throw new Error(response.info?.help_text || 'Chunk upload failed.');
            }
        } catch (error) {
            TB.ui.Loader.hide(loaderId);
            TB.ui.Toast.showError(`Upload failed: ${error.message}`);
            return;
        }
    }

    TB.ui.Loader.hide(loaderId);
    TB.ui.Toast.showSuccess(`'${file.name}' uploaded. Adding to canvas...`);

    // Now that the file is in BlobStorage, create a share link to get a stable URL for it
    try {
        const shareResponse = await TB.api.request('FileWidget', 'create_share_link', { file_path: file.name });
        if (shareResponse && shareResponse.result?.data?.share_link) {
            const fileUrl = shareResponse.result.data.share_link;
            await addImageToCanvas(fileUrl, file.type.startsWith('application/pdf'));
        } else {
            throw new Error('Could not create a shareable link for the uploaded file.');
        }
    } catch (error) {
         TB.ui.Toast.showError(`Failed to place file on canvas: ${error.message}`);
    }
}

async function addImageToCanvas(sourceUrl, options = {}) {
const { isPdf = false, ...extraProps } = options;
    finalizeTextInput();
    const loaderId = TB.ui.Loader.show("Loading asset...");

    try {
        // Calculate center of the current view in world coordinates
        const viewCenterX = (canvas.width / (window.devicePixelRatio || 1)) / 2;
        const viewCenterY = (canvas.height / (window.devicePixelRatio || 1)) / 2;
        const worldCenterX = (viewCenterX - canvasAppState.offsetX) / canvasAppState.zoom;
        const worldCenterY = (viewCenterY - canvasAppState.offsetY) / canvasAppState.zoom;

        let imageElement;

        if (isPdf) {
            const pdfDoc = await pdfjsLib.getDocument(sourceUrl).promise;
            const page = await pdfDoc.getPage(1);
            const viewport = page.getViewport({ scale: 1.5 }); // Higher scale for better quality

            imageElement = {
                id: TB.utils.uniqueId('pdf_'), type: 'image', src: sourceUrl,
                x: worldCenterX - viewport.width / 2, y: worldCenterY - viewport.height / 2,
                width: viewport.width, height: viewport.height,
                originalWidth: viewport.width, originalHeight: viewport.height,
                opacity: 1.0, angle: 0,
                isPdf: true, pdfPageNum: 1, pdfTotalPages: pdfDoc.numPages,
                pdfDoc: pdfDoc // Store live PDF document object (not serialized)
            };
            // Render the first page onto a temporary canvas to get an image object
            imageElement.imgObject = await renderPdfPageToImage(page, viewport);

        } else { // It's a regular image
            const imgObject = await loadImageAsync(sourceUrl);
            const aspectRatio = imgObject.width / imgObject.height;
            const defaultWidth = 300 / canvasAppState.zoom;
            const defaultHeight = defaultWidth / aspectRatio;

            imageElement = {
                id: TB.utils.uniqueId('image_'), type: 'image', src: sourceUrl,
                x: worldCenterX - defaultWidth / 2, y: worldCenterY - defaultHeight / 2,
                width: defaultWidth, height: defaultHeight,
                originalWidth: imgObject.width, originalHeight: imgObject.heigh,
                imgObject: imgObject, opacity: 1.0, angle: 0, isPdf: false,
                ...extraProps
            };
        }

        canvasElements.push(imageElement);
        pushToHistory("Add Asset");
        renderCanvas();

        // Exclude non-serializable properties before sending to server
        const { imgObject, pdfDoc, ...elementToSend } = imageElement;
        sendActionToServer("element_add", elementToSend);
        clearOwnPreviewOnServer();
        TB.ui.Toast.showSuccess("Asset added to canvas.");

    } catch (err) {
        TB.logger.error("Failed to load asset from URL:", sourceUrl, err);
        TB.ui.Toast.showError(`Could not load asset: ${err.message}`);
    } finally {
        TB.ui.Loader.hide(loaderId);
    }
}

// NEW Helper function to render a PDF page
async function renderPdfPageToImage(page, viewport) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = viewport.width;
    tempCanvas.height = viewport.height;
    await page.render({ canvasContext: tempCtx, viewport: viewport }).promise;

    const img = new Image();
    img.src = tempCanvas.toDataURL();
    await new Promise(resolve => { img.onload = resolve; });
    return img;
}
async function updateCanvasElementWithNewAsset(elementId, newSourceUrl, extraProps = {}) {
    const elementIndex = canvasElements.findIndex(el => el.id === elementId);
    if (elementIndex === -1) {
        TB.logger.error("Could not find element to update:", elementId);
        return;
    }

    const loaderId = TB.ui.Loader.show("Updating element...");
    try {
        const element = canvasElements[elementIndex];
        const newImgObject = await loadImageAsync(newSourceUrl);

        // Update properties
        element.src = newSourceUrl;
        element.imgObject = newImgObject;

        // Retain current scale
        const oldOriginalWidth = element.originalWidth || element.width;
        const currentScale = element.width / oldOriginalWidth;

        element.originalWidth = newImgObject.width;
        element.originalHeight = newImgObject.height;

        // Recalculate width/height based on new aspect ratio but same scale
        element.width = element.originalWidth * currentScale;
        element.height = element.originalHeight * currentScale;

        // Apply any other property updates
        Object.assign(element, extraProps);

        pushToHistory("Update Element Asset");
        renderCanvas();

        const { imgObject, pdfDoc, ...elementToSend } = element;
        sendActionToServer("element_update", elementToSend);

    } catch (err) {
        TB.logger.error("Failed to update canvas element asset:", err);
        TB.ui.Toast.showError("Failed to update element.");
    } finally {
        TB.ui.Loader.hide(loaderId);
    }
}
async function handleAddMarkdown() {
    let existingMarkdown = '';
    let existingOptions = {
        fontSize: 14,
        bgColor: '#ffffff',
        textColor: '#000000',
        width: 400
    };

    const selected = selectedElements.length === 1 ? selectedElements[0] : null;
    if (selected && selected.isMarkdown) {
        existingMarkdown = selected.originalMarkdown || '';
        // Load existing styles if they are stored on the element, otherwise use defaults
        existingOptions.fontSize = selected.mdFontSize || 14;
        existingOptions.bgColor = selected.mdBgColor || '#ffffff';
        existingOptions.textColor = selected.mdTextColor || '#000000';
        existingOptions.width = selected.mdWidth || 400;
    }

    // Use TB.ui.Modal.show for a custom form
    const modalId = TB.utils.uniqueId('mdModal_');
    TB.ui.Modal.show({
        id: modalId,
        title: selected ? "Edit Markdown Text" : "Add Markdown Text",
        content: `
            <div class="tb-form-group tb-mb-3">
                <textarea id="markdownInput_${modalId}" class="tb-input" style="width: 100%; height: 200px; min-height: 150px; font-family: monospace;" placeholder="Enter your Markdown here...">${TB.utils.escapeHtml(existingMarkdown)}</textarea>
            </div>
            <div class="tb-d-flex tb-justify-between tb-gap-3">
                <div class="tb-form-group">
                    <label for="mdFontSize_${modalId}" class="tb-form-label tb-form-label-sm">Font Size</label>
                    <input type="number" id="mdFontSize_${modalId}" class="tb-input tb-input-sm" value="${existingOptions.fontSize}" min="8" max="72">
                </div>
                <div class="tb-form-group">
                    <label for="mdWidth_${modalId}" class="tb-form-label tb-form-label-sm">Width</label>
                    <input type="number" id="mdWidth_${modalId}" class="tb-input tb-input-sm" value="${existingOptions.width}" min="100" max="2000">
                </div>
                <div class="tb-form-group">
                    <label for="mdTextColor_${modalId}" class="tb-form-label tb-form-label-sm">Text Color</label>
                    <input type="color" id="mdTextColor_${modalId}" class="tb-input tb-input-sm" value="${existingOptions.textColor}">
                </div>
                <div class="tb-form-group">
                    <label for="mdBgColor_${modalId}" class="tb-form-label tb-form-label-sm">BG Color</label>
                    <input type="color" id="mdBgColor_${modalId}" class="tb-input tb-input-sm" value="${existingOptions.bgColor}">
                </div>
            </div>
        `,
        buttons: [
            { text: "Cancel", variant: 'secondary', action: modal => modal.hide() },
            {
                text: selected ? "Update on Canvas" : "Add to Canvas",
                variant: 'primary',
                action: async (modal) => {
                    const markdownText = document.getElementById(`markdownInput_${modalId}`).value;
                    const fontSize = document.getElementById(`mdFontSize_${modalId}`).value;
                    const width = document.getElementById(`mdWidth_${modalId}`).value;
                    const textColor = document.getElementById(`mdTextColor_${modalId}`).value;
                    const bgColor = document.getElementById(`mdBgColor_${modalId}`).value;

                    modal.hide(); // Hide modal immediately for better UX

                    if (selected && !markdownText.trim()) {
                        deleteSelectedElements();
                        return;
                    }
                    if (!markdownText.trim()) return;

                    const loaderId = TB.ui.Loader.show("Converting Markdown...");
                    try {
                        const payload = {
                            markdown_text: markdownText,
                            width: parseInt(width),
                            font_size: parseInt(fontSize),
                            text_color: textColor,
                            bg_color: bgColor
                        };

                        const response = await TB.api.request('Canvas', 'markdown_to_svg', payload);

                        if (response && response.result?.data?.svg_data_url) {
                            const data = response.result.data;
                            const newAssetProps = {
                                isMarkdown: true,
                                originalMarkdown: data.original_markdown,
                                mdFontSize: parseInt(fontSize),
                                mdWidth: parseInt(width),
                                mdTextColor: textColor,
                                mdBgColor: bgColor
                            };

                            if (selected) {
                                await updateCanvasElementWithNewAsset(selected.id, data.svg_data_url, newAssetProps);
                            } else {
                                await addImageToCanvas(data.svg_data_url, newAssetProps);
                            }
                        } else {
                            throw new Error(response.info?.help_text || "Server failed to convert Markdown.");
                        }
                    } catch (error) {
                        TB.ui.Toast.showError(`Markdown conversion failed: ${error.message}`);
                    } finally {
                        TB.ui.Loader.hide(loaderId);
                    }
                }
            }
        ]
    });
}

    const init = () => {
        initializeCanvasStudio();
        checkAndJoinCollabSessionFromUrl();
    }
    window.TB.onLoaded(init);
</script>
"""
