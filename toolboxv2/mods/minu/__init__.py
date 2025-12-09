"""
Minu UI Framework - Enhanced Toolbox Module Integration
=======================================================
Complete SSR support with Toolbox integration
"""

import asyncio
import json
import weakref
from typing import Dict, Any, Optional, Callable, Type
from dataclasses import dataclass, field

from toolboxv2 import App, Result, RequestData, get_app

from .core import (
    MinuView,
    MinuSession,
    Component,
    Card,
    Text,
    Heading,
    Button,
    Row,
    Column,
)
from .flow_integration import scan_and_register_flows
from .examples import get_demo_page
from .flows import (
    ui_for_data,
    data_card,
    data_table,
    form_for,
    stats_grid,
    action_bar,
    ui_result,
)
from .shared_api import get_shared_websocket_handlers

# Module metadata
Name = "Minu"
export = get_app(f"{Name}.Export").tb
version = "0.1.0"

# Global session storage (per-user sessions)
_sessions: Dict[str, MinuSession] = {}
_view_registry: Dict[str, Type[MinuView]] = {}


# ============================================================================
# VIEW REGISTRY
# ============================================================================


def register_view(name: str, view_class: Type[MinuView]):
    """
    Register a view class for later instantiation.

    Usage in your module:
        from minu import register_view

        class MyDashboard(MinuView):
            ...

        register_view("my_dashboard", MyDashboard)
    """
    _view_registry[name] = view_class


def get_view_class(name: str) -> Optional[Type[MinuView]]:
    """Get a registered view class by name"""
    return _view_registry.get(name)


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================


def get_or_create_session(session_id: str) -> MinuSession:
    """Get existing session or create new one"""
    if session_id not in _sessions:
        _sessions[session_id] = MinuSession(session_id)
    return _sessions[session_id]


def cleanup_session(session_id: str):
    """Remove a session"""
    if session_id in _sessions:
        del _sessions[session_id]


# ============================================================================
# ENHANCED RENDER ENDPOINT WITH FULL SSR
# ============================================================================

@export(mod_name=Name, name="render", api=True, version=version, request_as_kwarg=True)
async def render_view(
    app: App,
    request: RequestData,
    view: str = None,
    props: Optional[Dict[str, Any]] = None,
    ssr: Optional[str] = None,
    format: str = "auto",  # auto, json, html, full-html
    **kwargs
) -> Result:
    """
    Enhanced render endpoint with full SSR support.

    Modes:
    - JSON (default): Returns view definition for client-side rendering
    - SSR HTML: Returns pre-rendered HTML fragment
    - Full HTML: Returns complete HTML document

    GET /api/Minu/render?view=my_dashboard&ssr=true&format=full-html
    POST /api/Minu/render {"view": "my_dashboard", "props": {...}, "ssr": "true"}

    Args:
        view: View name to render
        props: Optional props for the view
        ssr: Enable server-side rendering ("true", "1", or any truthy value)
        format: Output format ("auto", "json", "html", "full-html")
            - auto: JSON for API calls, full-html for browser requests
            - json: Always return JSON (for AJAX)
            - html: Return HTML fragment only
            - full-html: Return complete HTML document

    Returns:
        Result object with rendered content
    """
    # Get session ID from request
    session_data = request.session if hasattr(request, "session") else {}
    session_id = session_data.get("session_id", "anonymous")
    view_name = view or kwargs.get("view", kwargs.get("view_name", ""))

    if not view_name:
        error_msg = "View name is required"
        return Result.default_user_error(
            info=error_msg, exec_code=400
        ) if format == "json" else Result.html(
            f'<div class="alert alert-error">{error_msg}</div>'
        )

    # Determine if SSR should be used
    use_ssr = ssr is not None or format in ("html")

    if use_ssr:
        format = "html"

    # Auto-detect format from request headers if "auto"
    if format == "auto":
        accept_header = request.request.headers.accept
        is_browser_request = "text/html" in accept_header

        if use_ssr and is_browser_request:
            format = "full-html"
        elif use_ssr:
            format = "html"
        else:
            format = "json"

    # Get or create session
    session = get_or_create_session(session_id)

    # Get view class
    view_class = get_view_class(view_name)
    if not view_class:
        error_msg = f"View '{view_name}' not registered"
        app.logger.error(f"[Minu] {error_msg}")

        if format == "json":
            return Result.default_user_error(info=error_msg, exec_code=404)
        else:
            return Result.html(
                f'''<div class="alert alert-error" role="alert">
                    <strong>View Not Found</strong>
                    <p>{error_msg}</p>
                    <p class="text-sm text-secondary mt-2">
                        Available views: {", ".join(_view_registry.keys()) or "None"}
                    </p>
                </div>'''
            )

    try:
        # Instantiate view
        view_instance = view_class()

        # Apply props if provided
        if props:
            for key, value in props.items():
                if hasattr(view_instance, key):
                    attr = getattr(view_instance, key)
                    if hasattr(attr, "value"):
                        attr.value = value

        # Register view in session (for future WebSocket updates)
        session.register_view(view_instance)

        # Render based on format
        if format == "json":
            # Return JSON representation for client-side rendering
            return Result.json(
                data={
                    "view": view_instance.to_dict(),
                    "sessionId": session.session_id,
                    "viewId": view_instance._view_id,
                    "mode": "client-side",
                }
            )

        elif format == "html":
            # Return HTML fragment only (for HTMX swaps)
            props_json = json.dumps(props or {})
            html_bootloader = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Minu: {view_name}</title>
                <style>
                    body {{ margin: 0; padding: 0; background-color: #f9fafb; font-family: system-ui, -apple-system, sans-serif; }}
                    #minu-root {{ padding: 1rem; max-width: 1200px; margin: 0 auto; }}
                    .minu-loading {{
                        display: flex; justify-content: center; align-items: center;
                        height: 50vh; color: #6b7280; flex-direction: column; gap: 1rem;
                    }}
                    .spinner {{
                        width: 2rem; height: 2rem; border: 3px solid #e5e7eb;
                        border-top-color: #3b82f6; border-radius: 50%; animation: spin 1s linear infinite;
                    }}
                    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
                </style>
            </head>
            <body>
                <div id="minu-root">
                    <div class="minu-loading">
                        <div class="spinner"></div>
                        <p>Loading View: <strong>{view_name}</strong>...</p>
                    </div>
                </div>

                <script type="module" unsave="true">
                    // Bootloader Logic
                    async function boot() {{
                        const root = document.getElementById('minu-root') || document.getElementById('MainContent');
                        const viewName = "{view_name}";
                        const initialProps = {props_json};

                        try {{
                            // 1. Wait for Toolbox (TB) global object
                            // Usually injected by the platform, or we wait a bit
                            let attempts = 0;
                            while (!window.TB && attempts < 20) {{
                                await new Promise(r => setTimeout(r, 100));
                                attempts++;
                            }}

                            if (!window.TB || !window.TB.ui) {{
                                // Fallback: If not inside Toolbox shell, we might fail or need to load script manually
                                // For now, show specific error
                                throw new Error("Toolbox Framework (TBJS) not found. Please access via CloudM.");
                            }}

                            // 2. Mount the view using the client-side library
                            await window.TB.ui.mountMinuView(root, viewName, initialProps);

                        }} catch (err) {{
                            console.error("[Minu Boot] Error:", err);
                            root.innerHTML = `
                                <div style="background:#fee2e2; color:#991b1b; padding:1rem; border-radius:8px; border:1px solid #fecaca;">
                                    <strong>Error loading view:</strong><br>
                                    ${{err.message}}
                                </div>
                            `;
                        }}
                    }}

                    // Run bootloader
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', boot);
                    }} else {{
                        boot();
                    }}
                </script>
            </body>
            </html>
                        """
            return Result.html(html_bootloader)


    except Exception as e:
        app.logger.error(f"[Minu] Error rendering view {view_name}: {e}", exc_info=True)
        error_html = f'''
<div class="alert alert-error" role="alert">
    <strong>Render Error</strong>
    <p>Failed to render view '{view_name}'</p>
    <details class="mt-2">
        <summary class="cursor-pointer text-sm">Error details</summary>
        <pre class="mt-2 p-2 bg-neutral-800 text-neutral-100 rounded text-xs overflow-x-auto">
{str(e)}
        </pre>
    </details>
</div>'''

        return Result.default_internal_error(
            info=str(e)
        ) if format == "json" else Result.html(error_html)


# ============================================================================
# REMAINING ENDPOINTS (unchanged but updated for consistency)
# ============================================================================


@export(mod_name=Name, name="sync_flows", api=True, version=version)
async def sync_flow_uis(app: App) -> Result:
    """
    Scans all available Toolbox Flows and registers UI views for them.

    GET /api/Minu/sync_flows
    """
    try:
        html_content = scan_and_register_flows(app)
        return Result.html(html_content)
    except Exception as e:
        return Result.default_internal_error(info=str(e))

@export(
    mod_name=Name,
    name="list_flows",
    api=True,
    api_methods=["GET", "POST"],
    version=version,
    request_as_kwarg=True,
)
async def list_flows(
    app: App, request: RequestData, only_custom_ui: bool = True, **kwargs
) -> Result:
    """
    List all available flows for the dashboard.

    Args:
        only_custom_ui: If True, only return flows with custom UI (default: True)

    Returns:
        List of flow info objects with name, title, description, icon, path, auth

    GET /api/Minu/list_flows
    GET /api/Minu/list_flows?only_custom_ui=false
    """

    # 1. Load all flows
    try:
        from toolboxv2.flows import flows_dict

        all_flows = flows_dict()
    except Exception as e:
        app.logger.error(f"[Minu] Could not load flows: {e}")
        return Result.default_user_error(info=f"Could not load flows: {e}")

    # 2. Load custom UIs
    try:
        from toolboxv2.flows import flows_dict as get_flows

        custom_uis = get_flows(ui=True)
    except:
        custom_uis = {}

    scan_and_register_flows(app)
    # 3. Build flow list
    flows_list = []

    for flow_name, run_func in all_flows.items():
        has_custom_ui = flow_name in custom_uis

        # Skip if only_custom_ui and no custom UI
        if only_custom_ui and not has_custom_ui:
            continue

        # Extract docstring for description
        doc = ""
        if run_func.__doc__:
            doc = run_func.__doc__.strip().split("\n")[0]
            if len(doc) > 120:
                doc = doc[:117] + "..."

        # Build flow info
        flow_info = {
            "name": flow_name,
            "title": flow_name.replace("_", " ").title(),
            "description": doc or "Interactive Flow Application",
            "icon": "account_tree",  # Default icon
            "path": f"/api/Minu/render?view={flow_name}&ssr=True",
            "auth": False,  # Can be extended to check flow-specific auth
            "has_custom_ui": has_custom_ui,
            "type": "flow",
        }

        # Check for custom metadata in the UI function
        custom_ui_func = custom_uis.get(flow_name, {}).get("ui")
        if custom_ui_func and hasattr(custom_ui_func, "_minu_meta"):
            meta = custom_ui_func._minu_meta
            flow_info.update(
                {
                    "title": meta.get("title", flow_info["title"]),
                    "description": meta.get("description", flow_info["description"]),
                    "icon": meta.get("icon", flow_info["icon"]),
                    "auth": meta.get("auth", flow_info["auth"]),
                    "bg_img_url": meta.get("bg_img_url", flow_info["bg_img_url"])
                }
            )
        flow_info.update(custom_uis.get(flow_name, {}))
        if "ui" in flow_info:
            del flow_info["ui"]

        # Register the view if not already registered
        if flow_name not in _view_registry:
            try:
                custom_ui = custom_uis.get(flow_name)

                # Import here to avoid circular imports
                from .flow_integration import FlowWrapperView

                def make_init(fn, rf, cu):
                    def __init__(self):
                        FlowWrapperView.__init__(self, fn, rf, cu)

                    return __init__

                DynamicView = type(
                    f"FlowView_{flow_name}",
                    (FlowWrapperView,),
                    {"__init__": make_init(flow_name, run_func, custom_ui)},
                )

                register_view(flow_name, DynamicView)

            except Exception as e:
                app.logger.warning(f"[Minu] Could not register view for {flow_name}: {e}")

        flows_list.append(flow_info)

    # 4. Sort by title
    flows_list.sort(key=lambda x: x["title"].lower())

    return Result.ok(data=flows_list)


# ============================================================================
# DECORATOR FÜR FLOW METADATA
# ============================================================================


def flow_ui_meta(
    title: str = None, description: str = None, icon: str = None, auth: bool = False, bg_img_url: str = None
):
    """
    Decorator to add metadata to a flow UI function.

    Usage in your flow file:
        @flow_ui_meta(title="My Cool App", icon="rocket", auth=True)
        def ui(view):
            return Column(...)
    """

    def decorator(func):
        func._minu_meta = {
            "title": title,
            "description": description,
            "icon": icon,
            "auth": auth,
            "bg_img_url": bg_img_url
        }
        return func

    return decorator


@export(
    mod_name=Name,
    name="event",
    api=True,
    api_methods=["POST"],
    version=version,
    request_as_kwarg=True,
)
async def handle_event(
    app: App,
    request: RequestData,
    session_id: str,
    view_id: str,
    handler: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Result:
    """
    Handle a UI event from the frontend.

    POST /api/Minu/event
    {
        "session_id": "...",
        "view_id": "...",
        "handler": "button_clicked",
        "payload": {...}
    }
    """
    session = _sessions.get(session_id)
    if not session:
        return Result.default_user_error(
            info=f"Session '{session_id}' not found", exec_code=404
        )

    event_data = {
        "type": "event",
        "viewId": view_id,
        "handler": handler,
        "payload": payload or {},
    }

    result = await session.handle_event(event_data, request=request, app=app)

    if "error" in result:
        return Result.default_user_error(info=result["error"])

    return Result.json(data=result)


@export(
    mod_name=Name,
    name="state",
    api=True,
    api_methods=["POST"],
    version=version,
    request_as_kwarg=True,
)
async def update_state(
    app: App, request: RequestData, session_id: str, view_id: str, path: str, value: Any
) -> Result:
    """
    Update view state from the frontend (two-way binding).

    POST /api/Minu/state
    {
        "session_id": "...",
        "view_id": "...",
        "path": "name",
        "value": "New Value"
    }
    """
    session = _sessions.get(session_id)
    if not session:
        return Result.default_user_error(info=f"Session '{session_id}' not found")

    view = session.get_view(view_id)
    if not view:
        return Result.default_user_error(info=f"View '{view_id}' not found")

    # Parse path and update state
    parts = path.split(".")
    state_name = parts[-1] if len(parts) == 1 else parts[0]

    if hasattr(view, state_name):
        state = getattr(view, state_name)
        if hasattr(state, "value"):
            state.value = value
            return Result.json(data={"success": True, "path": path, "value": value})

    return Result.default_user_error(info=f"State '{path}' not found in view")


@export(mod_name=Name, name="list_views", api=True, version=version)
async def list_registered_views(app: App) -> Result:
    """
    List all registered view classes.

    GET /api/Minu/list_views
    """
    views = []
    for name, view_class in _view_registry.items():
        views.append(
            {
                "name": name,
                "className": view_class.__name__,
                "docstring": view_class.__doc__ or "",
            }
        )

    return Result.json(data={"views": views})


# ============================================================================
# WEBSOCKET HANDLER
# ============================================================================


@export(mod_name=Name, websocket_handler="ui", request_as_kwarg=True)
def register_ui_websocket(app: App, request: RequestData = None):
    shared_handlers = get_shared_websocket_handlers(app)
    async def on_connect(session: Dict[str, Any], conn_id=None, **kwargs):
        conn_id = conn_id or session.get("connection_id", "unknown")
        app.logger.info(f"[Minu] WebSocket connected: {conn_id}")

        session = get_or_create_session(conn_id)


        async def send_message(msg: str):
            await app.ws_send(conn_id, json.loads(msg))

        session.set_send_callback(send_message)

        await app.ws_send(
            conn_id,
            {
                "type": "connected",
                "sessionId": session.session_id,
                "message": "Connected to Minu UI",
            },
        )

        return {"accept": True}

    async def on_message(
        payload: dict, session: Dict[str, Any], conn_id=None, **kwargs
    ):
        """Handle incoming WebSocket messages."""
        conn_id = conn_id or session.get("connection_id", "unknown")
        session = _sessions.get(conn_id)

        if not session:
            app.logger.warning(f"[Minu] No session for connection: {conn_id}")
            return

        try:
            msg_type = payload.get("type")

            if msg_type.startswith("shared_"):
                result = await shared_handlers["handle_message"](
                    conn_id, msg_type, payload, request
                )
                if result:
                    await app.ws_send(conn_id, result)
                    return

            elif msg_type == "subscribe":
                view_name = payload.get("viewName")
                view_class = get_view_class(view_name)

                if view_class:
                    view = view_class()

                    # Props anwenden
                    props = payload.get("props", {})
                    if props:
                        for key, value in props.items():
                            if hasattr(view, key):
                                attr = getattr(view, key)
                                if hasattr(attr, "value"):
                                    attr.value = value

                    session.register_view(view)
                    await session.send_full_render(view)
                else:
                    await app.ws_send(
                        conn_id,
                        {"type": "error", "message": f"View '{view_name}' not found"},
                    )

            elif msg_type == "event":
                view_id = payload.get("viewId")
                handler_name = payload.get("handler")
                event_payload = payload.get("payload", {})

                # Event verarbeiten
                result = await session.handle_event(payload, request=kwargs.get("request"), app=app)

                # Prüfe ob es ein Error gab
                if isinstance(result, dict) and result.get("error"):
                    await app.ws_send(
                        conn_id,
                        {
                            "type": "event_result",
                            "viewId": view_id,
                            "handler": handler_name,
                            "result": result,
                        },
                    )
                    return

                # Für Flow-Views: Komplettes Re-Render senden
                # weil sich der gesamte UI-State ändern kann
                view = session.get_view(view_id)
                if view:
                    # Re-Render senden
                    await session.send_full_render(view)

                # Event-Result senden
                await app.ws_send(
                    conn_id,
                    {
                        "type": "event_result",
                        "viewId": view_id,
                        "handler": handler_name,
                        "result": result
                        if isinstance(result, dict)
                        else {"success": True},
                    },
                )

            elif msg_type == "state_update":
                view_id = payload.get("viewId")

                path = payload.get("path")

                value = payload.get("value")

                view = session.get_view(view_id)

                if view:
                    parts = path.split(".")

                    # Prüfe ob es ein direktes State-Attribut ist

                    if len(parts) == 1:
                        state_name = parts[0]

                        # 1. Direkte State-Attribute (z.B. status, error_msg)

                        if hasattr(view, state_name) and hasattr(
                            getattr(view, state_name), "value"
                        ):
                            getattr(view, state_name).value = value

                        # 2. Binding-Felder -> in inputs.value Dict speichern

                        elif hasattr(view, "inputs") and hasattr(view.inputs, "value"):
                            if view.inputs.value is None:
                                view.inputs.value = {}

                            # Merge statt überschreiben

                            current = (
                                view.inputs.value.copy()
                                if isinstance(view.inputs.value, dict)
                                else {}
                            )

                            current[state_name] = value

                            view.inputs.value = current

                    else:
                        # Nested path: viewId.stateName

                        state_name = parts[1] if parts[0] == view_id else parts[0]

                        if hasattr(view, state_name) and hasattr(
                            getattr(view, state_name), "value"
                        ):
                            getattr(view, state_name).value = value

                    # Patches flushen

                    await session.force_flush()

        except Exception as e:
            import traceback

            traceback.print_exc()
            app.logger.error(f"[Minu] WebSocket error: {e}")
            await app.ws_send(conn_id, {"type": "error", "message": str(e)})

    async def on_disconnect(session: Dict[str, Any], conn_id=None, **kwargs):
        conn_id = conn_id or session.get("connection_id", "unknown")
        app.logger.info(f"[Minu] WebSocket disconnected: {conn_id}")
        shared_handlers["cleanup"](conn_id)
        cleanup_session(conn_id)

    return {
        "on_connect": on_connect,
        "on_message": on_message,
        "on_disconnect": on_disconnect,
    }


# ============================================================================
# SSE ENDPOINT (Alternative to WebSocket)
# ============================================================================


@export(
    mod_name=Name,
    name="stream",
    api=True,
    api_methods=["GET"],
    version=version,
    request_as_kwarg=True,
)
async def stream_updates(
    app: App,
    request: RequestData,
    view_name: str,
    props: Optional[str] = None,
) -> Result:
    """
    SSE endpoint for real-time UI updates.

    GET /api/Minu/stream?view_name=dashboard&props={"key":"value"}
    """
    parsed_props = {}
    if props:
        try:
            parsed_props = json.loads(props)
        except:
            pass

    session_data = request.session if hasattr(request, "session") else {}
    session_id = session_data.get("session_id", f"sse-{id(request)}")

    session = get_or_create_session(session_id)

    view_class = get_view_class(view_name)
    if not view_class:
        return Result.default_user_error(info=f"View '{view_name}' not registered")

    view = view_class()
    if parsed_props:
        for key, value in parsed_props.items():
            if hasattr(view, key):
                attr = getattr(view, key)
                if hasattr(attr, "value"):
                    attr.value = value

    session.register_view(view)

    async def event_generator():
        yield {"event": "render", "data": view.to_dict()}

        update_queue = asyncio.Queue()

        async def queue_update(msg: str):
            await update_queue.put(json.loads(msg))

        session.set_send_callback(queue_update)

        try:
            while True:
                try:
                    update = await asyncio.wait_for(update_queue.get(), timeout=30)
                    yield {"event": update.get("type", "update"), "data": update}
                except asyncio.TimeoutError:
                    yield {
                        "event": "heartbeat",
                        "data": {"sessionId": session.session_id},
                    }
        except asyncio.CancelledError:
            pass
        finally:
            cleanup_session(session_id)

    return Result.sse(stream_generator=event_generator())


# ============================================================================
# HELPER EXPORTS
# ============================================================================

from .core import (
    State,
    ReactiveState,
    MinuView,
    MinuSession,
    Component,
    ComponentType,
    ComponentStyle,
    Card,
    Text,
    Heading,
    Button,
    Input,
    Select,
    Checkbox,
    Switch,
    Row,
    Column,
    Grid,
    Spacer,
    Divider,
    Alert,
    Progress,
    Spinner,
    Table,
    List,
    ListItem,
    Icon,
    Image,
    Badge,
    Modal,
    Widget,
    Form,
    Tabs,
    Textarea,
    Custom,
    Dynamic,
)

__all__ = [
    # Module functions
    "register_view",
    "get_view_class",
    "get_or_create_session",
    "cleanup_session",
    # Core re-exports
    "State",
    "ReactiveState",
    "MinuView",
    "MinuSession",
    "Component",
    "ComponentType",
    "ComponentStyle",
    "Card",
    "Text",
    "Heading",
    "Button",
    "Input",
    "Select",
    "Checkbox",
    "Switch",
    "Row",
    "Column",
    "Grid",
    "Textarea",
    "Spacer",
    "Divider",
    "Alert",
    "Progress",
    "Spinner",
    "Table",
    "List",
    "ListItem",
    "Icon",
    "Image",
    "Badge",
    "Modal",
    "Widget",
    "Form",
    "Tabs",
    "Custom",
    "Dynamic"
]
