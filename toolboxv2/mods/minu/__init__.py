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
    use_ssr = ssr is not None or format in ("html", "full-html")

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
                        const root = document.getElementById('minu-root');
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

@export(mod_name=Name, name="initialize_flows", initial=True)
def initialize_flows(app: App) -> Result:
    """Initialize module and register all views"""
    # Register UI route
    app.run_any(
        ("CloudM", "add_ui"),
        name="MinuFlows",
        title="Minu UI Flows",
        path=f"/api/{Name}/sync_flows",
        description="Minu UI Framework Flows hub",
        auth=True
    )

    return Result.ok(info="Minu UI Framework initialized")


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

    result = await session.handle_event(event_data)

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


@export(mod_name=Name, websocket_handler="ui")
def register_ui_websocket(app: App):
    async def on_connect(session_data: Dict[str, Any], conn_id=None, **kwargs):
        conn_id = conn_id or session_data.get("connection_id", "unknown")
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
        payload: dict, session_data: Dict[str, Any], conn_id=None, **kwargs
    ):
        conn_id = conn_id or session_data.get("connection_id", "unknown")
        session = _sessions.get(conn_id)

        if not session:
            return

        try:
            msg_type = payload.get("type")

            if msg_type == "subscribe":
                view_name = payload.get("viewName")
                view_class = get_view_class(view_name)

                if view_class:
                    view = view_class()
                    # Apply props if provided
                    props = payload.get("props", {})
                    if props:
                        for key, value in props.items():
                            if hasattr(view, key):
                                attr = getattr(view, key)
                                if hasattr(attr, "value"):
                                    attr.value = value

                    session.register_view(view)
                    await session.send_full_render(view)
                    await session.force_flush()
                else:
                    await app.ws_send(
                        conn_id,
                        {
                            "type": "error",
                            "message": f"View '{view_name}' not found"
                        }
                    )

            elif msg_type == "event":
                await session.handle_event(payload)

                await app.ws_send(
                    conn_id,
                    {
                        "type": "event_result",
                        "viewId": payload.get("viewId"),
                        "handler": payload.get("handler"),
                        "result": {"success": True},
                    },
                )

            elif msg_type == "state_update":
                view_id = payload.get("viewId")
                path = payload.get("path")
                value = payload.get("value")

                view = session.get_view(view_id)
                if view:
                    parts = path.split(".")
                    state_name = parts[-1] if len(parts) == 1 else parts[0]
                    if hasattr(view, state_name):
                        getattr(view, state_name).value = value
                        await session.force_flush()

        except Exception as e:
            app.logger.error(f"[Minu] WebSocket error: {e}", exc_info=True)
            await app.ws_send(conn_id, {"type": "error", "message": str(e)})

    async def on_disconnect(session_data: Dict[str, Any], conn_id=None, **kwargs):
        conn_id = conn_id or session_data.get("connection_id", "unknown")
        app.logger.info(f"[Minu] WebSocket disconnected: {conn_id}")
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
    Custom,
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
]
