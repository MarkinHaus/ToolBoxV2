#!/usr/bin/env python3
"""
minu_bridge.py - FastTB ↔ Minu UI Bridge
=========================================

Connects MinuView classes to FastTB routing with zero boilerplate.
Generates HTTP + WebSocket endpoints from a single decorator.

Usage:
    from toolboxv2.utils.workers.fast_tb import FastTB
    from toolboxv2.mods.Minu.minu_bridge import MinuBridge

    app = FastTB(title="MyApp")
    minu = MinuBridge(app)

    @minu.view("/dashboard")
    class Dashboard(MinuView):
        greeting = State("Hello!")

        def render(self):
            return Column(
                Heading(self.greeting.value),
                Text(f"User: {self.user.name}"),
            )

    # This auto-registers:
    #   GET  /dashboard          → SSR HTML (browser) or JSON (API)
    #   POST /dashboard/event    → event dispatch (button clicks etc.)
    #   WS   /ws/dashboard       → live updates via WebSocket

Generated endpoints:
    GET  {path}           Render view (HTML if Accept: text/html, else JSON)
    GET  {path}?format=json  Force JSON response
    GET  {path}?format=html  Force HTML response
    POST {path}/event     Dispatch UI event to view handler
    WS   {path}           WebSocket for live state updates
"""

import asyncio
import json
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class MinuBridge:
    """Bridge between FastTB and Minu UI framework.

    Provides a @minu.view() decorator that auto-generates
    HTTP + WS endpoints for a MinuView class.
    """

    def __init__(self, fast_tb_app, app_instance=None):
        """
        Args:
            fast_tb_app: FastTB instance to register routes on
            app_instance: Optional ToolBoxV2 App instance (for session verification, mod loading)
        """
        from toolboxv2.utils.workers.fast_tb import FastTB
        self._ftb: FastTB = fast_tb_app
        self._app = app_instance or fast_tb_app.app_instance
        self._sessions: Dict[str, Any] = {}  # session_id -> MinuSession
        self._view_registry: Dict[str, Type] = {}  # path -> view class

    def view(self, path: str, require_auth: bool = False):
        """Register a MinuView as a FastTB route.

        Args:
            path: URL path (e.g. "/dashboard")
            require_auth: If True, returns 401 for anonymous users

        Returns:
            Decorator that accepts a MinuView subclass
        """
        def decorator(view_class):
            self._view_registry[path] = view_class
            self._register_http_routes(path, view_class, require_auth)
            self._register_ws_route(path, view_class)
            return view_class

        return decorator

    # =========================================================================
    # HTTP Route Generation
    # =========================================================================

    def _register_http_routes(self, path: str, view_class: Type, require_auth: bool):
        """Register GET (render) and POST (event) routes."""
        from toolboxv2.utils.workers.server_worker import ParsedRequest
        from toolboxv2.utils.workers.session import SessionData

        bridge = self

        # --- GET: Render view ---
        @self._ftb.get(path, name=f"minu_{view_class.__name__}_render")
        async def render_view(request: ParsedRequest, session: SessionData,
                              format: str = "auto", **kwargs):
            # Auth gate
            if require_auth and (not session or not session.is_authenticated):
                return (401, {"error": "Authentication required"})

            view, minu_session = bridge._create_view(
                view_class, session, request
            )

            # Run on_mount if defined
            if hasattr(view, "on_mount") and callable(view.on_mount):
                try:
                    await view.on_mount()
                except Exception as e:
                    logger.error(f"[MinuBridge] on_mount error: {e}")

            # Determine output format
            if format == "auto":
                accept = request.headers.get("accept", "")
                format = "html" if "text/html" in accept else "json"

            view_dict = view.to_dict()

            if format == "json":
                return {
                    "view": view_dict,
                    "sessionId": minu_session.session_id,
                    "viewId": view._view_id,
                }

            # HTML: render bootloader page
            return bridge._render_html(view_class.__name__, view_dict, minu_session)

        # --- POST: Event dispatch ---
        @self._ftb.post(f"{path}/event", name=f"minu_{view_class.__name__}_event")
        async def handle_event(request: ParsedRequest, session: SessionData):
            if require_auth and (not session or not session.is_authenticated):
                return (401, {"error": "Authentication required"})

            body = request.json_data or {}
            view_id = body.get("viewId", "")
            handler_name = body.get("handler", "")
            event_data = body.get("data", {})

            if not handler_name:
                return (400, {"error": "Missing handler name"})

            minu_session = bridge._get_session(session)
            view = minu_session.get_view(view_id) if minu_session else None

            if not view:
                return (404, {"error": f"View {view_id} not found in session"})

            handler = getattr(view, handler_name, None)
            if not handler or not callable(handler):
                return (404, {"error": f"Handler '{handler_name}' not found"})

            try:
                result = handler(event_data)
                if asyncio.iscoroutine(result):
                    result = await result

                # Collect state patches
                patches = view.get_patches()
                return {
                    "ok": True,
                    "patches": patches,
                    "result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(result),
                }
            except Exception as e:
                logger.error(f"[MinuBridge] Event handler error: {e}")
                return (500, {"error": str(e)})

    # =========================================================================
    # WebSocket Route Generation
    # =========================================================================

    def _register_ws_route(self, path: str, view_class: Type):
        """Register WebSocket handler for live updates."""
        bridge = self
        ws_path = f"/ws{path}" if not path.startswith("/ws") else path

        @self._ftb.websocket(ws_path)
        class MinuWSHandler:
            async def on_connect(self_, conn_id, session):
                logger.info(f"[MinuBridge WS] {conn_id} connected to {ws_path}")

            async def on_message(self_, payload, conn_id, session, request=None):
                msg_type = payload.get("type", "")

                if msg_type == "init":
                    # Client sends init with session info
                    session_id = payload.get("sessionId", conn_id)
                    minu_session = bridge._sessions.get(session_id)

                    if minu_session:
                        # Set up send callback for this connection
                        async def ws_send(msg_str, _conn=conn_id):
                            from toolboxv2 import get_app
                            app = get_app(from_="minu_bridge_ws")
                            await app.ws_send(_conn, json.loads(msg_str) if isinstance(msg_str, str) else msg_str)

                        minu_session.set_send_callback(ws_send)
                        return {"type": "init_ok", "sessionId": session_id}
                    return {"type": "error", "message": "Session not found"}

                elif msg_type == "event":
                    return await bridge._handle_ws_event(payload, conn_id)

                elif msg_type == "state_update":
                    return await bridge._handle_ws_state_update(payload, conn_id)

            async def on_disconnect(self_, conn_id, session):
                logger.info(f"[MinuBridge WS] {conn_id} disconnected")

    async def _handle_ws_event(self, payload: dict, conn_id: str) -> dict:
        """Handle event message over WebSocket."""
        session_id = payload.get("sessionId", "")
        view_id = payload.get("viewId", "")
        handler_name = payload.get("handler", "")
        event_data = payload.get("data", {})

        minu_session = self._sessions.get(session_id)
        if not minu_session:
            return {"type": "error", "message": "Session not found"}

        view = minu_session.get_view(view_id)
        if not view:
            return {"type": "error", "message": f"View {view_id} not found"}

        handler = getattr(view, handler_name, None)
        if not handler:
            return {"type": "error", "message": f"Handler '{handler_name}' not found"}

        try:
            result = handler(event_data)
            if asyncio.iscoroutine(result):
                result = await result

            patches = view.get_patches()
            return {"type": "event_result", "patches": patches}
        except Exception as e:
            return {"type": "error", "message": str(e)}

    async def _handle_ws_state_update(self, payload: dict, conn_id: str) -> dict:
        """Handle client-side state update."""
        session_id = payload.get("sessionId", "")
        view_id = payload.get("viewId", "")
        path = payload.get("path", "")
        value = payload.get("value")

        minu_session = self._sessions.get(session_id)
        if not minu_session:
            return {"type": "error", "message": "Session not found"}

        view = minu_session.get_view(view_id)
        if not view:
            return {"type": "error", "message": f"View {view_id} not found"}

        # Set state on view
        parts = path.split(".")
        state_name = parts[-1]

        if hasattr(view, state_name):
            attr = getattr(view, state_name)
            if hasattr(attr, "value"):
                attr.value = value

        # Flush patches
        await minu_session.force_flush()
        return {"type": "state_ok"}

    # =========================================================================
    # Session & View Factory
    # =========================================================================

    def _create_view(self, view_class, session, request=None):
        """Create view instance and register in session."""
        from toolboxv2.mods.Minu.core import MinuSession

        minu_session = self._get_or_create_session(session)
        view = view_class()

        # Wire up app reference and request data
        if self._app:
            view.set_app(self._app)
        if request:
            # Build lightweight RequestData-compatible object
            view.request_data = self._build_request_data(session, request)

        minu_session.register_view(view, app=self._app)
        return view, minu_session

    def _get_or_create_session(self, session) -> "MinuSession":
        from toolboxv2.mods.Minu.core import MinuSession

        sid = getattr(session, "session_id", "") or f"anon-{uuid.uuid4().hex[:12]}"

        if sid not in self._sessions:
            self._sessions[sid] = MinuSession(sid)
        return self._sessions[sid]

    def _get_session(self, session) -> Optional["MinuSession"]:
        sid = getattr(session, "session_id", "")
        return self._sessions.get(sid)

    @staticmethod
    def _build_request_data(session, request=None):
        """Build a minimal RequestData-compatible object for MinuView."""
        class _MinimalRequestData:
            def __init__(self, session_dict, req):
                self.session = session_dict
                self.request = req
                self._cached_minu_user = None

        session_dict = session.to_dict() if hasattr(session, "to_dict") else {}
        return _MinimalRequestData(session_dict, request)

    # =========================================================================
    # HTML Rendering
    # =========================================================================

    @staticmethod
    def _render_html(view_name: str, view_dict: dict, minu_session) -> str:
        """Render HTML bootloader page for a MinuView."""
        view_json = json.dumps(view_dict, default=str)
        session_id = minu_session.session_id

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{view_name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: system-ui, -apple-system, sans-serif; background: #f9fafb; color: #1e293b; }}
        #minu-root {{ max-width: 960px; margin: 0 auto; padding: 2rem; }}
        .minu-loading {{ display: flex; justify-content: center; align-items: center; height: 40vh; color: #6b7280; flex-direction: column; gap: 1rem; }}
        .spinner {{ width: 2rem; height: 2rem; border: 3px solid #e5e7eb; border-top-color: #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
</head>
<body>
    <div id="minu-root">
        <div class="minu-loading">
            <div class="spinner"></div>
            <p>Loading {view_name}...</p>
        </div>
    </div>
    <script>
        const MINU_VIEW = {view_json};
        const MINU_SESSION = "{session_id}";
        // Minu client renderer picks this up
        document.addEventListener('DOMContentLoaded', () => {{
            if (window.TB && window.TB.ui) {{
                window.TB.ui.renderView(document.getElementById('minu-root'), MINU_VIEW, MINU_SESSION);
            }} else {{
                document.getElementById('minu-root').innerHTML =
                    '<pre style="padding:1rem;background:#f1f5f9;border-radius:8px;overflow:auto;font-size:13px;">' +
                    JSON.stringify(MINU_VIEW, null, 2) + '</pre>';
            }}
        }});
    </script>
</body>
</html>"""

    # =========================================================================
    # Utility
    # =========================================================================

    def list_views(self) -> List[Dict[str, str]]:
        """List all registered Minu views."""
        return [
            {"path": path, "view": cls.__name__}
            for path, cls in self._view_registry.items()
        ]
