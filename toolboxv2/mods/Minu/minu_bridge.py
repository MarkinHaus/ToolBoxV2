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
from typing import Any, Callable, Dict, List, Optional, Type

from toolboxv2 import get_logger, get_app

logger = get_logger()


class MinuBridge:
    """Bridge between FastTB and Minu UI framework.

    Provides a @minu.view() decorator that auto-generates
    HTTP + WS endpoints for a MinuView class.
    """

    with_3d = False
    style_toggle = True   # Show Glass ↔ Paper toggle button in nav
    default_style = "paper"  # "glass" or "paper" — initial style when no user preference saved
    public = True  # Default: views are public (no auth). False = all views require auth
    ws_public = True  # Default: WS connections allowed without auth. False = WS requires auth

    def __init__(self, fast_tb_app, app_instance=None,
                 registration_name:str|None = None,
                 registration_title:str|None = None,
                 registration_description:str|None = None,
                 index_view_path: str|None = None):
        """
        Args:
            fast_tb_app: FastTB instance to register routes on
            app_instance: Optional ToolBoxV2 App instance
        """
        import os
        from toolboxv2.utils.workers.fast_tb import FastTB
        from toolboxv2 import get_app

        self._ftb: FastTB = fast_tb_app
        self._app = app_instance or fast_tb_app.app_instance or get_app()
        self._sessions: Dict[str, Any] = {}  # session_id -> MinuSession
        self._view_registry: Dict[str, Dict[str, Any]] = {}  # path -> {class, icon, label, ...}
        self._nav_items: List[Dict[str, str]] = []  # extra static nav items [{path, icon, label}]
        self._nav_items_bottom: List[Dict[str, str]] = []  # items after views (Terms etc.)

        # Mount dist directory for TBJS assets (static serving)
        dist_dir = os.getenv("DIST_DIR", os.path.join(
            self._app.data_dir.split(".data")[0], "dist"
        )) if hasattr(self._app, "data_dir") else None

        if dist_dir and os.path.isdir(dist_dir):
            self._ftb.mount_static("/", dist_dir)

        # Auto-watch Minu module directory for hot-reload
        if self._ftb.hot_reload:
            minu_dir = os.path.dirname(os.path.abspath(__file__))
            self._ftb.watch(minu_dir)

        # Store bridge reference on FastTB for HMR access
        self._ftb._minu_bridge = self

        # Register Minu starter page + /docs if no user routes
        from toolboxv2.utils.workers.fast_tb_defaults import register_minu_defaults, register_defaults
        register_defaults(self._ftb)
        register_minu_defaults(self._ftb, self)

        self.registration_name = registration_name
        self.registration_title = registration_title or "My FastTB Minu App"
        self.registration_description = registration_description
        self.index_view_path = index_view_path

    def get_wsgi(self, link=True, enable_ws=True):
        if link:
            self._app.run_any(
                ("CloudM", "add_ui"),
                name=self.registration_name,
                title=self.registration_title,
                path=self.index_view_path,
                description=self.registration_description,
                auth=not self.public  # Kein Auth für Demo
            )
        from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
        return FastTBHandler(self).as_wsgi_app(enable_ws=enable_ws)

    def view(
        self,
        path: str,
        require_auth: bool | None = None,
        ws_public: bool | None = None,
        icon: str | None = None,
        label: str | None = None,
        nav: bool = True,
        is_index: bool =False,
    ):
        """Register a MinuView as a FastTB route.

        Args:
            path: URL path (e.g. "/dashboard")
            require_auth: If True, returns 401 for anonymous users.
                           None = use bridge default (inverse of self.public)
            ws_public: If True, WS connection allowed without auth.
                        None = use bridge default (self.ws_public)
            icon: Material Symbols icon name for NavMenu (e.g. "home", "group")
            label: Display label in NavMenu (defaults to class name)
            nav: If False, view is excluded from auto-generated NavMenu

        Returns:
            Decorator that accepts a MinuView subclass
        """
        # Resolve defaults from bridge-level settings
        _require_auth = require_auth if require_auth is not None else (not self.public)
        _ws_public = ws_public if ws_public is not None else self.ws_public
        if _ws_public and not 'open' in path.lower():
            path = "/open" + path[1].upper() + path[2:]
        if is_index:
            self.index_view_path = path
        def decorator(view_class):
            self._view_registry[path] = {
                "class": view_class,
                "icon": icon,
                "label": label or view_class.__name__,
                "require_auth": _require_auth,
                "ws_public": _ws_public,
                "nav": nav,
            }
            self._register_http_routes(path, view_class, _require_auth)
            self._register_ws_route(path, view_class)
            return view_class

        return decorator

    def add_nav_item(self, path: str, label: str, icon: str = "link", bottom: bool = False):
        """Add a static navigation item (not backed by a MinuView).

        Use for links to non-Minu pages like Home, Login, Config.

        Args:
            path: URL path (e.g. "/web/core0/index.html")
            label: Display label in NavMenu
            icon: Material Symbols icon name
            bottom: If True, placed after view items (e.g. Terms & Conditions)

        Usage:
            minu.add_nav_item("/", "Home", icon="home")
            minu.add_nav_item("/terms", "Terms", icon="description", bottom=True)
        """
        item = {"path": path, "label": label, "icon": icon}
        if bottom:
            self._nav_items_bottom.append(item)
        else:
            self._nav_items.append(item)

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

            view, minu_session, is_new = bridge._get_or_create_view(
                path, view_class, session, request
            )

            logger.info(
                f"[MinuBridge] GET {path}: {'created' if is_new else 'reused'} "
                f"view {view._view_id} in session {minu_session.session_id}"
            )

            # Run on_mount only for freshly created views
            if is_new and hasattr(view, "on_mount") and callable(view.on_mount):
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

            # Return HTML fragment (not full page).
            # TBJS Router loads this into #MainContent — TBJS is already running.
            # For direct browser access, TBJS Router redirects to / first, then navigates here.
            # Detect WS port from config
            _ws_port = 0
            try:
                from toolboxv2.utils.workers.config import load_config
                _ws_port = load_config().ws_worker.port
            except Exception:
                pass
            return bridge._render_html(view_class.__name__, view_dict, minu_session, ws_port=_ws_port)

        # --- POST: Event dispatch ---
        @self._ftb.post(f"{path}/event", name=f"minu_{view_class.__name__}_event")
        async def handle_event(request: ParsedRequest, session: SessionData, **kwargs):
            if require_auth and (not session or not session.is_authenticated):
                return (401, {"error": "Authentication required"})

            body = request.json_data or {}
            view_id = body.get("viewId", "")
            handler_name = body.get("handler", "")
            event_data = body.get("data", {})

            if not handler_name:
                return (400, {"error": "Missing handler name"})

            minu_session = bridge._get_session(session)

            # Fallback 1: find session that owns this view by viewId
            if not minu_session:
                for sid, ms in bridge._sessions.items():
                    if ms.get_view(view_id) is not None:
                        minu_session = ms
                        break

            # Fallback 2: find session that has a view for this path
            if not minu_session:
                lookup_key = f"_path:{path}"
                for sid, ms in bridge._sessions.items():
                    if ms.get_view_by_key(lookup_key) is not None:
                        minu_session = ms
                        break

            view = minu_session.get_view(view_id) if minu_session else None

            # Fallback 3: view_id mismatch (page reload created new view) — find by path
            if not view and minu_session:
                lookup_key = f"_path:{path}"
                view = minu_session.get_view_by_key(lookup_key)
                if view:
                    logger.info(f"[MinuBridge] View found via path fallback: {view._view_id} (client sent {view_id})")
            # Fallback: search all sessions by viewId (session cookie may differ between GET and POST)
            if not view:
                view = bridge._find_view_by_id(view_id)

            if not view:
                # Debug dump for troubleshooting
                session_dump = {}
                for sid, ms in bridge._sessions.items():
                    session_dump[sid] = list(ms._views.keys()) if hasattr(ms, '_views') else "no _views"
                logger.error(
                    f"[MinuBridge] View {view_id} not found. "
                    f"Session from cookie: {getattr(session, 'session_id', 'N/A')}. "
                    f"Sessions in bridge: {session_dump}"
                )
                return (404, {"error": f"View {view_id} not found in session"})

            # Handle state_update from client-side bindings
            if handler_name == "__state_update__":
                state_path = event_data.get("path", "")
                state_value = event_data.get("value")
                parts = state_path.split(".")
                state_name = parts[-1]
                if hasattr(view, state_name):
                    attr = getattr(view, state_name)
                    if hasattr(attr, "value"):
                        attr.value = state_value
                return {"ok": True, "patches": view.get_patches()}

            handler = getattr(view, handler_name, None)
            if not handler or not callable(handler):
                return (404, {"error": f"Handler '{handler_name}' not found"})

            try:
                result = handler(event_data)
                if asyncio.iscoroutine(result):
                    result = await result

                # Flush pending state changes.
                # If WS _send_callback is set, this pushes patches live via WS.
                # If not, patches accumulate and we return them in the HTTP response.
                if minu_session:
                    await minu_session.force_flush()

                # Collect any remaining patches not yet flushed
                patches = view.get_patches()

                # Re-render view so client can do a full update if needed
                view_dict = view.to_dict()

                return {
                    "ok": True,
                    "patches": patches,
                    "view": view_dict,
                    "result": result if isinstance(result, (dict, list, str, int, float, bool, type(None))) else str(
                        result),
                }
            except Exception as e:
                import traceback as _tb
                import difflib
                tb_str = _tb.format_exc()

                tb_obj = e.__traceback__
                source_info = ""
                while tb_obj and tb_obj.tb_next:
                    tb_obj = tb_obj.tb_next
                if tb_obj:
                    frame = tb_obj.tb_frame
                    source_info = f"{frame.f_code.co_filename}:{tb_obj.tb_lineno} in {frame.f_code.co_name}"

                err_type = type(e).__name__
                err_msg = str(e)
                hint = ""

                if "has no attribute" in err_msg:
                    bad_attr = err_msg.split("'")[-2] if "'" in err_msg else ""
                    if bad_attr and view:
                        valid_attrs = [a for a in dir(view) if
                                       not a.startswith("_") and hasattr(getattr(view, a, None), "value")]
                        matches = difflib.get_close_matches(bad_attr, valid_attrs, n=1, cutoff=0.5)
                        hint = f"Did you mean '{matches[0]}'?" if matches else f"Available states: {', '.join(valid_attrs)}"
                elif "unsupported operand" in err_msg:
                    hint = f"Type mismatch in '{handler_name}' — check types (e.g. += 1 not += '1')"

                logger.error(
                    f"[MinuBridge] Handler '{handler_name}' on {view.__class__.__name__ if view else '?'} failed:\n"
                    f"  {err_type}: {err_msg}\n"
                    f"  Location: {source_info}\n"
                    f"  Hint: {hint}\n{tb_str}"
                )

                return {
                    "ok": False,
                    "error": f"{err_type}: {err_msg}",
                    "handler": handler_name,
                    "hint": hint,
                    "source": source_info,
                    "traceback": tb_str,
                }

    # =========================================================================
    # WebSocket Route Generation
    # =========================================================================

    def _register_ws_route(self, path: str, view_class: Type):
        """Register WebSocket handler for live updates."""
        bridge = self
        ws_path = f"/ws{path}" if not path.startswith("/ws") else path

        # Track conn_id -> session_id mapping for cleanup
        _conn_sessions: Dict[str, str] = {}

        # AccessController checks function name: 'open*' prefix = public, else auth required.
        # ws_public=True  → openMinuWSHandler → anonymous WS allowed
        # ws_public=False → MinuWSHandler      → auth required for WS
        ws_is_public = self._view_registry.get(path, {}).get("ws_public", self.ws_public)
        handler_cls_name = "openMinuWSHandler" if ws_is_public else "MinuWSHandler"

        @self._ftb.websocket(ws_path)
        class openMinuWSHandler:
            async def on_connect(self_, conn_id, session):
                logger.info(f"[MinuBridge WS] {conn_id} connected to {ws_path}")

            async def on_message(self_, payload, conn_id, session, request=None):
                msg_type = payload.get("type", "")

                if msg_type == "init":
                    session_id = payload.get("sessionId", conn_id)
                    logger.info(
                        f"[MinuBridge WS] init: client sessionId={session_id}, "
                        f"conn_id={conn_id}, bridge sessions={list(bridge._sessions.keys())}"
                    )

                    minu_session = bridge._sessions.get(session_id)

                    # Fallback: try conn_id as session key
                    if not minu_session:
                        minu_session = bridge._sessions.get(conn_id)

                    # Fallback: search for session with a view on this path
                    if not minu_session:
                        lookup_key = f"_path:{path}"
                        for sid, ms in bridge._sessions.items():
                            if ms.get_view_by_key(lookup_key) is not None:
                                minu_session = ms
                                session_id = sid
                                logger.info(f"[MinuBridge WS] init: found session via path lookup: {sid}")
                                break

                    if not minu_session:
                        logger.warning(
                            f"[MinuBridge WS] init: NO session found! "
                            f"tried={session_id}, conn={conn_id}, "
                            f"available={list(bridge._sessions.keys())}"
                        )
                        return {"type": "error", "message": f"Session not found (tried: {session_id})"}

                    _conn_sessions[conn_id] = session_id

                    # Set up send callback bound to this specific conn_id.
                    # This enables live push: any state change during an async handler
                    # (e.g. auto_increment with sleep) is pushed immediately.
                    async def ws_send(msg_str, _conn=conn_id):
                        from toolboxv2 import get_app
                        app = get_app(from_="minu_bridge_ws")
                        data = json.loads(msg_str) if isinstance(msg_str, str) else msg_str
                        await app.ws_send(_conn, data)

                    minu_session.set_send_callback(ws_send)
                    return {"type": "init_ok", "sessionId": session_id}

                elif msg_type == "event":
                    # Enrich payload with session_id from our mapping
                    if "sessionId" not in payload or not payload["sessionId"]:
                        payload["sessionId"] = _conn_sessions.get(conn_id, "")
                    return await bridge._handle_ws_event(payload, conn_id)

                elif msg_type == "state_update":
                    if "sessionId" not in payload or not payload["sessionId"]:
                        payload["sessionId"] = _conn_sessions.get(conn_id, "")
                    return await bridge._handle_ws_state_update(payload, conn_id)

            async def on_disconnect(self_, conn_id, session):
                logger.info(f"[MinuBridge WS] {conn_id} disconnected from {ws_path}")
                sid = _conn_sessions.pop(conn_id, None)

        # Rename class so FastTB registers it with the right name
        openMinuWSHandler.__name__ = handler_cls_name
        openMinuWSHandler.__qualname__ = handler_cls_name
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
            # Fallback: search all sessions
            view = self._find_view_by_id(view_id)
            if not view:
                return {"type": "error", "message": f"View {view_id} not found"}

        handler = getattr(view, handler_name, None)
        if not handler:
            return {"type": "error", "message": f"Handler '{handler_name}' not found"}

        try:
            result = handler(event_data)
            if asyncio.iscoroutine(result):
                result = await result

            # Flush any state changes pushed during the handler (incl. async delays).
            # If _send_callback is set, force_flush already pushed patches via WS.
            await minu_session.force_flush()

            # Collect remaining patches (for handlers that didn't trigger _send_callback)
            patches = view.get_patches()

            # Re-render view for full sync
            view_dict = view.to_dict()

            return {
                "type": "event_result",
                "patches": patches,
                "view": view_dict,
            }
        except Exception as e:
            import traceback as _tb
            tb_str = _tb.format_exc()

            # Extract source location from traceback
            tb_obj = e.__traceback__
            source_info = ""
            while tb_obj and tb_obj.tb_next:
                tb_obj = tb_obj.tb_next
            if tb_obj:
                frame = tb_obj.tb_frame
                source_info = f"{frame.f_code.co_filename}:{tb_obj.tb_lineno} in {frame.f_code.co_name}"

            # Build developer-friendly error message
            err_type = type(e).__name__
            hint = ""
            err_msg = str(e)

            if "has no attribute" in err_msg:
                # Extract the typo and suggest correction
                import difflib
                bad_attr = err_msg.split("'")[-2] if "'" in err_msg else ""
                if bad_attr and view:
                    valid_attrs = [a for a in dir(view) if
                                   not a.startswith("_") and hasattr(getattr(view, a, None), "value")]
                    matches = difflib.get_close_matches(bad_attr, valid_attrs, n=1, cutoff=0.5)
                    if matches:
                        hint = f"Did you mean '{matches[0]}'? (typo: '{bad_attr}')"
                    else:
                        hint = f"Available state attributes: {', '.join(valid_attrs)}"

            elif "unsupported operand" in err_msg:
                hint = f"Type mismatch in handler '{handler_name}'. Check that you're using the correct types (e.g. += 1 not += '1')"

            elif "TypeError" in err_type:
                hint = f"Check argument types in '{handler_name}'"

            # Log full traceback for terminal
            logger.error(
                f"[MinuBridge] Handler '{handler_name}' on {view.__class__.__name__} failed:\n"
                f"  Error: {err_type}: {err_msg}\n"
                f"  Location: {source_info}\n"
                f"  Hint: {hint}\n"
                f"  Traceback:\n{tb_str}"
            )

            return {
                "type": "error",
                "message": f"{err_type}: {err_msg}",
                "handler": handler_name,
                "hint": hint,
                "source": source_info,
                "traceback": tb_str,
            }

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

    def _get_or_create_view(self, path, view_class, session, request=None):
        """Get existing view for this session+path, or create a new one.

        Searches the current session first, then falls back to all sessions
        (handles standalone mode where session_id may change between requests).

        Returns:
            (view, minu_session, is_new) — is_new=True if freshly created
        """
        from toolboxv2.mods.Minu.core import MinuSession

        minu_session = self._get_or_create_session(session)
        lookup_key = f"_path:{path}"

        # 1. Check current session
        existing_view = minu_session.get_view_by_key(lookup_key)
        if existing_view is not None:
            if request:
                existing_view.request_data = self._build_request_data(session, request)
            return existing_view, minu_session, False

        # 2. Fallback: search ALL sessions (handles unstable session IDs)
        for other_session in self._sessions.values():
            if other_session is minu_session:
                continue
            existing_view = other_session.get_view_by_key(lookup_key)
            if existing_view is not None:
                # Move view to current session
                other_session.unregister_view(existing_view._view_id)
                if request:
                    existing_view.request_data = self._build_request_data(session, request)
                minu_session.register_view(existing_view, app=self._app, key=lookup_key)
                return existing_view, minu_session, False

        # 3. Create new view
        view = view_class()
        if self._app:
            view.set_app(self._app)
        if request:
            view.request_data = self._build_request_data(session, request)

        minu_session.register_view(view, app=self._app, key=lookup_key)
        return view, minu_session, True

    def _get_or_create_session(self, session) -> "MinuSession":
        from toolboxv2.mods.Minu.core import MinuSession

        sid = getattr(session, "session_id", "") or f"anon-{uuid.uuid4().hex[:12]}"

        if sid not in self._sessions:
            self._sessions[sid] = MinuSession(sid)
        return self._sessions[sid]

    def _get_session(self, session) -> Optional["MinuSession"]:
        sid = getattr(session, "session_id", "")
        return self._sessions.get(sid)

    def _find_view_by_id(self, view_id: str):
        """Search all sessions for a view by its viewId.

        Needed because in standalone mode (no persistent cookies),
        the session_id may differ between the GET that created the view
        and the POST that dispatches events.
        """
        for minu_session in self._sessions.values():
            view = minu_session.get_view(view_id)
            if view is not None:
                return view
        return None

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

    def _build_nav_items_html(self) -> str:
        """Build NavMenu list items HTML from static nav items + view registry."""
        items = []

        # Static nav items first (Home, Login, Config, etc.)
        for nav in self._nav_items:
            items.append(
                f'<li><a href="{nav["path"]}" class="nav-item">'
                f'<span class="material-symbols-outlined nav-icon">{nav["icon"]}</span>'
                f'<span class="nav-text">{nav["label"]}</span>'
                f'</a></li>'
            )

        # Separator if both static and view items exist
        if self._nav_items and self._view_registry:
            items.append('<li style="border-top:1px solid var(--border-subtle,rgba(255,255,255,0.08));margin:4px 0;"></li>')

        # View items from registry
        for path, meta in self._view_registry.items():
            if not meta.get("nav", True):
                continue
            icon = meta.get("icon") or "article"
            label = meta.get("label") or meta["class"].__name__
            auth_badge = ' <span class="badge badge-default" style="font-size:9px;margin-left:4px;">auth</span>' if meta.get("require_auth") else ""
            items.append(
                f'<li><a href="{path}" class="nav-item">'
                f'<span class="material-symbols-outlined nav-icon">{icon}</span>'
                f'<span class="nav-text">{label}{auth_badge}</span>'
                f'</a></li>'
            )

        # Bottom items (Terms, etc.)
        if self._nav_items_bottom:
            items.append('<li style="border-top:1px solid var(--border-subtle,rgba(255,255,255,0.08));margin:4px 0;"></li>')
            for nav in self._nav_items_bottom:
                items.append(
                    f'<li><a href="{nav["path"]}" class="nav-item">'
                    f'<span class="material-symbols-outlined nav-icon">{nav["icon"]}</span>'
                    f'<span class="nav-text">{nav["label"]}</span>'
                    f'</a></li>'
                )

        return "\n".join(items)

    def _render_html(self, view_name: str, view_dict: dict, minu_session, ws_port: int = 0) -> str:
        """Render HTML fragment for TBJS Router to inject into #MainContent.

        NOT a full page — no <!DOCTYPE>, no <html>, no web_context().
        TBJS is already loaded and running. This fragment contains:
        1. Nav controls: menu trigger (#links) + theme toggle
        2. NavMenu initialized via TB.ui.NavMenu with auto-generated links
        3. minu-root container
        4. View data + boot script + HTTP event dispatch
        5. Background config (3D or color) applied immediately
        """
        from toolboxv2.mods.Minu.core import MinuJSONEncoder

        view_json = json.dumps(view_dict, cls=MinuJSONEncoder, default=str)
        session_id = minu_session.session_id
        view_id = view_dict.get("viewId", "")
        bg_type = "'3d'" if self.with_3d else "'color'"
        nav_items_html = self._build_nav_items_html()

        # Escape for JS string embedding (single quotes in HTML are fine,
        # but backticks and ${} would break template literals)
        nav_html_escaped = nav_items_html.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

        default_style = self.default_style or "glass"

        # Style is ALWAYS applied (even without toggle button)
        # so a dev can ship a paper-only app via default_style="paper"

        style_toggle_btn = ""
        style_toggle_js = ""
        if self.style_toggle:
            style_toggle_btn = """
    <button id="minuStyleToggle" class="nav-toggle" type="button" aria-label="Toggle style">
        <span class="material-symbols-outlined" style="font-size: 20px;">palette</span>
    </button>"""
            style_toggle_js = """
    // --- Style toggle (Glass ↔ Paper) ---
    (function() {
        var STYLE_KEY = 'tbjs_style_preference';
        var root = document.documentElement;
        var btn = document.getElementById('minuStyleToggle');
        if (btn) {
            var iconEl = btn.querySelector('.material-symbols-outlined');
            var updateStyleIcon = function() {
                var current = root.getAttribute('data-style') || 'glass';
                if (iconEl) iconEl.textContent = current === 'paper' ? 'blur_on' : 'palette';
                btn.title = current === 'paper' ? 'Switch to Glass' : 'Switch to Paper';
            };
            updateStyleIcon();
            btn.addEventListener('click', function() {
                var current = root.getAttribute('data-style') || 'glass';
                var next = current === 'paper' ? 'glass' : 'paper';
                root.setAttribute('data-style', next);
                localStorage.setItem(STYLE_KEY, next);
                updateStyleIcon();
            });
        }
    })();"""

            # Inject data-style onto <html> via inline script BEFORE any rendering
            style_attr_script = f"""<script>
(function(){{var k='tbjs_style_preference',r=document.documentElement;r.setAttribute('data-style',localStorage.getItem(k)||'{default_style}');}})();
</script>"""
            hot_reload_js = self._ftb.hot_reload_script()
            return get_app().web_context() + style_attr_script + hot_reload_js + f"""
<nav class="nav-controls" id="Nav-Main" aria-label="Main navigation">
    <button id="links" class="nav-toggle" type="button" aria-label="Menu" aria-expanded="false">
        <span class="material-symbols-outlined" style="font-size: 20px;">menu</span>
    </button>
    <button id="minuThemeToggle" class="nav-toggle" type="button" aria-label="Toggle theme">
        <span class="material-symbols-outlined" style="font-size: 20px;">light_mode</span>
    </button>{style_toggle_btn}
</nav>

<div id="minu-root"></div>

<script unSave=true>
window.__MINU_BRIDGE__ = {{
    view: {view_json},
    viewName: "{view_name}",
    sessionId: "{session_id}",
    viewId: "{view_id}"
}};
(function() {{
    // --- Theme + background config ---
    var theme = window.TB && window.TB.ui && window.TB.ui.theme;
    if (theme) {{
        if (theme._config && theme._config.background) {{
            theme._config.background.type = {bg_type};
        }}
        if (theme._applyEffectiveTheme) {{
            theme._applyEffectiveTheme();
        }} else if (theme._applyBackground) {{
            theme._applyBackground();
        }}

        var toggleBtn = document.getElementById('minuThemeToggle');
        if (toggleBtn) {{
            var iconEl = toggleBtn.querySelector('.material-symbols-outlined');
            var updateIcon = function() {{
                if (!iconEl) return;
                var mode = theme.getCurrentMode ? theme.getCurrentMode() : 'dark';
                iconEl.textContent = mode === 'dark' ? 'light_mode' : 'dark_mode';
            }};
            updateIcon();
            toggleBtn.addEventListener('click', function() {{
                if (theme.togglePreference) theme.togglePreference();
                setTimeout(updateIcon, 50);
            }});
            if (window.TB && window.TB.events) {{
                window.TB.events.on('theme:changed', updateIcon);
            }}
        }}
    }}
{style_toggle_js}
    // --- NavMenu from registered views ---
    var NavMenu = window.TB && window.TB.ui && window.TB.ui.NavMenu;
    if (NavMenu) {{
        // Destroy any previous bridge NavMenu instance
        if (window.__MINU_NAV__) {{
            window.__MINU_NAV__.destroy();
            window.__MINU_NAV__ = null;
        }}
        // Remove ALL NavMenu DOM artifacts (including TBJS default)
        var menuId = 'tb-nav-menu-modal';
        var el;
        el = document.getElementById(menuId);
        if (el && el.parentNode) el.parentNode.removeChild(el);
        el = document.getElementById(menuId + '-overlay');
        if (el && el.parentNode) el.parentNode.removeChild(el);

        // Clone trigger button to strip old event listeners (TBJS default NavMenu)
        var oldTrigger = document.getElementById('links');
        if (oldTrigger) {{
            var newTrigger = oldTrigger.cloneNode(true);
            oldTrigger.parentNode.replaceChild(newTrigger, oldTrigger);
        }}

        window.__MINU_NAV__ = NavMenu.init({{
            triggerSelector: '#links',
            menuContentHtml: `<ul class="nav-list" style="list-style:none;padding:0;margin:0;">{nav_html_escaped}</ul>`
        }});
    }}

    // --- Minu renderer ---
    var R = window.TB && window.TB.ui && window.TB.ui.MinuRenderer;
    if (!R) {{ console.error('[MinuBridge] TB.ui.MinuRenderer not available'); return; }}
    var bridge = window.__MINU_BRIDGE__;
    var root = document.getElementById('minu-root');
    var renderer = new R();
    renderer.container = root;
    if (renderer._injectStyles) renderer._injectStyles();
    renderer._renderView(bridge.view);

    // --- WebSocket event dispatch (live mode) ---
    var eventUrl = window.location.pathname.replace(/\\/$/, '') + '/event';
    var wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    var wsPort = window.__TB_WS_PORT__ || '{ws_port}' || window.location.port;
    var wsUrl = wsProto + '//' + window.location.hostname + ':' + wsPort + '/ws' + window.location.pathname.replace(/\\/$/, '');
    var _ws = null;
    var _wsReady = false;
    var _wsQueue = [];

    function _wsSend(msg) {{
        if (_ws && _ws.readyState === 1) {{
            _ws.send(JSON.stringify(msg));
        }} else {{
            _wsQueue.push(msg);
        }}
    }}

    function _wsConnect() {{
        try {{
            _ws = new WebSocket(wsUrl);
        }} catch(e) {{
            console.warn('[MinuBridge] WS connect failed, using HTTP fallback');
            _wsReady = false;
            return;
        }}
        _ws.onopen = function() {{
            _wsSend({{ type: 'init', sessionId: bridge.sessionId }});
        }};
        _ws.onmessage = function(evt) {{
            var msg;
            try {{ msg = JSON.parse(evt.data); }} catch(e) {{ return; }}
            if (msg.type === 'init_ok') {{
                _wsReady = true;
                console.log('[MinuBridge] WS connected, session:', msg.sessionId);
                // Drain queue
                while (_wsQueue.length) _ws.send(JSON.stringify(_wsQueue.shift()));
            }} else if (msg.type === 'patches') {{
                if (msg.patches && msg.patches.length) renderer._applyPatches(msg.patches);
            }} else if (msg.type === 'render') {{
                if (msg.view) renderer._renderView(msg.view);
            }} else if (msg.type === 'event_result') {{
                if (msg.patches && msg.patches.length) renderer._applyPatches(msg.patches);
                if (msg.view) renderer._renderView(msg.view);
            }} else if (msg.type === 'hot_reload') {{
                console.log('%c[HMR]%c ' + msg.file + ' changed',
                    'background:#f59e0b;color:black;padding:1px 6px;border-radius:3px;font-weight:bold',
                    'color:#f59e0b');
                var Toast = window.TB && window.TB.ui && window.TB.ui.Toast;
                if (msg.ext === '.py') {{
                    // Python change: server already hot-swapped the view class.
                    // Fetch fresh view JSON and re-render (preserves WS connection + state).
                    if (Toast) Toast.show({{
                        message: msg.file,
                        variant: 'info',
                        title: '↻ Hot Update',
                        duration: 2000
                    }});
                    fetch(window.location.pathname + '?format=json')
                        .then(function(r) {{ return r.json(); }})
                        .then(function(data) {{
                            if (data.view) {{
                                renderer._renderView(data.view);
                                console.log('%c[HMR]%c View re-rendered',
                                    'background:#10b981;color:white;padding:1px 6px;border-radius:3px',
                                    'color:#10b981');
                            }}
                        }})
                        .catch(function(e) {{
                            console.warn('[HMR] Fetch failed, full reload:', e);
                            window.location.reload();
                        }});
                }} else {{
                    // CSS/JS/HTML change: full page reload needed
                    if (Toast) Toast.show({{
                        message: msg.file + ' changed',
                        variant: 'info',
                        title: '↻ Reloading',
                        duration: 1500
                    }});
                    setTimeout(function() {{ window.location.reload(); }}, 300);
                }}
            }} else if (msg.type === 'error') {{
                var Toast = window.TB && window.TB.ui && window.TB.ui.Toast;
                var isAuth = msg.code === 'ACCESS_DENIED' || (msg.message && msg.message.indexOf('uthentication') !== -1);

                if (isAuth) {{
                    console.warn('[MinuBridge] Auth required for WS');
                    if (Toast) Toast.show({{
                        message: 'Live updates require login. Using HTTP fallback.',
                        variant: 'warning', title: 'WebSocket', duration: 5000
                    }});
                    _wsReady = false;
                    try {{ _ws.close(); }} catch(e) {{}}
                }} else {{
                    // Developer error — show detailed toast + console output
                    var handler = msg.handler || '?';
                    var hint = msg.hint || '';
                    var source = msg.source || '';
                    var toastMsg = msg.message || 'Unknown error';

                    // Rich console output for devs
                    console.error(
                        '%c[Minu] Handler Error%c ' + handler + '\\n' +
                        '%c' + toastMsg + '%c\\n' +
                        (hint ? '💡 ' + hint + '\\n' : '') +
                        (source ? '📍 ' + source + '\\n' : '') +
                        (msg.traceback ? '\\n' + msg.traceback : ''),
                        'background:#e53e3e;color:white;padding:2px 6px;border-radius:3px;font-weight:bold',
                        'color:#e53e3e;font-weight:bold',
                        'color:#c53030', 'color:inherit'
                    );

                    if (Toast) {{
                        var body = toastMsg;
                        if (hint) body += '\\n💡 ' + hint;
                        Toast.show({{
                            message: body,
                            variant: 'error',
                            title: '⚠ ' + handler + '()',
                            duration: 8000
                        }});
                    }}
                }}
            }}
        }};
        _ws.onclose = function() {{
            _wsReady = false;
            console.log('[MinuBridge] WS closed, falling back to HTTP');
        }};
        _ws.onerror = function() {{ _wsReady = false; }};
    }}

    // HTTP fallback for when WS is not available
    function _handleHttpError(resp) {{
        if (resp && resp.ok === false && resp.error) {{
            var Toast = window.TB && window.TB.ui && window.TB.ui.Toast;
            var handler = resp.handler || '?';
            var hint = resp.hint || '';
            var source = resp.source || '';

            console.error(
                '%c[Minu] Handler Error%c ' + handler + '\\n' +
                '%c' + resp.error + '%c\\n' +
                (hint ? '💡 ' + hint + '\\n' : '') +
                (source ? '📍 ' + source + '\\n' : '') +
                (resp.traceback ? '\\n' + resp.traceback : ''),
                'background:#e53e3e;color:white;padding:2px 6px;border-radius:3px;font-weight:bold',
                'color:#e53e3e;font-weight:bold',
                'color:#c53030', 'color:inherit'
            );

            if (Toast) {{
                var body = resp.error;
                if (hint) body += '\\n💡 ' + hint;
                Toast.show({{
                    message: body,
                    variant: 'error',
                    title: '⚠ ' + handler + '()',
                    duration: 8000
                }});
            }}
            return true;
        }}
        return false;
    }}

    function _httpSend(data) {{
        if (data.type === 'event') {{
            fetch(eventUrl, {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    viewId: data.viewId,
                    handler: data.handler,
                    data: data.payload || {{}}
                }})
            }})
            .then(function(r) {{ return r.json(); }})
            .then(function(resp) {{
                if (_handleHttpError(resp)) return;
                if (resp.patches && resp.patches.length) renderer._applyPatches(resp.patches);
                if (resp.view) renderer._renderView(resp.view);
            }})
            .catch(function(e) {{ console.error('[MinuBridge] HTTP request failed:', e); }});
        }} else if (data.type === 'state_update') {{
            fetch(eventUrl, {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    viewId: data.viewId,
                    handler: '__state_update__',
                    data: {{ path: data.path, value: data.value }}
                }})
            }}).catch(function(e) {{ console.error('[MinuBridge] State sync failed:', e); }});
        }}
    }}

    renderer._send = function(data) {{
        if (_wsReady) {{
            try {{
                _wsSend(data);
            }} catch(e) {{
                console.warn('[MinuBridge] WS send failed, using HTTP:', e);
                _wsReady = false;
                var Toast = window.TB && window.TB.ui && window.TB.ui.Toast;
                if (Toast) Toast.show({{
                    message: 'WebSocket connection lost. Switching to HTTP fallback.',
                    variant: 'warning',
                    title: 'Connection',
                    duration: 4000
                }});
                _httpSend(data);
            }}
        }} else {{
            _httpSend(data);
        }}
    }};

try {{ _wsConnect(); }} catch(e) {{ console.log('[MinuBridge] WS not available, using HTTP'); }}
    console.log('[MinuBridge] View rendered:', bridge.viewName, '(WS:', wsUrl, ')');
}})();
</script>"""

    # =========================================================================
    # Utility
    # =========================================================================

    def list_views(self) -> List[Dict[str, str]]:
        """List all registered Minu views."""
        return [
            {
                "path": path,
                "view": meta["class"].__name__,
                "icon": meta.get("icon") or "article",
                "label": meta.get("label") or meta["class"].__name__,
                "nav": meta.get("nav", True),
            }
            for path, meta in self._view_registry.items()
        ]


"""

# App-Level Defaults
app = FastTB(title="MyApp")
minu = MinuBridge(app)
minu.public = True       # All views public by default
minu.ws_public = True    # WS open for all by default

# Oder: Private App
minu.public = False      # All views require auth by default
minu.ws_public = False   # WS requires auth by default

# Per-View Override
@minu.view("/counter")                              # uses bridge defaults
class Counter(MinuView): ...

@minu.view("/admin", require_auth=True)              # HTTP auth-gated, WS uses bridge default
class Admin(MinuView): ...

@minu.view("/dashboard", ws_public=False)             # HTTP public, WS requires auth
class Dashboard(MinuView): ...

@minu.view("/monitor", require_auth=True, ws_public=True)  # HTTP auth-gated, WS public
class Monitor(MinuView): ...

"""
