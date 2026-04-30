#!/usr/bin/env python3
"""
fast_tb.py - FastAPI-like Wrapper for ToolBoxV2 Worker System

Provides decorator-based routing (@app.get, @app.post, @app.websocket)
that integrates with the existing WSGI/Waitress + ZMQ infrastructure.

Can run:
  A) Inside HTTPWorker (WSGI/Waitress) — zero migration cost
  B) Standalone via Uvicorn (ASGI) — using a2wsgi bridge

Usage:
    from toolboxv2.utils.workers.fast_tb import FastTB

    app = FastTB(title="MyApp")

    @app.get("/hello/{name}")
    async def hello(name: str, request: ParsedRequest):
        return {"message": f"Hello {name}"}

    @app.post("/items")
    async def create_item(request: ParsedRequest, session: SessionData):
        data = request.json_data
        return {"created": True, "user": session.user_name}

    @app.websocket("/ws/chat")
    class ChatHandler:
        async def on_connect(self, conn_id, session): ...
        async def on_message(self, payload, conn_id, session, request): ...
        async def on_disconnect(self, conn_id, session): ...

    # Standalone ASGI:
    #   uvicorn myapp:app --port 8000
"""

import asyncio
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Pattern


@dataclass
class Route:
    """Internal route registration."""
    path: str                    # Original path template, e.g. "/users/{id}"
    method: str                  # HTTP method (GET, POST, PUT, DELETE, PATCH)
    handler: Callable            # The endpoint function
    pattern: "re.Pattern"        # Compiled regex for matching
    param_names: List[str]       # Extracted path parameter names
    name: str = ""               # Optional route name


@dataclass
class WSRoute:
    """WebSocket route registration."""
    path: str
    handler_obj: Any             # Class instance or dict with on_connect/on_message/on_disconnect
    pattern: "re.Pattern"
    param_names: List[str]


class WebSocketContext:
    """Context object passed to WebSocket handlers registered via @app.websocket().

    Provides connection info, authenticated user data, and a send() helper.
    """

    def __init__(
        self,
        conn_id: str,
        channel_id: Optional[str] = None,
        user: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        headers: Optional[Dict[str, Any]] = None,
        cookies: Optional[Dict[str, Any]] = None,
        _send_fn: Optional[Callable] = None,
        _broadcast_fn: Optional[Callable] = None,
    ):
        self.conn_id = conn_id
        self.channel_id = channel_id
        self.user = user or {}
        self.session_id = session_id
        self.headers = headers or {}
        self.cookies = cookies or {}
        self._send_fn = _send_fn
        self._broadcast_fn = _broadcast_fn

    async def send(self, data: dict) -> bool:
        """Send message back to this connection."""
        if self._send_fn:
            return await self._send_fn(self.conn_id, data)
        return False

    async def broadcast(self, channel: str, data: dict) -> bool:
        """Broadcast to a channel, excluding this connection."""
        if self._broadcast_fn:
            return await self._broadcast_fn(channel, data, self.conn_id)
        return False

    @property
    def is_authenticated(self) -> bool:
        return bool(self.user and (self.user.get("id") or self.user.get("user_id")))

    @property
    def user_id(self) -> Optional[str]:
        return self.user.get("id") or self.user.get("user_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conn_id": self.conn_id,
            "user": self.user,
            "session_id": self.session_id,
            "authenticated": self.is_authenticated,
        }

    @classmethod
    def from_event_payload(cls, payload: dict, send_fn=None, broadcast_fn=None) -> "WebSocketContext":
        """Build context from ZMQ WS_MESSAGE event payload."""
        return cls(
            conn_id=payload.get("conn_id", ""),
            user={
                "user_id": payload.get("user_id", ""),
                "level": payload.get("level", 0),
                "cloudm_user_id": payload.get("cloudm_user_id", ""),
            } if payload.get("authenticated") else {},
            session_id=payload.get("session_id", ""),
            _send_fn=send_fn,
            _broadcast_fn=broadcast_fn,
        )


def _path_to_regex(path: str) -> Tuple["re.Pattern", List[str]]:
    """Convert FastAPI-style path to regex.

    "/users/{user_id}/posts/{post_id}" ->
        re.compile(r"^/users/(?P<user_id>[^/]+)/posts/(?P<post_id>[^/]+)$")
    """
    param_names = re.findall(r"\{(\w+)\}", path)
    regex = path
    for name in param_names:
        regex = regex.replace(f"{{{name}}}", f"(?P<{name}>[^/]+)")
    regex = f"^{regex}$"
    return re.compile(regex), param_names


class FastTB:
    """FastAPI-compatible routing wrapper for ToolBoxV2."""

    def __init__(self, title: str = "FastTB", app_instance=None):
        """
        Args:
            title: App title (metadata only)
            app_instance: Optional ToolBoxV2 App instance for module access
        """
        self.title = title
        self.app_instance = app_instance

        self._routes: List[Route] = []
        self._ws_routes: List[WSRoute] = []
        self._static_mounts: List[Tuple[str, str]] = []  # (url_prefix, fs_directory)

        # Pre-built index: method -> list of routes (populated on first resolve)
        self._route_index: Dict[str, List[Route]] = {}
        self._index_dirty = True

        # Hot-reload: auto-enabled in development, disabled in production
        env = os.getenv("TB_ENV", "development").lower()
        self.hot_reload: bool = env != "production"
        self.inject_style: bool = True  # Inject Paper CSS into default pages (/docs, welcome)
        self._watch_dirs: List[str] = []
        self._hot_reload_running = False

    # =========================================================================
    # Route Registration Decorators
    # =========================================================================

    def get(self, path: str, name: str = ""):
        """Register GET endpoint."""
        return self._route_decorator(path, "GET", name)

    def post(self, path: str, name: str = ""):
        """Register POST endpoint."""
        return self._route_decorator(path, "POST", name)

    def put(self, path: str, name: str = ""):
        """Register PUT endpoint."""
        return self._route_decorator(path, "PUT", name)

    def delete(self, path: str, name: str = ""):
        """Register DELETE endpoint."""
        return self._route_decorator(path, "DELETE", name)

    def patch(self, path: str, name: str = ""):
        """Register PATCH endpoint."""
        return self._route_decorator(path, "PATCH", name)

    def route(self, path: str, methods: List[str] | None = None, name: str = ""):
        """Register endpoint for multiple methods."""
        methods = methods or ["GET"]
        def decorator(func):
            for m in methods:
                self._register_route(path, m.upper(), func, name)
            return func
        return decorator

    def sse(self, path: str, name: str = ""):
        """Register SSE (Server-Sent Events) endpoint.

        The decorated function must be an async generator (yield items).
        SSE headers are set automatically.

        IMPORTANT: Each SSE stream blocks one Waitress thread for its
        entire duration. Keep streams finite and set max_concurrent >= 50.
        For high-concurrency real-time updates, prefer WebSocket.

        Usage:
            @app.sse("/events")
            async def stream_events(request: ParsedRequest):
                for i in range(10):
                    yield {"event": "tick", "data": {"count": i}}
                    await asyncio.sleep(1)
        """
        def decorator(func):
            # Wrap the handler to set SSE response format
            async def sse_wrapper(**kwargs):
                gen = func(**kwargs)
                # Import here to avoid circular imports at module level
                from toolboxv2.utils.workers.server_worker import format_sse_event
                headers = {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                }

                def _fmt(item):
                    """Format item as SSE. Extracts event/id keys from dicts."""
                    if isinstance(item, dict) and "event" in item:
                        evt = item.get("event")
                        eid = item.get("id")
                        payload = item.get("data", item)
                        return format_sse_event(payload, event=evt, event_id=eid)
                    return format_sse_event(item)

                async def sse_gen():
                    try:
                        if inspect.isasyncgen(gen):
                            async for item in gen:
                                yield _fmt(item).encode("utf-8")
                        elif inspect.isgenerator(gen):
                            for item in gen:
                                yield _fmt(item).encode("utf-8")
                                await asyncio.sleep(0)
                        else:
                            yield _fmt(gen).encode("utf-8")
                    finally:
                        yield b"event: stream_end\ndata: {}\n\n"

                return 200, headers, sse_gen()

            sse_wrapper.__name__ = func.__name__
            sse_wrapper.__qualname__ = func.__qualname__
            # Preserve original signature for DI
            sse_wrapper.__wrapped__ = func
            self._register_route(path, "GET", sse_wrapper, name or func.__name__)
            return func
        return decorator

    def _route_decorator(self, path: str, method: str, name: str):
        def decorator(func):
            self._register_route(path, method, func, name)
            return func
        return decorator

    def _register_route(self, path: str, method: str, handler: Callable, name: str = ""):
        pattern, param_names = _path_to_regex(path)
        route = Route(
            path=path,
            method=method,
            handler=handler,
            pattern=pattern,
            param_names=param_names,
            name=name or handler.__name__,
        )
        self._routes.append(route)
        self._index_dirty = True

    # =========================================================================
    # Static File Mounting
    # =========================================================================

    def mount_static(self, url_prefix: str, directory: str):
        """Mount a local directory for static file serving.

        Serves files from `directory` under `url_prefix` with path traversal
        protection (realpath must stay inside directory).

        Args:
            url_prefix: URL path prefix, e.g. "/static" or "/dist"
            directory: Absolute filesystem path to serve from

        Usage:
            app.mount_static("/dist", "/home/user/project/dist")
            # GET /dist/main.js  →  /home/user/project/dist/main.js
        """
        import os
        url_prefix = url_prefix.rstrip("/")
        directory = os.path.abspath(directory)
        self._static_mounts.append((url_prefix, directory))

    def watch(self, *directories: str):
        """Add directories to watch for hot-reload.

        Changes to .py, .js, .css, .html files trigger a browser reload
        via WebSocket broadcast.

        Automatically called by MinuBridge for its module directory.
        Can also be called manually:

            app.watch("./my_app", "./templates")

        Only active when self.hot_reload is True (default in development).
        """
        import os
        for d in directories:
            abspath = os.path.abspath(d)
            if os.path.isdir(abspath) and abspath not in self._watch_dirs:
                self._watch_dirs.append(abspath)
        return self

    def hot_reload_script(self) -> str:
        """Return a <script> tag that listens for hot_reload WS messages.

        Automatically connects to the WS worker and reloads on file changes.
        Returns empty string if hot_reload is disabled or no WS routes exist.

        Usage in custom HTML handlers:
            @app.get("/")
            def index():
                return f'''<html><body>
                    <h1>My App</h1>
                    {app.hot_reload_script()}
                </body></html>'''
        """
        if not self.hot_reload or not self._ws_routes:
            return ""

        return """<script>(function(){
var proto=location.protocol==='https:'?'wss:':'ws:';
var port=window.__TB_WS_PORT__||'8100';
var ws;
function connect(){
    try{ws=new WebSocket(proto+'//'+location.hostname+':'+port+'/ws/open_hot_reload');}catch(e){return;}
    ws.onmessage=function(e){
        var d;try{d=JSON.parse(e.data);}catch(x){return;}
        if(d.type==='hot_reload'){
            console.log('%c[HMR]%c '+d.file+' → reloading',
                'background:#f59e0b;color:black;padding:1px 6px;border-radius:3px;font-weight:bold',
                'color:#f59e0b');
            var T=window.TB&&window.TB.ui&&window.TB.ui.Toast;
            if(T)T.show({message:d.file+' changed',variant:'info',title:'↻ Reloading',duration:1500});
            setTimeout(function(){location.reload();},300);
        }
    };
    ws.onclose=function(){setTimeout(connect,2000);};
}
connect();
})();</script>"""

    def resolve_static(self, path: str) -> "str | None":
        """Resolve a request path to a local file, or None.

        Returns absolute file path if:
          1. Path starts with a registered prefix
          2. Resolved path stays inside the mount directory (no traversal)
          3. File exists and is a regular file
        """
        import os
        for url_prefix, directory in self._static_mounts:
            if path.startswith(url_prefix + "/") or path == url_prefix:
                # Strip prefix to get relative path
                rel = path[len(url_prefix):].lstrip("/")
                if not rel:
                    continue

                # Resolve and verify no traversal
                candidate = os.path.normpath(os.path.join(directory, rel))
                if not candidate.startswith(directory + os.sep) and candidate != directory:
                    continue  # path traversal attempt

                if os.path.isfile(candidate):
                    return candidate

        return None

    # =========================================================================
    # WebSocket Registration
    # =========================================================================

    def websocket(self, path: str):
        """Register WebSocket handler.

        Accepts either a class with on_connect/on_message/on_disconnect methods
        or a dict with those keys.

        Usage:
            @app.websocket("/ws/chat")
            class ChatHandler:
                async def on_connect(self, conn_id, session): ...
                async def on_message(self, payload, conn_id, session, request): ...
                async def on_disconnect(self, conn_id, session): ...
        """
        def decorator(cls_or_dict):
            pattern, param_names = _path_to_regex(path)

            # If it's a class, instantiate it
            if isinstance(cls_or_dict, type):
                handler_obj = cls_or_dict()
            else:
                handler_obj = cls_or_dict

            ws_route = WSRoute(
                path=path,
                handler_obj=handler_obj,
                pattern=pattern,
                param_names=param_names,
            )
            self._ws_routes.append(ws_route)
            return cls_or_dict

        return decorator

    # =========================================================================
    # Route Resolution
    # =========================================================================

    def _rebuild_index(self):
        self._route_index.clear()
        for route in self._routes:
            self._route_index.setdefault(route.method, []).append(route)
        self._index_dirty = False

    def resolve_route(self, path: str, method: str) -> Optional[Tuple[Route, Dict[str, str]]]:
        """Find matching route and extract path params.

        Returns:
            (Route, path_params) or None
        """
        if self._index_dirty:
            self._rebuild_index()

        candidates = self._route_index.get(method.upper(), [])
        for route in candidates:
            m = route.pattern.match(path)
            if m:
                return route, m.groupdict()

        return None

    def has_route(self, path: str, method: str) -> bool:
        """Check if a route exists for this path+method (including static mounts)."""
        if self.resolve_route(path, method) is not None:
            return True
        if method.upper() == "GET" and self.resolve_static(path) is not None:
            return True
        return False

    def resolve_ws_route(self, path: str) -> Optional[Tuple[WSRoute, Dict[str, str]]]:
        """Find matching WebSocket route."""
        for ws_route in self._ws_routes:
            m = ws_route.pattern.match(path)
            if m:
                return ws_route, m.groupdict()
        return None

    # =========================================================================
    # WebSocket Handler Export (for WebSocketMessageHandler integration)
    # =========================================================================

    def get_websocket_handlers(self) -> Dict[str, Dict[str, Callable]]:
        """Export WebSocket handlers in ToolBoxV2 format.

        Returns dict compatible with app.websocket_handlers:
            { "handler_id": { "on_connect": fn, "on_message": fn, "on_disconnect": fn } }
        """
        handlers = {}
        for ws_route in self._ws_routes:
            # Use path as handler_id (strip leading /ws/ if present)
            handler_id = ws_route.path.strip("/")
            obj = ws_route.handler_obj

            entry = {}
            for method_name in ("on_connect", "on_message", "on_disconnect"):
                fn = getattr(obj, method_name, None)
                if fn and callable(fn):
                    entry[method_name] = fn

            if entry:
                handlers[handler_id] = entry

        return handlers

    # =========================================================================
    # ASGI Entrypoint (for standalone uvicorn usage)
    # =========================================================================

    async def __call__(self, scope, receive, send):
        """ASGI entrypoint — allows `uvicorn myapp:app`.

        Wraps the internal WSGI dispatch via a2wsgi.WSGIMiddleware.
        """
        try:
            from a2wsgi import WSGIMiddleware
        except ImportError:
            raise RuntimeError(
                "a2wsgi required for ASGI mode. Install with: pip install a2wsgi"
            )

        from .fast_tb_handler import FastTBHandler

        handler = FastTBHandler(self)
        wsgi_app = handler.as_wsgi_app()
        asgi_app = WSGIMiddleware(wsgi_app)
        await asgi_app(scope, receive, send)

    # =========================================================================
    # Utility
    # =========================================================================

    def list_routes(self) -> List[Dict[str, str]]:
        """List all registered routes (for debugging/docs)."""
        routes = []
        for r in self._routes:
            routes.append({
                "method": r.method,
                "path": r.path,
                "name": r.name,
                "handler": r.handler.__qualname__,
            })
        for ws in self._ws_routes:
            routes.append({
                "method": "WS",
                "path": ws.path,
                "name": type(ws.handler_obj).__name__,
                "handler": type(ws.handler_obj).__qualname__,
            })
        return routes
