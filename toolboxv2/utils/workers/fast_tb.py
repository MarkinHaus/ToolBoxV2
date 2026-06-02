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
import errno
import inspect
import json
import os
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Pattern

from toolboxv2 import Spinner


@dataclass
class Route:
    """Internal route registration."""
    path: str                    # Original path template, e.g. "/users/{id}"
    method: str                  # HTTP method (GET, POST, PUT, DELETE, PATCH)
    handler: Callable            # The endpoint function
    pattern: "re.Pattern"        # Compiled regex for matching
    param_names: List[str]       # Extracted path parameter names
    name: str = ""               # Optional route name
    auth: Optional[bool] = None  # Per-route auth override; None = inherit app.auth


@dataclass
class WSRoute:
    """WebSocket route registration."""
    path: str
    handler_obj: Any             # Class instance or dict with on_connect/on_message/on_disconnect
    pattern: "re.Pattern"
    param_names: List[str]
    auth: Optional[bool] = None  # Per-route auth override; None = inherit app.auth


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
        # One-Port-Collective: merged sub-apps tracked per source for clean unmount
        self._mounted_sources: Dict[str, dict] = {}  # source -> {routes, static, prefix}
        self._server = None  # waitress server handle (set by serve())

        # Pre-built index: method -> list of routes (populated on first resolve)
        self._route_index: Dict[str, List[Route]] = {}
        self._index_dirty = True

        # Hot-reload: auto-enabled in development, disabled in production
        env = os.getenv("TB_ENV", "development").lower()
        self.hot_reload: bool = env != "production"
        self.inject_style: bool = True  # Inject Paper CSS into default pages (/docs, welcome)
        # Auth: when True, all routes require an authenticated session except /health.
        # Per-route override via @app.get("/x", auth=False/True). Default off.
        self.auth: bool = False
        # Serve the login UI assets (dist/web) under /web. When True the mount is
        # installed permanently at startup. When False the mount is created lazily
        # only when an auth-gated route is hit unauthenticated, so the local login
        # page is reachable without permanently exposing /web.
        self.serve_login_assets: bool = False
        # True only if WE installed the /web mount (vs. the app having mounted it
        # itself). Guards against touching an app-provided /web mount.
        self._web_mount_owned: bool = False
        self._watch_dirs: List[str] = []
        self._hot_reload_running = False

    # =========================================================================
    # Route Registration Decorators
    # =========================================================================

    def get(self, path: str, name: str = "", auth: Optional[bool] = None):
        """Register GET endpoint."""
        return self._route_decorator(path, "GET", name, auth)

    def post(self, path: str, name: str = "", auth: Optional[bool] = None):
        """Register POST endpoint."""
        return self._route_decorator(path, "POST", name, auth)

    def put(self, path: str, name: str = "", auth: Optional[bool] = None):
        """Register PUT endpoint."""
        return self._route_decorator(path, "PUT", name, auth)

    def delete(self, path: str, name: str = "", auth: Optional[bool] = None):
        """Register DELETE endpoint."""
        return self._route_decorator(path, "DELETE", name, auth)

    def patch(self, path: str, name: str = "", auth: Optional[bool] = None):
        """Register PATCH endpoint."""
        return self._route_decorator(path, "PATCH", name, auth)

    def route(self, path: str, methods: List[str] | None = None, name: str = "", auth: Optional[bool] = None):
        """Register endpoint for multiple methods."""
        methods = methods or ["GET"]
        def decorator(func):
            for m in methods:
                self._register_route(path, m.upper(), func, name, auth)
            return func
        return decorator

    def sse(self, path: str, name: str = "", auth: Optional[bool] = None):
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
            self._register_route(path, "GET", sse_wrapper, name or func.__name__, auth)
            return func
        return decorator

    def _route_decorator(self, path: str, method: str, name: str, auth: Optional[bool] = None):
        def decorator(func):
            self._register_route(path, method, func, name, auth)
            return func
        return decorator

    def _register_route(self, path: str, method: str, handler: Callable, name: str = "", auth: Optional[bool] = None):
        pattern, param_names = _path_to_regex(path)
        route = Route(
            path=path,
            method=method,
            handler=handler,
            pattern=pattern,
            param_names=param_names,
            name=name or handler.__name__,
            auth=auth,
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

    def _ensure_web_mounted(self) -> bool:
        """Idempotently mount the login UI assets (dist/web) under /web.

        Used by the auth flow so the local login page is reachable. If /web is
        already mounted (by the app or a previous call) this is a no-op and the
        existing mount is left untouched. Returns True if /web is now available.
        """
        import os
        if self._web_mount_owned:
            return True
        try:
            from toolboxv2 import tb_root_dir
        except Exception:
            return False
        directory = os.path.join(os.path.abspath(str(tb_root_dir)), "dist")
        if not os.path.isdir(directory):
            return False
        self._static_mounts.append(("", directory))
        self._web_mount_owned = True
        return True

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

    def websocket(self, path: str, auth: Optional[bool] = None):
        """Register WebSocket handler.

        Accepts either a class with on_connect/on_message/on_disconnect methods
        or a dict with those keys.

        Args:
            path: WS route path, e.g. "/ws/chat".
            auth: Per-route auth override. None inherits app.auth. When effective
                  auth is True, unauthenticated connections are rejected at connect.

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
                auth=auth,
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

    # =========================================================================
    # WebSocket send/broadcast — thin delegation to the ToolBox app's WS bridge
    # =========================================================================

    def _ws_bridge_target(self):
        """Resolve the object that carries the WS bridge methods.

        Prefer app_instance (set at WS startup on the served instance). Fall back
        to get_app() so a MOUNTED instance — which is never served itself and thus
        never gets app_instance set — still reaches the same singleton ToolBox app
        on which install_ws_bridge() ran. Returns None if neither is available.
        """
        inst = self.app_instance
        if inst is None:
            try:
                from toolboxv2 import get_app
                inst = get_app()
            except Exception:
                return None
        return inst

    def ws_broadcast(self, channel_id: str, payload: dict, source_conn_id: str = ""):
        """Broadcast to a WS channel via the ToolBox app's bridge.

        Returns the underlying awaitable (await it) or False if WS is unavailable.
        """
        inst = self._ws_bridge_target()
        fn = getattr(inst, "ws_broadcast", None) if inst is not None else None
        if fn is None:
            return False
        return fn(channel_id, payload, source_conn_id)

    def ws_send(self, conn_id: str, payload: dict):
        """Send to a single WS connection via the ToolBox app's bridge.

        Returns the underlying awaitable (await it) or False if WS is unavailable.
        """
        inst = self._ws_bridge_target()
        fn = getattr(inst, "ws_send", None) if inst is not None else None
        if fn is None:
            return False
        return fn(conn_id, payload)

    # =========================================================================
    # One-Port-Collective: mount / unmount sibling FastTB apps (Modell A)
    # =========================================================================

    @staticmethod
    def _root_pattern(fallback_path: str) -> "re.Pattern":
        """Compile a relocated-root pattern that matches /prefix and /prefix/."""
        return re.compile(rf"^{re.escape(fallback_path)}/?$")

    def mount_conflicts(self, other: "FastTB", fallback_path: str) -> List[str]:
        """Non-root route collisions between `other` and this app.

        Root '/' is auto-relocatable to `fallback_path`; everything else must
        not collide on (method, path). Also flags the relocated root landing
        on an already-occupied path. Empty list => compatible (mergeable).
        """
        own = {(r.method, r.path) for r in self._routes}
        bad = []
        for r in other._routes:
            target = fallback_path if r.path == "/" else r.path
            if (r.method, target) in own:
                bad.append(f"{r.method} {target}")
        # WS routes merge top-level (unchanged path) — flag path collisions too.
        own_ws = {w.path for w in self._ws_routes}
        for w in other._ws_routes:
            if w.path in own_ws:
                bad.append(f"WS {w.path}")
        return bad

    def mount_app(self, other: "FastTB", fallback_prefix: str, source: str) -> List[str]:
        """Merge `other`'s routes into this app (Modell A).

        - other '/'  -> relocated to '/<fallback_prefix>' (matches with/without
          trailing slash)
        - all other routes -> merged top-level, unchanged
        - static mounts -> merged (dedup by url_prefix)

        Returns [] on success, or a list of conflicting routes (caller then
        serves `other` on its own port as a specialist). Idempotent per source:
        re-mounting the same source first unmounts it.
        """
        fp = "/" + fallback_prefix.strip("/")
        conflicts = self.mount_conflicts(other, fp)
        if conflicts:
            return conflicts

        if source in self._mounted_sources:
            self.unmount_app(source)

        added_routes, added_static = [], []
        for r in other._routes:
            if r.path == "/":
                nr = Route(
                    path=fp, method=r.method, handler=r.handler,
                    pattern=self._root_pattern(fp), param_names=[],
                    name=r.name, auth=r.auth,
                )
            else:
                nr = r  # reuse object as-is; top-level merge
            self._routes.append(nr)
            added_routes.append(nr)

        existing_prefixes = {p for p, _ in self._static_mounts}
        for url_prefix, directory in other._static_mounts:
            if url_prefix not in existing_prefixes:
                self._static_mounts.append((url_prefix, directory))
                added_static.append((url_prefix, directory))

        # WS routes merge top-level, path unchanged (clients connect to the
        # original WS path, e.g. "/ws/openLive"); they are NOT relocated under
        # the prefix. Dedup by path so re-mount can't duplicate a handler.
        added_ws = []
        own_ws_paths = {w.path for w in self._ws_routes}
        for w in other._ws_routes:
            if w.path not in own_ws_paths:
                self._ws_routes.append(w)
                added_ws.append(w)
                own_ws_paths.add(w.path)

        self._mounted_sources[source] = {
            "routes": added_routes, "static": added_static,
            "ws_routes": added_ws, "prefix": fp,
        }
        self._index_dirty = True
        return []

    def unmount_app(self, source: str) -> bool:
        """Remove all routes/static mounts a source contributed. Enables clean
        reload-from-src: unmount, re-import module, mount_app again."""
        meta = self._mounted_sources.pop(source, None)
        if not meta:
            return False
        rids = {id(r) for r in meta["routes"]}
        self._routes = [r for r in self._routes if id(r) not in rids]
        wids = {id(w) for w in meta.get("ws_routes", [])}
        if wids:
            self._ws_routes = [w for w in self._ws_routes if id(w) not in wids]
        for sm in meta["static"]:
            try:
                self._static_mounts.remove(sm)
            except ValueError:
                pass
        self._index_dirty = True
        return True

    def list_mounted(self) -> List[Dict[str, str]]:
        """Sources currently merged into this owner app (for UI display)."""
        return [{"source": src, "prefix": meta["prefix"]}
                for src, meta in self._mounted_sources.items()]

    def relocate_root(self, fallback_prefix: str) -> bool:
        """Manual trigger: move THIS app's own '/' to '/<fallback_prefix>'.

        Used by the first app to step aside voluntarily. Returns False if no
        own root route exists."""
        fp = "/" + fallback_prefix.strip("/")
        moved = False
        for r in self._routes:
            if r.path == "/":
                r.path = fp
                r.pattern = self._root_pattern(fp)
                r.param_names = []
                moved = True
        if moved:
            self._index_dirty = True
        return moved

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
            { "handler_id": { "on_connect": fn, "on_message": fn, "on_disconnect": fn,
                              "auth": bool } }

        "auth" is the resolved effective requirement (route.auth if set, else
        app.auth). The WS connect gate reads it to reject unauth connections.
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
                entry["auth"] = ws_route.auth if ws_route.auth is not None else self.auth
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

    # =========================================================================
    # Server Runner (WSGI Only) — One-Port-Collective aware
    # =========================================================================

    def _join_existing(self, host: str, port: int, source=None,
                       module_path=None, app_attr="app",
                       fallback_prefix=None) -> bool:
        """Ask the owner on host:port to import+mount this app from src.
        Returns True only if the owner reports a successful merge."""
        if not module_path:
            return False  # owner needs the import path to re-load from src
        import urllib.request
        base = os.environ.get("TB_TRAY_URL") or f"http://{host}:{port}"
        body = {"command": "mount_app", "args": {
            "module": module_path, "attr": app_attr,
            "prefix": fallback_prefix or "",
            "source": source or module_path,
        }}
        try:
            req = urllib.request.Request(
                f"{base.rstrip('/')}/tray/command",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=2.0)
            data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False
        return bool(data.get("ok") and (data.get("result") or {}).get("merged"))

    @staticmethod
    def request_unmount(source: str, host: str = "127.0.0.1",
                        port: int = 8080) -> bool:
        """Joiner-side: tell the owner to unload this source (clean deregister).
        Call when the UI section is closed / no longer needed."""
        import urllib.request
        base = os.environ.get("TB_TRAY_URL") or f"http://{host}:{port}"
        body = {"command": "unmount_app", "args": {"source": source}}
        try:
            req = urllib.request.Request(
                f"{base.rstrip('/')}/tray/command",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=2.0)
            data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return False
        return bool(data.get("ok") and (data.get("result") or {}).get("unmounted"))

    def serve(self, host: str = "127.0.0.1", port: int = 8080,
              blocking: bool = True, enable_ws: Optional[bool] = None,
              allow_specialist: bool = True, source: Optional[str] = None,
              module_path: Optional[str] = None, app_attr: str = "app",
              fallback_prefix: Optional[str] = None, standby: bool = False,
              standby_retry_s: float = 3.0, standby_jitter_s: float = 0.5):
        """Start this FastTB app via Waitress (WSGI) with One-Port-Collective role resolution.

        On bind, the role is decided at runtime by the bind mutex on `host:port`
        (no flag chooses it — whoever binds first wins), in this order:

          1. PORT FREE        -> OWNER. Binds the port, serves '/' plus built-in
                                 endpoints (/live, /api/*, auth, /health) via
                                 as_wsgi_app. Accepts joiners if the process also
                                 called register_collective_commands().
          2. PORT BUSY + standby=True
                              -> STANDBY. Does NOT join and does NOT move. Waits
                                 standby_retry_s (+jitter) and retries the SAME
                                 port forever. When the current owner dies, one
                                 standby binds and becomes owner; the rest keep
                                 waiting. This is the live-UI failover path.
          3. PORT BUSY + module_path set (+ not standby)
                              -> JOINER. Asks the owner (via POST /tray/command
                                 mount_app) to re-import this module from src and
                                 merge its routes (Modell A: own '/' relocates to
                                 '/<fallback_prefix>', all other routes merge
                                 top-level). On a successful merge this process
                                 has nothing left to serve and serve() returns
                                 None. On conflict (owner already owns one of this
                                 app's non-root routes) it falls through to (4).
          4. PORT BUSY + allow_specialist=True (+ not standby, no merge)
                              -> SPECIALIST. Binds the next free port
                                 (port+1..port+49) and serves itself there. Used
                                 when an app is incompatible with the owner.
                                 If allow_specialist=False, raises OSError instead.

        Shutdown is forced: close() shuts the server immediately, does NOT wait
        for open client connections to drain, and never calls os._exit — so it is
        safe to run inside an owner process that hosts other mounted sub-apps.

        Args:
            host: Interface to bind, e.g. "127.0.0.1" (local only) or "0.0.0.0".
            port: Preferred port. Role resolution (above) decides whether this
                exact port is bound (owner/standby) or an offset is used
                (specialist).
            blocking: True  -> run in the calling thread until shutdown, then
                              clean up and return.
                      False -> start Waitress in a daemon thread and return the
                              server handle immediately (non-blocking).
            enable_ws: Force the WS infrastructure (ZMQ broker + WS worker +
                event bridge) on/off. None = auto: enabled iff this app has
                @app.websocket() routes registered.
            allow_specialist: If the port is busy and no merge/standby applies,
                allow binding the next free port. False -> raise OSError instead
                of moving (use for fixed-port services like the live-UI).
            source: Stable identity of this app for the owner's mount registry.
                Used as the key for mount_app/unmount_app so the owner can later
                unload exactly this contribution (reload-from-src). Defaults to
                module_path when omitted.
            module_path: Importable module path the OWNER re-imports from src to
                obtain this app's FastTB instance (e.g.
                "toolboxv2.utils.workers.fast.local_ui"). REQUIRED for the joiner
                path (3); without it a busy port can only go standby or
                specialist, never merge.
            app_attr: Attribute name of the FastTB instance inside module_path
                (default "app"). The owner does getattr(module, app_attr).
            fallback_prefix: Prefix this app's own '/' relocates to when merging
                under an owner that already owns '/', e.g. "app2" -> '/app2'.
                Empty -> the owner derives a slug from this app's title.
            standby: Enable the STANDBY role (2). When True, a busy port is never
                joined or escaped — the process waits and retries the same port,
                providing hot failover for the live-UI owner.
            standby_retry_s: Base wait between standby rebind attempts (seconds).
            standby_jitter_s: Upper bound of random jitter added to each standby
                wait, to avoid thundering-herd when several standbys race for a
                freed port.

        Returns:
            - blocking=True:  None (returns only after shutdown).
            - blocking=False: the Waitress server handle (already running in a
                              background thread).
            - joiner merged:  None (the owner serves this app; nothing local to
                              run).

        Raises:
            OSError: no bindable port within port..port+49 (specialist path), or
                the port is busy with allow_specialist=False and no standby/merge.
        """
        import socket as _socket
        from .fast_tb_handler import FastTBHandler
        from waitress.server import create_server

        handler = FastTBHandler(self)
        wsgi_app = handler.as_wsgi_app(enable_ws=enable_ws)

        server = None
        bound_port = port
        attempt = 0
        while True:
            try:
                server = create_server(
                    wsgi_app, host=host, port=bound_port,
                    clear_untrusted_proxy_headers=True,
                )
                break
            except OSError as e:
                if e.errno not in (errno.EADDRINUSE,
                                   getattr(errno, "WSAEADDRINUSE", 10048)):
                    raise
                # Standby: wait for the SAME port (live-UI failover). Never join,
                # never move. First replica to bind owns '/'; rest keep waiting
                # and take over when that one dies.
                if standby:
                    wait_s = standby_retry_s + random.uniform(0.0, standby_jitter_s)
                    print(f"[FastTB] {host}:{port} busy, standby {wait_s:.1f}s")
                    time.sleep(wait_s)
                    continue
                # First attempt: try to join the owner as a sub-app.
                if attempt == 0 and self._join_existing(
                    host, port, source, module_path, app_attr, fallback_prefix
                ):
                    print(f"[FastTB] joined owner on {host}:{port}")
                    return None
                if not allow_specialist:
                    raise
                bound_port += 1  # specialist: next free port
                attempt += 1
                if attempt >= 50:
                    raise OSError(f"[FastTB] no free port in {port}..{port + 49}")

        if server is None:
            raise OSError(f"[FastTB] no free port in {port}..{port + 49}")
        self._server = server
        role = "owner" if bound_port == port else f"specialist:{bound_port}"
        print(f"[FastTB] Serving on http://{host}:{bound_port} "
              f"({role}, {'blocking' if blocking else 'non-blocking'})")

        if not blocking:
            t = threading.Thread(target=server.run, daemon=True,
                                 name="fasttb-waitress")
            t.start()
            return server

        # Register SIGINT/SIGTERM so Strg+C reliably unblocks the Waitress accept
        # loop. On Windows `except KeyboardInterrupt` around server.run() does NOT
        # fire — calling server.close() is what makes run() return.
        import signal as _signal

        def _shutdown_handler(_sig, _frame):
            print(f"\n[FastTB] signal {_sig} — shutting down…")
            try:
                self.close()  # unblocks server.run() -> finally runs a_exit
            except Exception as e:
                print(e)
                pass

        if threading.current_thread() is threading.main_thread():
            try:
                _signal.signal(_signal.SIGINT, _shutdown_handler)
                _signal.signal(_signal.SIGTERM, _shutdown_handler)
            except (ValueError, RuntimeError) as e:
                print(f"[FastTB] could not register signal handlers: {e}")

        try:
            server.run()
        except KeyboardInterrupt:
            print("\n[FastTB] Strg+C — shutting down…")
        finally:
            # Guaranteed on-exit save: run the async app exit flow, THEN force-close.
            # run() has already returned here, so the main thread is free for asyncio.run.
            self.close()  # forced close, no os._exit
        return None

    async def serve_async(self, host: str = "127.0.0.1", port: int = 8080,
                          blocking: bool = True, enable_ws: Optional[bool] = None,
                          allow_specialist: bool = True, **kwargs):
        """Async wrapper: starts Waitress in a background thread."""
        import asyncio
        server = self.serve(host=host, port=port, blocking=False,
                            enable_ws=enable_ws, allow_specialist=allow_specialist, **kwargs)
        if server is None or not blocking:
            return server
        try:
            while True:
                await asyncio.sleep(3600)
        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n[FastTB] Async shutdown…")
        finally:
            self.close()
        return None

    def close(self):
        """Stop WS infrastructure, then the server. Idempotent, no os._exit.

        Stops the ZMQ event managers + WS worker so their ZMQ contexts are
        term()'d and the daemon threads exit (otherwise pyzmq blocks at
        interpreter exit -> process hangs).
        """
        import asyncio as _asyncio

        def _stop_on(manager, loop, *, stop_loop=False):
            """Run manager.stop() on its own loop from this sync thread."""
            if manager is None or loop is None or loop.is_closed():
                return
            try:
                fut = _asyncio.run_coroutine_threadsafe(manager.stop(), loop)
                fut.result(timeout=2)
            except TimeoutError:
                pass
            except Exception as e:
                print(f"[FastTB] stop({getattr(manager, 'worker_id', manager)}) failed: {e}")
            if stop_loop:
                with Spinner("closing worker loops"):
                    try:
                        loop.call_soon_threadsafe(loop.stop)
                    except Exception as e:
                        print(e)
                        pass

        # HTTP em: shared fasttb-loop -> stop manager only (loop is daemon).
        _stop_on(getattr(self, "_ws_em", None), getattr(self, "_ws_em_loop", None))
        # WS worker: stop() closes serve_forever -> run_until_complete returns -> thread exits.
        _stop_on(getattr(self, "_ws_worker", None), getattr(self, "_ws_worker_loop", None))
        # Broker: run_forever -> must also stop its loop so the thread exits.
        _stop_on(getattr(self, "_ws_broker", None), getattr(self, "_ws_broker_loop", None), stop_loop=False)

        self._ws_em = self._ws_worker = self._ws_broker = None

        srv = self._server
        self._server = None
        if srv is not None:
            try:
                srv.close()
            except Exception as e:
                print(e)
                pass
