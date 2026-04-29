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

import inspect
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

        # Pre-built index: method -> list of routes (populated on first resolve)
        self._route_index: Dict[str, List[Route]] = {}
        self._index_dirty = True

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
        """Check if a route exists for this path+method."""
        return self.resolve_route(path, method) is not None

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
