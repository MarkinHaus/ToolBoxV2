#!/usr/bin/env python3
"""
fast_tb_handler.py - Request Dispatcher for FastTB

Resolves routes, injects parameters via inspect.signature,
and bridges between WSGI environ and FastTB endpoint handlers.

Integration points:
  1. HTTPWorker.wsgi_app — insert after auth/health checks, before 404
  2. Standalone WSGI app — via FastTBHandler.as_wsgi_app()
"""

import asyncio
import inspect
import json
import logging
import time
import traceback
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional, Tuple

from toolboxv2 import get_logger
from toolboxv2.utils.workers.server_worker import (
    ParsedRequest,
    json_response,
    html_response,
    error_response,
    parse_request,
)
from toolboxv2.utils.workers.session import SessionData

logger = get_logger()


def _file_iter(file_obj, chunk_size: int = 65536):
    """Generator for streaming a file object as WSGI response body."""
    try:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        file_obj.close()


def _is_hashed_filename(path: str) -> bool:
    """Check if filename contains a content hash (e.g. main-5d3f7ed2.js).

    Hashed files are immutable and can be cached forever.
    """
    import os
    import re
    name = os.path.basename(path)
    return bool(re.search(r'[-_.][0-9a-f]{6,}\.', name))


class FastTBHandler:
    """Dispatch engine for FastTB routes.

    Responsibilities:
      - Match incoming path+method to a registered Route
      - Inspect handler signature and inject the right parameters
      - Convert return values to WSGI-compatible (status, headers, body) tuples
    """

    def __init__(self, fast_tb_app: "FastTB", session_manager=None):
        """
        Args:
            fast_tb_app: FastTB instance with registered routes
            session_manager: Optional SessionManager for cookie handling in standalone mode
        """
        from .fast_tb import FastTB
        self._app: FastTB = fast_tb_app
        self._session_manager = session_manager

    def has_route(self, path: str, method: str) -> bool:
        """Check if FastTB can handle this path+method."""
        return self._app.has_route(path, method)

    async def handle_request(
        self,
        request: ParsedRequest,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Resolve route, inject params, execute handler, format response.

        Also serves static files from mounted directories.
        """
        # Static file check (GET only)
        if request.method.upper() == "GET":
            static_path = self._app.resolve_static(request.path)
            if static_path is not None:
                return self._serve_static_file(static_path)

        result = self._app.resolve_route(request.path, request.method)
        if result is None:
            return error_response("Not Found", 404, "NotFound")

        route, path_params = result

        try:
            kwargs = self._build_kwargs(route.handler, request, path_params)
            return await self._execute_and_format(route.handler, kwargs)

        except Exception as e:
            logger.error(f"FastTB handler error [{route.name}]: {e}")
            traceback.print_exc()
            return error_response(str(e), 500)

    # =========================================================================
    # Parameter Injection
    # =========================================================================

    def _build_kwargs(
        self,
        handler: Callable,
        request: ParsedRequest,
        path_params: Dict[str, str],
    ) -> Dict[str, Any]:
        """Inspect handler signature and build kwargs.

        Resolution order for each parameter name:
          1. "request" / type-hint ParsedRequest -> inject ParsedRequest
          2. "session" / type-hint SessionData   -> inject request.session
          3. Path parameter (from URL regex)      -> inject as str
          4. Query parameter                      -> inject from query_params
          5. Body field (from json_data)           -> inject from json_data
        """
        sig = inspect.signature(getattr(handler, '__wrapped__', handler))
        kwargs: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            # Skip self for bound methods
            if param_name == "self":
                continue

            # Skip *args and **kwargs — not injectable
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            annotation = param.annotation
            has_default = param.default is not inspect.Parameter.empty

            # 1. Request injection
            if param_name == "request" or annotation is ParsedRequest:
                kwargs[param_name] = request
                continue

            # 2. Session injection
            if param_name == "session" or annotation is SessionData:
                kwargs[param_name] = request.session or SessionData()
                continue

            # 3. Path parameters
            if param_name in path_params:
                kwargs[param_name] = self._coerce(path_params[param_name], annotation)
                continue

            # 4. Query parameters
            if param_name in request.query_params:
                vals = request.query_params[param_name]
                raw = vals[0] if len(vals) == 1 else vals
                kwargs[param_name] = self._coerce(raw, annotation)
                continue

            # 5. JSON body fields
            if request.json_data and isinstance(request.json_data, dict):
                if param_name in request.json_data:
                    kwargs[param_name] = request.json_data[param_name]
                    continue

            # 6. Form data fields
            if request.form_data and param_name in request.form_data:
                kwargs[param_name] = request.form_data[param_name]
                continue

            # 7. Default or skip
            if has_default:
                continue  # inspect.signature handles default

            # Missing required parameter
            return self._missing_param_error(param_name)

        return kwargs

    @staticmethod
    def _coerce(value: str, annotation) -> Any:
        """Coerce string value to annotated type."""
        if annotation is inspect.Parameter.empty or annotation is str:
            return value
        try:
            if annotation is int:
                return int(value)
            if annotation is float:
                return float(value)
            if annotation is bool:
                return value.lower() in ("true", "1", "yes")
        except (ValueError, TypeError):
            pass
        return value

    @staticmethod
    def _missing_param_error(param_name: str):
        """Return error dict that _execute_and_format will detect."""
        raise ValueError(f"Missing required parameter: {param_name}")

    # =========================================================================
    # Execution & Response Formatting
    # =========================================================================

    async def _execute_and_format(
        self,
        handler: Callable,
        kwargs: Dict[str, Any],
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Execute handler and convert return value to response tuple."""

        # Call handler (async or sync)
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**kwargs)
        else:
            result = handler(**kwargs)

        return self._format_result(result)

    @staticmethod
    def _format_result(result) -> Tuple[int, Dict[str, str], bytes]:
        """Convert handler return value to (status, headers, body).

        Supported return types:
          - tuple(status, headers, body)  -> pass through (body can be generator/async gen)
          - tuple(status, data)           -> json with status
          - Result object                 -> delegate to _format_tb_result
          - dict / list                   -> json_response(200)
          - str starting with '<'         -> html_response
          - str                           -> json {"result": str}
          - None                          -> json {"status": "ok"}
          - bytes                         -> raw binary
          - async generator / generator   -> streaming body (SSE or raw)
        """
        if result is None:
            return json_response({"status": "ok"})

        # Explicit tuple: (status, headers, body)
        if isinstance(result, tuple):
            if len(result) == 3:
                return result
            if len(result) == 2:
                status, data = result
                if isinstance(data, dict):
                    return json_response(data, status=status)
                return (status, {"Content-Type": "text/plain"}, str(data).encode())

        # ToolBoxV2 Result object (has is_error/get/result attributes)
        if hasattr(result, "is_error") and hasattr(result, "get") and hasattr(result, "result"):
            return FastTBHandler._format_tb_result(result)

        # Dict or list -> JSON
        if isinstance(result, (dict, list)):
            return json_response(result)

        # String
        if isinstance(result, str):
            if result.strip().startswith("<"):
                return html_response(result)
            return json_response({"result": result})

        # Bytes
        if isinstance(result, bytes):
            return (200, {"Content-Type": "application/octet-stream"}, result)

        # Async generator or generator — return as streaming body
        # The wsgi_app layer handles wrapping these
        if inspect.isasyncgen(result) or inspect.isgenerator(result):
            return (200, {"Content-Type": "application/octet-stream"}, result)

        # Fallback
        return json_response({"result": str(result)})

    @staticmethod
    def _format_tb_result(result) -> Tuple[int, Dict[str, str], Any]:
        """Convert a ToolBoxV2 Result object to (status, headers, body).

        Handles: stream, html, special_html, redirect, file_path, file, binary, json.
        """
        import mimetypes as _mt
        import os as _os

        if result.is_error():
            status = getattr(result.info, "exec_code", 500)
            if status <= 0:
                status = 500
            return error_response(
                getattr(result.info, "help_text", "Error"), status
            )

        data_type = getattr(result.result, "data_type", "")
        data = result.get()

        # Streaming (SSE or raw)
        if data_type == "stream":
            stream_info = data
            headers = stream_info.get("headers", {}).copy()
            content_type = stream_info.get("content_type", "text/event-stream")
            headers["Content-Type"] = content_type
            if content_type == "text/event-stream":
                headers.setdefault("Cache-Control", "no-cache")
                headers.setdefault("X-Accel-Buffering", "no")
            return (200, headers, stream_info.get("generator"))

        if data_type == "html":
            return html_response(data, status=getattr(result.info, "exec_code", 200) or 200)

        if data_type == "special_html":
            return html_response(data.get("html", ""), headers=data.get("headers", {}))

        if data_type == "redirect":
            code = getattr(result.info, "exec_code", 302)
            return (code, {"Location": data, "Content-Type": "text/plain"}, b"")

        if data_type == "file_path":
            if not isinstance(data, str) or not _os.path.isfile(data):
                return error_response(f"File not found: {data}", 404, "NotFound")

            info_str = getattr(result.result, "data_info", "")
            filename = _os.path.basename(data)
            if info_str and "filename=" in info_str:
                filename = info_str.split("filename=", 1)[1].strip()

            ct, _ = _mt.guess_type(data)
            ct = ct or "application/octet-stream"
            size = _os.path.getsize(data)

            headers = {
                "Content-Type": ct,
                "Content-Length": str(size),
                "Content-Disposition": f'attachment; filename="{filename}"',
            }
            # Return file_obj — wsgi_app handles streaming via file_wrapper
            return (200, headers, ("__file_stream__", open(data, "rb"), {}))

        if data_type == "file":
            import base64
            file_data = base64.b64decode(data) if isinstance(data, str) else data
            info_str = getattr(result.result, "data_info", "")
            filename = info_str.replace("File download: ", "") if info_str else "download"
            ct, _ = _mt.guess_type(filename)
            ct = ct or "application/octet-stream"
            return (200, {
                "Content-Type": ct,
                "Content-Length": str(len(file_data)),
                "Content-Disposition": f'attachment; filename="{filename}"',
            }, file_data)

        if data_type == "binary":
            if isinstance(data, dict):
                bd = data.get("data", b"")
                ct = data.get("content_type", "application/octet-stream")
                fn = data.get("filename")
            else:
                bd = data if isinstance(data, bytes) else str(data).encode()
                ct = "application/octet-stream"
                fn = None
            headers = {"Content-Type": ct, "Content-Length": str(len(bd))}
            if fn:
                headers["Content-Disposition"] = f'attachment; filename="{fn}"'
            return (200, headers, bd)

        # Default: JSON serialization of the full Result
        return json_response(result.as_dict() if hasattr(result, "as_dict") else {"result": str(data)})

    # =========================================================================
    # Static File Serving
    # =========================================================================

    @staticmethod
    def _serve_static_file(file_path: str) -> Tuple[int, Dict[str, str], bytes]:
        """Serve a static file with correct content-type.

        Args:
            file_path: Absolute path to the file (already validated by resolve_static)

        Returns:
            (status, headers, body)
        """
        import mimetypes
        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                body = f.read()
        except OSError as e:
            return error_response(str(e), 500)

        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Cache-Control": "public, max-age=31536000, immutable"
            if _is_hashed_filename(file_path)
            else "public, max-age=3600",
        }
        return (200, headers, body)

    # =========================================================================
    # Standalone WSGI App (reuses HTTPWorker infrastructure)
    # =========================================================================

    def as_wsgi_app(self, config=None, app=None, enable_ws: bool | None = None) -> Callable:
        """Return a WSGI app that wraps HTTPWorker with FastTB routes.

        Reuses HTTPWorker's full infrastructure:
        - Session management (cookies + Bearer fallback)
        - Auth endpoints (/validateSession, /auth/*, etc.)
        - CORS headers
        - /api/client-logs, /health, /metrics, /api/ping, /api/ip, /api/geo
        - ToolBoxHandler for /api/* module calls

        FastTB routes are checked first, then HTTPWorker for built-in endpoints.

        When WebSocket routes are registered (or enable_ws=True), automatically
        starts ZMQ broker + WS worker + event bridge so @app.websocket() handlers
        work out of the box.

        Args:
            config: WorkerConfig. If None, loads default via load_config().
            app: ToolBoxV2 App instance. If None, uses get_app().
            enable_ws: Force WS infrastructure on/off. None = auto (on if ws_routes exist).

        Returns:
            WSGI callable (environ, start_response) -> [bytes]
        """
        from toolboxv2.utils.workers.server_worker import HTTPWorker, ToolBoxHandler, AsyncGenToSyncIter
        from toolboxv2.utils.workers.config import load_config
        from toolboxv2.utils.system.getting_and_closing_app import get_app as tb_get_app
        import threading

        config = config or load_config()
        app = app or self._app.app_instance or tb_get_app(from_="FastTBStandalone")

        # Create HTTPWorker but don't start its HTTP server
        worker = HTTPWorker("fasttb_standalone", config, app=app)
        worker._init_session_manager()
        worker._init_access_controller()
        worker._init_auth_handler()
        worker._toolbox_handler = ToolBoxHandler(
            app, config, worker._access_controller, config.toolbox.api_prefix
        )

        # Register WS handlers from FastTB into the app
        ws_handlers = self._app.get_websocket_handlers()
        if ws_handlers:
            if not hasattr(app, "websocket_handlers"):
                app.websocket_handlers = {}
            app.websocket_handlers.update(ws_handlers)

        # Background event loop for async FastTB handlers + event manager
        _loop = None
        _loop_lock = threading.Lock()

        def _get_loop():
            nonlocal _loop
            if _loop is None or _loop.is_closed():
                with _loop_lock:
                    if _loop is None or _loop.is_closed():
                        _loop = asyncio.new_event_loop()
                        t = threading.Thread(
                            target=_loop.run_forever, daemon=True, name="fasttb-loop"
                        )
                        t.start()
            return _loop

        # ---- WS Infrastructure (ZMQ broker + WS worker + bridge) ----
        has_ws = enable_ws if enable_ws is not None else bool(self._app._ws_routes)

        if has_ws:
            self._start_ws_infrastructure(config, app, worker, _get_loop)

        ftb_handler = self
        original_wsgi = worker.wsgi_app

        def patched_wsgi(environ, start_response):
            """FastTB routes first, then HTTPWorker for built-in endpoints."""
            path = environ.get("PATH_INFO", "/")
            method = environ.get("REQUEST_METHOD", "GET")

            # Check if FastTB can handle this (path + method only, no body read)
            if ftb_handler.has_route(path, method):
                request = parse_request(environ)

                loop = _get_loop()
                future = asyncio.run_coroutine_threadsafe(
                    ftb_handler.handle_request(request), loop
                )
                try:
                    status, headers, body = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"FastTB standalone error: {e}")
                    status, headers, body = error_response(str(e), 500)

                # Session cookie
                if worker._session_manager and request.session:
                    cookie_header = worker._session_manager.get_set_cookie_header(
                        request.session
                    )
                    if cookie_header:
                        headers["Set-Cookie"] = cookie_header

                # CORS
                if hasattr(worker, "_get_cors_headers"):
                    cors = worker._get_cors_headers(environ)
                    headers.update(cors)

                status_line = f"{status} {HTTPStatus(status).phrase}"
                response_headers = [(k, v) for k, v in headers.items()]
                start_response(status_line, response_headers)

                # Handle different body types
                if isinstance(body, bytes):
                    return [body]

                # File streaming
                if isinstance(body, tuple) and len(body) == 3 and body[0] == "__file_stream__":
                    _, file_obj, req_environ = body
                    if 'wsgi.file_wrapper' in environ:
                        return environ['wsgi.file_wrapper'](file_obj, 65536)
                    return _file_iter(file_obj)

                # Async generator (SSE / streaming)
                if inspect.isasyncgen(body):
                    return AsyncGenToSyncIter(body, loop)

                # Sync generator
                if inspect.isgenerator(body):
                    return body

                return [str(body).encode()]

            # No FastTB match — delegate to HTTPWorker (auth, health, API, etc.)
            return original_wsgi(environ, start_response)

        return patched_wsgi

    def _start_ws_infrastructure(self, config, app, worker, get_loop_fn):
        """Start ZMQ broker, WS worker, and event bridge in background threads.

        Called by as_wsgi_app() when WebSocket routes are registered.
        """
        import threading

        logger.info("[FastTB] Starting WebSocket infrastructure...")

        # 1. Start ZMQ broker in background
        def _run_broker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from toolboxv2.utils.workers.event_manager import ZMQEventManager
            broker = ZMQEventManager(
                worker_id="fasttb_broker",
                pub_endpoint=config.zmq.pub_endpoint,
                sub_endpoint=config.zmq.sub_endpoint,
                req_endpoint=config.zmq.req_endpoint,
                rep_endpoint=config.zmq.rep_endpoint,
                http_to_ws_endpoint=config.zmq.http_to_ws_endpoint,
                is_broker=True,
            )
            loop.run_until_complete(broker.start())
            loop.run_forever()

        threading.Thread(target=_run_broker, daemon=True, name="fasttb-zmq-broker").start()

        import time
        time.sleep(0.3)  # Wait for broker to bind

        # 2. Start WS worker in background
        def _run_ws_worker():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from toolboxv2.utils.workers.ws_worker import WSWorker
            ws = WSWorker("fasttb_ws", config)
            loop.run_until_complete(ws.start())

        threading.Thread(target=_run_ws_worker, daemon=True, name="fasttb-ws-worker").start()

        # 3. Start event manager + WS bridge on the FastTB event loop
        loop = get_loop_fn()

        async def _init_bridge():
            from toolboxv2.utils.workers.event_manager import ZMQEventManager
            from toolboxv2.utils.workers.ws_bridge import install_ws_bridge
            from toolboxv2.utils.workers.server_worker import WebSocketMessageHandler

            em = ZMQEventManager(
                worker_id="fasttb_http",
                pub_endpoint=config.zmq.pub_endpoint,
                sub_endpoint=config.zmq.sub_endpoint,
                req_endpoint=config.zmq.req_endpoint,
                rep_endpoint=config.zmq.rep_endpoint,
                http_to_ws_endpoint=config.zmq.http_to_ws_endpoint,
                is_broker=False,
            )
            await em.start()

            # Install WS bridge on app (ws_send, ws_broadcast etc.)
            install_ws_bridge(app, em, "fasttb_http")

            # Wire up WS message handler
            ac = worker._access_controller
            ws_handler = WebSocketMessageHandler(app, em, ac)

            from toolboxv2.utils.workers.event_manager import EventType, Event

            @em.on(EventType.WS_CONNECT)
            async def _on_connect(event: Event):
                await ws_handler.handle_ws_connect(event)

            @em.on(EventType.WS_MESSAGE)
            async def _on_message(event: Event):
                await ws_handler.handle_ws_message(event)

            @em.on(EventType.WS_DISCONNECT)
            async def _on_disconnect(event: Event):
                await ws_handler.handle_ws_disconnect(event)

            worker._event_manager = em
            worker._event_loop = loop
            logger.info(f"[FastTB] WS bridge ready (ws port {config.ws_worker.port})")

        future = asyncio.run_coroutine_threadsafe(_init_bridge(), loop)
        future.result(timeout=10)  # Block until bridge is ready
