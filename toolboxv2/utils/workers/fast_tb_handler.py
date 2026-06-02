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
    redirect_response,
    parse_request,
)
from toolboxv2.utils.workers.session import SessionData

logger = get_logger()
_inject_style_enabled = True  # Overridden by as_wsgi_app() from app.inject_style

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


def _async_gen_to_sync(async_gen, loop):
    """Convert async generator to sync iterator for WSGI streaming.

    Each __next__ blocks the Waitress thread until the next chunk arrives.
    This is unavoidable with WSGI — SSE streams occupy one thread each.

    To prevent thread starvation:
    - Waitress max_concurrent should be set high enough (50-100)
    - SSE streams should be finite (not infinite)
    - For high-concurrency SSE, use the WS worker instead

    Timeout per chunk: 30s. Sends SSE keepalive comment on timeout
    to prevent proxy/browser disconnect.
    """
    import asyncio

    aiter = async_gen.__aiter__()
    CHUNK_TIMEOUT = 30

    def _gen():
        while True:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    aiter.__anext__(), loop
                )
                data = future.result(timeout=CHUNK_TIMEOUT)

                if isinstance(data, str):
                    yield data.encode("utf-8")
                elif isinstance(data, bytes):
                    yield data
                else:
                    yield str(data).encode("utf-8")

            except StopAsyncIteration:
                return
            except TimeoutError:
                yield b": keepalive\n\n"
            except GeneratorExit:
                # Client disconnected — cancel the async generator
                future = asyncio.run_coroutine_threadsafe(
                    aiter.aclose(), loop
                )
                try:
                    future.result(timeout=2)
                except Exception:
                    pass
                return
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"SSE stream error: {e}")
                return

    return _gen()

def _maybe_inject_style(html_str: str) -> str:
    """Inject Paper CSS into HTML responses that lack TBJS web_context.

    Skips injection if:
    - Not a string or empty
    - inject_style globally disabled
    - HTML already contains TBJS markers (tbjs-main, TB.init, web_context)
    - HTML already has substantial user CSS (>200 chars in a <style> block
      that isn't from _SHARED_CSS / ftb-wrap)

    Injects: fonts + main.css + paper.css + data-style="paper" on <html>.
    """
    if not html_str or not isinstance(html_str, str):
        return html_str

    if not globals().get('_inject_style_enabled', True):
        return html_str

    lower = html_str[:3000].lower()

    # Already has TBJS — don't touch
    if any(m in lower for m in ("tbjs-main", "tb.init", "web_context", "tbjs.js")):
        return html_str

    # Already has substantial user styling (not our own ftb- classes)
    if "<style>" in lower:
        style_start = lower.index("<style>") + 7
        style_end = lower.find("</style>", style_start)
        if style_end == -1:
            style_end = len(lower)
        style_content = html_str[style_start:style_end]
        # Our own defaults use .ftb- prefixed classes — don't count those
        if len(style_content) > 200 and ".ftb-" not in style_content[:100]:
            return html_str

    from toolboxv2.utils.workers.fast_tb_defaults import _MAIN_CSS, _PAPER_CSS, _FONTS

    style_block = f"{_FONTS}\n<style>{_MAIN_CSS}</style>\n<style>{_PAPER_CSS}</style>\n"

    # Ensure data-style="paper" on <html>
    if "<html" in html_str:
        if 'data-style=' not in html_str[:500]:
            html_str = html_str.replace("<html", '<html data-style="paper"', 1)
    else:
        # No <html> tag — wrap minimally
        html_str = '<html data-style="paper">\n' + html_str

    # Inject before </head>
    if "</head>" in html_str:
        return html_str.replace("</head>", style_block + "</head>", 1)

    # Inject before <body>
    if "<body" in html_str:
        idx = html_str.index("<body")
        return html_str[:idx] + "<head>" + style_block + "</head>\n" + html_str[idx:]

    # No head/body — prepend after <html>
    idx = html_str.index(">", html_str.index("<html")) + 1
    return html_str[:idx] + "\n<head>" + style_block + "</head>\n" + html_str[idx:]

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
        # Auth gate (enforcement variant iii: global). When effective auth is on,
        # every served path requires an authenticated session except /health.
        # Effective auth = route.auth if set, else app-level self._app.auth. For
        # static files / unknown paths (no Route) the app-level flag applies.
        if request.path.startswith("/vendors-") or request.path.startswith("/main-"):
            pass
        elif request.path.startswith("/web/"):
            pass
        elif request.path == "/helper.html":
            pass
        elif request.path == "/index.html":
            pass
        elif request.path == "/favicon.ico":
            pass
        elif request.path != "/health":
            _match = self._app.resolve_route(request.path, request.method)
            if _match is not None and _match[0].auth is not None:
                _eff_auth = _match[0].auth
            else:
                _eff_auth = self._app.auth
            if _eff_auth:
                _sess = request.session
                if not (_sess is not None and _sess.is_authenticated):
                    # Browser (HTML) requests are redirected to the login page so
                    # the user can authenticate; ?next carries the original path so
                    # tbjs (_handlePostAuthRedirect) returns here after login.
                    # Non-HTML (fetch/API) requests get a JSON 401 instead.
                    _accept = request.headers.get("accept", '').lower()
                    if "text/html" in _accept:
                        # Ensure the local login assets are reachable. If the app
                        # didn't permanently mount /web, mount it now for the flow.
                        self._app._ensure_web_mounted()
                        from urllib.parse import quote
                        _next = quote(request.path, safe="/")
                        return redirect_response(
                            f"/web/assets/login.html?next={_next}", 302
                        )
                    return error_response("Authentication required", 401, "Unauthorized")


        result = self._app.resolve_route(request.path, request.method)

        # Static file check (GET only)
        if result is None and request.method.upper() == "GET":
            static_path = self._app.resolve_static(request.path)
            if static_path is not None:
                return self._serve_static_file(static_path)

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
        import functools

        # Call handler (async or sync)
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**kwargs)
        else:
            result = await asyncio.to_thread(functools.partial(handler, **kwargs))

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
                return html_response(_maybe_inject_style(result))
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
            return html_response(_maybe_inject_style(data), status=getattr(result.info, "exec_code", 200) or 200)

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

        # Permanently mount login UI assets when requested (e.g. remote base is
        # local). Idempotent: skipped if the app already mounted /web itself.
        if self._app.auth and self._app.serve_login_assets:
            self._app._ensure_web_mounted()

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
                # Attach session before parse_request reads environ["tb.session"].
                # Mirrors server_worker.wsgi_app: cookie/API-key/Bearer -> SessionData,
                # never None (anonymous fallback). Bearer fallback for cross-origin
                # (e.g. Tauri) where the cookie is not sent.
                if worker._session_manager:
                    session = worker._session_manager.get_session_from_request_sync(environ)
                    if (not session or session.anonymous) and environ.get(
                        "HTTP_AUTHORIZATION", "").startswith("Bearer "):
                        bearer_token = environ["HTTP_AUTHORIZATION"][7:]
                        try:
                            valid, jwt_session = worker._session_manager.verify_session_token(bearer_token)
                            if valid and jwt_session:
                                session = jwt_session
                                session.mark_dirty()
                        except Exception as e:
                            logger.debug(f"Bearer fallback failed: {e}")
                    environ["tb.session"] = session

                request = parse_request(environ)

                loop = _get_loop()
                future = asyncio.run_coroutine_threadsafe(
                    ftb_handler.handle_request(request), loop
                )
                try:
                    status, headers, body = future.result(timeout=15)
                except TimeoutError as e:
                    logger.error(f"FastTB standalone error: {e}")
                    status, headers, body = error_response(str(e), 500)
                except Exception as e:
                    import traceback
                    _e = traceback.format_exc()
                    logger.error(f"FastTB standalone error: {e} {_e}")
                    status, headers, body = error_response(str(_e), 500)

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
                    return _async_gen_to_sync(body, loop)

                # Sync generator
                if inspect.isgenerator(body):
                    return body

                return [str(body).encode()]

            # No FastTB match — delegate to HTTPWorker (auth, health, API, etc.)
            return original_wsgi(environ, start_response)

        # Register default pages (/docs, welcome)
        from toolboxv2.utils.workers.fast_tb_defaults import register_defaults
        register_defaults(self._app)

        # Store inject_style flag for _maybe_inject_style
        global _inject_style_enabled
        _inject_style_enabled = getattr(self._app, 'inject_style', True)

        # Warn if thread count is too low for SSE/streaming
        if self._app.hot_reload:
            self._start_hot_reload(app, config, _get_loop)
        _has_sse = any(
            r.handler.__name__ == 'sse_wrapper'
            for r in self._app._routes
            if hasattr(r, 'handler')
        )
        if _has_sse and config.http_worker.max_concurrent < 20:
            logger.warning(
                f"[FastTB] SSE routes detected but max_concurrent={config.http_worker.max_concurrent}. "
                f"Each SSE stream blocks a Waitress thread. "
                f"Recommend: config.http_worker.max_concurrent >= 50"
            )

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
                cluster_secret=getattr(config.zmq, "cluster_secret", ""),
            )
            self._app._ws_broker = broker  # retain for shutdown
            self._app._ws_broker_loop = loop
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
            self._app._ws_worker = ws  # retain for shutdown
            self._app._ws_worker_loop = loop
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
                cluster_secret=getattr(config.zmq, "cluster_secret", ""),
            )
            await em.start()

            # Install WS bridge on app (ws_send, ws_broadcast etc.)
            install_ws_bridge(app, em, "fasttb_http")
            # Point this FastTB instance at the ToolBox app that now carries the
            # WS bridge, so FastTB.ws_broadcast/ws_send can delegate to it.
            self._app.app_instance = app

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
            self._app._ws_em = em  # retain for shutdown
            self._app._ws_em_loop = loop
            logger.info(f"[FastTB] WS bridge ready (ws port {config.ws_worker.port})")

        future = asyncio.run_coroutine_threadsafe(_init_bridge(), loop)
        future.result(timeout=10)  # Block until bridge is ready

    def _start_hot_reload(self, app, config, get_loop_fn):
        """Start file watcher for hot-reload in development mode.

        Uses watchdog to monitor registered directories. On file change,
        broadcasts a reload command to all connected WS clients.

        Falls back to no-op if watchdog is not installed.
        """
        import threading

        watch_dirs = list(self._app._watch_dirs)

        # Auto-add the directory of the main script
        import sys
        import os
        main_mod = sys.modules.get("__main__")
        if main_mod and hasattr(main_mod, "__file__") and main_mod.__file__:
            main_dir = os.path.dirname(os.path.abspath(main_mod.__file__))
            if main_dir not in watch_dirs:
                watch_dirs.append(main_dir)

        if not watch_dirs:
            logger.info("[FastTB] Hot-reload enabled but no directories to watch")
            return

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            logger.warning(
                "[FastTB] Hot-reload enabled but watchdog not installed. "
                "Install with: pip install watchdog"
            )
            return

        WATCH_EXTENSIONS = {".py", ".js", ".css", ".html", ".jsx", ".ts", ".tsx", ".vue", ".svelte"}
        _last_reload = [0.0]  # debounce tracker
        app_ref = self._app

        class ReloadHandler(FileSystemEventHandler):
            def on_modified(self, event):
                self._handle(event)

            def on_created(self, event):
                self._handle(event)

            def _handle(self, event):
                import time
                import importlib
                if event.is_directory:
                    return

                ext = os.path.splitext(event.src_path)[1].lower()
                if ext not in WATCH_EXTENSIONS:
                    return

                path_str = event.src_path.replace("\\", "/")
                if any(skip in path_str for skip in ("__pycache__", ".git/", "node_modules/", ".pyc")):
                    return

                now = time.time()
                if now - _last_reload[0] < 0.5:
                    return
                _last_reload[0] = now

                rel_path = os.path.relpath(event.src_path)
                logger.info(f"[FastTB] File changed: {rel_path}")

                # For .py files: reload the module so new code is picked up
                if ext == ".py":
                    try:
                        self._reload_python_module(event.src_path)
                    except Exception as e:
                        logger.error(f"[HMR] reload error: {e}", exc_info=True)

                # Broadcast reload to all WS clients
                if hasattr(app, "ws_broadcast_all"):
                    loop = get_loop_fn()
                    asyncio.run_coroutine_threadsafe(
                        app.ws_broadcast_all({
                            "type": "hot_reload",
                            "file": rel_path,
                            "ext": ext,
                        }),
                        loop,
                    )

            def _exec_reload(self, filepath):
                """Reload a Python file by extracting only function/class definitions.

                Parses the file AST to find top-level function and class defs,
                then compiles and executes ONLY those — skipping all module-level
                side effects (server start, global assignments, etc.).
                """
                import ast
                import types

                main_mod = sys.modules.get("__main__")
                if not main_mod:
                    return None

                with open(filepath, "r", encoding="utf-8") as f:
                    source = f.read()

                try:
                    tree = ast.parse(source, filepath)
                except SyntaxError as e:
                    logger.error(f"[HMR] Syntax error in {filepath}: {e}")
                    return None

                # Extract only FunctionDef, AsyncFunctionDef, ClassDef at top level
                safe_nodes = []
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        safe_nodes.append(node)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        safe_nodes.append(node)

                if not safe_nodes:
                    logger.debug(f"[HMR] No reloadable definitions in {filepath}")
                    return None

                # Build a new module with only the safe definitions
                new_tree = ast.Module(body=safe_nodes, type_ignores=[])
                ast.fix_missing_locations(new_tree)

                code = compile(new_tree, filepath, "exec")

                # Execute in a namespace that inherits existing globals
                namespace = dict(vars(main_mod))
                namespace["__name__"] = "__hmr_reload__"  # prevent if __name__=="__main__" blocks

                try:
                    exec(code, namespace)
                except Exception as e:
                    logger.error(f"[HMR] Exec error in {filepath}: {e}")
                    return None

                # Update __main__ with new definitions only
                updated = []
                for k, v in namespace.items():
                    if k.startswith("_"):
                        continue
                    old = getattr(main_mod, k, None)
                    if old is None:
                        continue
                    # Only update functions and classes, not instances or data
                    if isinstance(v, type) and isinstance(old, type):
                        setattr(main_mod, k, v)
                        updated.append(k)
                    elif callable(v) and callable(old) and not isinstance(old, type):
                        setattr(main_mod, k, v)
                        updated.append(k)

                # Return a namespace-like object for route updating
                result = types.SimpleNamespace(**{k: getattr(main_mod, k) for k in updated})
                logger.info(f"[HMR] Exec-reloaded {os.path.basename(filepath)}: {', '.join(updated) or 'nothing'}")
                return result

            def _reload_python_module(self, filepath):
                """Reload a Python module and update MinuBridge view classes.

                Preserves existing view state (ReactiveState values) while
                swapping in the new class definition (render, handlers).
                """
                import importlib

                # Find the module that corresponds to this file
                abs_path = os.path.abspath(filepath)
                target_module = None
                for name, mod in list(sys.modules.items()):
                    if name == "__main__":
                        continue  # Skip __main__ — reload would re-execute everything
                    mod_file = getattr(mod, "__file__", None)
                    if mod_file and os.path.abspath(mod_file) == abs_path:
                        target_module = mod
                        break

                if not target_module:
                    # For __main__ or unregistered modules: manually exec the file
                    # to get updated function/class definitions without full re-execute
                    logger.info(f"[HMR] Exec-reloading: {filepath}")
                    try:
                        reloaded = self._exec_reload(abs_path)
                    except Exception as e:
                        logger.error(f"[HMR] Exec-reload failed: {e}")
                        return
                else:
                    mod_name = target_module.__name__
                    logger.info(f"[HMR] Reloading module: {mod_name}")
                    try:
                        reloaded = importlib.reload(target_module)
                    except Exception as e:
                        logger.error(f"[HMR] Reload failed for {mod_name}: {e}")
                        return

                # Update MinuBridge view classes with new definitions

                bridge = getattr(app_ref, '_minu_bridge', None)
                if bridge:
                    for path, meta in bridge._view_registry.items():
                        old_class = meta["class"]
                        new_class = getattr(reloaded, old_class.__name__, None)

                        if new_class is None or new_class is old_class:
                            continue

                        logger.info(f"[HMR] Updating view class: {old_class.__name__} on {path}")
                        meta["class"] = new_class

                        # Hot-swap: update existing view instances with new methods
                        # while preserving their state
                        for sid, minu_session in bridge._sessions.items():
                            lookup_key = f"_path:{path}"
                            view = minu_session.get_view_by_key(lookup_key)
                            if view is None:
                                continue

                            # Preserve state values
                            saved_state = {}
                            for attr_name, state in view._state_attrs.items():
                                saved_state[attr_name] = state.value

                            # Swap the class — this updates render(), handlers etc.
                            view.__class__ = new_class

                            # Re-init state descriptors from new class
                            # (in case new states were added)
                            from toolboxv2.mods.Minu.core import ReactiveState
                            for attr_name in dir(new_class):
                                if attr_name.startswith("_"):
                                    continue
                                class_attr = getattr(new_class, attr_name, None)
                                if isinstance(class_attr, ReactiveState):
                                    if attr_name in saved_state:
                                        # Existing state — restore value
                                        if attr_name in view._state_attrs:
                                            view._state_attrs[attr_name].value = saved_state[attr_name]
                                    else:
                                        # New state added in updated code
                                        new_state = ReactiveState(
                                            class_attr._value,
                                            path=f"{view._view_id}.{attr_name}"
                                        )
                                        new_state.bind(view)
                                        setattr(view, attr_name, new_state)
                                        view._state_attrs[attr_name] = new_state

                            logger.info(
                                f"[HMR] View {view._view_id} hot-swapped "
                                f"(preserved {len(saved_state)} states)"
                            )

                # Update FastTB route handlers with new function references
                for route in app_ref._routes:
                    old_handler = route.handler
                    # Check the original function (unwrap SSE wrapper etc.)
                    orig = getattr(old_handler, '__wrapped__', old_handler)
                    handler_name = getattr(orig, '__name__', '')
                    handler_qual = getattr(orig, '__qualname__', '')

                    # Find matching function in reloaded module
                    new_fn = getattr(reloaded, handler_name, None)
                    if new_fn is None or new_fn is orig:
                        continue
                    if not callable(new_fn):
                        continue

                    # For SSE-wrapped handlers, re-wrap with new function
                    if hasattr(old_handler, '__wrapped__'):
                        old_handler.__wrapped__ = new_fn
                        logger.info(
                            f"[HMR] Updated wrapped route: {route.method} {route.path} → {handler_name}")
                    else:
                        route.handler = new_fn
                        logger.info(f"[HMR] Updated route: {route.method} {route.path} → {handler_name}")

                # Mark route index as dirty so next resolve picks up changes
                app_ref._index_dirty = True

        handler = ReloadHandler()
        observer = Observer()

        for d in watch_dirs:
            observer.schedule(handler, d, recursive=True)
            logger.info(f"[FastTB] Watching: {d}")

        observer.daemon = True
        observer.start()
        self._app._hot_reload_running = True
        logger.info(f"[FastTB] Hot-reload active ({len(watch_dirs)} directories)")
