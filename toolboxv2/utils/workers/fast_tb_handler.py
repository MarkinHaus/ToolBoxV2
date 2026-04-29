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

from toolboxv2.utils.workers.server_worker import (
    ParsedRequest,
    json_response,
    html_response,
    error_response,
    parse_request,
)
from toolboxv2.utils.workers.session import SessionData

logger = logging.getLogger(__name__)


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

        Args:
            request: Parsed WSGI request

        Returns:
            (status_code, headers_dict, body_bytes)
        """
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
        sig = inspect.signature(handler)
        kwargs: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            # Skip self for bound methods
            if param_name == "self":
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
          - tuple(status, headers, body)  -> pass through
          - tuple(status, data)           -> json with status
          - dict / list                   -> json_response(200)
          - str starting with '<'         -> html_response
          - str                           -> json {"result": str}
          - None                          -> json {"status": "ok"}
          - Response (has .status_code)   -> extract fields
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

        # Fallback
        return json_response({"result": str(result)})

    # =========================================================================
    # Standalone WSGI App (for ASGI bridge or direct Waitress usage)
    # =========================================================================

    def as_wsgi_app(self) -> Callable:
        """Return a WSGI-compatible callable.

        This wraps the async handle_request in a sync WSGI interface,
        handling its own event loop for standalone usage.
        """
        _loop = None
        _loop_lock = __import__("threading").Lock()

        def _get_loop():
            nonlocal _loop
            if _loop is None or _loop.is_closed():
                with _loop_lock:
                    if _loop is None or _loop.is_closed():
                        _loop = asyncio.new_event_loop()
                        t = __import__("threading").Thread(
                            target=_loop.run_forever, daemon=True, name="fasttb-loop"
                        )
                        t.start()
            return _loop

        handler = self

        def wsgi_app(environ, start_response):
            # Session handling
            if handler._session_manager:
                session = handler._session_manager.get_session_from_request_sync(environ)
                environ["tb.session"] = session

            request = parse_request(environ)

            if not handler.has_route(request.path, request.method):
                status, headers, body = error_response("Not Found", 404, "NotFound")
            else:
                loop = _get_loop()
                future = asyncio.run_coroutine_threadsafe(
                    handler.handle_request(request), loop
                )
                try:
                    status, headers, body = future.result(timeout=30)
                except Exception as e:
                    logger.error(f"FastTB WSGI error: {e}")
                    status, headers, body = error_response(str(e), 500)

            # Cookie update
            if handler._session_manager and request.session:
                cookie_header = handler._session_manager.get_set_cookie_header(request.session)
                if cookie_header:
                    headers["Set-Cookie"] = cookie_header

            status_line = f"{status} {HTTPStatus(status).phrase}"
            response_headers = [(k, v) for k, v in headers.items()]
            start_response(status_line, response_headers)

            if isinstance(body, bytes):
                return [body]
            return [str(body).encode()]

        return wsgi_app
