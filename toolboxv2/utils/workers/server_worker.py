#!/usr/bin/env python3
"""
server_worker.py - High-Performance HTTP Worker for ToolBoxV2

Raw WSGI implementation without frameworks.
Features:
- Raw WSGI (no framework)
- Async request processing
- Signed cookie sessions
- ZeroMQ event integration
- ToolBoxV2 module routing
- SSE streaming support
- WebSocket message handling via ZMQ
- Auth endpoints (validateSession, IsValidSession, logout, api_user_data)
- Access Control (open_modules, open* functions, level system)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import parse_qs, unquote

from toolboxv2.utils.workers.event_manager import (
    ZMQEventManager,
    Event,
    EventType,
)
from toolboxv2.utils.system.types import RequestData

logger = logging.getLogger(__name__)


# ============================================================================
# Access Control Constants
# ============================================================================

class AccessLevel:
    """User access levels."""
    ADMIN = -1
    NOT_LOGGED_IN = 0
    LOGGED_IN = 1
    TRUSTED = 2


# ============================================================================
# Request Parsing
# ============================================================================


@dataclass
class ParsedRequest:
    """Parsed HTTP request."""
    method: str
    path: str
    query_params: Dict[str, List[str]]
    headers: Dict[str, str]
    content_type: str
    content_length: int
    body: bytes
    form_data: Dict[str, Any] | None = None
    json_data: Any | None = None
    session: Any = None
    client_ip: str = "unknown"
    client_port: str = "unknown"

    @property
    def is_htmx(self) -> bool:
        return self.headers.get("hx-request", "").lower() == "true"

    def get_bearer_token(self) -> Optional[str]:
        """Extract Bearer token from Authorization header."""
        auth = self.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:]
        return None

    def get_session_token(self) -> Optional[str]:
        """Get session token from body or Authorization header."""
        # From body (JSON)
        if self.json_data and isinstance(self.json_data, dict):
            token = self.json_data.get("session_token") or self.json_data.get("Jwt_claim")
            if token:
                return token
        # From Authorization header
        return self.get_bearer_token()

    def get_clerk_user_id(self) -> Optional[str]:
        """Get Clerk user ID from body."""
        if self.json_data and isinstance(self.json_data, dict):
            return self.json_data.get("clerk_user_id") or self.json_data.get("Username")
        return None

    def to_toolbox_request(self) -> Dict[str, Any]:
        """Convert to ToolBoxV2 RequestData format."""
        return {
            "request": {
                "content_type": self.content_type,
                "headers": self.headers,
                "method": self.method,
                "path": self.path,
                "query_params": {k: v[0] if len(v) == 1 else v
                                 for k, v in self.query_params.items()},
                "form_data": self.form_data,
                "body": self.body.decode("utf-8", errors="replace") if self.body else None,
                "client_ip": self.client_ip,
            },
            "session": self.session.to_dict() if self.session else {
                "SiID": "", "level": "0", "spec": "", "user_name": "anonymous",
            },
            "session_id": self.session.session_id if self.session else "",
        }


def parse_request(environ: Dict) -> ParsedRequest:
    """Parse WSGI environ into structured request."""
    method = environ.get("REQUEST_METHOD", "GET")
    path = unquote(environ.get("PATH_INFO", "/"))
    query_string = environ.get("QUERY_STRING", "")
    query_params = parse_qs(query_string, keep_blank_values=True)

    headers = {}
    for key, value in environ.items():
        if key.startswith("HTTP_"):
            headers[key[5:].replace("_", "-").lower()] = value
        elif key in ("CONTENT_TYPE", "CONTENT_LENGTH"):
            headers[key.replace("_", "-").lower()] = value

    content_type = environ.get("CONTENT_TYPE", "")
    try:
        content_length = int(environ.get("CONTENT_LENGTH", 0))
    except (ValueError, TypeError):
        content_length = 0

    body = b""
    if content_length > 0:
        wsgi_input = environ.get("wsgi.input")
        if wsgi_input:
            body = wsgi_input.read(content_length)

    form_data = None
    json_data = None

    if body:
        if "application/x-www-form-urlencoded" in content_type:
            try:
                form_data = {k: v[0] if len(v) == 1 else v
                             for k, v in parse_qs(body.decode("utf-8")).items()}
            except Exception:
                pass
        elif "application/json" in content_type:
            try:
                json_data = json.loads(body.decode("utf-8"))
            except Exception:
                pass

    session = environ.get("tb.session")

    # Extract client IP (check X-Forwarded-For for proxy)
    client_ip = headers.get("x-forwarded-for", "").split(",")[0].strip()
    if not client_ip:
        client_ip = headers.get("x-real-ip", "")
    if not client_ip:
        remote_addr = environ.get("REMOTE_ADDR", "unknown")
        client_ip = remote_addr.split(":")[0] if ":" in remote_addr else remote_addr

    client_port = environ.get("REMOTE_PORT", "unknown")

    return ParsedRequest(
        method=method, path=path, query_params=query_params,
        headers=headers, content_type=content_type,
        content_length=content_length, body=body,
        form_data=form_data, json_data=json_data, session=session,
        client_ip=client_ip, client_port=str(client_port),
    )


# ============================================================================
# Response Helpers
# ============================================================================


def json_response(data: Any, status: int = 200, headers: Dict = None) -> Tuple:
    resp_headers = {"Content-Type": "application/json"}
    if headers:
        resp_headers.update(headers)
    body = json.dumps(data, separators=(",", ":"), default=str).encode()
    return (status, resp_headers, body)


def html_response(content: str, status: int = 200, headers: Dict = None) -> Tuple:
    resp_headers = {"Content-Type": "text/html; charset=utf-8"}
    if headers:
        resp_headers.update(headers)
    return (status, resp_headers, content.encode())


def error_response(message: str, status: int = 500, error_type: str = "InternalError") -> Tuple:
    return json_response({"error": error_type, "message": message}, status=status)


def redirect_response(url: str, status: int = 302) -> Tuple:
    return (status, {"Location": url, "Content-Type": "text/plain"}, b"")


def api_result_response(
    error: Optional[str] = None,
    origin: Optional[List[str]] = None,
    data: Any = None,
    data_info: Optional[str] = None,
    data_type: Optional[str] = None,
    exec_code: int = 0,
    help_text: str = "OK",
    status: int = 200,
) -> Tuple:
    """Create a ToolBoxV2-style API result response."""
    result = {
        "error": error,
        "origin": origin,
        "result": {
            "data_to": "API",
            "data_info": data_info,
            "data": data,
            "data_type": data_type,
        } if data is not None or data_info else None,
        "info": {
            "exec_code": exec_code,
            "help_text": help_text,
        } if exec_code != 0 or help_text != "OK" else None,
    }
    return json_response(result, status=status)


def format_sse_event(data: Any, event: str = None, event_id: str = None) -> str:
    lines = []
    if event:
        lines.append(f"event: {event}")
    if event_id:
        lines.append(f"id: {event_id}")
    data_str = json.dumps(data) if isinstance(data, dict) else str(data)
    for line in data_str.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Access Control
# ============================================================================


class AccessController:
    """
    Controls access to API endpoints based on:
    - open_modules: Modules that are publicly accessible
    - Function names: Functions starting with 'open' are public
    - User level: -1=Admin, 0=not logged in, 1=logged in, 2=trusted
    """

    def __init__(self, config):
        self.config = config
        self._open_modules: Set[str] = set()
        self._load_config()

    def _load_config(self):
        """Load open modules from config."""
        if hasattr(self.config, 'toolbox'):
            modules = getattr(self.config.toolbox, 'open_modules', [])
            self._open_modules = set(modules)
            logger.info(f"Open modules: {self._open_modules}")

    def is_public_endpoint(self, module_name: str, function_name: str) -> bool:
        """Check if endpoint is publicly accessible (no auth required)."""
        # Module in open_modules list
        if module_name in self._open_modules:
            return True

        # Function starts with 'open'
        if function_name and function_name.lower().startswith("open"):
            return True

        return False

    def check_access(
        self,
        module_name: str,
        function_name: str,
        user_level: int,
        required_level: int = AccessLevel.LOGGED_IN,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user has access to endpoint.

        Returns:
            Tuple of (allowed: bool, error_message: Optional[str])
        """
        # Public endpoints
        if self.is_public_endpoint(module_name, function_name):
            return True, None

        # Not logged in
        if user_level == AccessLevel.NOT_LOGGED_IN:
            return False, "Authentication required"

        # Admin has access to everything
        if user_level == AccessLevel.ADMIN:
            return True, None

        # Check level requirement
        if user_level >= required_level:
            return True, None

        return False, f"Insufficient permissions (level {user_level}, required {required_level})"

    def get_user_level(self, session) -> int:
        """Extract user level from session."""
        if not session:
            return AccessLevel.NOT_LOGGED_IN

        # Try to get level from session
        level = None
        if hasattr(session, 'level'):
            level = session.level
        elif hasattr(session, 'live_data') and isinstance(session.live_data, dict):
            level = session.live_data.get('level')
        elif hasattr(session, 'to_dict'):
            data = session.to_dict()
            level = data.get('level')

        if level is None:
            return AccessLevel.NOT_LOGGED_IN

        try:
            return int(level)
        except (ValueError, TypeError):
            return AccessLevel.NOT_LOGGED_IN


# ============================================================================
# Auth Handlers
# ============================================================================


class AuthHandler:
    """
    Handles authentication endpoints equivalent to Rust handlers:
    - /validateSession (POST)
    - /IsValidSession (GET)
    - /web/logoutS (POST)
    - /api_user_data (GET)
    """

    def __init__(self, session_manager, app, config):
        self.session_manager = session_manager
        self.app = app
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.AuthHandler")

    async def validate_session(self, request: ParsedRequest) -> Tuple:
        """
        Validate session with Clerk token.
        Equivalent to validate_session_handler in Rust.
        """
        client_ip = request.client_ip
        token = request.get_session_token()
        clerk_user_id = request.get_clerk_user_id()

        self._logger.info(
            f"[Session] Validation request - IP: {client_ip}, "
            f"User: {clerk_user_id}, Has Token: {token is not None}"
        )

        # Token must be present
        if not token:
            self._logger.warning("[Session] No token provided")
            if request.session:
                request.session.invalidate()
            return api_result_response(
                error="No authentication token provided",
                status=401,
            )

        # Get or create session
        session = request.session
        session_id = session.session_id if session else None

        if not session_id:
            self._logger.info("[Session] Creating new session for validation")
            session_id = self.session_manager.create_session(
                client_ip=client_ip,
                token=token,
                clerk_user_id=clerk_user_id,
            )
            session = self.session_manager.get_session(session_id)

        # Verify session with Clerk
        self._logger.info(f"[Session] Verifying session {session_id} with Clerk")
        valid, user_data = await self._verify_with_clerk(token)

        if not valid:
            self._logger.warning(f"[Session] Validation FAILED for session {session_id}")
            self.session_manager.delete_session(session_id)
            return api_result_response(
                error="Invalid or expired session",
                status=401,
            )

        self._logger.info(f"[Session] ✓ Validation SUCCESS for session {session_id}")

        # Update session with user data
        if user_data:
            session.user_id = user_data.get("user_id", clerk_user_id)
            session.clerk_user_id = clerk_user_id
            session.level = user_data.get("level", AccessLevel.LOGGED_IN)
            session.user_name = user_data.get("user_name", "")
            session.validated = True
            session.anonymous = False
            self.session_manager.update_session(session)

        # Return success response
        return api_result_response(
            error="none",
            data={
                "authenticated": True,
                "session_id": session_id,
                "clerk_user_id": clerk_user_id,
                "user_name": session.user_name if session else "",
                "level": session.level if session else AccessLevel.LOGGED_IN,
            },
            data_info="Valid Session",
            exec_code=0,
            help_text="Valid Session",
            status=200,
        )

    async def is_valid_session(self, request: ParsedRequest) -> Tuple:
        """
        Check if current session is valid.
        Equivalent to is_valid_session_handler in Rust.
        """
        session = request.session

        if session and session.validated and not session.anonymous:
            return api_result_response(
                error="none",
                data_info="Valid Session",
                exec_code=0,
                help_text="Valid Session",
                status=200,
            )
        else:
            return api_result_response(
                error="Invalid Auth data.",
                status=401,
            )

    async def logout(self, request: ParsedRequest) -> Tuple:
        """
        Logout user and invalidate session.
        Equivalent to logout_handler in Rust.
        """
        session = request.session

        if not session or not session.validated:
            return api_result_response(
                error="Invalid Auth data.",
                status=403,
            )

        session_id = session.session_id

        # Call Clerk sign out if available
        try:
            await self._clerk_sign_out(session_id)
        except Exception as e:
            self._logger.debug(f"Clerk sign out failed: {e}")

        # Delete session
        self.session_manager.delete_session(session_id)

        # Redirect to logout page
        return redirect_response("/web/logout", status=302)

    async def get_user_data(self, request: ParsedRequest) -> Tuple:
        """
        Get user data from Clerk.
        Equivalent to get_user_data_handler in Rust.
        """
        session = request.session

        if not session or not session.validated:
            return api_result_response(
                error="Unauthorized: Session invalid.",
                status=401,
            )

        # Get clerk_user_id
        clerk_user_id = None
        if hasattr(session, 'clerk_user_id'):
            clerk_user_id = session.clerk_user_id
        elif hasattr(session, 'live_data') and isinstance(session.live_data, dict):
            clerk_user_id = session.live_data.get('clerk_user_id')

        if not clerk_user_id:
            return api_result_response(
                error="No Clerk user ID found in session.",
                status=400,
            )

        # Get user data from Clerk
        user_data = await self._get_clerk_user_data(clerk_user_id)

        if user_data:
            return api_result_response(
                error="none",
                data=user_data,
                data_info="User data retrieved",
                data_type="json",
                exec_code=0,
                help_text="Success",
                status=200,
            )
        else:
            return api_result_response(
                error="User data not found.",
                status=404,
            )

    async def _verify_with_clerk(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verify session token with CloudM.AuthClerk."""
        try:
            auth_module = getattr(self.config.toolbox, 'auth_module', 'CloudM.AuthClerk')
            verify_func = getattr(self.config.toolbox, 'verify_session_func', 'verify_session')

            result = await self.app.a_run_any(
                (auth_module, verify_func),
                session_token=token,
                get_results=True,
            )

            if hasattr(result, 'is_error') and result.is_error():
                self._logger.debug(f"Clerk verification returned error: {result}")
                return False, None

            data = result.get() if hasattr(result, 'get') else result

            # Check for 'authenticated' key (returned by verify_session)
            # Also support legacy 'valid' key for backwards compatibility
            if not data:
                return False, None

            is_authenticated = data.get('authenticated', data.get('valid', False))
            if not is_authenticated:
                self._logger.debug(f"Clerk verification: not authenticated, data={data}")
                return False, None

            return True, data

        except Exception as e:
            self._logger.error(f"Clerk verification error: {e}")
            return False, None

    async def _clerk_sign_out(self, session_id: str):
        """Call Clerk sign out."""
        try:
            auth_module = getattr(self.config.toolbox, 'auth_module', 'CloudM.AuthClerk')

            await self.app.a_run_any(
                (auth_module, "on_sign_out"),
                session_id=session_id,
                get_results=False,
            )
        except Exception as e:
            self._logger.debug(f"Clerk sign out error: {e}")

    async def _get_clerk_user_data(self, clerk_user_id: str) -> Optional[Dict]:
        """Get user data from Clerk."""
        try:
            auth_module = getattr(self.config.toolbox, 'auth_module', 'CloudM.AuthClerk')

            result = await self.app.a_run_any(
                (auth_module, "get_user_data"),
                clerk_user_id=clerk_user_id,
                get_results=True,
            )

            if hasattr(result, 'is_error') and result.is_error():
                return None

            return result.get() if hasattr(result, 'get') else result

        except Exception as e:
            self._logger.error(f"Get user data error: {e}")
            return None


# ============================================================================
# ToolBoxV2 Handler (with Access Control)
# ============================================================================


class ToolBoxHandler:
    """Handler for ToolBoxV2 module calls with access control."""

    def __init__(self, app, config, access_controller: AccessController, api_prefix: str = "/api"):
        self.app = app
        self.config = config
        self.access_controller = access_controller
        self.api_prefix = api_prefix

    def is_api_request(self, path: str) -> bool:
        return path.startswith(self.api_prefix)

    def parse_api_path(self, path: str) -> Tuple[str | None, str | None]:
        """Parse /api/Module/function into (module, function)."""
        stripped = path[len(self.api_prefix):].strip("/")
        if not stripped:
            return None, None
        parts = stripped.split("/", 1)
        if len(parts) == 1:
            return parts[0], None
        return parts[0], parts[1]

    async def handle_api_call(
        self,
        request: ParsedRequest,
    ) -> Tuple[int, Dict[str, str], bytes]:
        """Handle API call to ToolBoxV2 module with access control."""
        module_name, function_name = self.parse_api_path(request.path)

        if not module_name:
            return error_response("Missing module name", 400, "BadRequest")

        if not function_name:
            return error_response("Missing function name", 400, "BadRequest")

        # Access control check
        user_level = self.access_controller.get_user_level(request.session)
        allowed, error_msg = self.access_controller.check_access(
            module_name, function_name, user_level
        )

        if not allowed:
            logger.warning(
                f"Access denied: {module_name}.{function_name} "
                f"(user_level={user_level}): {error_msg}"
            )
            return error_response(error_msg, 401 if user_level == 0 else 403, "Forbidden")

        # Build kwargs from request
        kwargs = {}

        if request.query_params:
            for k, v in request.query_params.items():
                kwargs[k] = v[0] if len(v) == 1 else v

        if request.form_data:
            kwargs.update(request.form_data)

        if request.json_data and isinstance(request.json_data, dict):
            kwargs.update(request.json_data)

        # Add request context - convert to RequestData object for modules
        request_dict = request.to_toolbox_request()
        kwargs["request"] = RequestData.from_dict(request_dict)

        try:
            result = await self.app.a_run_any(
                (module_name, function_name),
                get_results=True,
                **kwargs
            )
            # result.print(show=True)
            data = self._process_result(result, request)
            return data
        except Exception as e:
            logger.error(f"API call error: {e}")
            traceback.print_exc()
            return error_response(str(e), 500)

    def _process_result(self, result, request: ParsedRequest) -> Tuple:
        """Process ToolBoxV2 Result into HTTP response."""
        if result is None:
            return json_response({"status": "ok"})

        # Check if Result object
        if hasattr(result, "is_error") and hasattr(result, "get"):
            if result.is_error():
                status = getattr(result.info, "exec_code", 500)
                if status <= 0:
                    status = 500
                return error_response(
                    getattr(result.info, "help_text", "Error"),
                    status
                )

            # Check result type
            data_type = getattr(result.result, "data_type", "")
            data = result.get()

            if data_type == "html":
                return html_response(data, status=getattr(result.info, "exec_code", 200) or 200)

            if data_type == "special_html":
                html_data = data.get("html", "")
                extra_headers = data.get("headers", {})
                return html_response(html_data, headers=extra_headers)

            if data_type == "redirect":
                return redirect_response(data, getattr(result.info, "exec_code", 302))

            if data_type == "file":
                import base64
                file_data = base64.b64decode(data) if isinstance(data, str) else data
                info = getattr(result.result, "data_info", "")
                filename = info.replace("File download: ", "") if info else "download"
                return (
                    200,
                    {
                        "Content-Type": "application/octet-stream",
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    },
                    file_data
                )

            # Default JSON response
            return json_response(result.as_dict())

        # Plain data
        if isinstance(result, (dict, list)):
            return json_response(result)

        if isinstance(result, str):
            if result.strip().startswith("<"):
                return html_response(result)
            return json_response({"result": result})

        return json_response({"result": str(result)})


# ============================================================================
# WebSocket Message Handler
# ============================================================================


class WebSocketMessageHandler:
    """
    Handles WebSocket messages forwarded from WS workers via ZMQ.
    Routes messages to registered websocket_handler functions in ToolBoxV2.
    """

    def __init__(self, app, event_manager: ZMQEventManager, access_controller: AccessController):
        self.app = app
        self.event_manager = event_manager
        self.access_controller = access_controller
        self._logger = logging.getLogger(f"{__name__}.WSHandler")

    async def handle_ws_connect(self, event: Event):
        """Handle WebSocket connect event."""
        conn_id = event.payload.get("conn_id")
        path = event.payload.get("path", "/ws")

        self._logger.info(f"WS Connect: {conn_id} on {path}")
        self._logger.info(f"Available WS handlers: {list(self.app.websocket_handlers.keys())}")

        handler_id = self._get_handler_from_path(path)
        if not handler_id:
            self._logger.warning(f"No handler found for path: {path}")
            return

        self._logger.info(f"Found handler: {handler_id}")
        handler = self.app.websocket_handlers.get(handler_id, {}).get("on_connect")
        if handler:
            try:
                session = {"connection_id": conn_id, "path": path}
                result = await self._call_handler(handler, session=session, conn_id=conn_id)

                if isinstance(result, dict) and not result.get("accept", True):
                    self._logger.info(f"Connection {conn_id} rejected by handler")

            except Exception as e:
                self._logger.error(f"on_connect handler error: {e}", exc_info=True)

    async def handle_ws_message(self, event: Event):
        """Handle WebSocket message event with access control."""
        conn_id = event.payload.get("conn_id")
        user_id = event.payload.get("user_id", "")
        session_id = event.payload.get("session_id", "")
        data = event.payload.get("data", "")
        path = event.payload.get("path", "/ws")

        self._logger.info(f"WS Message from {conn_id} on path {path}: {data[:200] if isinstance(data, str) else str(data)[:200]}...")

        # Parse JSON message
        try:
            payload = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError:
            payload = {"raw": data}

        # Determine handler
        handler_id = self._get_handler_from_path(path)
        self._logger.info(f"Handler from path: {handler_id}")
        if not handler_id:
            handler_id = self._get_handler_from_message(payload)
            self._logger.info(f"Handler from message: {handler_id}")

        if not handler_id:
            self._logger.warning(f"No handler found for path {path}, available handlers: {list(self.app.websocket_handlers.keys())}")
            return

        # Access control for WS handlers
        # Extract module/function from handler_id (format: Module/handler)
        parts = handler_id.split("/", 1)
        if len(parts) == 2:
            module_name, function_name = parts
            # Get user level from event payload
            user_level = int(event.payload.get("level", AccessLevel.NOT_LOGGED_IN))
            authenticated = event.payload.get("authenticated", False)

            self._logger.info(f"WS Access check: handler={handler_id}, user_level={user_level}, authenticated={authenticated}")
            self._logger.info(f"WS Access check: open_modules={self.access_controller._open_modules}")

            allowed, error_msg = self.access_controller.check_access(
                module_name, function_name, user_level
            )

            self._logger.info(f"WS Access result: allowed={allowed}, error={error_msg}")

            if not allowed:
                self._logger.warning(f"WS access denied: {handler_id}: {error_msg}")
                try:
                    await self.app.ws_send(conn_id, {
                        "type": "error",
                        "message": error_msg,
                        "code": "ACCESS_DENIED",
                    })
                except Exception:
                    pass
                return

        handler = self.app.websocket_handlers.get(handler_id, {}).get("on_message")
        if handler:
            try:
                session = {
                    "connection_id": conn_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "path": path,
                }

                # Build RequestData object for WebSocket handlers
                # Extract additional session info from event payload
                user_level = int(event.payload.get("level", AccessLevel.NOT_LOGGED_IN))
                authenticated = event.payload.get("authenticated", False)
                clerk_user_id = event.payload.get("clerk_user_id", "")

                request_dict = {
                    "request": {
                        "content_type": "application/json",
                        "headers": {},
                        "method": "WEBSOCKET",
                        "path": path,
                        "query_params": {},
                        "form_data": None,
                        "body": None,
                    },
                    "session": {
                        "SiID": session_id,
                        "level": user_level,
                        "spec": "ws",
                        "user_name": user_id or "anonymous",
                        "user_id": user_id,
                        "session_id": session_id,
                        "clerk_user_id": clerk_user_id,
                        "validated": authenticated,
                        "anonymous": not authenticated,
                    },
                    "session_id": session_id,
                }
                request = RequestData.from_dict(request_dict)

                result = await self._call_handler(
                    handler,
                    payload=payload,
                    session=session,
                    conn_id=conn_id,
                    request=request,
                )

                if result and isinstance(result, dict):
                    await self.app.ws_send(conn_id, result)

            except Exception as e:
                self._logger.error(f"on_message handler error: {e}", exc_info=True)
                try:
                    await self.app.ws_send(conn_id, {
                        "type": "error",
                        "message": str(e),
                    })
                except Exception:
                    pass

    async def handle_ws_disconnect(self, event: Event):
        """Handle WebSocket disconnect event."""
        conn_id = event.payload.get("conn_id")
        user_id = event.payload.get("user_id", "")

        self._logger.debug(f"WS Disconnect: {conn_id}")

        for handler_id, handlers in self.app.websocket_handlers.items():
            handler = handlers.get("on_disconnect")
            if handler:
                try:
                    session = {"connection_id": conn_id, "user_id": user_id}
                    await self._call_handler(handler, session=session, conn_id=conn_id)
                except Exception as e:
                    self._logger.error(f"on_disconnect handler error: {e}", exc_info=True)

    def _get_handler_from_path(self, path: str) -> str | None:
        """Extract handler ID from WebSocket path.

        Supports paths like:
        - /ws/ModuleName/handler_name -> "ModuleName/handler_name"
        - /ws/handler_name -> searches for "*/{handler_name}" in registered handlers
        """
        path = path.strip("/")
        parts = path.split("/")

        if len(parts) >= 2 and parts[0] == "ws":
            if len(parts) >= 3:
                # Full path: /ws/ModuleName/handler_name
                handler_id = f"{parts[1]}/{parts[2]}"
                if handler_id in self.app.websocket_handlers:
                    return handler_id
                # Also try case-insensitive match
                for registered_id in self.app.websocket_handlers:
                    if registered_id.lower() == handler_id.lower():
                        return registered_id
            else:
                # Short path: /ws/handler_name - search for matching handler
                handler_name = parts[1]
                for handler_id in self.app.websocket_handlers:
                    if handler_id.endswith(f"/{handler_name}"):
                        return handler_id

        return None

    def _get_handler_from_message(self, payload: dict) -> str | None:
        """Try to find handler based on message content.

        Looks for 'handler' field in the payload that specifies which handler to use.
        """
        handler = payload.get("handler")
        if handler and handler in self.app.websocket_handlers:
            return handler

        return None

    async def _call_handler(self, handler: Callable, **kwargs) -> Any:
        """Call a handler function (sync or async)."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(**kwargs)
        else:
            return handler(**kwargs)


# ============================================================================
# HTTP Worker
# ============================================================================


class HTTPWorker:
    """HTTP Worker with raw WSGI application and auth endpoints."""

    # Auth endpoint paths
    AUTH_ENDPOINTS = {
        "/validateSession": "validate_session",
        "/IsValidSession": "is_valid_session",
        "/web/logoutS": "logout",
        "/api_user_data": "get_user_data",
    }

    def __init__(
        self,
        worker_id: str,
        config,
        app=None,
    ):
        self._server = None
        self.worker_id = worker_id
        self.config = config
        self._app = app
        self._toolbox_handler: ToolBoxHandler | None = None
        self._auth_handler: AuthHandler | None = None
        self._access_controller: AccessController | None = None
        self._ws_handler: WebSocketMessageHandler | None = None
        self._session_manager = None
        self._event_manager: ZMQEventManager | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._running = False
        self._event_loop = None
        self._event_loop_thread = None

        # Request metrics
        self._metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "requests_auth": 0,
            "requests_denied": 0,
            "ws_messages_handled": 0,
            "latency_sum": 0.0,
        }

    def _init_toolbox(self):
        """Initialize ToolBoxV2 app."""
        if self._app is not None:
            return

        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            from ..system.getting_and_closing_app  import get_app
            instance_id = f"{self.config.toolbox.instance_id}_{self.worker_id}"
            self._app = get_app(name=instance_id, from_="HTTPWorker")
            logger.info(f"ToolBoxV2 initialized: {instance_id}")
        except Exception as e:
            logger.error(f"ToolBoxV2 init failed: {e}")
            raise

    def _init_session_manager(self):
        """Initialize session manager."""
        from ..workers.session import SessionManager

        secret = self.config.session.cookie_secret
        if not secret:
            if self.config.environment == "production":
                raise ValueError("Cookie secret required in production!")
            secret = "dev_secret_" + "x" * 40

        self._session_manager = SessionManager(
            cookie_secret=secret,
            cookie_name=self.config.session.cookie_name,
            cookie_max_age=self.config.session.cookie_max_age,
            cookie_secure=self.config.session.cookie_secure,
            cookie_httponly=self.config.session.cookie_httponly,
            cookie_samesite=self.config.session.cookie_samesite,
            app=self._app,
            clerk_enabled=self.config.auth.clerk_enabled,
        )

    def _init_access_controller(self):
        """Initialize access controller."""
        self._access_controller = AccessController(self.config)

    def _init_auth_handler(self):
        """Initialize auth handler."""
        self._auth_handler = AuthHandler(
            self._session_manager,
            self._app,
            self.config,
        )

    async def _init_event_manager(self):
        """Initialize ZeroMQ event manager and WS bridge."""
        await self._app.load_all_mods_in_file()
        self._event_manager = ZMQEventManager(
            worker_id=self.worker_id,
            pub_endpoint=self.config.zmq.pub_endpoint,
            sub_endpoint=self.config.zmq.sub_endpoint,
            req_endpoint=self.config.zmq.req_endpoint,
            rep_endpoint=self.config.zmq.rep_endpoint,
            http_to_ws_endpoint=self.config.zmq.http_to_ws_endpoint,
            is_broker=False,
        )
        await self._event_manager.start()

        from toolboxv2.utils.workers.ws_bridge import install_ws_bridge
        install_ws_bridge(self._app, self._event_manager, self.worker_id)

        self._ws_handler = WebSocketMessageHandler(
            self._app, self._event_manager, self._access_controller
        )

        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register ZMQ event handlers."""

        @self._event_manager.on(EventType.CONFIG_RELOAD)
        async def handle_config_reload(event):
            logger.info("Config reload requested")
            self._access_controller._load_config()

        @self._event_manager.on(EventType.SHUTDOWN)
        async def handle_shutdown(event):
            logger.info("Shutdown requested")
            self._running = False

        @self._event_manager.on(EventType.WS_CONNECT)
        async def handle_ws_connect(event: Event):
            logger.info(f"[HTTP] Received WS_CONNECT event: conn_id={event.payload.get('conn_id')}, path={event.payload.get('path')}")
            if self._ws_handler:
                await self._ws_handler.handle_ws_connect(event)
            else:
                logger.warning("[HTTP] No WS handler configured!")

        @self._event_manager.on(EventType.WS_MESSAGE)
        async def handle_ws_message(event: Event):
            logger.info(f"[HTTP] Received WS_MESSAGE event: conn_id={event.payload.get('conn_id')}, data={str(event.payload.get('data', ''))[:100]}...")
            self._metrics["ws_messages_handled"] += 1
            if self._ws_handler:
                await self._ws_handler.handle_ws_message(event)
            else:
                logger.warning("[HTTP] No WS handler configured!")

        @self._event_manager.on(EventType.WS_DISCONNECT)
        async def handle_ws_disconnect(event: Event):
            logger.info(f"[HTTP] Received WS_DISCONNECT event: conn_id={event.payload.get('conn_id')}")
            if self._ws_handler:
                await self._ws_handler.handle_ws_disconnect(event)
            else:
                logger.warning("[HTTP] No WS handler configured!")

    def _is_auth_endpoint(self, path: str) -> bool:
        """Check if path is an auth endpoint."""
        return path in self.AUTH_ENDPOINTS

    async def _handle_auth_endpoint(self, request: ParsedRequest) -> Tuple:
        """Handle auth endpoint request."""
        handler_name = self.AUTH_ENDPOINTS.get(request.path)
        if not handler_name:
            return error_response("Unknown auth endpoint", 404, "NotFound")

        handler = getattr(self._auth_handler, handler_name, None)
        if not handler:
            return error_response("Handler not implemented", 501, "NotImplemented")

        self._metrics["requests_auth"] += 1
        return await handler(request)

    def _get_cors_headers(self, environ: Dict) -> Dict[str, str]:
        """Get CORS headers for the response."""
        origin = environ.get("HTTP_ORIGIN", "*")
        # Allow requests from Tauri and localhost
        allowed_origins = [
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
            "http://localhost",
            "https://localhost",
            "http://127.0.0.1",
            "https://127.0.0.1",
        ]
        # Also allow any localhost port
        if origin and (origin in allowed_origins or
                       origin.startswith("http://localhost:") or
                       origin.startswith("http://127.0.0.1:") or
                       origin.startswith("https://localhost:") or
                       origin.startswith("https://127.0.0.1:")):
            allow_origin = origin
        else:
            allow_origin = "*"

        return {
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin, X-Session-Token",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
        }

    def wsgi_app(self, environ: Dict, start_response: Callable) -> List[bytes]:
        """Raw WSGI application entry point."""
        start_time = time.time()
        self._metrics["requests_total"] += 1

        try:
            # Handle CORS preflight requests
            if environ.get("REQUEST_METHOD") == "OPTIONS":
                cors_headers = self._get_cors_headers(environ)
                status_line = "204 No Content"
                response_headers = [(k, v) for k, v in cors_headers.items()]
                start_response(status_line, response_headers)
                return [b""]

            # Add session to environ
            if self._session_manager:
                session = self._session_manager.get_session_from_request_sync(environ)
                environ["tb.session"] = session

            # Parse request
            request = parse_request(environ)

            # Route request
            if self._is_auth_endpoint(request.path):
                # Auth endpoints
                status, headers, body = self._run_async(
                    self._handle_auth_endpoint(request)
                )
            elif self._toolbox_handler and self._toolbox_handler.is_api_request(request.path):
                # API endpoints
                status, headers, body = self._run_async(
                    self._toolbox_handler.handle_api_call(request)
                )
            elif request.path == "/health":
                status, headers, body = self._handle_health()
            elif request.path == "/metrics":
                status, headers, body = self._handle_metrics()
            else:
                status, headers, body = error_response("Not Found", 404, "NotFound")

            # Update session cookie if needed
            if self._session_manager and request.session:
                cookie_header = self._session_manager.get_set_cookie_header(request.session)
                if cookie_header:
                    headers["Set-Cookie"] = cookie_header

            # Add CORS headers to all responses
            cors_headers = self._get_cors_headers(environ)
            headers.update(cors_headers)

            # Build response
            status_line = f"{status} {HTTPStatus(status).phrase}"
            response_headers = [(k, v) for k, v in headers.items()]

            start_response(status_line, response_headers)

            self._metrics["requests_success"] += 1
            self._metrics["latency_sum"] += time.time() - start_time

            if isinstance(body, bytes):
                return [body]
            elif isinstance(body, Generator):
                return body
            else:
                return [str(body).encode()]

        except Exception as e:
            logger.error(f"Request error: {e}")
            traceback.print_exc()
            self._metrics["requests_error"] += 1

            # Add CORS headers even to error responses
            cors_headers = self._get_cors_headers(environ)
            status_line = "500 Internal Server Error"
            response_headers = [("Content-Type", "application/json")] + [(k, v) for k, v in cors_headers.items()]
            start_response(status_line, response_headers)

            return [json.dumps({"error": "InternalError", "message": str(e)}).encode()]

    def _run_async(self, coro) -> Any:
        """Run async coroutine from sync context using the background event loop."""
        # Use the background event loop thread if available
        if self._event_loop and self._event_loop.is_running():
            # Schedule coroutine in the background event loop and wait for result
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            try:
                # Wait for result with timeout
                return future.result(timeout=self.config.http_worker.timeout or 30)
            except Exception as e:
                logger.error(f"Async run error (threadsafe): {e}")
                raise
        else:
            # Fallback: create new event loop for this thread
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            except Exception as e:
                try:
                    self._app.run_bg_task(coro)
                except Exception:
                    logger.error(f"Async run error (fallback): {e}")
                    raise

    def _handle_health(self) -> Tuple:
        """Health check endpoint."""
        return json_response({
            "status": "healthy",
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "timestamp": time.time(),
        })

    def _handle_metrics(self) -> Tuple:
        """Metrics endpoint."""
        avg_latency = 0
        if self._metrics["requests_total"] > 0:
            avg_latency = self._metrics["latency_sum"] / self._metrics["requests_total"]

        metrics = {
            "worker_id": self.worker_id,
            "requests_total": self._metrics["requests_total"],
            "requests_success": self._metrics["requests_success"],
            "requests_error": self._metrics["requests_error"],
            "requests_auth": self._metrics["requests_auth"],
            "requests_denied": self._metrics["requests_denied"],
            "ws_messages_handled": self._metrics["ws_messages_handled"],
            "avg_latency_ms": avg_latency * 1000,
        }

        if self._event_manager:
            metrics["zmq"] = self._event_manager.get_metrics()

        return json_response(metrics)

    def run(self, host: str = None, port: int = None, do_run=True):
        """Run the HTTP worker."""
        host = host or self.config.http_worker.host
        port = port or self.config.http_worker.port

        logger.info(f"Starting HTTP worker {self.worker_id} on {host}:{port}")

        # Initialize components
        self._init_toolbox()
        self._init_session_manager()
        self._init_access_controller()
        self._init_auth_handler()

        self._toolbox_handler = ToolBoxHandler(
            self._app,
            self.config,
            self._access_controller,
            self.config.toolbox.api_prefix,
        )

        # Initialize event manager in a background thread with its own event loop
        import threading
        loop_ready_event = threading.Event()

        def run_event_loop():
            """Run the event loop in a background thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._event_loop = loop

            try:
                # Initialize event manager
                loop.run_until_complete(self._init_event_manager())
                logger.info(f"[HTTP] Event manager initialized, starting event loop")

                # Signal that the loop is ready
                loop_ready_event.set()

                # Keep the event loop running to process events
                loop.run_forever()
            except Exception as e:
                logger.error(f"Event loop error: {e}", exc_info=True)
                loop_ready_event.set()  # Unblock main thread even on error
            finally:
                loop.close()
                logger.info("[HTTP] Event loop stopped")

        try:
            self._event_loop_thread = threading.Thread(target=run_event_loop, daemon=True, name="event-loop")
            self._event_loop_thread.start()

            # Wait for the event loop to be ready (with timeout)
            if not loop_ready_event.wait(timeout=10.0):
                logger.warning("[HTTP] Event loop initialization timed out, continuing anyway")

            logger.info(f"[HTTP] Event loop thread started: {self._event_loop_thread.is_alive()}, loop running: {self._event_loop and self._event_loop.is_running()}")
        except Exception as e:
            logger.error(f"Event manager init failed: {e}", exc_info=True)

        self._running = True
        self._server = None

        # Run WSGI server
        try:
            from waitress import create_server

            self._server = create_server(
                self.wsgi_app,
                host=host,
                port=port,
                threads=self.config.http_worker.max_concurrent,
                connection_limit=self.config.http_worker.backlog,
                channel_timeout=self.config.http_worker.timeout,
                ident="ToolBoxV2",
            )

            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, shutting down...")
                self._running = False
                if self._server:
                    self._server.close()

            # Only register signal handlers in main thread
            try:
                import threading
                if threading.current_thread() is threading.main_thread():
                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
                else:
                    logger.info("[HTTP] Running in non-main thread, skipping signal handlers")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"[HTTP] Could not register signal handlers: {e}")

            logger.info(f"Serving on http://{host}:{port}")
            self._server.run()

        except ImportError:
            from wsgiref.simple_server import make_server, WSGIServer
            import threading

            logger.warning("Using wsgiref (dev only), install waitress for production")

            class ShutdownableWSGIServer(WSGIServer):
                allow_reuse_address = True
                timeout = 0.5

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._shutdown_event = threading.Event()

                def serve_forever(self):
                    try:
                        while not self._shutdown_event.is_set():
                            self.handle_request()
                    except Exception:
                        pass

                def shutdown(self):
                    self._shutdown_event.set()

            self._server = make_server(
                host, port, self.wsgi_app, server_class=ShutdownableWSGIServer
            )

            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, shutting down...")
                self._running = False
                if self._server:
                    self._server.shutdown()

            # Only register signal handlers in main thread
            try:
                if threading.current_thread() is threading.main_thread():
                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
                else:
                    logger.info("[HTTP] Running in non-main thread, skipping signal handlers")
            except (ValueError, RuntimeError) as e:
                logger.warning(f"[HTTP] Could not register signal handlers: {e}")

            if do_run:
                logger.info(f"Serving on http://{host}:{port}")
                self._server.serve_forever()

        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            self._running = False
            if self._server:
                self._server.close()

        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        # Stop the event loop and event manager
        if self._event_loop and self._event_manager:
            try:
                # Schedule stop on the event loop
                async def stop_manager():
                    await self._event_manager.stop()

                if self._event_loop.is_running():
                    # Schedule the stop coroutine
                    asyncio.run_coroutine_threadsafe(stop_manager(), self._event_loop)
                    # Stop the event loop
                    self._event_loop.call_soon_threadsafe(self._event_loop.stop)

                    # Wait for the thread to finish
                    if self._event_loop_thread and self._event_loop_thread.is_alive():
                        self._event_loop_thread.join(timeout=2.0)
            except Exception as e:
                logger.warning(f"Error stopping event manager: {e}")

        if self._executor:
            self._executor.shutdown(wait=False)

        logger.info(f"HTTP worker {self.worker_id} stopped")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ToolBoxV2 HTTP Worker", prog="tb http_worker")
    parser.add_argument("-c", "--config", help="Config file path")
    parser.add_argument("-H", "--host", help="Host to bind")
    parser.add_argument("-p", "--port", type=int, help="Port to bind")
    parser.add_argument("-w", "--worker-id", help="Worker ID")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    from toolboxv2.utils.workers.config import load_config
    config = load_config(args.config)

    worker_id = args.worker_id or f"http_{os.getpid()}"

    worker = HTTPWorker(worker_id, config)
    worker.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
