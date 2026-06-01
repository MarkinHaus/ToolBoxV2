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
import mimetypes
import os
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import parse_qs, unquote
from dotenv import load_dotenv
load_dotenv()

import requests

# Multipart parsing - Standard library für file uploads
try:
    from multipart import parse_form_data, is_form_request, MultipartPart

    MULTIPART_AVAILABLE = True
except ImportError:
    MULTIPART_AVAILABLE = False
    # Warning wird später geloggt

from toolboxv2.utils.workers.event_manager import (
    ZMQEventManager,
    Event,
    EventType,
)
from toolboxv2.utils.system.types import RequestData

logger = logging.getLogger(__name__)

# Multipart warning loggen falls nicht verfügbar
if not MULTIPART_AVAILABLE:
    logger.warning("multipart library not installed - file uploads disabled. Install with: pip install multipart")


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
class UploadedFile:
    """Wrapper für hochgeladene Dateien."""
    filename: str
    content_type: str
    size: int
    temp_path: str  # Pfad zur temp Datei auf Disk
    field_name: str

    def read(self) -> bytes:
        """Liest gesamte Datei in Memory - Vorsicht bei großen Dateien!"""
        with open(self.temp_path, 'rb') as f:
            return f.read()

    def save_to(self, destination: str) -> str:
        """Verschiebt Datei zum Ziel. Gibt finalen Pfad zurück."""
        import shutil
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(self.temp_path, destination)
        return destination

    def copy_to(self, destination: str) -> str:
        """Kopiert Datei zum Ziel. Gibt finalen Pfad zurück."""
        import shutil
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.temp_path, destination)
        return destination

    def stream(self, chunk_size: int = 65536):
        """Generator zum Streaming der Datei."""
        with open(self.temp_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk


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
    files: Dict[str, UploadedFile] = field(default_factory=dict)  # NEU: Uploaded files
    session: Any = None
    client_ip: str = "unknown"
    client_port: str = "unknown"
    environ: Dict = field(default_factory=dict)  # NEU: Original WSGI environ für file_wrapper

    @property
    def is_htmx(self) -> bool:
        return self.headers.get("hx-request", "").lower() == "true"

    @property
    def has_files(self) -> bool:
        """Prüft ob Files hochgeladen wurden."""
        return bool(self.files)

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

    def get_user_id_from_body(self) -> Optional[str]:
        """Get user ID from body."""
        if self.json_data and isinstance(self.json_data, dict):
            return self.json_data.get("user_id") or self.json_data.get("cloudm_user_id") or self.json_data.get("Username")
        return None

    def to_toolbox_request(self) -> Dict[str, Any]:
        """Convert to ToolBoxV2 RequestData format."""
        # Files als serialisierbare Info (nicht die Objekte selbst)
        files_info = {}
        for name, uploaded_file in self.files.items():
            files_info[name] = {
                "filename": uploaded_file.filename,
                "content_type": uploaded_file.content_type,
                "size": uploaded_file.size,
                "temp_path": uploaded_file.temp_path,
            }

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
                "files": files_info,  # NEU: File metadata
            },
            "session": self.session.to_dict() if self.session else {
                "SiID": "", "level": "0", "spec": "", "user_name": "anonymous",
            },
            "session_id": self.session.session_id if self.session else "",
        }


def parse_request(environ: Dict, upload_temp_dir: str = None) -> ParsedRequest:
    """Parse WSGI environ into structured request.

    Args:
        environ: WSGI environment dict
        upload_temp_dir: Temp directory for file uploads (default: system temp)
    """
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
    form_data = None
    json_data = None
    files = {}

    # Multipart form-data (File Uploads) - MUSS VOR body read kommen!
    if MULTIPART_AVAILABLE and "multipart/form-data" in content_type:
        try:
            # Parse mit disk buffering für große Files
            # spool_limit: Files > 64KB werden auf Disk gespeichert
            # disk_limit: Max 1.5 GB pro Upload
            forms_multi, files_multi = parse_form_data(
                environ,
                charset='utf-8',
                disk_limit=int(1.5 * 1024 ** 3),  # 1.5 GB
                mem_limit=64 * 1024 ** 2,  # 64 MB total in memory
                memfile_limit=64 * 1024,  # 64 KB pro part bevor disk
            )

            # Forms in dict konvertieren
            form_data = {}
            for key in forms_multi:
                values = forms_multi.getall(key)
                form_data[key] = values[0] if len(values) == 1 else values

            # Files verarbeiten
            for key in files_multi:
                part = files_multi[key]
                if part.filename:  # Nur echte File-Uploads
                    # Temp file Pfad ermitteln
                    if hasattr(part, 'file') and hasattr(part.file, 'name'):
                        temp_path = part.file.name
                    else:
                        # Fallback: in temp dir speichern
                        import tempfile
                        temp_dir = upload_temp_dir or tempfile.gettempdir()
                        Path(temp_dir).mkdir(parents=True, exist_ok=True)
                        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{part.filename}")
                        part.save_as(temp_path)

                    files[key] = UploadedFile(
                        filename=part.filename,
                        content_type=part.content_type or 'application/octet-stream',
                        size=part.size,
                        temp_path=temp_path,
                        field_name=key,
                    )

        except Exception as e:
            logger.error(f"Multipart parsing error: {e}")
            # Fallback: body als raw lesen falls multipart failed
            wsgi_input = environ.get("wsgi.input")
            if wsgi_input and content_length > 0:
                body = wsgi_input.read(content_length)

    # Nicht-multipart: Body normal lesen
    elif content_length > 0:
        wsgi_input = environ.get("wsgi.input")
        if wsgi_input:
            body = wsgi_input.read(content_length)

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
        form_data=form_data, json_data=json_data, files=files,
        session=session, client_ip=client_ip, client_port=str(client_port),
        environ=environ,  # Für wsgi.file_wrapper
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

    def is_cm_auth(self, module_name: str, function_name: str) -> bool:
        if module_name != "CloudM.Auth":
            return False

        save_fuctions = [
            "passkey_login_start", "verify_magic_link",
            "check_magic_link_status", "verify_device_invite",
            "passkey_register_start", "passkey_register_finish",
            "passkey_login_start", "passkey_login_finish",
            "verify_session_token", "get_discord_auth_url",
            "get_google_auth_url", "login_discord", "login_google",
        ]
        if function_name in save_fuctions:
            return True

        return False

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

        # ClaudM Auth
        if self.is_cm_auth(module_name, function_name):
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
    Handles authentication endpoints:
    - /validateSession (POST) - JWT token validation
    - /IsValidSession (GET) - Session check
    - /web/logoutS (POST) - Logout with token blacklist
    - /api_user_data (GET) - User data retrieval
    - /auth/discord/url (GET) - Discord OAuth URL
    - /auth/discord/callback (GET) - Discord OAuth callback
    - /auth/google/url (GET) - Google OAuth URL
    - /auth/google/callback (GET) - Google OAuth callback
    - /auth/magic/verify (GET) - Magic link verification

    Provider-agnostic: uses config.toolbox.auth_module (default: CloudM.Auth)
    """

    def __init__(self, session_manager, app, config):
        self.session_manager = session_manager
        self.app = app
        self.config = config
        self.auth_module = getattr(config.toolbox, 'auth_module', 'CloudM.Auth')
        self.verify_func = getattr(config.toolbox, 'verify_session_func', 'validate_session')
        self._logger = logging.getLogger(f"{__name__}.AuthHandler")

    # ================================================================
    # Core Auth Endpoints
    # ================================================================

    async def validate_session(self, request: ParsedRequest) -> Tuple:
        """Validate JWT token via CloudM.Auth.validate_session."""
        client_ip = request.client_ip
        token = request.get_session_token()
        user_id = request.get_user_id_from_body()

        self._logger.info(
            f"[Session] Validation request - IP: {client_ip}, "
            f"User: {user_id}, Has Token: {token is not None}"
        )

        if not token:
            self._logger.warning("[Session] No token provided")
            if request.session:
                request.session.invalidate()
            return api_result_response(
                error="No authentication token provided",
                status=401,
            )

        session = request.session
        session_id = session.session_id if session else None

        if not session_id:
            self._logger.info("[Session] Creating new session for validation")
            session_id = self.session_manager.create_session(
                client_ip=client_ip,
                token=token,
                provider_user_id=user_id or "",
            )
            session = self.session_manager.get_session(session_id)

        valid, user_data = await self._verify_token(token)

        if not valid:
            self._logger.warning(f"[Session] Validation FAILED for session {session_id}")
            self.session_manager.delete_session(session_id)
            return api_result_response(
                error="Invalid or expired session",
                status=401,
            )

        self._logger.info(f"[Session] Validation SUCCESS for session {session_id}")

        if user_data:
            session.user_id = user_data.get("user_id", user_id or "")
            session.provider_user_id = user_data.get("user_id", "")
            session.level = user_data.get("level", AccessLevel.LOGGED_IN)
            session.user_name = user_data.get("user_name", "")
            session.validated = True
            session.anonymous = False
            self.session_manager.update_session(session)

        return api_result_response(
            error="none",
            data={
                "authenticated": True,
                "session_id": session_id,
                "user_id": session.user_id if session else "",
                "user_name": session.user_name if session else "",
                "level": session.level if session else AccessLevel.LOGGED_IN,
            },
            data_info="Valid Session",
            exec_code=0,
            help_text="Valid Session",
            status=200,
        )

    async def is_valid_session(self, request: ParsedRequest) -> Tuple:
        """Check if current session is valid."""
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
        """Logout: blacklist token + invalidate session."""
        session = request.session

        if not session or not session.validated:
            return api_result_response(
                error="Invalid Auth data.",
                status=403,
            )

        session_id = session.session_id

        # Blacklist token via CloudM.Auth.logout
        token = request.get_session_token()
        try:
            await self.app.a_run_any(
                (self.auth_module, "logout"),
                token=token,
                get_results=False,
            )
        except Exception as e:
            self._logger.debug(f"Auth logout call failed: {e}")

        self.session_manager.delete_session(session_id)
        return redirect_response("/web/logout", status=302)

    async def get_user_data(self, request: ParsedRequest) -> Tuple:
        """Get user data from CloudM.Auth."""
        session = request.session

        if not session or not session.validated:
            return api_result_response(
                error="Unauthorized: Session invalid.",
                status=401,
            )

        user_id = session.user_id or getattr(session, 'provider_user_id', None)
        if not user_id:
            if hasattr(session, 'live_data') and isinstance(session.live_data, dict):
                user_id = session.live_data.get('provider_user_id', '')

        if not user_id:
            return api_result_response(
                error="No user ID found in session.",
                status=400,
            )

        user_data = await self._get_user_data(user_id)

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

    # ================================================================
    # OAuth Endpoints
    # ================================================================

    async def get_discord_auth_url(self, request: ParsedRequest) -> Tuple:
        """GET /auth/discord/url - Returns Discord OAuth authorization URL."""
        redirect_after = None
        if request.query_params:
            redirect_after = request.query_params.get("redirect_after", [None])[0]

        result = await self.app.a_run_any(
            (self.auth_module, "get_discord_auth_url"),
            redirect_after=redirect_after,
            get_results=True,
        )

        if hasattr(result, 'is_error') and result.is_error():
            return error_response("Discord OAuth not configured", 500)

        data = result.get() if hasattr(result, 'get') else result
        # Return JSON with auth_url (NOT a 302 redirect) so fetch() can read it
        return json_response(data)

    async def discord_callback(self, request: ParsedRequest) -> Tuple:
        """GET /auth/discord/callback - Discord OAuth callback."""
        code = request.query_params.get("code", [None])[0] if request.query_params else None
        state = request.query_params.get("state", [None])[0] if request.query_params else None

        if not code:
            return error_response("Authorization code required", 400, "BadRequest")

        result = await self.app.a_run_any(
            (self.auth_module, "login_discord"),
            code=code,
            state=state,
            get_results=True,
        )

        return self._handle_oauth_result(result, request)

    async def get_google_auth_url(self, request: ParsedRequest) -> Tuple:
        """GET /auth/google/url - Returns Google OAuth authorization URL."""
        redirect_after = None
        if request.query_params:
            redirect_after = request.query_params.get("redirect_after", [None])[0]

        result = await self.app.a_run_any(
            (self.auth_module, "get_google_auth_url"),
            redirect_after=redirect_after,
            get_results=True,
        )

        if hasattr(result, 'is_error') and result.is_error():
            return error_response("Google OAuth not configured", 500)

        data = result.get() if hasattr(result, 'get') else result
        # Return JSON with auth_url (NOT a 302 redirect) so fetch() can read it
        return json_response(data)

    async def google_callback(self, request: ParsedRequest) -> Tuple:
        """GET /auth/google/callback - Google OAuth callback."""
        code = request.query_params.get("code", [None])[0] if request.query_params else None
        state = request.query_params.get("state", [None])[0] if request.query_params else None

        if not code:
            return error_response("Authorization code required", 400, "BadRequest")

        result = await self.app.a_run_any(
            (self.auth_module, "login_google"),
            code=code,
            state=state,
            get_results=True,
        )

        return self._handle_oauth_result(result, request)

    async def magic_link_verify(self, request: ParsedRequest) -> Tuple:
        """GET /auth/magic/verify?token=... - Verify magic link."""
        token = request.query_params.get("token", [None])[0] if request.query_params else None

        if not token:
            return error_response("Token required", 400, "BadRequest")

        result = await self.app.a_run_any(
            (self.auth_module, "verify_magic_link"),
            token=token,
            get_results=True,
        )

        return self._handle_oauth_result(result, request)

    # ================================================================
    # Internal Helpers
    # ================================================================

    def _handle_oauth_result(self, result, request: ParsedRequest) -> Tuple:
        """Process OAuth/magic link result: create session + return token bridge page.

        Returns an inline HTML page that stores auth tokens in localStorage
        and redirects to the appropriate app page. This works in both web
        and Tauri contexts since the HTML is served directly by the worker.
        """
        if hasattr(result, 'is_error') and result.is_error():
            error_msg = "Authentication failed"
            if hasattr(result, 'info') and hasattr(result.info, 'help_text'):
                error_msg = result.info.help_text
            error_html = self._build_token_bridge_html(error=error_msg)
            return html_response(error_html, status=200)

        data = result.get() if hasattr(result, 'get') else result
        if not data or not isinstance(data, dict):
            error_html = self._build_token_bridge_html(error="Authentication failed")
            return html_response(error_html, status=200)

        # Create server-side session
        user_id = data.get("user_id", "")
        username = data.get("username", "")
        level = data.get("level", AccessLevel.LOGGED_IN)

        session, cookie_header = self.session_manager.create_authenticated_session(
            user_id=user_id,
            user_name=username,
            level=level,
            provider_user_id=user_id,
        )

        access_token = data.get("access_token", "")
        refresh_token = data.get("refresh_token", "")
        email = data.get("email", "")
        redirect_after = data.get("redirect_after", "")

        bridge_html = self._build_token_bridge_html(
            token=access_token,
            refresh_token=refresh_token,
            user_id=user_id,
            username=username,
            email=email,
            redirect_after=redirect_after,
        )

        extra_headers = {}
        if cookie_header:
            extra_headers["Set-Cookie"] = cookie_header

        return html_response(bridge_html, status=200, headers=extra_headers)

    @staticmethod
    def _build_token_bridge_html(
        token: str = "",
        refresh_token: str = "",
        user_id: str = "",
        username: str = "",
        email: str = "",
        error: str = "",
        redirect_after: str = "",
    ) -> str:
        """Build an inline HTML page that stores auth tokens and redirects.

        The redirect_after parameter is the origin URL of the frontend that
        initiated the OAuth flow (e.g. 'http://tauri.localhost', 'http://localhost:80').
        Tokens are passed via URL params to the target origin so that
        TB.user._checkAuthCallback() or CustomAuth can pick them up.
        """
        import json as _json
        auth_data = _json.dumps({
            "token": token,
            "refresh_token": refresh_token,
            "user_id": user_id,
            "username": username,
            "email": email,
            "error": error,
            "redirect_after": redirect_after,
        })
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Authenticating...</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #fff;
       display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
.card {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 40px;
         text-align: center; max-width: 400px; border: 1px solid rgba(255,255,255,0.1); }}
.spinner {{ width: 32px; height: 32px; border: 3px solid rgba(255,255,255,0.1);
            border-top-color: #6366f1; border-radius: 50%; animation: spin 0.8s linear infinite;
            margin: 0 auto 16px; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.error {{ color: #fca5a5; }}
a {{ color: #a5b4fc; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head><body>
<div class="card" id="card">
  <div class="spinner" id="spinner"></div>
  <p id="msg">Authenticating...</p>
</div>
<script>
(function() {{
  var data = {auth_data};
  var origin = data.redirect_after || '';

  // Build token query string for cross-origin transfer
  var tokenParams = 'token=' + encodeURIComponent(data.token || '') +
    '&refresh_token=' + encodeURIComponent(data.refresh_token || '') +
    '&user_id=' + encodeURIComponent(data.user_id || '') +
    '&username=' + encodeURIComponent(data.username || '');

  if (data.error) {{
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('msg').className = 'error';
    document.getElementById('msg').textContent = 'Error: ' + data.error;
    var errorTarget = (origin || '') + '/web/assets/login.html?error=' + encodeURIComponent(data.error);
    setTimeout(function() {{ window.location.href = errorTarget; }}, 2000);
    return;
  }}

  // Store auth data in localStorage (works if same origin as app)
  try {{
    var userState = {{
      isAuthenticated: true,
      token: data.token,
      refreshToken: data.refresh_token,
      userId: data.user_id,
      username: data.username,
      email: data.email,
      userLevel: 2
    }};
    localStorage.setItem('tbjs_user_session', JSON.stringify(userState));
  }} catch(e) {{ /* cross-origin — tokens will be passed via URL params */ }}

  document.getElementById('msg').textContent = 'Login successful! Redirecting...';

  // Redirect to the frontend origin with token params.
  // TB.user._checkAuthCallback() will pick up the params on the target page.
  var target;
  if (origin && origin !== window.location.origin) {{
    // Cross-origin redirect (e.g. Tauri, or nginx on different port)
    target = origin + '/web/assets/login.html?' + tokenParams;
  }} else if (origin) {{
    // Same origin — use relative path
    target = '/web/assets/login.html?' + tokenParams;
  }} else {{
    // No redirect_after provided — best effort
    // If we're on the worker port directly, stay here and show success
    if (window.location.port === '8000' || window.location.port === '5000') {{
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('msg').innerHTML =
        'Login successful! Tokens saved.<br><br>' +
        '<a href="/web/assets/login.html?' + tokenParams + '">Continue to app</a>';
      return;
    }}
    target = '/web/assets/login.html?' + tokenParams;
  }}

  setTimeout(function() {{ window.location.href = target; }}, 500);
}})();
</script></body></html>"""

    async def _verify_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verify JWT token via auth module."""
        try:
            result = await self.app.a_run_any(
                (self.auth_module, self.verify_func),
                token=token,
                session_token=token,
                get_results=True,
            )

            if hasattr(result, 'is_error') and result.is_error():
                self._logger.debug(f"Token verification returned error: {result}")
                return False, None

            data = result.get() if hasattr(result, 'get') else result

            if not data:
                return False, None

            is_authenticated = data.get('authenticated', data.get('valid', False))
            if not is_authenticated:
                self._logger.debug(f"Token verification: not authenticated")
                return False, None

            return True, data

        except Exception as e:
            self._logger.error(f"Token verification error: {e}")
            return False, None

    async def _get_user_data(self, user_id: str) -> Optional[Dict]:
        """Get user data via auth module."""
        try:
            result = await self.app.a_run_any(
                (self.auth_module, "get_user_data"),
                user_id=user_id,
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

        # NEU: Files als kwargs übergeben (UploadedFile Objekte)
        if request.files:
            kwargs["files"] = request.files

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

            if data_type == "stream":
                # Data structure from Result.stream():
                # { "type": "stream", "generator": async_gen, "content_type": "...", "headers": {...} }
                stream_info = data

                headers = stream_info.get("headers", {}).copy()
                content_type = stream_info.get("content_type", "text/event-stream")

                headers["Content-Type"] = content_type
                # Ensure no buffering for SSE
                if content_type == "text/event-stream":
                    headers["Cache-Control"] = "no-cache"
                    headers["X-Accel-Buffering"] = "no"

                # Return tuple: (status, headers, async_generator)
                # The wsgi_app method will wrap the async_generator
                return (200, headers, stream_info.get("generator"))

            if data_type == "html":
                return html_response(data, status=getattr(result.info, "exec_code", 200) or 200)

            if data_type == "special_html":
                html_data = data.get("html", "")
                extra_headers = data.get("headers", {})
                return html_response(html_data, headers=extra_headers)

            if data_type == "redirect":
                return redirect_response(data, getattr(result.info, "exec_code", 302))

            # NEU: file_path - Streaming direkt aus Datei
            if data_type == "file_path":
                if not isinstance(data, str) or not os.path.isfile(data):
                    return error_response(f"File not found: {data}", 404, "NotFound")

                info = getattr(result.result, "data_info", "")
                # Parse filename aus data_info (Format: "filename=xyz.pdf")
                filename = os.path.basename(data)
                if info and "filename=" in info:
                    filename = info.split("filename=", 1)[1].strip()

                # Content-Type ermitteln
                content_type, _ = mimetypes.guess_type(data)
                content_type = content_type or "application/octet-stream"

                file_size = os.path.getsize(data)

                headers = {
                    "Content-Type": content_type,
                    "Content-Length": str(file_size),
                    "Content-Disposition": f'attachment; filename="{filename}"',
                }

                # File streaming - nutze wsgi.file_wrapper wenn verfügbar
                file_obj = open(data, 'rb')

                # Markiere als streaming file für wsgi_app
                return (200, headers, ("__file_stream__", file_obj, request.environ))

            if data_type == "file":
                import base64
                # Legacy: base64 encoded file data
                file_data = base64.b64decode(data) if isinstance(data, str) else data
                info = getattr(result.result, "data_info", "")
                filename = info.replace("File download: ", "") if info else "download"

                # Content-Type ermitteln
                content_type, _ = mimetypes.guess_type(filename)
                content_type = content_type or "application/octet-stream"

                return (
                    200,
                    {
                        "Content-Type": content_type,
                        "Content-Length": str(len(file_data)),
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    },
                    file_data
                )

            # Binary data response (Result.binary())
            if data_type == "binary":
                # data ist dict: {"data": bytes, "content_type": str, "filename": str|None}
                if isinstance(data, dict):
                    binary_data = data.get("data", b"")
                    content_type = data.get("content_type", "application/octet-stream")
                    filename = data.get("filename")
                else:
                    # Fallback: data direkt als bytes
                    binary_data = data if isinstance(data, bytes) else str(data).encode()
                    content_type = "application/octet-stream"
                    filename = None

                headers = {
                    "Content-Type": content_type,
                    "Content-Length": str(len(binary_data)),
                }

                if filename:
                    headers["Content-Disposition"] = f'attachment; filename="{filename}"'

                return (200, headers, binary_data)

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

        # Auth gate (Decision A): if the WS route requires auth and the connection
        # is not authenticated, do NOT call on_connect. Force-close the connection
        # via the WS worker instead (hard close, no protocol-level 401).
        handler_entry = self.app.websocket_handlers.get(handler_id, {})
        if handler_entry.get("auth") and not event.payload.get("authenticated", False):
            self._logger.warning(f"WS connect denied (unauthenticated): {handler_id} conn={conn_id}")
            try:
                from toolboxv2.utils.workers.event_manager import create_ws_close_event
                await self.event_manager.send_to_ws(
                    create_ws_close_event(
                        source="fasttb_ws_gate",
                        conn_id=conn_id,
                        code=1008,
                        reason="Unauthorized",
                    )
                )
            except Exception as e:
                self._logger.error(f"WS close emit failed for {conn_id}: {e}", exc_info=True)
            return

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
        path = event.payload.get("path", "ws")

        if path.startswith("/"):
            path = path[1:]

        self._logger.info(
            f"WS Message from {conn_id} on path {path}: {data[:200] if isinstance(data, str) else str(data)[:200]}...")

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
            self._logger.warning(
                f"No handler found for path {path}, available handlers: {list(self.app.websocket_handlers.keys())}")
            return

        # Access control for WS handlers
        # Extract module/function from handler_id (format: Module/handler)
        parts = handler_id.split("/", 1)
        if len(parts) == 2:
            module_name, function_name = parts
            # Get user level from event payload
            user_level = int(event.payload.get("level", AccessLevel.NOT_LOGGED_IN))
            authenticated = event.payload.get("authenticated", False)

            self._logger.info(
                f"WS Access check: handler={handler_id}, user_level={user_level}, authenticated={authenticated}")
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
                provider_user_id = event.payload.get("user_id", event.payload.get("cloudm_user_id", ""))

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
                        "provider_user_id": provider_user_id,
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
        Also handles special HUD action messages.
        """
        handler = payload.get("handler")
        if handler and handler in self.app.websocket_handlers:
            return handler

        # Check for HUD action messages - route to internal handler
        msg_type = payload.get("type")
        if msg_type in ("widget_action", "get_widget", "get_widgets", "get_status"):
            return "__hud_internal__"

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

class AsyncGenToSyncIter:
    """
    Adapter that converts an AsyncGenerator into a synchronous Iterator
    compatible with WSGI, running the async tasks on a specific event loop.
    """

    def __init__(self, async_gen, loop):
        self.async_gen = async_gen
        self.loop = loop
        self._iterator = self.async_gen.__aiter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            # Schedule the fetching of the next item on the background event loop
            future = asyncio.run_coroutine_threadsafe(
                self._iterator.__anext__(),
                self.loop
            )
            # Block specifically for this chunk with a reasonable timeout (optional)
            # Note: SSE connections are long-lived, so standard timeouts apply to "time between events"
            data = future.result()

            # WSGI expects bytes
            if isinstance(data, str):
                return data.encode('utf-8')
            return data

        except StopAsyncIteration:
            raise StopIteration
        except Exception as e:
            # Log error if needed, then stop
            logger.error(f"Stream error: {e}")
            raise StopIteration


def get_location(ip_address):
    response = requests.get(f"https://ipapi.co/{ip_address}/json/").json()
    return response


class HTTPWorker:
    """HTTP Worker with raw WSGI application and auth endpoints."""

    # Auth endpoint paths
    AUTH_ENDPOINTS = {
        "/validateSession": "validate_session",
        "/IsValidSession": "is_valid_session",
        "/web/logoutS": "logout",
        "/api_user_data": "get_user_data",
        # OAuth Routes (Custom Auth)
        "/auth/discord/url": "get_discord_auth_url",
        "/auth/discord/callback": "discord_callback",
        "/auth/google/url": "get_google_auth_url",
        "/auth/google/callback": "google_callback",
        # Magic Link
        "/auth/magic/verify": "magic_link_verify",
    }

    # Client-Log level mapping (browser string → Python level)
    _CLIENT_LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "AUDIT": logging.INFO,  # audit entries go through INFO
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
            from ..system.getting_and_closing_app import get_app
            instance_id = f"{self.config.toolbox.instance_id}_{self.worker_id}"
            self._app = get_app(name=instance_id, from_="HTTPWorker")
            self._audit_logger = self._app.audit_logger
            logger.info(f"ToolBoxV2 initialized: {instance_id}")
        except Exception as e:
            logger.error(f"ToolBoxV2 init failed: {e}")
            raise

    def _init_session_manager(self):
        """Initialize session manager."""
        from ..workers.session import SessionManager

        secret = self.config.session.cookie_secret or os.getenv("TOKEN_SECRET", os.getenv("TB_COOKIE_SECRET"))
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
            cluster_secret=getattr(self.config.zmq, "cluster_secret", ""),
            heartbeat_period_s=getattr(self.config.zmq, "heartbeat_period_s", 1.0),
            heartbeat_timeout_s=getattr(self.config.zmq, "heartbeat_timeout_s", 3.0),
            takeover_jitter_max_s=getattr(self.config.zmq, "takeover_jitter_max_s", 0.5),
            status_period_s=getattr(self.config.zmq, "status_period_s", 5.0),
            topology_broadcast_period_s=getattr(
                self.config.zmq, "topology_broadcast_period_s", 2.0
            ),
        )
        # Provide HTTP-side status info before start() so first WORKER_STATUS is complete
        self._event_manager.set_status_providers(
            worker_type="HTTP",
            route_provider=self._list_http_routes,
        )
        await self._event_manager.start()

        from toolboxv2.utils.workers.ws_bridge import install_ws_bridge
        ws_bridge = install_ws_bridge(self._app, self._event_manager, self.worker_id)

        # NotificationSystem mit WebSocket Bridge verbinden
        from toolboxv2.utils.extras.notification import setup_web_notifications
        setup_web_notifications(ws_bridge)
        logger.info("[HTTP] NotificationSystem linked to WebSocket Bridge")

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
            logger.info(
                f"[HTTP] Received WS_CONNECT event: conn_id={event.payload.get('conn_id')}, path={event.payload.get('path')}")
            if self._ws_handler:
                await self._ws_handler.handle_ws_connect(event)
            else:
                logger.warning("[HTTP] No WS handler configured!")

        @self._event_manager.on(EventType.WS_MESSAGE)
        async def handle_ws_message(event: Event):
            logger.info(
                f"[HTTP] Received WS_MESSAGE event: conn_id={event.payload.get('conn_id')}, data={str(event.payload.get('data', ''))[:100]}...")
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

                # Bearer token fallback for cross-origin requests (e.g. Tauri).
                # Cookie is set on worker origin (localhost:5000) but requests come
                # from tauri.localhost — cookie not sent. Fall back to JWT in header.
                if (not session or session.anonymous) and environ.get("HTTP_AUTHORIZATION", "").startswith(
                    "Bearer "):
                    bearer_token = environ["HTTP_AUTHORIZATION"][7:]
                    try:
                        valid, jwt_session = self._session_manager.verify_session_token(bearer_token)
                        if valid and jwt_session:
                            session = jwt_session
                            session.mark_dirty()
                    except Exception as e:
                        logger.debug(f"Bearer fallback failed: {e}")

                environ["tb.session"] = session

            # Parse request - mit upload temp directory aus app config
            upload_temp_dir = None
            if self._app and hasattr(self._app, 'data_dir'):
                upload_temp_dir = os.path.join(self._app.data_dir, 'uploads', 'temp')
            request = parse_request(environ, upload_temp_dir=upload_temp_dir)

            # Route request
            if self._is_auth_endpoint(request.path):
                # Auth endpoints
                status, headers, body = self._run_async(
                    self._handle_auth_endpoint(request)
                )
            elif request.path == "/health":
                status, headers, body = self._handle_health()
            elif request.path == "/metrics":
                status, headers, body = self._handle_metrics()
            elif request.path == "/live" or request.path == "/live/":
                status, headers, body = self._handle_live(request)
            elif request.path == "/live/snapshot":
                status, headers, body = self._handle_live_snapshot(request)
            elif request.path.startswith("/live/mgr/"):
                status, headers, body = self._handle_live_mgr(request)
            elif request.path == "/api/ip":
                status, headers, body = self._handle_ip_request(request)
            elif request.path == "/api/ping":
                status, headers, body = self._handle_ping_request()
            elif request.path == "/api/geo":
                status, headers, body = self._handle_geo_request(request)
            elif request.path == "/api/client-logs":
                status, headers, body = self._handle_client_logs(request)
            elif self._toolbox_handler and self._toolbox_handler.is_api_request(request.path):
                # API endpoints
                status, headers, body = self._run_async(
                    self._toolbox_handler.handle_api_call(request)
                )
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
            import inspect

            if isinstance(body, bytes):
                return [body]

            # NEU: File streaming mit wsgi.file_wrapper
            elif isinstance(body, tuple) and len(body) == 3 and body[0] == "__file_stream__":
                _, file_obj, req_environ = body
                # Nutze wsgi.file_wrapper für optimiertes sendfile() wenn verfügbar
                if 'wsgi.file_wrapper' in req_environ:
                    return req_environ['wsgi.file_wrapper'](file_obj, 65536)
                else:
                    # Fallback: Generator-basiertes Streaming
                    return self._file_iterator(file_obj)

            elif inspect.isasyncgen(body):
                if not self._event_loop:
                    # Fallback if loop is missing (shouldn't happen in worker)
                    logger.error("Cannot stream: No event loop available")
                    return [b"Error: Internal Event Loop Missing"]

                return AsyncGenToSyncIter(body, self._event_loop)
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

    def _file_iterator(self, file_obj, chunk_size: int = 65536):
        """Generator für File-Streaming als WSGI Fallback.

        Args:
            file_obj: Geöffnetes File-Object
            chunk_size: Größe der Chunks (default 64KB)

        Yields:
            bytes chunks
        """
        try:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            file_obj.close()

    def _list_http_routes(self) -> List[str]:
        """Return the list of routes this HTTP worker handles (for WORKER_STATUS payload)."""
        base = [
            "/health", "/metrics", "/live", "/live/snapshot",
            "/api/ip", "/api/ping", "/api/geo", "/api/client-logs",
        ]
        auth_routes = [
            "/api/login", "/api/logout", "/api/register",
            "/auth/", "/validateSession", "/IsValidSession",
            "/web/logoutS", "/api_user_data",
        ]
        api_prefix = getattr(self.config.toolbox, "api_prefix", "/api")
        return base + auth_routes + [f"{api_prefix}/*"]

    def _live_dashboard_key(self) -> str:
        """Resolve /live auth key: env > config.manager.live_dashboard_key."""
        env_key = os.environ.get("LIVE_DASHBOARD_KEY", "")
        if env_key:
            return env_key
        return getattr(self.config.manager, "live_dashboard_key", "") or ""

    def _is_live_authorized(self, request) -> bool:
        configured = self._live_dashboard_key()
        if not configured:
            return False  # Route disabled when no key configured
        provided = None
        if getattr(request, "query_params", None):
            provided = request.query_params.get("key", [None])[0]
        if not provided and getattr(request, "headers", None):
            provided = request.headers.get("X-Live-Key")
        return provided == configured

    def _handle_live(self, request) -> Tuple:
        """Serve the /live dashboard HTML (TBJS Glass v3.0 style)."""
        if not self._live_dashboard_key():
            return error_response("Live dashboard disabled", 404, "Disabled")
        if not self._is_live_authorized(request):
            return error_response("Unauthorized", 401, "Unauthorized")
        html = self._LIVE_DASHBOARD_HTML
        return 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}, html.encode("utf-8")

    def _handle_live_snapshot(self, request) -> Tuple:
        """Return current topology snapshot as JSON (auth-gated)."""
        if not self._is_live_authorized(request):
            return error_response("Unauthorized", 401, "Unauthorized")
        em = self._event_manager
        if em is None:
            return json_response({"leader": None, "workers": {}, "chain": []})
        return json_response(em.get_topology_snapshot())

    # ---- Manager UI integration (proxies to cli_worker_manager) ----

    # Map external /live/mgr/<sub> -> internal manager /api/<sub>
    _LIVE_MGR_GET_MAP = {
        "status": "/api/status",
        "workers": "/api/workers",
        "metrics": "/api/metrics",
        "health": "/api/health",
    }
    _LIVE_MGR_POST_MAP = {
        "scale": "/api/scale",
        "rolling-update": "/api/rolling-update",
        "shutdown": "/api/shutdown",
        "nginx/reload": "/api/nginx/reload",
        "workers/start": "/api/workers/start",
        "workers/stop": "/api/workers/stop",
        "workers/restart": "/api/workers/restart",
    }

    # Cached manager port — discovered at first successful request (run_web_ui
    # auto-increments the port on collision, so the live config value may be wrong).
    _mgr_port_cached: Optional[int] = None

    def _handle_live_mgr(self, request) -> Tuple:
        if not self._is_live_authorized(request):
            return error_response("Unauthorized", 401, "Unauthorized")
        # Strip /live/mgr/ prefix
        sub = request.path[len("/live/mgr/"):].rstrip("/")
        method = (getattr(request, "method", "GET") or "GET").upper()

        # Direct broker control — handled locally via ZMQ, not proxied to the
        # manager API. Asks the current ZMQ leader to step down (auto re-elect).
        if method == "POST" and sub == "broker/relinquish":
            return self._handle_live_relinquish()

        if method == "GET":
            target = self._LIVE_MGR_GET_MAP.get(sub)
        elif method == "POST":
            target = self._LIVE_MGR_POST_MAP.get(sub)
        else:
            target = None
        if target is None:
            return error_response(f"Unknown manager endpoint: {sub}", 404, "Not Found")

        # Read body for POST
        body_bytes = b""
        if method == "POST":
            body = getattr(request, "body", None)
            if isinstance(body, (bytes, bytearray)):
                body_bytes = bytes(body)
            elif isinstance(body, str):
                body_bytes = body.encode("utf-8")
            elif isinstance(body, dict):
                body_bytes = json.dumps(body).encode("utf-8")

        base_port = int(getattr(self.config.manager, "web_ui_port", 9005))
        mgr_host = getattr(self.config.manager, "web_ui_host", "127.0.0.1") or "127.0.0.1"
        # Force IPv4 numeric host on Windows (localhost resolves to ::1 there)
        if mgr_host in ("localhost", ""):
            mgr_host = "127.0.0.1"

        # Try cached port first, then base..base+4 (matches run_web_ui's retry range)
        candidate_ports = []
        if self._mgr_port_cached is not None:
            candidate_ports.append(self._mgr_port_cached)
        for p in range(base_port, base_port + 5):
            if p not in candidate_ports:
                candidate_ports.append(p)

        import http.client as _hc
        last_err = None
        for port in candidate_ports:
            try:
                conn = _hc.HTTPConnection(mgr_host, port, timeout=3)
                headers = {"Content-Type": "application/json"} if method == "POST" else {}
                conn.request(method, target, body=body_bytes if method == "POST" else None, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                content_type = resp.getheader("Content-Type", "application/json")
                status_code = resp.status
                conn.close()
                # Cache the working port for next time
                HTTPWorker._mgr_port_cached = port
                return status_code, {"Content-Type": content_type, "Cache-Control": "no-store"}, data
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                # On Windows EADDRNOTAVAIL/refused: try next port; the cached port may be stale
                if port == self._mgr_port_cached:
                    HTTPWorker._mgr_port_cached = None
                continue
            except Exception as e:
                last_err = e
                break

        logger.warning(
            f"/live/mgr proxy {method} {target} -> {mgr_host}:{candidate_ports} all failed: {last_err}"
        )
        return error_response(
            f"Manager API unreachable at {mgr_host}:{base_port}{target}: {last_err}",
            502, "Bad Gateway",
        )

    _LIVE_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SimpleCore · /live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --raw-primary: 55% 0.18 230;
  --raw-success: 65% 0.2 145;
  --raw-warning: 75% 0.18 85;
  --raw-error: 55% 0.22 25;
  --primary: oklch(var(--raw-primary));
  --success: oklch(var(--raw-success));
  --warning: oklch(var(--raw-warning));
  --error: oklch(var(--raw-error));
  --bg-base: #08080d;
  --bg-surface: rgba(10,10,18,0.8);
  --bg-elevated: rgba(15,15,25,0.9);
  --bg-sunken: rgba(0,0,0,0.3);
  --glass-bg: rgba(255,255,255,0.02);
  --glass-border: rgba(255,255,255,0.05);
  --glass-blur: 12px;
  --border-subtle: rgba(255,255,255,0.08);
  --text-main: rgba(255,255,255,0.85);
  --text-label: rgba(255,255,255,0.4);
  --text-muted: rgba(255,255,255,0.25);
  --surface-badge: color-mix(in oklch, var(--primary) 15%, transparent);
  --surface-hover: color-mix(in oklch, var(--primary) 5%, transparent);
  --surface-active: color-mix(in oklch, var(--primary) 10%, transparent);
  --border-active: color-mix(in oklch, var(--primary) 30%, transparent);
  --highlight-inset: inset 0 1px 0 rgba(255,255,255,0.05);
  --shadow-micro: 0 2px 4px rgba(0,0,0,0.5);
  --radius-sm: 2px;
  --radius-md: 6px;
  --radius-lg: 12px;
  --radius-full: 9999px;
  --space-1: 0.25rem; --space-2: 0.5rem; --space-3: 0.75rem; --space-4: 1rem;
  --space-5: 1.5rem; --space-6: 2rem; --space-8: 3rem;
  --font-sans: 'IBM Plex Sans', system-ui, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, Consolas, monospace;
  --text-base: 13px; --text-sm: 11px; --text-xs: 9px;
  --duration-fast: 120ms; --duration-normal: 200ms; --duration-slow: 400ms;
  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  font-family: var(--font-sans);
  font-size: var(--text-base);
  color: var(--text-main);
  background:
    radial-gradient(ellipse at 20% 10%, color-mix(in oklch, var(--primary) 8%, transparent), transparent 60%),
    radial-gradient(ellipse at 80% 90%, color-mix(in oklch, oklch(60% 0.15 280) 6%, transparent), transparent 55%),
    var(--bg-base);
  overflow: hidden;
}
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
  background-size: 32px 32px;
  pointer-events: none;
  z-index: -1;
  animation: bgdrift 60s linear infinite;
}
@keyframes bgdrift { from { background-position: 0 0, 0 0; } to { background-position: 32px 32px, 32px 32px; } }

.label, h6 {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2.5px;
  color: var(--text-label);
  margin: 0;
  user-select: none;
}

header.bar {
  position: fixed; top: var(--space-4); left: var(--space-4); right: var(--space-4);
  z-index: 100;
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: var(--space-5);
  align-items: center;
  padding: var(--space-3) var(--space-5);
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  backdrop-filter: blur(var(--glass-blur));
  -webkit-backdrop-filter: blur(var(--glass-blur));
  border-radius: var(--radius-lg);
  box-shadow: var(--highlight-inset), var(--shadow-micro);
}
.brand { display: flex; align-items: center; gap: var(--space-3); }
.brand .dot {
  width: 10px; height: 10px; border-radius: var(--radius-full);
  background: var(--success);
  box-shadow: 0 0 8px color-mix(in oklch, var(--success) 60%, transparent);
  animation: heartbeat 1s ease-in-out infinite;
}
.brand .dot.stale { background: var(--error); box-shadow: 0 0 8px color-mix(in oklch, var(--error) 60%, transparent); }
.brand h1 { margin: 0; font-size: 14px; font-weight: 700; letter-spacing: -0.02em; }
.brand small { color: var(--text-muted); font-family: var(--font-mono); font-size: var(--text-xs); }

.kpis { display: flex; gap: var(--space-5); justify-content: center; }
.kpi { display: flex; flex-direction: column; gap: 2px; align-items: center; }
.kpi .v { font-family: var(--font-mono); font-size: 14px; font-weight: 600; color: var(--text-main); }
.kpi.warn .v { color: var(--warning); }
.kpi.err .v { color: var(--error); }

.controls { display: flex; gap: var(--space-2); align-items: center; }
.led-bars { display: inline-flex; gap: 3px; align-items: flex-end; height: 16px; }
.led-bars span { width: 3px; background: var(--border-subtle); border-radius: 1px; transition: background var(--duration-normal); }
.led-bars span:nth-child(1) { height: 30%; } .led-bars span:nth-child(2) { height: 50%; }
.led-bars span:nth-child(3) { height: 70%; } .led-bars span:nth-child(4) { height: 90%; } .led-bars span:nth-child(5) { height: 100%; }
.led-bars.health-3 span:nth-child(-n+5) { background: var(--success); }
.led-bars.health-2 span:nth-child(-n+4) { background: var(--warning); }
.led-bars.health-1 span:nth-child(-n+3) { background: var(--warning); }
.led-bars.health-0 span:nth-child(-n+1) { background: var(--error); }

main {
  position: fixed;
  inset: 80px var(--space-4) var(--space-4);
  display: grid;
  grid-template-columns: 1fr 340px 320px;
  gap: var(--space-4);
}
@media (max-width: 1280px) {
  main { grid-template-columns: 1fr 320px; }
  #manager-panel { display: none; }
}
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; grid-template-rows: 1fr auto; inset: 80px var(--space-2) var(--space-2); }
  #detail-panel, #manager-panel { max-height: 40vh; }
}

.panel {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  backdrop-filter: blur(var(--glass-blur));
  -webkit-backdrop-filter: blur(var(--glass-blur));
  border-radius: var(--radius-lg);
  box-shadow: var(--highlight-inset), var(--shadow-micro);
  overflow: hidden;
  display: flex; flex-direction: column;
}
.panel .head {
  padding: var(--space-3) var(--space-5);
  border-bottom: 1px solid var(--border-subtle);
  display: flex; justify-content: space-between; align-items: center;
}
.panel .body { flex: 1; overflow: hidden; position: relative; }
.panel .body.scroll { overflow: auto; padding: var(--space-4) var(--space-5); }

#graph { width: 100%; height: 100%; display: block; }
#graph .edge { stroke: color-mix(in oklch, var(--primary) 25%, transparent); stroke-width: 1; }
#graph .edge.chain-next { stroke: color-mix(in oklch, var(--primary) 60%, transparent); stroke-width: 1.5; stroke-dasharray: 3 3; animation: dashflow 1.5s linear infinite; }
@keyframes dashflow { to { stroke-dashoffset: -12; } }
#graph .node circle.halo { fill: none; stroke: color-mix(in oklch, var(--primary) 40%, transparent); stroke-width: 1; opacity: 0; }
#graph .node.leader circle.halo { opacity: 1; animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse {
  0% { r: 24; opacity: 0.6; }
  100% { r: 42; opacity: 0; }
}
#graph .node circle.core {
  fill: var(--bg-elevated);
  stroke: var(--border-subtle); stroke-width: 1.5;
  transition: stroke var(--duration-normal), fill var(--duration-normal);
}
#graph .node.leader circle.core { stroke: var(--primary); fill: color-mix(in oklch, var(--primary) 18%, var(--bg-elevated)); }
#graph .node.follower circle.core { stroke: color-mix(in oklch, var(--success) 50%, transparent); }
#graph .node.stale circle.core { stroke: var(--error); fill: color-mix(in oklch, var(--error) 12%, var(--bg-elevated)); }
#graph .node.selected circle.core { stroke-width: 2.5; filter: drop-shadow(0 0 6px color-mix(in oklch, var(--primary) 60%, transparent)); }
#graph .node text.id { font-family: var(--font-mono); font-size: 9px; fill: var(--text-main); text-anchor: middle; pointer-events: none; }
#graph .node text.kind { font-family: var(--font-mono); font-size: 8px; text-transform: uppercase; letter-spacing: 1px; fill: var(--text-label); text-anchor: middle; pointer-events: none; }
#graph .node { cursor: pointer; }
@keyframes heartbeat {
  0%, 100% { transform: scale(1); opacity: 0.85; }
  50% { transform: scale(1.25); opacity: 1; }
}

#detail h2 { font-size: 14px; font-weight: 700; margin: 0 0 var(--space-1); letter-spacing: -0.02em; }
#detail .sub { font-family: var(--font-mono); font-size: var(--text-xs); color: var(--text-muted); margin-bottom: var(--space-4); display: flex; gap: var(--space-2); flex-wrap: wrap; }
#detail .chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: var(--radius-sm);
  background: var(--surface-badge);
  font-family: var(--font-mono); font-size: var(--text-xs);
  text-transform: uppercase; letter-spacing: 1.5px;
  color: var(--text-main);
}
#detail .chip.leader { background: color-mix(in oklch, var(--primary) 25%, transparent); }
#detail .chip.stale { background: color-mix(in oklch, var(--error) 25%, transparent); }
#detail .placeholder { color: var(--text-muted); font-style: italic; padding: var(--space-5) 0; text-align: center; }
#detail details {
  border-top: 1px solid var(--border-subtle);
  padding: var(--space-3) 0;
}
#detail details:first-of-type { border-top: none; padding-top: 0; }
#detail summary {
  list-style: none;
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-label);
  display: flex; justify-content: space-between; align-items: center;
  padding: var(--space-1) 0;
  user-select: none;
  transition: color var(--duration-fast);
}
#detail summary:hover { color: var(--text-main); }
#detail summary::-webkit-details-marker { display: none; }
#detail summary::after { content: '▸'; transition: transform var(--duration-normal) var(--ease-default); color: var(--text-muted); }
#detail details[open] > summary::after { transform: rotate(90deg); }
#detail .kv {
  display: grid; grid-template-columns: minmax(80px, max-content) 1fr;
  gap: 4px var(--space-3); margin-top: var(--space-2);
  font-family: var(--font-mono); font-size: var(--text-sm);
}
#detail .kv dt { color: var(--text-label); }
#detail .kv dd { margin: 0; color: var(--text-main); word-break: break-all; }
#detail ul.routes, #detail ul.chain { list-style: none; padding: 0; margin: var(--space-2) 0 0; font-family: var(--font-mono); font-size: var(--text-sm); }
#detail ul.routes li, #detail ul.chain li {
  padding: 3px 8px; border-radius: var(--radius-sm); background: var(--bg-sunken);
  color: var(--text-main); margin-bottom: 3px; word-break: break-all;
}
#detail ul.chain li.current { background: var(--surface-badge); color: var(--text-main); border-left: 2px solid var(--primary); }

/* Action buttons (inspector + manager) */
.act-row { display: flex; gap: var(--space-2); margin-top: var(--space-2); flex-wrap: wrap; }
.act-btn {
  flex: 1 1 auto;
  min-width: 80px;
  padding: 6px 10px;
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-main);
  background: var(--surface-hover);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background var(--duration-fast), border-color var(--duration-fast), color var(--duration-fast);
}
.act-btn:hover { background: var(--surface-active); border-color: var(--border-active); }
.act-btn:active { transform: scale(0.98); }
.act-btn:disabled, .act-btn.loading { opacity: 0.5; cursor: wait; }
.act-btn.warn { color: var(--warning); }
.act-btn.warn:hover { background: color-mix(in oklch, var(--warning) 10%, transparent); border-color: color-mix(in oklch, var(--warning) 40%, transparent); }
.act-btn.sm { min-width: 32px; padding: 4px 8px; flex: 0 0 auto; }
.scale-row {
  display: flex; align-items: center; gap: var(--space-2);
  margin-top: var(--space-2);
  font-family: var(--font-mono); font-size: var(--text-sm);
}
.scale-row .lbl { color: var(--text-label); text-transform: uppercase; letter-spacing: 1.5px; font-size: var(--text-xs); min-width: 40px; }
.scale-row .count { min-width: 24px; text-align: center; color: var(--text-main); }
code { font-family: var(--font-mono); font-size: var(--text-xs); background: var(--bg-sunken); padding: 1px 4px; border-radius: var(--radius-sm); }

#banner {
  position: fixed; bottom: var(--space-4); left: 50%;
  transform: translateX(-50%) translateY(40px);
  padding: var(--space-2) var(--space-4);
  background: var(--bg-elevated);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  backdrop-filter: blur(var(--glass-blur));
  box-shadow: var(--highlight-inset), var(--shadow-micro);
  font-family: var(--font-mono); font-size: var(--text-sm);
  color: var(--text-main);
  opacity: 0; pointer-events: none;
  transition: opacity var(--duration-normal), transform var(--duration-normal);
  z-index: 200;
}
#banner.show { opacity: 1; transform: translateX(-50%) translateY(0); }
#banner.warn { border-color: color-mix(in oklch, var(--warning) 30%, var(--glass-border)); }
#banner.err { border-color: color-mix(in oklch, var(--error) 30%, var(--glass-border)); }

/* utility */
.mono { font-family: var(--font-mono); }
</style>
</head>
<body>
<header class="bar">
  <div class="brand">
    <span class="dot" id="health-dot"></span>
    <div>
      <h1>SimpleCore · <span class="mono" style="color: var(--text-label); font-weight: 500;">/live</span></h1>
      <small>leader: <span id="leader-id">—</span></small>
    </div>
  </div>
  <div class="kpis">
    <div class="kpi"><h6>workers</h6><span class="v" id="kpi-workers">0</span></div>
    <div class="kpi"><h6>conn</h6><span class="v" id="kpi-conn">0</span></div>
    <div class="kpi"><h6>req</h6><span class="v mono" id="kpi-req">—</span></div>
    <div class="kpi"><h6>latency</h6><span class="v mono" id="kpi-lat">—</span></div>
    <div class="kpi"><h6>routes</h6><span class="v" id="kpi-routes">0</span></div>
    <div class="kpi"><h6>updated</h6><span class="v mono" id="kpi-updated">—</span></div>
  </div>
  <div class="controls">
    <span class="led-bars" id="health-bars" aria-label="cluster health">
      <span></span><span></span><span></span><span></span><span></span>
    </span>
  </div>
</header>

<main>
  <section class="panel">
    <div class="head">
      <h6>topology</h6>
      <h6 id="conn-state">connecting…</h6>
    </div>
    <div class="body">
      <svg id="graph" preserveAspectRatio="xMidYMid meet"></svg>
    </div>
  </section>

  <section class="panel" id="detail-panel">
    <div class="head">
      <h6>inspector</h6>
      <h6 id="selected-hint">click a node</h6>
    </div>
    <div class="body scroll" id="detail">
      <p class="placeholder">No node selected.<br><small style="color: var(--text-muted);">Click any worker in the topology graph.</small></p>
    </div>
  </section>

  <section class="panel" id="manager-panel">
    <div class="head">
      <h6>manager</h6>
      <h6 id="mgr-hint">cluster</h6>
    </div>
    <div class="body scroll" id="mgr">
      <p class="placeholder">Loading…</p>
    </div>
  </section>
</main>

<div id="banner"></div>

<script>
(function(){
  const KEY = new URLSearchParams(location.search).get('key') || '';
  const STORE_KEY = 'live.state.v1';
  const RECONNECT_MS = 2000;

  const $graph = document.getElementById('graph');
  const $detail = document.getElementById('detail');
  const $selHint = document.getElementById('selected-hint');
  const $banner = document.getElementById('banner');
  const $dot = document.getElementById('health-dot');
  const $leader = document.getElementById('leader-id');
  const $kw = document.getElementById('kpi-workers');
  const $kc = document.getElementById('kpi-conn');
  const $kr = document.getElementById('kpi-routes');
  const $kreq = document.getElementById('kpi-req');
  const $klat = document.getElementById('kpi-lat');
  const $ku = document.getElementById('kpi-updated');
  const $bars = document.getElementById('health-bars');
  const $cstate = document.getElementById('conn-state');
  const $mgr = document.getElementById('mgr');

  let state = restoreState() || { leader: null, workers: {}, chain: [], updated_at: 0 };
  let mgrState = restoreMgrState() || { workers: [], metrics: {}, health: {}, nginx: {} };
  let connFailCount = 0;
  let reconnectInFlight = false;
  const CONN_FAIL_THRESHOLD = 3;
  let selectedId = sessionStorage.getItem('live.selected') || null;
  let nodes = new Map();  // worker_id -> {x, y, vx, vy}
  let lastDraw = 0;

  function saveState() {
    try { sessionStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch(e){}
  }
  function restoreState() {
    try { const s = sessionStorage.getItem(STORE_KEY); return s ? JSON.parse(s) : null; } catch(e){ return null; }
  }
  const MGR_STORE_KEY = 'live.mgrState.v1';
  function saveMgrState() { try { sessionStorage.setItem(MGR_STORE_KEY, JSON.stringify(mgrState)); } catch(e){} }
  function restoreMgrState() {
    try { const s = sessionStorage.getItem(MGR_STORE_KEY); return s ? JSON.parse(s) : null; } catch(e){ return null; }
  }
  // True if the given worker_id is the worker currently serving this page.
  function isSelf(workerId) {
    if (!workerId || !mgrState.workers) return false;
    const myPort = location.port || (location.protocol === 'https:' ? '443' : '80');
    const w = mgrState.workers.find(x => x.worker_id === workerId);
    return !!(w && w.port && String(w.port) === String(myPort));
  }
  // Try to redirect this page to another healthy HTTP worker's port when our
  // own worker has died. Returns true if a reconnect was scheduled.
  function maybeReconnectToOtherWorker() {
    if (reconnectInFlight) return true;
    const workers = (mgrState && mgrState.workers) || [];
    const myPort = location.port || (location.protocol === 'https:' ? '443' : '80');
    const others = workers.filter(w =>
      w.worker_type === 'http' && w.port && String(w.port) !== String(myPort)
      && (!w.state || w.state === 'running')
    );
    if (!others.length) return false;
    reconnectInFlight = true;
    const target = others[0];
    const url = new URL(location.href);
    url.port = String(target.port);
    banner('Worker on :' + myPort + ' down — reconnecting to :' + target.port + '…', 'warn');
    setTimeout(() => { location.href = url.toString(); }, 1500);
    return true;
  }

  function banner(msg, kind) {
    $banner.className = kind || '';
    $banner.classList.add('show');
    $banner.textContent = msg;
    clearTimeout(banner._t);
    banner._t = setTimeout(()=>$banner.classList.remove('show'), 3500);
  }

  function fmtAgo(ts) {
    if (!ts) return '—';
    const dt = Math.max(0, Date.now()/1000 - ts);
    if (dt < 2) return 'now';
    if (dt < 60) return Math.round(dt) + 's';
    if (dt < 3600) return Math.round(dt/60) + 'm';
    return Math.round(dt/3600) + 'h';
  }
  function fmtUptime(ts) {
    if (!ts) return '—';
    const s = Math.max(0, Date.now()/1000 - ts);
    const d = Math.floor(s/86400), h = Math.floor((s%86400)/3600), m = Math.floor((s%3600)/60);
    return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`;
  }

  function renderKPIs() {
    const wids = Object.keys(state.workers||{});
    $kw.textContent = wids.length;
    const m = mgrState.metrics || {};
    const fallbackConn = wids.reduce((a,k)=>a+(state.workers[k].connections||0),0);
    const fallbackRoutes = new Set(wids.flatMap(k=>state.workers[k].routes||[])).size;
    $kc.textContent = (m.total_connections != null) ? m.total_connections : fallbackConn;
    $kr.textContent = fallbackRoutes;
    if ($kreq) $kreq.textContent = m.total_requests != null ? m.total_requests : '—';
    if ($klat) $klat.textContent = m.avg_latency_ms != null ? (m.avg_latency_ms.toFixed(1) + 'ms') : '—';
    $ku.textContent = fmtAgo(state.updated_at);
    $leader.textContent = state.leader || '—';
    const now = Date.now()/1000;
    const stale = wids.filter(k => now - (state.workers[k].last_status_at||0) > 15).length;
    $dot.classList.toggle('stale', stale > 0 || !state.leader);
    const health = !state.leader ? 0 : stale === 0 ? 3 : stale < wids.length/2 ? 2 : 1;
    $bars.className = 'led-bars health-' + health;
  }

  // ---- simple force layout ----
  function tick() {
    const W = $graph.clientWidth, H = $graph.clientHeight;
    const cx = W/2, cy = H/2;
    const wids = Object.keys(state.workers||{});
    // Add missing nodes; remove gone ones
    for (const k of wids) {
      if (!nodes.has(k)) {
        const angle = Math.random() * Math.PI * 2;
        nodes.set(k, { x: cx + Math.cos(angle)*120, y: cy + Math.sin(angle)*120, vx: 0, vy: 0 });
      }
    }
    for (const k of [...nodes.keys()]) if (!wids.includes(k)) nodes.delete(k);

    const arr = [...nodes.entries()];
    // Repulsion
    for (let i = 0; i < arr.length; i++) {
      for (let j = i+1; j < arr.length; j++) {
        const [, a] = arr[i]; const [, b] = arr[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx*dx + dy*dy + 0.01;
        const f = 8000 / d2;
        const d = Math.sqrt(d2);
        const fx = (dx/d)*f, fy = (dy/d)*f;
        a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
      }
    }
    // Attraction to center + to leader
    const leaderNode = nodes.get(state.leader);
    for (const [k, n] of arr) {
      n.vx += (cx - n.x) * 0.012;
      n.vy += (cy - n.y) * 0.012;
      if (leaderNode && k !== state.leader) {
        n.vx += (leaderNode.x - n.x) * 0.008;
        n.vy += (leaderNode.y - n.y) * 0.008;
      }
    }
    // Integrate with damping
    for (const [, n] of arr) {
      n.vx *= 0.82; n.vy *= 0.82;
      n.x += n.vx; n.y += n.vy;
      n.x = Math.max(40, Math.min(W-40, n.x));
      n.y = Math.max(40, Math.min(H-40, n.y));
    }
  }

  function draw() {
    const W = $graph.clientWidth, H = $graph.clientHeight;
    $graph.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const wids = Object.keys(state.workers||{});
    const chain = state.chain || [];
    const now = Date.now()/1000;
    // edges (leader -> followers + chain hops)
    let svg = '';
    const leaderNode = nodes.get(state.leader);
    if (leaderNode) {
      for (const k of wids) {
        if (k === state.leader) continue;
        const n = nodes.get(k); if (!n) continue;
        svg += `<line class="edge" x1="${leaderNode.x}" y1="${leaderNode.y}" x2="${n.x}" y2="${n.y}"/>`;
      }
    }
    // chain hop overlay
    for (let i = 0; i < chain.length - 1; i++) {
      const a = nodes.get(chain[i]); const b = nodes.get(chain[i+1]);
      if (a && b) svg += `<line class="edge chain-next" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/>`;
    }
    // nodes
    for (const k of wids) {
      const n = nodes.get(k); if (!n) continue;
      const w = state.workers[k];
      const isLeader = k === state.leader;
      const stale = now - (w.last_status_at||0) > 15;
      const cls = 'node ' + (isLeader ? 'leader' : 'follower') + (stale ? ' stale' : '') + (k === selectedId ? ' selected' : '');
      const short = (k.length > 16 ? k.slice(0,14)+'…' : k);
      svg += `<g class="${cls}" data-id="${k}" transform="translate(${n.x},${n.y})">
        <circle class="halo" r="28"/>
        <circle class="core" r="22"/>
        <text class="kind" y="-4">${w.worker_type||'?'}</text>
        <text class="id" y="9">${short}</text>
      </g>`;
    }
    $graph.innerHTML = svg;
  }

  // Persist open/closed state of <details data-key="…"> across re-renders.
  // sessionStorage so it also survives reloads.
  const OPEN_STORE_KEY = 'live.openSections.v1';
  function loadOpenSet() {
    try {
      const raw = sessionStorage.getItem(OPEN_STORE_KEY);
      if (raw) return new Set(JSON.parse(raw));
    } catch(e){}
    // sensible defaults on first load
    return new Set(['process', 'mgr-overview']);
  }
  function saveOpenSet(set) {
    try { sessionStorage.setItem(OPEN_STORE_KEY, JSON.stringify([...set])); } catch(e){}
  }
  let openSections = loadOpenSet();
  function applyOpenStateTo(container) {
    container.querySelectorAll('details[data-key]').forEach(d => {
      d.open = openSections.has(d.dataset.key);
      // sync changes back to the set + storage
      d.addEventListener('toggle', () => {
        if (d.open) openSections.add(d.dataset.key);
        else openSections.delete(d.dataset.key);
        saveOpenSet(openSections);
      });
    });
  }

  function renderDetail() {
    if (!selectedId || !state.workers[selectedId]) {
      $selHint.textContent = 'click a node';
      $detail.innerHTML = '<p class="placeholder">No node selected.<br><small style="color: var(--text-muted);">Click any worker in the topology graph.</small></p>';
      return;
    }
    const w = state.workers[selectedId];
    const isLeader = selectedId === state.leader;
    const now = Date.now()/1000;
    const stale = now - (w.last_status_at||0) > 15;
    const chain = state.chain || [];
    const myChainIdx = chain.indexOf(selectedId);
    $selHint.textContent = selectedId.slice(0, 24);

    const chips = [
      isLeader ? '<span class="chip leader">LEADER</span>' : '<span class="chip">FOLLOWER</span>',
      stale ? '<span class="chip stale">STALE</span>' : '',
      `<span class="chip">${w.worker_type||'?'}</span>`,
      isSelf(selectedId) ? '<span class="chip" style="background: color-mix(in oklch, var(--warning) 25%, transparent);">SELF</span>' : '',
      myChainIdx >= 0 ? `<span class="chip">#${myChainIdx+1} / ${chain.length}</span>` : '',
    ].filter(Boolean).join('');

    let html = `<h2>${selectedId}</h2><div class="sub">${chips}</div>`;

    html += `<details data-key="process"><summary>process</summary><dl class="kv">
      <dt>pid</dt><dd>${w.pid ?? '—'}</dd>
      <dt>uptime</dt><dd>${fmtUptime(w.started_at)}</dd>
      <dt>cpu</dt><dd>${(w.cpu_percent ?? 0).toFixed(1)}%</dd>
      <dt>memory</dt><dd>${(w.memory_mb ?? 0).toFixed(1)} MB</dd>
      <dt>seen</dt><dd>${fmtAgo(w.last_status_at)} ago</dd>
    </dl></details>`;

    // Per-worker live metrics (from manager — keyed by worker_id)
    const wm = (mgrState.workers || []).find(x => x.worker_id === selectedId);
    if (wm && wm.metrics) {
      html += `<details data-key="metrics"><summary>metrics</summary><dl class="kv">
        <dt>requests</dt><dd>${wm.metrics.requests ?? 0}</dd>
        <dt>connections</dt><dd>${wm.metrics.connections ?? 0}</dd>
        <dt>errors</dt><dd>${wm.metrics.errors ?? 0}</dd>
        <dt>avg latency</dt><dd>${(wm.metrics.avg_latency_ms ?? 0).toFixed(2)} ms</dd>
      </dl></details>`;
    }

    if (w.routes && w.routes.length) {
      html += `<details data-key="routes"><summary>routes <span style="color: var(--text-muted);">${w.routes.length}</span></summary>
        <ul class="routes">${w.routes.map(r=>`<li>${r}</li>`).join('')}</ul></details>`;
    }

    html += `<details data-key="websocket"><summary>websocket</summary><dl class="kv">
      <dt>connections</dt><dd>${w.connections ?? 0}</dd>
    </dl></details>`;

    if (chain.length) {
      html += `<details data-key="chain"><summary>failover chain</summary><ul class="chain">${
        chain.map((k,i)=>`<li class="${k===selectedId?'current':''}"><span style="color: var(--text-muted);">#${i+1}</span> ${k}${k===state.leader?' <span class="chip leader" style="margin-left: var(--space-2);">L</span>':''}</li>`).join('')
      }</ul></details>`;
    }

    // Worker actions (proxied to manager)
    const selfWarn = isSelf(selectedId);
    html += `<details data-key="actions"><summary>actions</summary>
      <div class="act-row">
        <button class="act-btn" data-act="restart">restart</button>
        <button class="act-btn warn" data-act="stop"${selfWarn ? ' title="stops the worker serving this page"' : ''}>stop${selfWarn ? ' (self)' : ''}</button>
      </div>
    </details>`;

    html += `<details data-key="raw"><summary>raw</summary><pre style="font-size: var(--text-xs); color: var(--text-muted); white-space: pre-wrap; word-break: break-all; margin: var(--space-2) 0 0;">${JSON.stringify(w, null, 2)}</pre></details>`;

    $detail.innerHTML = html;
    applyOpenStateTo($detail);
    // bind worker-action buttons
    $detail.querySelectorAll('.act-btn').forEach(b => {
      b.addEventListener('click', () => workerAction(selectedId, b.dataset.act, b));
    });
  }

  function applySnapshot(snap) {
    if (!snap) return;
    const prevLeader = state.leader;
    state = snap;
    saveState();
    renderKPIs();
    renderDetail();
    if (prevLeader && prevLeader !== snap.leader) {
      banner('Leader changed: ' + prevLeader + ' → ' + (snap.leader || '∅'), 'warn');
    }
  }

  // initial fetch via REST
  function fetchSnapshot() {
    fetch('/live/snapshot' + (KEY ? '?key=' + encodeURIComponent(KEY) : ''))
      .then(r => { if (r.status === 401) { banner('Unauthorized — append ?key=…', 'err'); throw 0; } return r.json(); })
      .then(snap => { connFailCount = 0; applySnapshot(snap); })
      .catch(() => {
        connFailCount++;
        if (connFailCount >= CONN_FAIL_THRESHOLD) maybeReconnectToOtherWorker();
      });
  }

  // Manager (cluster-control) state — proxied through /live/mgr/*
  function fetchMgr() {
    const k = KEY ? ('?key=' + encodeURIComponent(KEY)) : '';
    Promise.all([
      fetch('/live/mgr/workers' + k).then(r => r.ok ? r.json() : null).catch(()=>null),
      fetch('/live/mgr/metrics' + k).then(r => r.ok ? r.json() : null).catch(()=>null),
      fetch('/live/mgr/health'  + k).then(r => r.ok ? r.json() : null).catch(()=>null),
      fetch('/live/mgr/status'  + k).then(r => r.ok ? r.json() : null).catch(()=>null),
    ]).then(([workers, metrics, health, status]) => {
      const ok = !!(workers || metrics || status);
      mgrState.available = ok;
      if (workers) mgrState.workers = workers;
      if (metrics) mgrState.metrics = metrics;
      if (health) mgrState.health = health;
      if (status) {
        mgrState.status = status;
        mgrState.nginx = status.nginx || {};
      }
      if (ok) { connFailCount = 0; saveMgrState(); }
      else    { connFailCount++; if (connFailCount >= CONN_FAIL_THRESHOLD) maybeReconnectToOtherWorker(); }
      renderMgr();
      renderDetail();
    });
  }

  function mgrPost(path, body, btnEl) {
    const k = KEY ? ('?key=' + encodeURIComponent(KEY)) : '';
    if (btnEl) { btnEl.disabled = true; btnEl.classList.add('loading'); }
    return fetch('/live/mgr' + path + k, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body || {})
    }).then(r => r.json().catch(()=>({status:'error'})))
      .then(j => {
        if (j.status === 'ok') banner('✓ ' + path, null);
        else banner('✗ ' + path + ': ' + (j.message || 'error'), 'err');
        fetchMgr();
        return j;
      })
      .catch(e => { banner('✗ ' + path + ': ' + e, 'err'); })
      .finally(() => { if (btnEl) { btnEl.disabled = false; btnEl.classList.remove('loading'); } });
  }

  function workerAction(workerId, act, btnEl) {
    if (act === 'stop' && isSelf(workerId)) {
      if (!confirm('This worker is currently serving this page. Stopping it will disconnect you — the UI will auto-reconnect to another worker if available. Continue?')) return;
    }
    if (act === 'restart') return mgrPost('/workers/restart', {worker_id: workerId}, btnEl);
    if (act === 'stop')    return mgrPost('/workers/stop',    {worker_id: workerId, graceful: true}, btnEl);
  }

  // Render the global manager panel (stats + actions)
  function renderMgr() {
    if (!$mgr) return;
    if (!mgrState.available) {
      $mgr.innerHTML = '<p class="placeholder">Manager API not reachable.<br><small style="color: var(--text-muted);">Make sure <code>tb workers start</code> is running.</small></p>';
      return;
    }
    const m = mgrState.metrics || {};
    const h = mgrState.health || {};
    const s = mgrState.status || {};
    const httpCount = (mgrState.workers || []).filter(w => w.worker_type === 'http').length;
    const wsCount   = (mgrState.workers || []).filter(w => w.worker_type === 'ws').length;
    const healthy   = h.healthy ? '<span class="chip" style="background: color-mix(in oklch, var(--success) 25%, transparent);">healthy</span>'
                                : '<span class="chip stale">degraded</span>';

    let html = `<details data-key="mgr-overview" open><summary>overview</summary><dl class="kv">
      <dt>healthy</dt><dd>${healthy}</dd>
      <dt>broker</dt><dd>${(h.broker && h.broker.alive) ? 'process' : (h.broker && h.broker.leader_id) ? 'p2p:' + h.broker.leader_id.slice(0,12) : 'none'}</dd>
      <dt>http workers</dt><dd>${httpCount}</dd>
      <dt>ws workers</dt><dd>${wsCount}</dd>
      <dt>requests</dt><dd>${m.total_requests ?? 0}</dd>
      <dt>connections</dt><dd>${m.total_connections ?? 0}</dd>
      <dt>errors</dt><dd>${m.total_errors ?? 0}</dd>
      <dt>avg latency</dt><dd>${(m.avg_latency_ms ?? 0).toFixed(2)} ms</dd>
      <dt>nginx</dt><dd>${(mgrState.nginx && mgrState.nginx.installed) ? 'installed' : '—'}</dd>
    </dl></details>`;

    html += `<details data-key="mgr-scale"><summary>scale</summary>
      <div class="scale-row"><span class="lbl">http</span>
        <button class="act-btn sm" data-scale="http" data-delta="-1">−</button>
        <span class="count">${httpCount}</span>
        <button class="act-btn sm" data-scale="http" data-delta="+1">+</button>
      </div>
      <div class="scale-row"><span class="lbl">ws</span>
        <button class="act-btn sm" data-scale="ws" data-delta="-1">−</button>
        <span class="count">${wsCount}</span>
        <button class="act-btn sm" data-scale="ws" data-delta="+1">+</button>
      </div>
    </details>`;

    html += `<details data-key="mgr-actions"><summary>actions</summary>
      <div class="act-row">
        <button class="act-btn" data-act="rolling">rolling update</button>
        <button class="act-btn" data-act="nginx">nginx reload</button>
      </div>
      <div class="act-row" style="margin-top: var(--space-2);">
        <button class="act-btn warn" data-act="shutdown">shutdown all</button>
      </div>
    </details>`;

    if (mgrState.workers && mgrState.workers.length) {
      html += `<details data-key="mgr-workers"><summary>workers <span style="color: var(--text-muted);">${mgrState.workers.length}</span></summary>
        <ul class="routes">${mgrState.workers.map(w => `<li><span style="color: var(--text-muted);">${w.worker_type}</span> ${w.worker_id} <span style="color: var(--text-muted);">:${w.port || '—'}</span></li>`).join('')}</ul>
      </details>`;
    }

    $mgr.innerHTML = html;
    applyOpenStateTo($mgr);
    // Bind scale +/-
    $mgr.querySelectorAll('[data-scale]').forEach(b => {
      b.addEventListener('click', () => {
        const t = b.dataset.scale;
        const delta = parseInt(b.dataset.delta);
        const current = t === 'http' ? httpCount : wsCount;
        const target = Math.max(0, current + delta);
        mgrPost('/scale', {type: t, count: target}, b);
      });
    });
    // Bind global actions
    $mgr.querySelectorAll('[data-act]').forEach(b => {
      b.addEventListener('click', () => {
        const a = b.dataset.act;
        if (a === 'rolling')   return mgrPost('/rolling-update', {}, b);
        if (a === 'nginx')     return mgrPost('/nginx/reload',   {}, b);
        if (a === 'shutdown')  { if (confirm('Shut down all workers?')) mgrPost('/shutdown', {}, b); }
      });
    });
  }

  // WS subscribe — channel "sys.topology"
  let ws = null;
  function connectWS() {
    $cstate.textContent = 'connecting…';
    try {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      // WS worker default port 8100; allow override via ?ws=host:port
      const wsTarget = new URLSearchParams(location.search).get('ws')
        || (location.hostname + ':8100');
      ws = new WebSocket(`${proto}//${wsTarget}/ws?channel=sys.topology`);
      ws.onopen = () => { $cstate.textContent = 'live'; banner('Connected', null); };
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m && m.type === 'sys.topology' && m.data) applySnapshot(m.data);
        } catch(e){}
      };
      ws.onclose = () => { $cstate.textContent = 'reconnecting…'; setTimeout(connectWS, RECONNECT_MS); };
      ws.onerror = () => { try { ws.close(); } catch(e){} };
    } catch(e) {
      $cstate.textContent = 'offline';
      setTimeout(connectWS, RECONNECT_MS);
    }
  }

  // render loop
  function loop() {
    tick();
    const now = performance.now();
    if (now - lastDraw > 16) { draw(); lastDraw = now; }
    requestAnimationFrame(loop);
  }

  // keep KPI updated_at fresh
  setInterval(renderKPIs, 1000);
  // refresh snapshot via REST every 5s as fallback
  setInterval(fetchSnapshot, 5000);
  // refresh manager state every 3s
  setInterval(fetchMgr, 3000);

  renderKPIs();
  renderDetail();
  renderMgr();
  fetchMgr();
  // Delegated pointer handler — bound once on the SVG parent. Uses pointerdown
  // (not click) because nodes move every frame; click requires down+up on the
  // same element, which fails on a moving target.
  $graph.addEventListener('pointerdown', (e) => {
    const node = e.target.closest('.node');
    if (!node) return;
    e.preventDefault();
    selectedId = node.dataset.id;
    sessionStorage.setItem('live.selected', selectedId);
    renderDetail();
    draw();
  });
  fetchSnapshot();
  connectWS();
  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""

    def _handle_live_relinquish(self) -> Tuple:
        """POST /live/mgr/broker/relinquish — ask the current ZMQ leader to step
        down. rpc_call reaches whoever owns the REP socket (= the leader); a
        follower then takes over via its watchdog. Returns the leader's ack."""
        from toolboxv2.utils.workers.event_manager import Event, EventType
        em = self._event_manager
        if em is None or not getattr(em, "_running", False):
            return error_response("event bus unavailable", 503, "Unavailable")
        ev = Event(type=EventType.SYS_RELINQUISH, source=self.worker_id, target="*")
        try:
            resp = self._run_async(em.rpc_call(ev, timeout=3.0))  # returns dict
        except Exception as e:
            return error_response(f"relinquish failed: {e}", 502, "BadGateway")
        return 200, {"Content-Type": "application/json"}, json.dumps(resp).encode()

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

    def _handle_ip_request(self, request: ParsedRequest) -> Tuple:
        """API für /api/ip"""
        return json_response({"ip": request.client_ip})

    def _handle_ping_request(self) -> Tuple:
        """API für /api/ping (Antwortet sofort für Latenzmessung)"""
        return json_response({"status": "pong", "timestamp": time.time()})

    def _handle_geo_request(self, request: ParsedRequest) -> Tuple:
        """API für /api/geo - Ersetzt deine alte get_location() Logik."""
        ip = request.client_ip
        geo_data = get_location(ip)
        if 'city' not in geo_data:
            geo_data = self._get_geo_locally(ip=ip)

        # Rückgabe als Dictionary wie gewünscht
        return json_response({
            "ip": ip,
            **geo_data
        })

    def _get_geo_locally(self, ip: str) -> dict:
        """Sucht die Location lokal in der MMDB Datei ohne externe API."""
        # Pfad aus config oder default
        db_path = getattr(self.config, "geoip_db_path", "data/GeoLite2-City.mmdb")

        default_data = {"city": "Unknown", "region": "Unknown", "country": "Unknown"}

        if ip in ["127.0.0.1", "localhost", "::1"] or ip.startswith("192.168."):
            return {"city": "Local", "region": "LAN", "country": "Internal"}

        try:
            import geoip2.database
            with geoip2.database.Reader(db_path) as reader:
                response = reader.city(ip)
                return {
                    "city": response.city.name or "Unknown",
                    "region": response.subdivisions.most_specific.name or "Unknown",
                    "country": response.country.name or "Unknown"
                }
        except Exception as e:
            logger.debug(f"Local GeoIP failed (check if .mmdb exists): {e}")
            return default_data

    def _handle_client_logs(self, request: ParsedRequest) -> Tuple:
        """POST /api/client-logs — ingest browser logs into the server logging pipeline.

        Accepts JSON body:
            { "logs": [ { type, timestamp, level, source, message, url, ... }, ... ] }

        Consent filtering happens client-side; this endpoint trusts the batch
        and caps at 200 entries per request to prevent abuse.
        """
        if request.method != "POST":
            return error_response("Method not allowed", 405, "MethodNotAllowed")

        logs = (request.json_data or {}).get("logs")
        if not logs or not isinstance(logs, list):
            return json_response({"status": "ok", "ingested": 0})

        client_ip = request.client_ip
        session = request.session
        session_id = session.session_id if session else ""
        user_name = ""
        if session and hasattr(session, "user_name"):
            user_name = session.user_name or ""

        ingested = 0
        audit_db = getattr(self, "_audit_logger", None)

        for entry in logs[:200]:
            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type", "log")
            level_str = entry.get("level", "INFO").upper()
            py_level = self._CLIENT_LEVEL_MAP.get(level_str, logging.INFO)
            message = entry.get("message", "")
            client_ts = entry.get("timestamp", "")
            page_url = entry.get("url", "")

            if entry_type == "audit":
                # Route audit entries through AuditLogger if available
                action = entry.get("action", "CLIENT_EVENT")
                resource = entry.get("resource", page_url)
                status = entry.get("status", "SUCCESS")
                details = entry.get("details")

                if audit_db:
                    audit_db.log_action(
                        user_id=user_name or session_id or client_ip,
                        action=action,
                        resource=resource,
                        status=status,
                        details={
                            **(details if isinstance(details, dict) else {}),
                            "source": "browser",
                            "client_ip": client_ip,
                            "client_ts": client_ts,
                        },
                    )
                else:
                    # Fallback: log as structured INFO
                    logger.info(
                        "[CLIENT:AUDIT] %s %s %s",
                        action, resource, status,
                        extra={
                            "source": "browser",
                            "entry_type": "audit",
                            "client_ip": client_ip,
                            "session_id": session_id,
                            "user_name": user_name,
                            "action": action,
                            "resource": resource,
                            "audit_status": status,
                            "details": details,
                            "client_ts": client_ts,
                        },
                    )
            else:
                # Regular log entry
                logger.log(
                    py_level,
                    "[CLIENT] %s",
                    message,
                    extra={
                        "source": "browser",
                        "entry_type": "log",
                        "client_ip": client_ip,
                        "session_id": session_id,
                        "user_name": user_name,
                        "page_url": page_url,
                        "client_ts": client_ts,
                    },
                )

            ingested += 1

        return json_response({"status": "ok", "ingested": ingested})

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

            logger.info(
                f"[HTTP] Event loop thread started: {self._event_loop_thread.is_alive()}, loop running: {self._event_loop and self._event_loop.is_running()}")
        except Exception as e:
            logger.error(f"Event manager init failed: {e}", exc_info=True)

        self._running = True
        self._server = None

        # Run WSGI server
        try:
            import errno as _errno
            import random as _random
            from waitress import create_server

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

            # P2P standby: if the port is owned by another worker, wait and
            # retry. When the owner dies, this worker takes over the live UI.
            _jitter = getattr(self.config.zmq, "takeover_jitter_max_s", 0.5)
            _retry = max(1.0, getattr(self.config.zmq, "heartbeat_timeout_s", 3.0))
            while self._running:
                try:
                    self._server = create_server(
                        self.wsgi_app,
                        host=host,
                        port=port,
                        threads=self.config.http_worker.max_concurrent,
                        connection_limit=self.config.http_worker.backlog,
                        channel_timeout=self.config.http_worker.timeout,
                        ident="ToolBoxV2",
                    )
                except OSError as e:
                    if e.errno not in (_errno.EADDRINUSE,
                                       getattr(_errno, "WSAEADDRINUSE", 10048)):
                        raise
                    wait_s = _retry + _random.uniform(0.0, _jitter)
                    logger.info(f"[HTTP-P2P] {host}:{port} busy, standby {wait_s:.2f}s")
                    time.sleep(wait_s)
                    continue

                logger.info(f"Serving on http://{host}:{port} (active)")
                self._server.run()  # blocks until close()
                self._server = None
                if self._running:
                    time.sleep(_random.uniform(0.0, _jitter))

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
