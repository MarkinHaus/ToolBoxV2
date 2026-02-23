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
            return self.json_data.get("user_id") or self.json_data.get("clerk_user_id") or self.json_data.get("Username")
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

        # Handle internal HUD messages
        if handler_id == "__hud_internal__":
            await self._handle_hud_message(event, payload, conn_id, user_id, session_id)
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
                provider_user_id = event.payload.get("user_id", event.payload.get("clerk_user_id", ""))

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

    async def _handle_hud_message(self, event: Event, payload: dict, conn_id: str, user_id: str, session_id: str):
        """Handle HUD-specific WebSocket messages (widget_action, get_widget, etc.)."""
        msg_type = payload.get("type")
        self._logger.info(f"[HUD] Handling message type: {msg_type}")

        try:
            # Build request object with session data for access control
            user_level = int(event.payload.get("level", AccessLevel.NOT_LOGGED_IN))
            authenticated = event.payload.get("authenticated", False)
            provider_user_id = event.payload.get("user_id", event.payload.get("clerk_user_id", ""))

            request_dict = {
                "request": {
                    "content_type": "application/json",
                    "headers": {},
                    "method": "WEBSOCKET",
                    "path": "/ws/hud",
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

            if msg_type == "widget_action":
                await self._handle_widget_action(payload, conn_id, request)
            elif msg_type == "get_widget":
                await self._handle_get_widget(payload, conn_id, request)
            elif msg_type == "get_widgets":
                await self._handle_get_widgets(conn_id, request)
            elif msg_type == "get_status":
                await self._handle_get_status(conn_id)
            else:
                self._logger.warning(f"[HUD] Unknown message type: {msg_type}")

        except Exception as e:
            self._logger.error(f"[HUD] Error handling message: {e}", exc_info=True)
            await self.app.ws_send(conn_id, {
                "type": "error",
                "message": str(e),
            })

    async def _handle_widget_action(self, payload: dict, conn_id: str, request: RequestData):
        """Handle widget action from HUD."""
        widget_id = payload.get("widget_id")
        action = payload.get("action")
        action_payload = payload.get("payload", {})
        action_id = payload.get("action_id")

        self._logger.info(f"[HUD] Widget action: {widget_id}.{action}")

        if not widget_id or not action:
            await self.app.ws_send(conn_id, {
                "type": "widget_response",
                "action_id": action_id,
                "widget_id": widget_id,
                "error": "Missing widget_id or action",
            })
            return

        # Try to find and call the widget action handler
        # Convention: Module has a function named hud_action or hud_{widget_id}_action
        # The function receives: action, payload, conn_id, request
        try:
            # First try: {widget_id}.hud_action
            result = await self.app.a_run_any(
                (widget_id, "hud_action"),
                action=action,
                payload=action_payload,
                conn_id=conn_id,
                request=request,
            )

            # Send response back
            response = {
                "type": "widget_response",
                "action_id": action_id,
                "widget_id": widget_id,
                "action": action,
                "result": result if isinstance(result, dict) else {"data": result},
            }
            await self.app.ws_send(conn_id, response)

        except Exception as e:
            self._logger.error(f"[HUD] Widget action error: {e}", exc_info=True)
            await self.app.ws_send(conn_id, {
                "type": "widget_response",
                "action_id": action_id,
                "widget_id": widget_id,
                "error": str(e),
            })

    async def _handle_get_widget(self, payload: dict, conn_id: str, request: RequestData):
        """Handle request for a single widget."""
        widget_id = payload.get("widget_id")

        if not widget_id:
            return

        self._logger.info(f"[HUD] Get widget: {widget_id}")

        try:
            # Try to call the widget's render function
            # Convention: hud_{widget_name} function returns HTML
            result = await self.app.a_run_any(
                (widget_id, f"hud_{widget_id}"),
                request=request,
            )

            html = result if isinstance(result, str) else str(result)

            await self.app.ws_send(conn_id, {
                "type": "single_widget_update",
                "widget_id": widget_id,
                "html": html,
            })

        except Exception as e:
            self._logger.error(f"[HUD] Get widget error: {e}", exc_info=True)

    async def _handle_get_widgets(self, conn_id: str, request: RequestData):
        """Handle request for all widgets."""
        self._logger.info(f"[HUD] Get all widgets")

        try:
            # Get HUD functions from CloudM
            result = await self.app.a_run_any(
                ("CloudM", "get_hud_functions"),
                request=request,
            )

            widgets = []
            if isinstance(result, list):
                for func in result:
                    widgets.append({
                        "widget_id": func.get("func_name", ""),
                        "title": func.get("display_name", "Widget"),
                        "mod_name": func.get("mod_name", ""),
                        "path": func.get("path", ""),
                    })

            await self.app.ws_send(conn_id, {
                "type": "widgets",
                "widgets": widgets,
            })

        except Exception as e:
            self._logger.error(f"[HUD] Get widgets error: {e}", exc_info=True)
            await self.app.ws_send(conn_id, {
                "type": "widgets",
                "widgets": [],
            })

    async def _handle_get_status(self, conn_id: str):
        """Handle status request."""
        self._logger.info(f"[HUD] Get status")

        try:
            status = {
                "type": "status",
                "running": True,
                "worker_running": True,
                "active_mods": len(self.app.functions) if hasattr(self.app, 'functions') else 0,
                "is_remote": False,
            }
            await self.app.ws_send(conn_id, status)

        except Exception as e:
            self._logger.error(f"[HUD] Get status error: {e}", exc_info=True)

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
            elif self._toolbox_handler and self._toolbox_handler.is_api_request(request.path):
                # API endpoints
                status, headers, body = self._run_async(
                    self._toolbox_handler.handle_api_call(request)
                )
            elif request.path == "/health":
                status, headers, body = self._handle_health()
            elif request.path == "/metrics":
                status, headers, body = self._handle_metrics()
            elif request.path == "/api/client-logs":
                status, headers, body = self._handle_client_logs(request)
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
