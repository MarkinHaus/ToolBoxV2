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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote

from toolboxv2.utils.workers import ZMQEventManager

logger = logging.getLogger(__name__)

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

    @property
    def is_htmx(self) -> bool:
        return self.headers.get("hx-request", "").lower() == "true"

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
            },
            "session": self.session.to_dict() if self.session else {
                "SiID": "", "level": "-1", "spec": "", "user_name": "anonymous",
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

    return ParsedRequest(
        method=method, path=path, query_params=query_params,
        headers=headers, content_type=content_type,
        content_length=content_length, body=body,
        form_data=form_data, json_data=json_data, session=session,
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
# ToolBoxV2 Handler
# ============================================================================

class ToolBoxHandler:
    """Handler for ToolBoxV2 module calls."""

    def __init__(self, app, api_prefix: str = "/api"):
        self.app = app
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
        """Handle API call to ToolBoxV2 module."""
        module_name, function_name = self.parse_api_path(request.path)

        if not module_name:
            return error_response("Missing module name", 400, "BadRequest")

        if not function_name:
            return error_response("Missing function name", 400, "BadRequest")

        # Build kwargs from request
        kwargs = {}

        if request.query_params:
            for k, v in request.query_params.items():
                kwargs[k] = v[0] if len(v) == 1 else v

        if request.form_data:
            kwargs.update(request.form_data)

        if request.json_data and isinstance(request.json_data, dict):
            kwargs.update(request.json_data)

        # Add request context
        kwargs["request"] = request.to_toolbox_request()

        try:
            result = await self.app.a_run_any(
                (module_name, function_name),
                get_results=True,
                **kwargs
            )
            return self._process_result(result, request)
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
                # Binary file download
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
            return json_response(data)

        # Plain data
        if isinstance(result, (dict, list)):
            return json_response(result)

        if isinstance(result, str):
            if result.strip().startswith("<"):
                return html_response(result)
            return json_response({"result": result})

        return json_response({"result": str(result)})


# ============================================================================
# HTTP Worker
# ============================================================================

class HTTPWorker:
    """HTTP Worker with raw WSGI application."""

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
        self._session_manager = None
        self._event_manager: ZMQEventManager | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._running = False

        # Request metrics
        self._metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "latency_sum": 0.0,
        }

    def _init_toolbox(self):
        """Initialize ToolBoxV2 app."""
        if self._app is not None:
            return

        import sys

        # Windows: Use SelectorEventLoop for ZMQ compatibility
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            from toolboxv2 import get_app
            instance_id = f"{self.config.toolbox.instance_id}_{self.worker_id}"
            self._app = get_app(name=instance_id, from_="HTTPWorker")
            logger.info(f"ToolBoxV2 initialized: {instance_id}")
        except Exception as e:
            logger.error(f"ToolBoxV2 init failed: {e}")
            raise

    def _init_session_manager(self):
        """Initialize session manager."""
        from toolboxv2.utils.workers.session import SessionManager

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

    async def _init_event_manager(self):
        """Initialize ZeroMQ event manager."""
        from toolboxv2.utils.workers.event_manager import ZMQEventManager

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

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register ZMQ event handlers."""
        from toolboxv2.utils.workers.event_manager import EventType

        @self._event_manager.on(EventType.CONFIG_RELOAD)
        async def handle_config_reload(event):
            logger.info("Config reload requested")
            # Reload config if needed

        @self._event_manager.on(EventType.SHUTDOWN)
        async def handle_shutdown(event):
            logger.info("Shutdown requested")
            self._running = False

    def wsgi_app(self, environ: Dict, start_response: Callable) -> List[bytes]:
        """Raw WSGI application entry point."""
        start_time = time.time()
        self._metrics["requests_total"] += 1

        try:
            # Add session to environ
            if self._session_manager:
                session = self._session_manager.get_session_from_request_sync(environ)
                environ["tb.session"] = session

            # Parse request
            request = parse_request(environ)

            # Route request
            if self._toolbox_handler and self._toolbox_handler.is_api_request(request.path):
                # API request - run async handler
                status, headers, body = self._run_async(
                    self._toolbox_handler.handle_api_call(request)
                )
            elif request.path == "/health":
                status, headers, body = self._handle_health()
            elif request.path == "/metrics":
                status, headers, body = self._handle_metrics()
            else:
                status, headers, body = error_response("Not Found", 404, "NotFound")

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

            status_line = "500 Internal Server Error"
            response_headers = [("Content-Type", "application/json")]
            start_response(status_line, response_headers)

            return [json.dumps({"error": "InternalError", "message": str(e)}).encode()]

    def _run_async(self, coro) -> Any:
        """Run async coroutine from sync context."""
        return self._app.run_bg_task(coro)

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

        return json_response({
            "worker_id": self.worker_id,
            "requests_total": self._metrics["requests_total"],
            "requests_success": self._metrics["requests_success"],
            "requests_error": self._metrics["requests_error"],
            "avg_latency_ms": avg_latency * 1000,
        })

    def run(self, host: str = None, port: int = None):
        """Run the HTTP worker."""
        host = host or self.config.http_worker.host
        port = port or self.config.http_worker.port

        logger.info(f"Starting HTTP worker {self.worker_id} on {host}:{port}")

        # Initialize components
        self._init_toolbox()
        self._init_session_manager()
        self._toolbox_handler = ToolBoxHandler(self._app, self.config.toolbox.api_prefix)

        # Create event loop

        # Initialize event manager
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self._init_event_manager())
        except Exception as e:
            print(f"Event manager init failed: {e}")
            self._app.run_bg_task(self._init_event_manager())
        self._running = True
        self._server = None

        # Run WSGI server
        try:
            from waitress import create_server

            # Use create_server instead of serve for controllable shutdown
            self._server = create_server(
                self.wsgi_app,
                host=host,
                port=port,
                threads=self.config.http_worker.max_concurrent,
                connection_limit=self.config.http_worker.backlog,
                channel_timeout=self.config.http_worker.timeout,
                ident="ToolBoxV2",
            )

            # Signal handlers - close server on signal
            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, shutting down...")
                self._running = False
                if self._server:
                    self._server.close()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            logger.info(f"Serving on http://{host}:{port}")
            self._server.run()

        except ImportError:
            # Fallback to wsgiref with shutdown support
            from wsgiref.simple_server import make_server, WSGIServer
            import threading
            import selectors

            logger.warning("Using wsgiref (dev only), install waitress for production")

            class ShutdownableWSGIServer(WSGIServer):
                """WSGIServer with clean shutdown support."""

                allow_reuse_address = True
                timeout = 0.5  # Check for shutdown every 0.5s

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._shutdown_event = threading.Event()

                def serve_forever(self):
                    """Handle requests until shutdown."""
                    try:
                        while not self._shutdown_event.is_set():
                            self.handle_request()
                    except Exception:
                        pass

                def shutdown(self):
                    """Signal shutdown."""
                    self._shutdown_event.set()

            self._server = make_server(
                host, port, self.wsgi_app, server_class=ShutdownableWSGIServer
            )

            def signal_handler(sig, frame):
                logger.info(f"Received signal {sig}, shutting down...")
                self._running = False
                if self._server:
                    self._server.shutdown()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

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
        if self._event_manager:
            self._app.run_bg_task(self._event_manager.stop)

        if self._executor:
            self._executor.shutdown(wait=False)


        logger.info(f"HTTP worker {self.worker_id} stopped")


# Required import for HTTPStatus
from http import HTTPStatus

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

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load config
    from toolboxv2.utils.workers.config import load_config
    config = load_config(args.config)

    # Worker ID
    worker_id = args.worker_id or f"http_{os.getpid()}"

    # Run worker
    worker = HTTPWorker(worker_id, config)
    worker.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
