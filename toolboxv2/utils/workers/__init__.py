"""
ToolBoxV2 Worker System

High-performance Python workers for ToolBoxV2:
- HTTP Worker: Raw WSGI, async request processing
- WS Worker: Minimal overhead WebSocket handler
- Event Manager: ZeroMQ-based IPC
- Session: Signed cookies (stateless)
- Manager: Nginx config, process orchestration, web UI

Usage:
    # Start all workers
    python -m tbv2_workers.cli_worker_manager start

    # Or import components
    from tbv2_workers import HTTPWorker, WSWorker, SessionManager
"""

__version__ = "0.1.0"

from .config import Config, load_config, get_default_config_yaml, main as cli_config
from .session import SessionData, SessionManager, SignedCookieSession, main as cli_session
from .event_manager import ZMQEventManager, Event, EventType, main as cli_event
from .server_worker import HTTPWorker, ParsedRequest, json_response, html_response, main as cli_http_worker
from .ws_worker import WSWorker, ConnectionManager, main as cli_ws_worker

__all__ = [
    # Config
    "Config",
    "load_config",
    "get_default_config_yaml",
    # Session
    "SessionData",
    "SessionManager",
    "SignedCookieSession",
    # Events
    "ZMQEventManager",
    "Event",
    "EventType",
    # Workers
    "HTTPWorker",
    "WSWorker",
    "ConnectionManager",
    # Helpers
    "ParsedRequest",
    "json_response",
    "html_response",
    # CLI
    "cli_config",
    "cli_session",
    "cli_event",
    "cli_http_worker",
    "cli_ws_worker",
]
