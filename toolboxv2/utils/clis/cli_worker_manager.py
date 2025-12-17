#!/usr/bin/env python3
"""
cli_worker_manager.py - Complete Worker Manager for ToolBoxV2

Cross-Platform (Windows/Linux/macOS) Worker Orchestration:
- Nginx installation and high-performance configuration
- HTTP and WebSocket worker processes
- ZeroMQ event broker with real metrics
- Zero-downtime rolling updates
- Cluster mode with remote workers
- SSL auto-discovery (Let's Encrypt)
- Health monitoring with active probing
- Minimal web UI
- CLI interface
"""

import argparse
import asyncio
import http.client
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# ZMQ optional import
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    zmq = None
    ZMQ_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# Constants & OS Detection
# ============================================================================

SYSTEM = platform.system().lower()
IS_WINDOWS = SYSTEM == "windows"
IS_LINUX = SYSTEM == "linux"
IS_MACOS = SYSTEM == "darwin"

if IS_WINDOWS:
    DEFAULT_NGINX_PATHS = [r"C:\nginx\nginx.exe", r"C:\Program Files\nginx\nginx.exe"]
    DEFAULT_CONF_PATH = r"C:\nginx\conf\nginx.conf"
    SOCKET_PREFIX = None
elif IS_MACOS:
    DEFAULT_NGINX_PATHS = ["/usr/local/bin/nginx", "/opt/homebrew/bin/nginx"]
    DEFAULT_CONF_PATH = "/usr/local/etc/nginx/nginx.conf"
    SOCKET_PREFIX = "/tmp"
else:
    DEFAULT_NGINX_PATHS = ["/usr/sbin/nginx", "/usr/local/sbin/nginx"]
    DEFAULT_CONF_PATH = "/etc/nginx/nginx.conf"
    SOCKET_PREFIX = "/tmp"


class WorkerType(str, Enum):
    HTTP = "http"
    WS = "ws"
    BROKER = "broker"


class WorkerState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class WorkerMetrics:
    requests: int = 0
    connections: int = 0
    errors: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    avg_latency_ms: float = 0.0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    last_update: float = 0.0


@dataclass
class WorkerInfo:
    worker_id: str
    worker_type: WorkerType
    pid: int
    port: int
    socket_path: str | None = None
    state: WorkerState = WorkerState.STOPPED
    started_at: float = 0.0
    restart_count: int = 0
    last_health_check: float = 0.0
    health_latency_ms: float = 0.0
    healthy: bool = False
    node: str = "local"
    metrics: WorkerMetrics = field(default_factory=WorkerMetrics)

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type.value,
            "pid": self.pid,
            "port": self.port,
            "socket_path": self.socket_path,
            "state": self.state.value,
            "started_at": self.started_at,
            "restart_count": self.restart_count,
            "last_health_check": self.last_health_check,
            "health_latency_ms": self.health_latency_ms,
            "healthy": self.healthy,
            "node": self.node,
            "uptime": time.time() - self.started_at if self.started_at > 0 else 0,
            "metrics": {
                "requests": self.metrics.requests,
                "connections": self.metrics.connections,
                "errors": self.metrics.errors,
                "avg_latency_ms": self.metrics.avg_latency_ms,
                "memory_mb": self.metrics.memory_mb,
            },
        }


@dataclass
class ClusterNode:
    node_id: str
    host: str
    port: int
    secret: str
    healthy: bool = False
    last_seen: float = 0.0
    workers: List[Dict] = field(default_factory=list)


# ============================================================================
# SSL Certificate Discovery
# ============================================================================

class SSLManager:
    def __init__(self, domain: str = None):
        self.domain = domain
        self._cert_path: str | None = None
        self._key_path: str | None = None

    def discover(self) -> bool:
        env_cert = os.environ.get("NGINX_CERT_PATH") or os.environ.get("SSL_CERT_PATH")
        env_key = os.environ.get("NGINX_KEY_PATH") or os.environ.get("SSL_KEY_PATH")

        if env_cert and env_key and os.path.exists(env_cert) and os.path.exists(env_key):
            self._cert_path, self._key_path = env_cert, env_key
            return True

        for cert_base in ["/etc/letsencrypt/live"]:
            if not os.path.isdir(cert_base):
                continue
            try:
                for name in os.listdir(cert_base):
                    cert = os.path.join(cert_base, name, "fullchain.pem")
                    key = os.path.join(cert_base, name, "privkey.pem")
                    if os.path.exists(cert) and os.path.exists(key):
                        self._cert_path, self._key_path = cert, key
                        return True
            except PermissionError:
                continue
        return False

    @property
    def available(self) -> bool:
        return self._cert_path is not None

    @property
    def cert_path(self) -> str | None:
        return self._cert_path

    @property
    def key_path(self) -> str | None:
        return self._key_path


# ============================================================================
# Nginx Manager
# ============================================================================

class NginxManager:
    def __init__(self, config):
        self.config = config.nginx
        self._manager = config.manager
        self._nginx_path = self._find_nginx()
        self._ssl = SSLManager(getattr(self.config, 'server_name', None))
        self._ssl.discover()

    def _find_nginx(self) -> str | None:
        env_path = os.environ.get("NGINX_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
        found = shutil.which("nginx")
        if found:
            return found
        for path in DEFAULT_NGINX_PATHS:
            if os.path.exists(path):
                return path
        return None

    def is_installed(self) -> bool:
        return self._nginx_path is not None

    def get_version(self) -> str | None:
        if not self._nginx_path:
            return None
        try:
            result = subprocess.run([self._nginx_path, "-v"], capture_output=True, text=True)
            return result.stderr.strip()
        except Exception:
            return None

    def install(self) -> bool:
        if IS_WINDOWS:
            logger.error("Windows: Download nginx from https://nginx.org/en/download.html")
            return False
        if IS_MACOS:
            try:
                subprocess.run(["brew", "install", "nginx"], check=True, capture_output=True)
                self._nginx_path = self._find_nginx()
                return self._nginx_path is not None
            except Exception:
                return False
        try:
            if shutil.which("apt-get"):
                subprocess.run(["sudo", "apt-get", "update"], check=True, capture_output=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "nginx"], check=True, capture_output=True)
            elif shutil.which("yum"):
                subprocess.run(["sudo", "yum", "install", "-y", "nginx"], check=True, capture_output=True)
            elif shutil.which("dnf"):
                subprocess.run(["sudo", "dnf", "install", "-y", "nginx"], check=True, capture_output=True)
            else:
                return False
            self._nginx_path = self._find_nginx()
            return self._nginx_path is not None
        except subprocess.CalledProcessError:
            return False

    def generate_config(
        self,
        http_ports: List[int],
        ws_ports: List[int],
        http_sockets: List[str] = None,
        ws_sockets: List[str] = None,
        remote_nodes: List[Tuple[str, int]] = None,
    ) -> str:
        """
        Generate Nginx configuration for ToolBoxV2 worker system.

        Features:
        - HTTP/WS upstream load balancing
        - Auth endpoint routing (secured)
        - API endpoint routing with access control
        - Unix socket support (Linux/macOS)
        - Rate limiting (different zones for auth/api)
        - Static file serving from dist/
        - SSL/TLS support
        - Gzip compression
        - WebSocket proxying with session auth
        - SSE streaming support

        Args:
            http_ports: List of HTTP worker ports
            ws_ports: List of WebSocket worker ports
            http_sockets: Optional Unix socket paths for HTTP workers
            ws_sockets: Optional Unix socket paths for WS workers
            remote_nodes: Optional list of (host, port) tuples for remote backends

        Returns:
            Complete nginx.conf content as string
        """
        cfg = self.config
        remote_nodes = remote_nodes or []
        http_sockets = http_sockets or []
        ws_sockets = ws_sockets or []

        http_servers, ws_server_list = [], []

        # Unix sockets on Linux/macOS (preferred for performance)
        if (IS_LINUX or IS_MACOS) and http_sockets:
            for sock in http_sockets:
                if sock:
                    http_servers.append(
                        f"server unix:{sock} weight=1 max_fails=3 fail_timeout=80s;"
                    )
        if (IS_LINUX or IS_MACOS) and ws_sockets:
            for sock in ws_sockets:
                if sock:
                    ws_server_list.append(f"server unix:{sock};")

        # TCP fallback / Windows
        for port in http_ports:
            if not http_sockets or IS_WINDOWS:
                http_servers.append(
                    f"server 127.0.0.1:{port} weight=1 max_fails=3 fail_timeout=80s;"
                )
        for port in ws_ports:
            if not ws_sockets or IS_WINDOWS:
                ws_server_list.append(f"server 127.0.0.1:{port};")

        # Remote nodes (backup servers)
        for host, port in remote_nodes:
            http_servers.append(f"server {host}:{port} backup;")

        http_upstream = (
            "\n        ".join(http_servers) if http_servers else "server 127.0.0.1:8000;"
        )
        ws_upstream = (
            "\n        ".join(ws_server_list)
            if ws_server_list
            else "server 127.0.0.1:8100;"
        )

        # OS-specific optimizations
        if IS_LINUX:
            event_use = "epoll"
            worker_processes = "auto"
            worker_rlimit = "worker_rlimit_nofile 65535;"
            worker_connections = "4096"
        elif IS_MACOS:
            event_use = "kqueue"
            worker_processes = "auto"
            worker_rlimit = "worker_rlimit_nofile 65535;"
            worker_connections = "4096"
        else:  # Windows
            event_use = "select"
            worker_processes = "1"
            worker_rlimit = ""
            worker_connections = "1024"

        # SSL configuration
        use_ssl = self._ssl.available and getattr(cfg, "ssl_enabled", False)
        ssl_block = ""
        ssl_redirect = ""
        if use_ssl:
            ssl_port = getattr(cfg, "listen_ssl_port", 443)
            ssl_block = f"""
            listen {ssl_port} ssl http2;
            ssl_certificate {self._ssl.cert_path};
            ssl_certificate_key {self._ssl.key_path};
            ssl_protocols TLSv1.2 TLSv1.3;
            ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
            ssl_prefer_server_ciphers off;
            ssl_session_cache shared:SSL:10m;
            ssl_session_timeout 1d;
            ssl_session_tickets off;"""

            listen_port = getattr(cfg, "listen_port", 80)
            ssl_redirect = f"""
        # HTTP to HTTPS redirect
        server {{
            listen {listen_port};
            server_name {getattr(cfg, "server_name", "_")};
            return 301 https://$host$request_uri;
        }}
    """

        listen_port = getattr(cfg, "listen_port", 80)
        upstream_http = getattr(cfg, "upstream_http", "toolbox_http")
        upstream_ws = getattr(cfg, "upstream_ws", "toolbox_ws")
        server_name = getattr(cfg, "server_name", "_")

        # Paths based on OS
        if IS_WINDOWS:
            mime_include = "include mime.types;"
            log_path = "logs"
            pid_directive = ""
        else:
            mime_include = "include /etc/nginx/mime.types;"
            log_path = "/var/log/nginx"
            pid_directive = "pid /run/nginx.pid;"

        # Rate limiting configuration
        rate_limit_enabled = getattr(cfg, "rate_limit_enabled", True)
        rate_limit_zone = getattr(cfg, "rate_limit_zone", "tb_limit")
        rate_limit_rate = getattr(cfg, "rate_limit_rate", "10r/s")
        rate_limit_burst = getattr(cfg, "rate_limit_burst", 20)

        # Auth rate limit (stricter)
        auth_rate_limit_rate = getattr(cfg, "auth_rate_limit_rate", "5r/s")
        auth_rate_limit_burst = getattr(cfg, "auth_rate_limit_burst", 10)

        rate_limit_zone_block = ""
        rate_limit_api_block = ""
        rate_limit_auth_block = ""
        if rate_limit_enabled:
            rate_limit_zone_block = f"""
        # Rate limiting zones
        limit_req_zone $binary_remote_addr zone={rate_limit_zone}:10m rate={rate_limit_rate};
        limit_req_zone $binary_remote_addr zone=tb_auth_limit:10m rate={auth_rate_limit_rate};
        limit_req_status 429;"""
            rate_limit_api_block = f"""
                limit_req zone={rate_limit_zone} burst={rate_limit_burst} nodelay;"""
            rate_limit_auth_block = f"""
                limit_req zone=tb_auth_limit burst={auth_rate_limit_burst} nodelay;"""

        # Static files configuration
        static_enabled = getattr(cfg, "static_enabled", True)
        static_root = getattr(cfg, "static_root", "./dist")

        static_block = ""
        if static_enabled:
            static_block = f"""
            # Static files (SPA frontend)
            location / {{
                root {static_root};
                try_files $uri $uri/ /index.html;

                # Cache static assets
                location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {{
                    expires 1h;
                    add_header Cache-Control "public, immutable";
                    access_log off;
                }}

                # Don't cache HTML
                location ~* \\.html$ {{
                    expires -1;
                    add_header Cache-Control "no-store, no-cache, must-revalidate";
                }}
            }}"""

        # Main listen directive
        main_listen = f"listen {listen_port};" if not (use_ssl and ssl_redirect) else ""
        if use_ssl and not ssl_redirect:
            main_listen = f"listen {listen_port};"

        # Auth endpoints block
        auth_endpoints_block = self._generate_auth_endpoints_block(
            upstream_http, rate_limit_auth_block
        )

        # API endpoints block with security
        api_endpoints_block = self._generate_api_endpoints_block(
            upstream_http, rate_limit_api_block
        )

        # WebSocket block with session validation
        ws_endpoints_block = self._generate_ws_endpoints_block(
            upstream_ws, upstream_http
        )

        admin_ui_port = getattr(self._manager, "web_ui_port", 9002)
        admin_block = self._generate_admin_ui_block(admin_ui_port)


        return f"""# ToolBoxV2 Nginx Configuration - {SYSTEM}
    # Generated automatically - do not edit manually
    # Regenerate with: tb manager nginx-config

    {pid_directive}
    worker_processes {worker_processes};
    {worker_rlimit}

    events {{
        worker_connections {worker_connections};
        use {event_use};
        multi_accept on;
    }}

    http {{
        {mime_include}
        default_type application/octet-stream;

        # Logging
        log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                        '$status $body_bytes_sent "$http_referer" '
                        '"$http_user_agent" "$http_x_forwarded_for" '
                        'rt=$request_time uct="$upstream_connect_time" '
                        'uht="$upstream_header_time" urt="$upstream_response_time"';

        access_log {log_path}/toolboxv2_access.log main;
        error_log {log_path}/toolboxv2_error.log warn;

        # Performance
        sendfile on;
        tcp_nopush on;
        tcp_nodelay on;
        keepalive_timeout 65;
        keepalive_requests 1000;
        types_hash_max_size 2048;

        # Buffers
        client_body_buffer_size 128k;
        client_max_body_size 50M;
        client_header_buffer_size 1k;
        large_client_header_buffers 4 16k;

        # Timeouts
        client_body_timeout 60s;
        client_header_timeout 60s;
        send_timeout 60s;

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_proxied any;
        gzip_comp_level 6;
        gzip_min_length 1000;
        gzip_types
            text/plain
            text/css
            text/xml
            text/javascript
            application/json
            application/javascript
            application/x-javascript
            application/xml
            application/xml+rss
            application/atom+xml
            image/svg+xml;
    {rate_limit_zone_block}

        # HTTP Backend Upstream
        upstream {upstream_http} {{
            least_conn;
            {http_upstream}
            keepalive 128;
            keepalive_requests 10000;
            keepalive_timeout 60s;
        }}

        # WebSocket Backend Upstream
        upstream {upstream_ws} {{
            # Consistent hashing for sticky sessions
            hash $request_uri consistent;
            {ws_upstream}
        }}
    {ssl_redirect}
        # Main Server Block
        server {{
            {main_listen}{ssl_block}
            server_name {server_name};

            # Security headers
            add_header X-Frame-Options "SAMEORIGIN" always;
            add_header X-Content-Type-Options "nosniff" always;
            add_header X-XSS-Protection "1; mode=block" always;
            add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    {static_block}
    {auth_endpoints_block}
    {api_endpoints_block}

            # SSE / Streaming endpoints
            location /sse/ {{
                proxy_pass http://{upstream_http};
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

                # Disable buffering for streaming
                proxy_buffering off;
                proxy_cache off;
                chunked_transfer_encoding on;

                proxy_read_timeout 3600s;
                proxy_send_timeout 3600s;
            }}
    {ws_endpoints_block}

            # Health check endpoint (no rate limit)
            location /health {{
                proxy_pass http://{upstream_http}/health;
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                access_log off;
            }}

            # Metrics endpoint (restricted access recommended)
            location /metrics {{
                proxy_pass http://{upstream_http}/metrics;
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                # Uncomment to restrict access:
                # allow 127.0.0.1;
                # deny all;
            }}


            {admin_block}

            # Error pages
            error_page 500 502 503 504 /50x.html;
            location = /50x.html {{
                root {static_root if static_enabled else "/usr/share/nginx/html"};
                internal;
            }}

            error_page 429 /429.html;
            location = /429.html {{
                default_type application/json;
                return 429 '{{"error": "TooManyRequests", "message": "Rate limit exceeded"}}';
            }}

            error_page 401 /401.html;
            location = /401.html {{
                default_type application/json;
                return 401 '{{"error": "Unauthorized", "message": "Authentication required"}}';
            }}

            error_page 403 /403.html;
            location = /403.html {{
                default_type application/json;
                return 403 '{{"error": "Forbidden", "message": "Access denied"}}';
            }}
        }}

    }}
    """

    def _generate_auth_endpoints_block(
        self, upstream_http: str, rate_limit_block: str
    ) -> str:
        """Generate auth endpoint configuration."""
        return f"""
            # ============================================================
            # Auth Endpoints (Clerk Integration)
            # ============================================================

            # Validate session with Clerk token (POST only)
            location = /validateSession {{
                limit_except POST {{
                    deny all;
                }}
    {rate_limit_block}
                proxy_pass http://{upstream_http}/validateSession;
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header Content-Type $content_type;

                proxy_connect_timeout 10s;
                proxy_send_timeout 30s;
                proxy_read_timeout 30s;
            }}

            # Check if session is valid (GET only)
            location = /IsValidSession {{
                limit_except GET {{
                    deny all;
                }}
                proxy_pass http://{upstream_http}/IsValidSession;
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header Cookie $http_cookie;

                proxy_connect_timeout 5s;
                proxy_send_timeout 10s;
                proxy_read_timeout 10s;
            }}

            # Logout endpoint (POST only)
            location = /web/logoutS {{
                limit_except POST {{
                    deny all;
                }}
    {rate_limit_block}
                proxy_pass http://{upstream_http}/web/logoutS;
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header Cookie $http_cookie;

                proxy_connect_timeout 5s;
                proxy_send_timeout 10s;
                proxy_read_timeout 10s;
            }}

            # Get user data endpoint (GET only, requires valid session)
            location = /api_user_data {{
                limit_except GET {{
                    deny all;
                }}
                proxy_pass http://{upstream_http}/api_user_data;
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header Cookie $http_cookie;
                proxy_set_header Authorization $http_authorization;

                proxy_connect_timeout 5s;
                proxy_send_timeout 15s;
                proxy_read_timeout 15s;
            }}

            # Logout redirect page (static or handled by frontend)
            location = /web/logout {{
                # Can be handled by SPA or redirect
                try_files $uri /index.html;
            }}"""

    def _generate_api_endpoints_block(
        self, upstream_http: str, rate_limit_block: str
    ) -> str:
        """Generate API endpoint configuration with security."""
        return f"""
            # ============================================================
            # API Endpoints
            # Access control is handled by the workers based on:
            # - open_modules: Publicly accessible modules
            # - open* functions: Functions starting with 'open' are public
            # - User level: -1=Admin, 0=not logged in, 1=logged in, 2=trusted
            # ============================================================

            location /api/ {{
                proxy_pass http://{upstream_http};
                proxy_http_version 1.1;
                proxy_set_header Connection "";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header X-Request-ID $request_id;
                proxy_set_header Cookie $http_cookie;
                proxy_set_header Authorization $http_authorization;

                # Pass session cookie for auth validation
                proxy_pass_header Set-Cookie;

                # Buffering for normal API requests
                proxy_buffering on;
                proxy_buffer_size 4k;
                proxy_buffers 8 16k;
                proxy_busy_buffers_size 24k;

                proxy_connect_timeout 10s;
                proxy_send_timeout 30s;
                proxy_read_timeout 30s;
    {rate_limit_block}
            }}"""

    def _generate_ws_endpoints_block(
        self, upstream_ws: str, upstream_http: str
    ) -> str:
        """Generate WebSocket endpoint configuration with auth subrequest."""
        return f"""
            # ============================================================
            # WebSocket Endpoint
            # Auth validated via subrequest to /IsValidSession
            # ============================================================

            # Internal auth check endpoint
            location = /_ws_auth {{
                internal;
                proxy_pass http://{upstream_http}/IsValidSession;
                proxy_pass_request_body off;
                proxy_set_header Content-Length "";
                proxy_set_header X-Original-URI $request_uri;
                proxy_set_header Cookie $http_cookie;
            }}

            # Main WebSocket endpoint
            location /ws {{
                # Optional: Require authentication for WebSocket
                # Uncomment the following lines to enable auth check:
                # auth_request /_ws_auth;
                # auth_request_set $auth_status $upstream_status;

                proxy_pass http://{upstream_ws};
                proxy_http_version 1.1;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header Cookie $http_cookie;

                # WebSocket specific timeouts
                proxy_connect_timeout 10s;
                proxy_send_timeout 3600s;
                proxy_read_timeout 3600s;

                # Disable buffering for WebSocket
                proxy_buffering off;
            }}

            # WebSocket with explicit path routing (e.g., /ws/Module/handler)
            location ~ ^/ws/([^/]+)/([^/]+)$ {{
                # Optional: Require authentication
                # auth_request /_ws_auth;

                proxy_pass http://{upstream_ws};
                proxy_http_version 1.1;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_set_header Cookie $http_cookie;

                proxy_connect_timeout 10s;
                proxy_send_timeout 3600s;
                proxy_read_timeout 3600s;
                proxy_buffering off;
            }}"""

    def _generate_admin_ui_block(self, web_port: int) -> str:
        """
        Generates a password-protected admin UI route on /admin/
        exposed on config.manager.web_ui_port.

        Password is read from ENV ADMIN_UI_PASSWORD.
        Proxies to DB (9000) and Worker Manager (9001) internally.
        Admin index must be outside static_root.
        """

        import os

        pwd = os.environ.get("ADMIN_UI_PASSWORD", "")
        if not pwd:
            raise RuntimeError("Environment variable ADMIN_UI_PASSWORD is missing.")

        # htpasswd hash generieren (bcrypt)
        from platform import system

        if system() == "Windows":
            import bcrypt, toolboxv2
            hashed = bcrypt.hashpw(
                pwd.encode("utf-8"), bcrypt.gensalt(rounds=12)
            ).decode()
            admin_htpasswd = toolboxv2.get_app().appdata + "/admin_htpasswd"
            admin_root = toolboxv2.get_app().appdata + "/admin_ui"
            auth_basic = ""
        else:
            import crypt

            hashed = crypt.crypt(pwd, crypt.mksalt(crypt.METHOD_BLOWFISH))
            admin_htpasswd = "/etc/nginx/admin_htpasswd"
            admin_root = "/var/lib/toolboxv2/admin_ui"

            auth_basic = f'auth_basic "Restricted Admin UI";\n                    auth_basic_user_file {admin_htpasswd};'

        # htpasswd speichern
        with open(admin_htpasswd, "w") as f:
            f.write(f"admin:{hashed}\n")

        # Admin root directory erstellen falls nicht vorhanden
        os.makedirs(admin_root, exist_ok=True)

        self._populate_admin_ui(admin_root, manager_port=web_port)

        return f"""
            # Admin UI Server (Basic Auth protected)
                # Admin UI mit Basic Auth
                location /admin/ {{
                    {auth_basic}

                    # Admin static files (außerhalb static_root!)
                    root {admin_root};
                    try_files $uri $uri/ /admin/index.html;
                }}

                # Proxy zu DB auf Port 9000 (nur mit Auth)
                location /admin/db/ {{
                    {auth_basic}

                    proxy_pass http://127.0.0.1:9000/;
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                }}

                # Proxy zu Worker Manager auf Port {web_port} (nur mit Auth)
                location /admin/manager/ {{
                    {auth_basic}

                    proxy_pass http://127.0.0.1:{web_port}/;
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                }}

        """

    def _populate_admin_ui(
        self,
        admin_root: str,
        minio_port: int = 9000,
        minio_console_port: int = 9001,
        manager_port: int = 9002,
    ) -> None:
        """
        Populates admin_ui directory with a minimal HUNIZED UI if not already present.
        Creates index.html that directly fetches from MinIO and Manager APIs.
        """
        import os

        admin_index = os.path.join(admin_root, "admin", "index.html")

        # Nur erstellen wenn noch nicht vorhanden
        if os.path.exists(admin_index):
            return

        # Directory struktur erstellen
        os.makedirs(os.path.dirname(admin_index), exist_ok=True)

        # Minimal HUNIZED Admin UI mit direkten API Calls
        html_content = f"""<!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ToolBoxV2 Admin</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0a0a0a;
                color: #e0e0e0;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }}

            header {{
                background: #111;
                border-bottom: 1px solid #222;
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            h1 {{
                font-size: 1.2rem;
                font-weight: 600;
                color: #fff;
            }}

            .tabs {{
                display: flex;
                gap: 0.5rem;
            }}

            .tab {{
                padding: 0.5rem 1rem;
                background: transparent;
                border: 1px solid #333;
                border-radius: 6px;
                color: #999;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.9rem;
            }}

            .tab:hover {{
                background: #1a1a1a;
                border-color: #444;
                color: #fff;
            }}

            .tab.active {{
                background: #2563eb;
                border-color: #2563eb;
                color: #fff;
            }}

            .content {{
                flex: 1;
                padding: 2rem;
                overflow-y: auto;
            }}

            .panel {{
                display: none;
            }}

            .panel.active {{
                display: block;
            }}

            .status {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.85rem;
                color: #666;
            }}

            .status-dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #22c55e;
                animation: pulse 2s infinite;
            }}

            .status-dot.error {{
                background: #ef4444;
            }}

            .status-dot.warning {{
                background: #f59e0b;
            }}

            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}

            .card {{
                background: #111;
                border: 1px solid #222;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1rem;
            }}

            .card h2 {{
                font-size: 1rem;
                margin-bottom: 1rem;
                color: #fff;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .badge {{
                padding: 0.25rem 0.75rem;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 500;
            }}

            .worker-list {{
                display: grid;
                gap: 1rem;
            }}

            .worker-item {{
                background: #0a0a0a;
                border: 1px solid #222;
                border-radius: 6px;
                padding: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: start;
                transition: border-color 0.2s;
            }}

            .worker-item:hover {{
                border-color: #333;
            }}

            .worker-item.unhealthy {{
                border-color: #ef4444;
            }}

            .worker-main {{
                flex: 1;
            }}

            .worker-header {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
            }}

            .worker-name {{
                font-weight: 600;
                color: #fff;
                font-family: 'Courier New', monospace;
            }}

            .worker-type {{
                padding: 0.125rem 0.5rem;
                background: #2563eb;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
            }}

            .worker-type.ws {{
                background: #8b5cf6;
            }}

            .worker-details {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 0.5rem;
                font-size: 0.85rem;
                color: #666;
            }}

            .worker-detail {{
                display: flex;
                flex-direction: column;
            }}

            .worker-detail-label {{
                font-size: 0.75rem;
                color: #555;
            }}

            .worker-detail-value {{
                color: #e0e0e0;
                font-family: 'Courier New', monospace;
            }}

            .worker-detail-value.healthy {{
                color: #22c55e;
            }}

            .worker-detail-value.unhealthy {{
                color: #ef4444;
            }}

            .worker-actions {{
                display: flex;
                gap: 0.5rem;
            }}

            button {{
                padding: 0.5rem 1rem;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 6px;
                color: #e0e0e0;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.85rem;
            }}

            button:hover {{
                background: #222;
                border-color: #444;
            }}

            button.danger {{
                border-color: #ef4444;
                color: #ef4444;
            }}

            button.danger:hover {{
                background: #ef4444;
                color: #fff;
            }}

            .loading {{
                text-align: center;
                padding: 2rem;
                color: #666;
            }}

            .error {{
                background: #1a0a0a;
                border: 1px solid #ef4444;
                border-radius: 6px;
                padding: 1rem;
                color: #ef4444;
            }}

            .minio-link {{
                display: inline-block;
                padding: 0.75rem 1.5rem;
                background: #c72c48;
                border-radius: 6px;
                color: #fff;
                text-decoration: none;
                font-weight: 500;
                transition: all 0.2s;
            }}

            .minio-link:hover {{
                background: #a81d38;
            }}

            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }}

            .stat-card {{
                background: #0a0a0a;
                border: 1px solid #222;
                border-radius: 6px;
                padding: 1rem;
            }}

            .stat-value {{
                font-size: 1.5rem;
                font-weight: 600;
                color: #2563eb;
            }}

            .stat-label {{
                font-size: 0.85rem;
                color: #666;
                margin-top: 0.25rem;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>ToolBoxV2 Admin</h1>
            <div class="tabs">
                <button class="tab active" onclick="switchTab('manager')">Workers</button>
                <button class="tab" onclick="switchTab('minio')">MinIO Storage</button>
            </div>
            <div class="status">
                <span class="status-dot" id="status-dot"></span>
                <span id="status-text">Online</span>
            </div>
        </header>

        <div class="content">
            <!-- Worker Manager Panel -->
            <div id="manager-panel" class="panel active">
                <div class="card">
                    <h2>
                        Worker Status
                        <span class="badge" id="worker-count">0 Workers</span>
                    </h2>
                    <div id="manager-content" class="loading">Loading...</div>
                </div>
            </div>

            <!-- MinIO Panel -->
            <div id="minio-panel" class="panel">
                <div class="card">
                    <h2>MinIO Object Storage</h2>
                    <p style="color: #666; margin-bottom: 1rem;">
                        MinIO Console für Bucket Management und Monitoring
                    </p>
                    <a href="http://127.0.0.1:{minio_console_port}" target="_blank" class="minio-link">
                        Open MinIO Console
                    </a>
                    <div id="minio-stats" class="stats-grid">
                        <!-- Stats werden hier geladen -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            const MINIO_PORT = {minio_port};
            const MINIO_CONSOLE_PORT = {minio_console_port};
            const MANAGER_PORT = {manager_port};

            let currentTab = 'manager';

            function switchTab(target) {{
                currentTab = target;

                document.querySelectorAll('.tab').forEach(tab => {{
                    tab.classList.remove('active');
                }});
                event.target.classList.add('active');

                document.querySelectorAll('.panel').forEach(panel => {{
                    panel.classList.remove('active');
                }});
                document.getElementById(target + '-panel').classList.add('active');

                if (target === 'manager') {{
                    loadManagerData();
                }} else if (target === 'minio') {{
                    loadMinioData();
                }}
            }}

            function formatUptime(seconds) {{
                const hours = Math.floor(seconds / 3600);
                const minutes = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                if (hours > 0) return `${{hours}}h ${{minutes}}m`;
                if (minutes > 0) return `${{minutes}}m ${{secs}}s`;
                return `${{secs}}s`;
            }}

            function formatLatency(ms) {{
                return `${{ms.toFixed(1)}}ms`;
            }}

            async function loadManagerData() {{
                const content = document.getElementById('manager-content');
                const countBadge = document.getElementById('worker-count');

                try {{
                    const response = await fetch(`http://127.0.0.1:${{MANAGER_PORT}}/admin/manager/api/workers`);
                    if (!response.ok) throw new Error('Failed to fetch workers');

                    const workers = await response.json();

                    if (!workers || workers.length === 0) {{
                        content.innerHTML = '<p style="color: #666;">No workers running</p>';
                        countBadge.textContent = '0 Workers';
                        return;
                    }}

                    countBadge.textContent = `${{workers.length}} Worker${{workers.length > 1 ? 's' : ''}}`;

                    const healthyCount = workers.filter(w => w.healthy).length;
                    const unhealthyCount = workers.length - healthyCount;

                    if (unhealthyCount > 0) {{
                        updateStatus('warning');
                    }} else {{
                        updateStatus('online');
                    }}

                    const workersHtml = workers.map(worker => `
                        <div class="worker-item ${{!worker.healthy ? 'unhealthy' : ''}}">
                            <div class="worker-main">
                                <div class="worker-header">
                                    <span class="worker-name">${{worker.worker_id}}</span>
                                    <span class="worker-type ${{worker.worker_type}}">${{worker.worker_type}}</span>
                                </div>
                                <div class="worker-details">
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">PID</span>
                                        <span class="worker-detail-value">${{worker.pid}}</span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Port</span>
                                        <span class="worker-detail-value">${{worker.port}}</span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Uptime</span>
                                        <span class="worker-detail-value">${{formatUptime(worker.uptime)}}</span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Health</span>
                                        <span class="worker-detail-value ${{worker.healthy ? 'healthy' : 'unhealthy'}}">
                                            ${{worker.healthy ? '✓ Healthy' : '✗ Unhealthy'}}
                                        </span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Latency</span>
                                        <span class="worker-detail-value">${{formatLatency(worker.health_latency_ms)}}</span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Requests</span>
                                        <span class="worker-detail-value">${{worker.metrics.requests}}</span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Errors</span>
                                        <span class="worker-detail-value">${{worker.metrics.errors}}</span>
                                    </div>
                                    <div class="worker-detail">
                                        <span class="worker-detail-label">Restarts</span>
                                        <span class="worker-detail-value">${{worker.restart_count}}</span>
                                    </div>
                                </div>
                            </div>
                            <div class="worker-actions">
                                <button onclick="restartWorker('${{worker.worker_id}}')">Restart</button>
                                <button class="danger" onclick="stopWorker('${{worker.worker_id}}')">Stop</button>
                            </div>
                        </div>
                    `).join('');

                    content.innerHTML = `<div class="worker-list">${{workersHtml}}</div>`;
                }} catch (error) {{
                    content.innerHTML = `<div class="error">Error: ${{error.message}}</div>`;
                    updateStatus('error');
                }}
            }}

            async function loadMinioData() {{
                const statsDiv = document.getElementById('minio-stats');

                try {{
                    // MinIO API erfordert auth, daher nur placeholder stats
                    statsDiv.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">Active</div>
                            <div class="stat-label">Status</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">:{minio_port}</div>
                            <div class="stat-label">API Port</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">:{minio_console_port}</div>
                            <div class="stat-label">Console Port</div>
                        </div>
                    `;
                    updateStatus('online');
                }} catch (error) {{
                    statsDiv.innerHTML = `<div class="error">Error loading MinIO stats</div>`;
                    updateStatus('error');
                }}
            }}

            async function restartWorker(workerId) {{
                try {{
                    const response = await fetch(`http://127.0.0.1:${{MANAGER_PORT}}/admin/manager/api/workers/${{workerId}}/restart`, {{
                        method: 'POST'
                    }});
                    if (!response.ok) throw new Error('Failed to restart worker');
                    setTimeout(loadManagerData, 1000);
                }} catch (error) {{
                    alert(`Error restarting worker: ${{error.message}}`);
                }}
            }}

            async function stopWorker(workerId) {{
                if (!confirm(`Stop worker ${{workerId}}?`)) return;
                try {{
                    const response = await fetch(`http://127.0.0.1:${{MANAGER_PORT}}/admin/manager/api/workers/${{workerId}}/stop`, {{
                        method: 'POST'
                    }});
                    if (!response.ok) throw new Error('Failed to stop worker');
                    setTimeout(loadManagerData, 1000);
                }} catch (error) {{
                    alert(`Error stopping worker: ${{error.message}}`);
                }}
            }}

            function updateStatus(status) {{
                const dot = document.getElementById('status-dot');
                const text = document.getElementById('status-text');

                dot.classList.remove('error', 'warning');

                if (status === 'online') {{
                    text.textContent = 'Online';
                }} else if (status === 'warning') {{
                    dot.classList.add('warning');
                    text.textContent = 'Warning';
                }} else {{
                    dot.classList.add('error');
                    text.textContent = 'Error';
                }}
            }}

            // Initial load
            loadManagerData();

            // Auto-refresh every 5 seconds
            setInterval(() => {{
                if (currentTab === 'manager') {{
                    loadManagerData();
                }}
            }}, 5000);
        </script>
    </body>
    </html>"""

        with open(admin_index, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✓ Admin UI created at {admin_index}")

    def write_config(self, http_ports: List[int], ws_ports: List[int],
                     http_sockets: List[str] = None, ws_sockets: List[str] = None,
                     remote_nodes: List[Tuple[str, int]] = None) -> bool:
        content = self.generate_config(http_ports, ws_ports, http_sockets, ws_sockets, remote_nodes)
        config_path = os.environ.get("NGINX_CONF_PATH") or getattr(self.config, 'config_path', DEFAULT_CONF_PATH)
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                f.write(content)
            logger.info(f"Nginx config written to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write nginx config: {e}")
            return False

    def test_config(self) -> bool:
        if not self._nginx_path:
            return False
        config_path = os.environ.get("NGINX_CONF_PATH") or getattr(self.config, 'config_path', DEFAULT_CONF_PATH)
        try:
            result = subprocess.run([self._nginx_path, "-t", "-c", str(config_path)], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    def reload(self) -> bool:
        if not self._nginx_path:
            return False
        try:
            subprocess.run([self._nginx_path, "-s", "reload"], check=True, capture_output=True)
            logger.info("Nginx reloaded")
            return True
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False

    def start(self) -> bool:
        if not self._nginx_path:
            return False
        config_path = os.environ.get("NGINX_CONF_PATH") or getattr(self.config, 'config_path', DEFAULT_CONF_PATH)
        try:
            subprocess.run([self._nginx_path, "-c", str(config_path)], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def stop(self) -> bool:
        if not self._nginx_path:
            return False
        try:
            subprocess.run([self._nginx_path, "-s", "stop"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    @property
    def ssl_available(self) -> bool:
        return self._ssl.available

    @property
    def platform_warning(self) -> str | None:
        if IS_WINDOWS:
            return "Nginx on Windows uses select() - expect ~10x slower than Linux"
        return None


# ============================================================================
# Metrics Collector - HTTP + ZMQ based
# ============================================================================

class MetricsCollector:
    """
    Collect metrics from workers via:
    - HTTP /metrics endpoint (for HTTP workers)
    - ZMQ HEALTH_CHECK events (for WS workers)
    """

    def __init__(self, zmq_pub_endpoint: str = "tcp://127.0.0.1:5555"):
        self._zmq_pub = zmq_pub_endpoint
        self._metrics: Dict[str, WorkerMetrics] = {}
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None
        self._workers: Dict[str, WorkerInfo] = {}
        self._zmq_ctx = None
        self._zmq_sub = None

    def start(self, workers: Dict[str, 'WorkerInfo']):
        """Start metrics collection."""
        self._workers = workers
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._zmq_sub:
            try:
                self._zmq_sub.close()
            except Exception:
                pass
        if self._zmq_ctx:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass

    def update_workers(self, workers: Dict[str, 'WorkerInfo']):
        """Update worker reference."""
        self._workers = workers

    def _collect_loop(self):
        """Background loop to collect metrics from workers."""
        # Setup ZMQ subscriber for WS worker WORKER_HEALTH events
        zmq_available = False
        if ZMQ_AVAILABLE:
            try:
                self._zmq_ctx = zmq.Context()
                self._zmq_sub = self._zmq_ctx.socket(zmq.SUB)
                self._zmq_sub.connect(self._zmq_pub)
                self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
                self._zmq_sub.setsockopt(zmq.RCVTIMEO, 100)
                zmq_available = True
            except Exception as e:
                logger.warning(f"ZMQ setup failed: {e}")
        else:
            logger.warning("ZMQ not installed - WS metrics via ZMQ disabled")

        while self._running:
            # Collect HTTP worker metrics via /metrics endpoint
            for wid, info in list(self._workers.items()):
                if info.worker_type == WorkerType.HTTP and info.state == WorkerState.RUNNING:
                    self._fetch_http_metrics(wid, info)

            # Process ZMQ events for WS worker metrics
            if zmq_available and self._zmq_sub:
                self._process_zmq_events()

            time.sleep(60)

    def _fetch_http_metrics(self, worker_id: str, info: 'WorkerInfo'):
        """Fetch metrics from HTTP worker via /metrics endpoint."""
        try:
            # Use socket for Unix socket support
            if info.socket_path and not IS_WINDOWS and os.path.exists(info.socket_path):
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(info.socket_path)
                sock.sendall(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                response = b""
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                sock.close()
            else:
                conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=2)
                conn.request("GET", "/metrics")
                resp = conn.getresponse()
                response = resp.read()
                conn.close()

            # Parse JSON from response body
            body_start = response.find(b"\r\n\r\n")
            if body_start > 0:
                json_body = response[body_start + 4:]
            else:
                json_body = response

            data = json.loads(json_body.decode())

            with self._lock:
                self._metrics[worker_id] = WorkerMetrics(
                    requests=data.get("requests_total", 0),
                    connections=data.get("requests_success", 0),
                    errors=data.get("requests_error", 0),
                    avg_latency_ms=data.get("avg_latency_ms", 0),
                    last_update=time.time()
                )
        except Exception as e:
            logger.debug(f"Failed to fetch metrics from {worker_id}: {e}")

    def _process_zmq_events(self):
        """Process ZMQ events for WORKER_HEALTH responses."""
        if not ZMQ_AVAILABLE or not self._zmq_sub:
            return
        try:
            while True:
                try:
                    msg = self._zmq_sub.recv(zmq.NOBLOCK)
                    data = json.loads(msg.decode())

                    # Check for WORKER_HEALTH event type
                    if data.get("type") == "worker.health":
                        payload = data.get("payload", {})
                        wid = payload.get("worker_id") or data.get("source")
                        if wid:
                            with self._lock:
                                self._metrics[wid] = WorkerMetrics(
                                    requests=payload.get("messages_received", 0),
                                    connections=payload.get("total_connections", 0),
                                    errors=payload.get("errors", 0),
                                    avg_latency_ms=0,
                                    last_update=time.time()
                                )
                except Exception:
                    break
        except Exception:
            pass

    def get_metrics(self, worker_id: str) -> WorkerMetrics:
        """Get metrics for a specific worker."""
        with self._lock:
            return self._metrics.get(worker_id, WorkerMetrics())

    def get_all_metrics(self) -> Dict[str, WorkerMetrics]:
        """Get all worker metrics."""
        with self._lock:
            return dict(self._metrics)


# ============================================================================
# Health Checker
# ============================================================================

class HealthChecker:
    def __init__(self, interval: float = 5.0):
        self._interval = interval
        self._running = False
        self._thread: Thread | None = None
        self._workers: Dict[str, WorkerInfo] = {}

    def start(self, workers: Dict[str, WorkerInfo]):
        self._workers = workers
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._check_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def update_workers(self, workers: Dict[str, WorkerInfo]):
        self._workers = workers

    def _check_loop(self):
        while self._running:
            for wid, info in list(self._workers.items()):
                if info.state != WorkerState.RUNNING:
                    continue
                healthy, latency = self._check_worker(info)
                info.healthy = healthy
                info.health_latency_ms = latency
                info.last_health_check = time.time()
            time.sleep(self._interval)

    def _check_worker(self, info: WorkerInfo) -> Tuple[bool, float]:
        start = time.perf_counter()
        try:
            # WebSocket workers need a different health check
            if info.worker_type == WorkerType.WS:
                return self._check_ws_worker(info, start)

            # HTTP workers use standard HTTP health check
            if info.socket_path and not IS_WINDOWS and os.path.exists(info.socket_path):
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect(info.socket_path)
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                resp = sock.recv(1024)
                sock.close()
                return b"200" in resp, (time.perf_counter() - start) * 1000
            else:
                conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=2)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                conn.close()
                return resp.status == 200, (time.perf_counter() - start) * 1000
        except Exception:
            return False, 0.0

    def _check_ws_worker(self, info: WorkerInfo, start: float) -> Tuple[bool, float]:
        """Check WebSocket worker health using HTTP request to /health endpoint.

        The WS worker has a process_request handler that responds to /health
        with HTTP 200 OK without performing a WebSocket handshake.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(("127.0.0.1", info.port))

            # Send HTTP/1.1 request to /health endpoint
            # The WS worker's process_request handler will respond with 200 OK
            request = (
                b"GET /health HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            sock.sendall(request)

            # Read response
            try:
                response = sock.recv(512)
                sock.close()

                # Check for HTTP 200 response
                response_str = response.decode('utf-8', errors='ignore')
                if "200" in response_str or "OK" in response_str:
                    return True, (time.perf_counter() - start) * 1000
                # Any response means server is alive, even if not 200
                return True, (time.perf_counter() - start) * 1000
            except socket.timeout:
                sock.close()
                return False, 0.0
        except Exception:
            return False, 0.0


# ============================================================================
# Cluster Manager
# ============================================================================

class ClusterManager:
    def __init__(self, secret: str = None):
        self._nodes: Dict[str, ClusterNode] = {}
        self._secret = secret or os.environ.get("CLUSTER_SECRET", uuid.uuid4().hex)
        self._lock = Lock()
        self._running = False
        self._thread: Thread | None = None

    def add_node(self, host: str, port: int, secret: str) -> bool:
        node_id = f"{host}:{port}"
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/admin/manager/api/cluster/verify", headers={"X-Cluster-Secret": secret})
            resp = conn.getresponse()
            conn.close()
            if resp.status != 200:
                return False
        except Exception:
            return False

        with self._lock:
            self._nodes[node_id] = ClusterNode(node_id=node_id, host=host, port=port, secret=secret, healthy=True, last_seen=time.time())
        logger.info(f"Cluster: Added node {node_id}")
        return True

    def remove_node(self, node_id: str):
        with self._lock:
            self._nodes.pop(node_id, None)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _monitor_loop(self):
        while self._running:
            for node_id, node in list(self._nodes.items()):
                try:
                    conn = http.client.HTTPConnection(node.host, node.port, timeout=5)
                    conn.request("GET", "/admin/manager/api/workers", headers={"X-Cluster-Secret": node.secret})
                    resp = conn.getresponse()
                    data = json.loads(resp.read().decode())
                    conn.close()
                    for w in data:
                        w["node"] = node_id
                    with self._lock:
                        node.workers = data
                        node.healthy = True
                        node.last_seen = time.time()
                except Exception:
                    with self._lock:
                        node.healthy = False
            time.sleep(10)

    def get_all_workers(self) -> List[Dict]:
        with self._lock:
            return [w for n in self._nodes.values() if n.healthy for w in n.workers]

    def get_remote_addresses(self) -> List[Tuple[str, int]]:
        with self._lock:
            return [(n.host, w["port"]) for n in self._nodes.values() if n.healthy
                    for w in n.workers if w.get("worker_type") == "http" and w.get("state") == "running"]

    @property
    def nodes(self) -> Dict[str, ClusterNode]:
        with self._lock:
            return dict(self._nodes)

    @property
    def secret(self) -> str:
        return self._secret


# ============================================================================
# Worker Process Functions
# ============================================================================

def _run_broker_process(config_dict: Dict):
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    from toolboxv2.utils.workers.config import Config
    from toolboxv2.utils.workers.event_manager import run_broker
    config = Config.from_dict(config_dict)
    try:
        asyncio.run(run_broker(config))
    except KeyboardInterrupt:
        pass


def _run_http_worker_process(worker_id: str, config_dict: Dict, port: int, socket_path: str = None):
    from toolboxv2.utils.workers.config import Config
    from toolboxv2.utils.workers.server_worker import HTTPWorker
    config = Config.from_dict(config_dict)
    worker = HTTPWorker(worker_id, config)
    if socket_path and not IS_WINDOWS:
        worker.run(socket_path=socket_path)
    else:
        worker.run(port=port)


def _run_ws_worker_process(worker_id: str, config_dict: Dict, port: int):
    """Run WebSocket worker in a separate process."""
    from toolboxv2.utils.workers.config import Config
    from toolboxv2.utils.workers.ws_worker import WSWorker
    config = Config.from_dict(config_dict)
    config.ws_worker.port = port
    worker = WSWorker(worker_id, config)
    # Use run_sync which handles event loop creation properly
    worker.run_sync()


# ============================================================================
# Worker Manager
# ============================================================================

class WorkerManager:
    def __init__(self, config):
        self.config = config
        self._workers: Dict[str, WorkerInfo] = {}
        self._processes: Dict[str, Process] = {}
        self._nginx = NginxManager(config)
        self._broker_process: Process | None = None
        self._running = False
        self._next_http_port = config.http_worker.port
        ws_base = config.ws_worker.port
        if ws_base < config.http_worker.port + 100:
            ws_base = config.http_worker.port + 100
        self._next_ws_port = ws_base

        # ZMQ endpoints from config
        zmq_pub = getattr(config.zmq, 'pub_endpoint', 'tcp://127.0.0.1:5555')
        self._metrics_collector = MetricsCollector(zmq_pub_endpoint=zmq_pub)
        self._health_checker = HealthChecker()
        self._cluster = ClusterManager()

    def _get_socket_path(self, worker_id: str) -> str | None:
        if IS_WINDOWS or not SOCKET_PREFIX:
            return None
        return os.path.join(SOCKET_PREFIX, f"tbv2_{worker_id}.sock")

    def start_broker(self) -> bool:
        if self._broker_process and self._broker_process.is_alive():
            return True
        self._broker_process = Process(target=_run_broker_process, args=(self.config.to_dict(),), name="zmq_broker")
        self._broker_process.start()
        time.sleep(0.5)
        if self._broker_process.is_alive():
            logger.info(f"ZMQ broker started (PID: {self._broker_process.pid})")
            return True
        return False

    def stop_broker(self):
        self._metrics_collector.stop()
        if self._broker_process and self._broker_process.is_alive():
            self._broker_process.terminate()
            self._broker_process.join(timeout=5)
            if self._broker_process.is_alive():
                self._broker_process.kill()

    def start_http_worker(self, worker_id: str = None, port: int = None) -> WorkerInfo | None:
        if not worker_id:
            worker_id = f"http_{uuid.uuid4().hex[:8]}"
        if not port:
            port = self._next_http_port
            self._next_http_port += 1

        socket_path = self._get_socket_path(worker_id)
        process = Process(target=_run_http_worker_process, args=(worker_id, self.config.to_dict(), port, socket_path), name=worker_id)
        process.start()

        info = WorkerInfo(worker_id=worker_id, worker_type=WorkerType.HTTP, pid=process.pid, port=port,
                          socket_path=socket_path, state=WorkerState.STARTING, started_at=time.time())
        self._workers[worker_id] = info
        self._processes[worker_id] = process

        time.sleep(0.5)
        if process.is_alive():
            info.state = WorkerState.RUNNING
            logger.info(f"HTTP worker started: {worker_id} (port {port})")
            return info
        info.state = WorkerState.FAILED
        return None

    def start_ws_worker(self, worker_id: str = None, port: int = None) -> WorkerInfo | None:
        if not worker_id:
            worker_id = f"ws_{uuid.uuid4().hex[:8]}"
        if not port:
            port = self._next_ws_port
            self._next_ws_port += 1

        process = Process(target=_run_ws_worker_process, args=(worker_id, self.config.to_dict(), port), name=worker_id)
        process.start()

        info = WorkerInfo(worker_id=worker_id, worker_type=WorkerType.WS, pid=process.pid, port=port,
                          state=WorkerState.STARTING, started_at=time.time())
        self._workers[worker_id] = info
        self._processes[worker_id] = process

        time.sleep(0.5)
        if process.is_alive():
            info.state = WorkerState.RUNNING
            logger.info(f"WS worker started: {worker_id} (port {port})")
            return info
        info.state = WorkerState.FAILED
        return None

    def stop_worker(self, worker_id: str, graceful: bool = True) -> bool:
        if worker_id not in self._processes:
            return False
        info = self._workers.get(worker_id)
        process = self._processes[worker_id]
        if info:
            info.state = WorkerState.STOPPING
            if info.socket_path:
                try:
                    if os.path.exists(info.socket_path):
                        os.unlink(info.socket_path)
                except Exception:
                    pass
        if graceful:
            process.terminate()
            process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if info:
            info.state = WorkerState.STOPPED
        del self._processes[worker_id]
        logger.info(f"Worker stopped: {worker_id}")
        return True

    def restart_worker(self, worker_id: str) -> WorkerInfo | None:
        if worker_id not in self._workers:
            return None
        info = self._workers[worker_id]
        port, wtype = info.port, info.worker_type
        self.stop_worker(worker_id)
        del self._workers[worker_id]
        if wtype == WorkerType.HTTP:
            return self.start_http_worker(worker_id, port)
        return self.start_ws_worker(worker_id, port)

    def _get_http_ports(self) -> List[int]:
        return [w.port for w in self._workers.values() if w.worker_type == WorkerType.HTTP and w.state == WorkerState.RUNNING]

    def _get_ws_ports(self) -> List[int]:
        return [w.port for w in self._workers.values() if w.worker_type == WorkerType.WS and w.state == WorkerState.RUNNING]

    def _get_http_sockets(self) -> List[str]:
        return [w.socket_path for w in self._workers.values() if w.worker_type == WorkerType.HTTP and w.state == WorkerState.RUNNING and w.socket_path]

    def _update_nginx_config(self):
        self._nginx.write_config(self._get_http_ports(), self._get_ws_ports(), self._get_http_sockets(), [], self._cluster.get_remote_addresses())

    def start_all(self) -> bool:
        logger.info("Starting all services...")
        if IS_WINDOWS:
            logger.warning("Windows: Nginx performance limited (~10x slower than Linux)")

        if not self.start_broker():
            return False

        for _ in range(self.config.http_worker.workers):
            self.start_http_worker()
        self.start_ws_worker()

        # Start metrics collector and health checker AFTER workers are created
        self._metrics_collector.start(self._workers)
        self._health_checker.start(self._workers)
        self._cluster.start()

        if self.config.nginx.enabled:
            self._update_nginx_config()
            if not self._nginx.is_installed():
                self._nginx.install()
            if self._nginx.test_config():
                self._nginx.reload()

        self._running = True
        logger.info("All services started")
        return True

    def stop_all(self, graceful: bool = True):
        logger.info("Stopping all services...")
        self._running = False
        self._health_checker.stop()
        self._cluster.stop()
        for wid in list(self._processes.keys()):
            self.stop_worker(wid, graceful)
        self.stop_broker()
        logger.info("All services stopped")

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "platform": SYSTEM,
            "platform_warning": self._nginx.platform_warning,
            "broker_alive": self._broker_process.is_alive() if self._broker_process else False,
            "workers": {wid: w.to_dict() for wid, w in self._workers.items()},
            "nginx": {"installed": self._nginx.is_installed(), "ssl_available": self._nginx.ssl_available, "version": self._nginx.get_version()},
            "cluster": {"nodes": len(self._cluster.nodes), "healthy_nodes": sum(1 for n in self._cluster.nodes.values() if n.healthy)},
        }

    def get_workers(self) -> List[Dict]:
        local = [w.to_dict() for w in self._workers.values()]
        for w in local:
            m = self._metrics_collector.get_metrics(w["worker_id"])
            w["metrics"] = {"requests": m.requests, "connections": m.connections, "errors": m.errors, "avg_latency_ms": m.avg_latency_ms}
        return local + self._cluster.get_all_workers()

    def get_metrics(self) -> Dict[str, Any]:
        all_m = self._metrics_collector.get_all_metrics()
        return {
            "total_workers": len(self._workers),
            "http_workers": sum(1 for w in self._workers.values() if w.worker_type == WorkerType.HTTP),
            "ws_workers": sum(1 for w in self._workers.values() if w.worker_type == WorkerType.WS),
            "total_requests": sum(m.requests for m in all_m.values()),
            "total_connections": sum(m.connections for m in all_m.values()),
            "total_errors": sum(m.errors for m in all_m.values()),
            "avg_latency_ms": sum(m.avg_latency_ms for m in all_m.values()) / len(all_m) if all_m else 0,
        }

    def get_health(self) -> Dict[str, Any]:
        return {
            "healthy": all(w.healthy for w in self._workers.values()) and (self._broker_process and self._broker_process.is_alive()),
            "broker": {"alive": self._broker_process.is_alive() if self._broker_process else False},
            "workers": {wid: {"healthy": w.healthy, "state": w.state.value, "latency_ms": w.health_latency_ms} for wid, w in self._workers.items()},
            "nginx": {"installed": self._nginx.is_installed(), "ssl": self._nginx.ssl_available},
        }

    def scale_workers(self, worker_type: str, target: int) -> Dict[str, Any]:
        wtype = WorkerType.HTTP if worker_type == "http" else WorkerType.WS
        current = [w for w in self._workers.values() if w.worker_type == wtype]
        started, stopped = [], []

        if target > len(current):
            for _ in range(target - len(current)):
                info = self.start_http_worker() if wtype == WorkerType.HTTP else self.start_ws_worker()
                if info:
                    started.append(info.worker_id)
        elif target < len(current):
            for w in current[:len(current) - target]:
                self.stop_worker(w.worker_id)
                stopped.append(w.worker_id)

        if started or stopped:
            self._update_nginx_config()
            self._health_checker.update_workers(self._workers)
            self._metrics_collector.update_workers(self._workers)
            self._nginx.reload()

        return {"status": "ok", "started": started, "stopped": stopped}

    def rolling_update(self, delay: float = 2.0, validate: bool = True):
        logger.info("Starting rolling update...")
        for info in [w for w in self._workers.values() if w.worker_type == WorkerType.HTTP]:
            new = self.start_http_worker()
            if not new:
                continue
            time.sleep(delay)
            if validate:
                try:
                    conn = http.client.HTTPConnection("127.0.0.1", new.port, timeout=2)
                    conn.request("GET", "/health")
                    if conn.getresponse().status != 200:
                        self.stop_worker(new.worker_id)
                        continue
                except Exception:
                    self.stop_worker(new.worker_id)
                    continue
            self._update_nginx_config()
            self._nginx.reload()
            info.state = WorkerState.DRAINING
            time.sleep(delay)
            self.stop_worker(info.worker_id)
        logger.info("Rolling update complete")

    def add_cluster_node(self, host: str, port: int, secret: str) -> bool:
        if self._cluster.add_node(host, port, secret):
            self._update_nginx_config()
            self._nginx.reload()
            return True
        return False

    @property
    def cluster_secret(self) -> str:
        return self._cluster.secret


# ============================================================================
# Web UI
# ============================================================================

class ManagerWebUI(BaseHTTPRequestHandler):
    manager: 'WorkerManager' = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path.startswith("/admin/manager"):
            path = path[14:]
        if path == "/":
            self._serve_dashboard()
        elif path == "/api/status":
            self._json(self.manager.get_status())
        elif path == "/api/workers":
            self._json(self.manager.get_workers())
        elif path == "/api/metrics":
            self._json(self.manager.get_metrics())
        elif path == "/api/health":
            self._json(self.manager.get_health())
        elif path == "/api/cluster/verify":
            secret = self.headers.get("X-Cluster-Secret")
            if secret == self.manager.cluster_secret:
                self._json({"status": "ok"})
            else:
                self.send_response(403)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}
        if path.startswith("/admin/manager"):
            path = path[14:]
        if path == "/api/workers/start":
            results = []
            for _ in range(data.get("count", 1)):
                info = self.manager.start_http_worker() if data.get("type", "http") == "http" else self.manager.start_ws_worker()
                if info:
                    results.append(info.to_dict())
            self._json({"status": "ok", "workers": results})
        elif path == "/api/workers/stop":
            self.manager.stop_worker(data.get("worker_id"), data.get("graceful", True))
            self._json({"status": "ok"})
        elif path == "/api/workers/restart":
            info = self.manager.restart_worker(data.get("worker_id"))
            self._json({"status": "ok", "worker": info.to_dict() if info else None})
        elif path == "/api/rolling-update":
            Thread(target=self.manager.rolling_update, daemon=True).start()
            self._json({"status": "ok"})
        elif path == "/api/scale":
            self._json(self.manager.scale_workers(data.get("type", "http"), data.get("count", 1)))
        elif path == "/api/shutdown":
            Thread(target=self.manager.stop_all, daemon=True).start()
            self._json({"status": "ok"})
        elif path == "/api/nginx/reload":
            self.manager._update_nginx_config()
            self._json({"status": "ok" if self.manager._nginx.reload() else "error"})
        elif path == "/api/cluster/join":
            self._json({"status": "ok" if self.manager.add_cluster_node(data.get("host"), data.get("port", 9000), data.get("secret")) else "error"})
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _serve_dashboard(self):
        html = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Worker Manager</title>
<style>:root{--bg:#0f172a;--card:#1e293b;--accent:#3b82f6;--ok:#22c55e;--err:#ef4444;--txt:#f1f5f9;--muted:#94a3b8}
*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui;background:var(--bg);color:var(--txt);padding:20px}
.h{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid #475569}
.badge{padding:4px 12px;border-radius:99px;font-size:.875rem}.ok{background:rgba(34,197,94,.2);color:var(--ok)}
.err{background:rgba(239,68,68,.2);color:var(--err)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-bottom:20px}
.card{background:var(--card);border-radius:12px;padding:20px;border:1px solid #475569}.title{font-size:.75rem;color:var(--muted);text-transform:uppercase;margin-bottom:12px}
.metrics{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.m{background:#334155;padding:12px;border-radius:8px}
.m-v{font-size:1.5rem;font-weight:700;color:var(--accent)}.m-l{font-size:.75rem;color:var(--muted)}
.btn{padding:8px 16px;border-radius:6px;border:none;font-size:.875rem;cursor:pointer;margin-right:8px;margin-bottom:8px}
.btn-p{background:var(--accent);color:#fff}.btn-d{background:rgba(239,68,68,.2);color:var(--err)}
.btn-s{background:#334155;color:var(--txt)}.wl{display:flex;flex-direction:column;gap:12px;max-height:400px;overflow-y:auto}
.wi{background:#334155;padding:16px;border-radius:8px;display:flex;justify-content:space-between;align-items:center}
.wi-id{font-family:monospace;color:var(--accent)}.wi-m{font-size:.75rem;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;margin-right:12px;display:inline-block}.dot.running{background:var(--ok)}.dot.stopped{background:var(--err)}
</style></head><body>
<div class="h"><h1>⚡ Worker Manager</h1><span class="badge" id="status">Loading...</span></div>
<div style="margin-bottom:20px">
<button class="btn btn-p" onclick="start('http')">+ HTTP</button>
<button class="btn btn-p" onclick="start('ws')">+ WS</button>
<button class="btn btn-s" onclick="update()">Rolling Update</button>
<button class="btn btn-s" onclick="reload()">Reload Nginx</button>
<button class="btn btn-d" onclick="shutdown()">Shutdown</button>
</div>
<div class="grid">
<div class="card"><div class="title">Metrics</div><div class="metrics">
<div class="m"><div class="m-v" id="reqs">0</div><div class="m-l">Requests</div></div>
<div class="m"><div class="m-v" id="conns">0</div><div class="m-l">Connections</div></div>
<div class="m"><div class="m-v" id="http">0</div><div class="m-l">HTTP Workers</div></div>
<div class="m"><div class="m-v" id="ws">0</div><div class="m-l">WS Workers</div></div>
</div></div>
<div class="card"><div class="title">System</div><div class="metrics">
<div class="m"><div class="m-v" id="nginx">-</div><div class="m-l">Nginx</div></div>
<div class="m"><div class="m-v" id="broker">-</div><div class="m-l">Broker</div></div>
<div class="m"><div class="m-v" id="platform">-</div><div class="m-l">Platform</div></div>
<div class="m"><div class="m-v" id="cluster">0</div><div class="m-l">Cluster</div></div>
</div></div>
</div>
<div class="card"><div class="title">Workers</div><div class="wl" id="workers"></div></div>
<script>
async function fetch_data(){try{
const[s,m,w]=await Promise.all([fetch('/admin/manager/api/status').then(r=>r.json()),fetch('/admin/manager/api/metrics').then(r=>r.json()),fetch('/admin/manager/api/workers').then(r=>r.json())]);
document.getElementById('status').className='badge '+(s.running?'ok':'err');
document.getElementById('status').textContent=s.running?'Running':'Stopped';
document.getElementById('reqs').textContent=m.total_requests;
document.getElementById('conns').textContent=m.total_connections;
document.getElementById('http').textContent=m.http_workers;
document.getElementById('ws').textContent=m.ws_workers;
document.getElementById('nginx').textContent=s.nginx.installed?'OK':'No';
document.getElementById('broker').textContent=s.broker_alive?'OK':'Down';
document.getElementById('platform').textContent=s.platform;
document.getElementById('cluster').textContent=s.cluster.healthy_nodes;
document.getElementById('workers').innerHTML=w.map(x=>`<div class="wi"><div><span class="dot ${x.state}"></span><span class="wi-id">${x.worker_id}</span><div class="wi-m">${x.worker_type} | Port ${x.port} | ${x.node||'local'}</div></div><div><button class="btn btn-s" onclick="restart('${x.worker_id}')">Restart</button><button class="btn btn-d" onclick="stop('${x.worker_id}')">Stop</button></div></div>`).join('');
}catch(e){}}
async function start(t){await fetch('/admin/manager/api/workers/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:t,count:1})});fetch_data()}
async function stop(id){await fetch('/admin/manager/api/workers/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({worker_id:id})});fetch_data()}
async function restart(id){await fetch('/admin/manager/api/workers/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({worker_id:id})});fetch_data()}
async function update(){await fetch('/admin/manager/api/rolling-update',{method:'POST'})}
async function reload(){await fetch('/admin/manager/api/nginx/reload',{method:'POST'})}
async function shutdown(){if(confirm('Shutdown?'))await fetch('/admin/manager/api/shutdown',{method:'POST'})}
fetch_data();setInterval(fetch_data,2000);
</script></body></html>'''
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


def run_web_ui(manager: WorkerManager, host: str, port: int):
    ManagerWebUI.manager = manager
    for _ in range(5):
        try:
            server = HTTPServer((host, port), ManagerWebUI)
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logger.info(f"Web UI on http://{host}:{port}")
            server.serve_forever()
            break
        except OSError:
            port += 1


# ============================================================================
# CLI
# ============================================================================

def main():
    if IS_WINDOWS:
        from multiprocessing import freeze_support
        freeze_support()

    parser = argparse.ArgumentParser(description="ToolBoxV2 Worker Manager", prog="tb workers")
    parser.add_argument("command", nargs="?", default="start", choices=["start", "stop", "restart", "status", "update", "nginx-config", "nginx-reload", "worker-start", "worker-stop", "cluster-join"])
    parser.add_argument("-c", "--config", help="Config file")
    parser.add_argument("-w", "--worker-id")
    parser.add_argument("-t", "--type", choices=["http", "ws"], default="http")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--secret")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from toolboxv2.utils.workers.config import load_config
    config = load_config(args.config)
    print(config.nginx.static_root)
    manager = WorkerManager(config)

    if args.command == "start":
        if not manager.start_all():
            sys.exit(1)
        if config.manager.web_ui_enabled and not args.no_ui:
            Thread(target=run_web_ui, args=(manager, config.manager.web_ui_host, config.manager.web_ui_port), daemon=True).start()
        try:
            while manager._running:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_all()
    elif args.command == "stop":
        manager.stop_all()
    elif args.command == "restart":
        manager.stop_all()
        time.sleep(2)
        manager.start_all()
    elif args.command == "status":
        print(json.dumps(manager.get_status(), indent=2))
    elif args.command == "update":
        manager.rolling_update()
    elif args.command == "nginx-config":
        print(manager._nginx.generate_config([config.http_worker.port + i for i in range(config.http_worker.workers)], [config.ws_worker.port]))
    elif args.command == "nginx-reload":
        manager._update_nginx_config()
        manager._nginx.reload()
    elif args.command == "worker-start":
        info = manager.start_http_worker(args.worker_id) if args.type == "http" else manager.start_ws_worker(args.worker_id)
        print(json.dumps(info.to_dict() if info else {"error": "failed"}, indent=2))
    elif args.command == "worker-stop":
        manager.stop_worker(args.worker_id)
    elif args.command == "cluster-join":
        if manager.add_cluster_node(args.host, args.port or 9000, args.secret):
            print(f"Joined {args.host}:{args.port or 9000}")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
