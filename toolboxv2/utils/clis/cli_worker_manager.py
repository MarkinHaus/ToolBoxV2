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
import textwrap
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

from toolboxv2 import tb_root_dir
from toolboxv2.utils.workers.debug_runner import run_debug_server

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
"""
NginxManager - Updated version with separate site config generation
Supports standard nginx pattern: nginx.conf includes sites-enabled/*
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import platform

logger = logging.getLogger(__name__)

SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_LINUX = SYSTEM == "Linux"
IS_MACOS = SYSTEM == "Darwin"

DEFAULT_NGINX_PATHS = [
    "/usr/sbin/nginx",
    "/usr/local/bin/nginx",
    "/opt/homebrew/bin/nginx",
    "C:\\nginx\\nginx.exe",
]

DEFAULT_CONF_PATH = "/etc/nginx/nginx.conf" if not IS_WINDOWS else "C:\\nginx\\conf\\nginx.conf"
DEFAULT_SITES_AVAILABLE = "/etc/nginx/sites-available"
DEFAULT_SITES_ENABLED = "/etc/nginx/sites-enabled"
DEFAULT_BOX_AVAILABLE = "/etc/nginx/box-available"
DEFAULT_BOX_ENABLED = "/etc/nginx/box-enabled"


class SSLManager:
    """Placeholder - import your actual SSLManager"""

    def __init__(self, server_name):
        self.server_name = server_name
        self.cert_path = None
        self.key_path = None
        self.available = False

    def discover(self):
        # Check for Let's Encrypt certs
        if self.server_name:
            cert_path = f"/etc/letsencrypt/live/{self.server_name}/fullchain.pem"
            key_path = f"/etc/letsencrypt/live/{self.server_name}/privkey.pem"
            if os.path.exists(cert_path) and os.path.exists(key_path):
                self.cert_path = cert_path
                self.key_path = key_path
                self.available = True

def _hash_password_for_nginx(pwd: str) -> Optional[str]:
    """
    Versucht in Reihenfolge: passlib → bcrypt → sha1-fallback.
    Alle drei werden von nginx auth_basic akzeptiert.
    """
    # Beste Option: passlib APR1-MD5 (identisch mit htpasswd-Tool)
    try:
        from passlib.hash import apr_md5_crypt
        return apr_md5_crypt.hash(pwd)
    except ImportError:
        pass

    # Zweite Option: bcrypt
    try:
        import bcrypt
        return bcrypt.hashpw(
            pwd.encode("utf-8"),
            bcrypt.gensalt(rounds=12, prefix=b"2b")  # nginx erwartet $2y$
        ).decode("utf-8")
    except ImportError:
        pass

    # Fallback: SHA1 — von nginx unterstützt, schwächer aber funktional
    import hashlib, base64
    digest = hashlib.sha1(pwd.encode("utf-8")).digest()
    return "{SHA}" + base64.b64encode(digest).decode("utf-8")

class NginxManager:
    _PROXY_COMMON = (
        "proxy_http_version 1.1;\n"
        "            proxy_set_header Connection \"\";\n"
        "            proxy_set_header Host $host;\n"
        "            proxy_set_header X-Real-IP $remote_addr;\n"
        "            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\n"
        "            proxy_set_header X-Forwarded-Proto $scheme;"
    )
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
    # =========================================================================
    # NEW: Generate site config (for sites-available/toolbox)
    # =========================================================================
    def generate_site_config(
        self,
        max_http_workers: int,
        max_ws_workers: int,
        base_http_port: int,
        base_ws_port: int,
        remote_nodes: List[Tuple[str, int]] = None,
    ) -> str:
        cfg = self.config
        remote_nodes = remote_nodes or []

        http_servers = "\n        ".join(
            f"server localhost:{base_http_port + i} max_fails=1 fail_timeout=5s;"
            for i in range(max_http_workers)
        )
        ws_servers = "\n        ".join(
            f"server localhost:{base_ws_port + i} max_fails=1 fail_timeout=5s;"
            for i in range(max_ws_workers)
        )
        for host, port in remote_nodes:
            http_servers += f"\n        server {host}:{port} backup;"

        upstream_http = getattr(cfg, "upstream_http", "toolbox_http")
        upstream_ws = getattr(cfg, "upstream_ws", "toolbox_ws")
        server_name = getattr(cfg, "server_name", "_")
        listen_port = getattr(cfg, "listen_port", 80)
        static_root = getattr(cfg, "static_root", "./tb_dist")
        rate_zone = getattr(cfg, "rate_limit_zone", "tb_limit")
        rate_burst = getattr(cfg, "rate_limit_burst", 20)
        auth_burst = getattr(cfg, "auth_rate_limit_burst", 10)
        admin_port = getattr(self._manager, "web_ui_port", 9005)

        admin_block = self._generate_admin_ui_block(admin_port)

        return f"""# ToolBoxV2 Site Config
    # Ports pre-allocated: HTTP {base_http_port}-{base_http_port + max_http_workers - 1} | WS {base_ws_port}-{base_ws_port + max_ws_workers - 1}
    # Offline workers skipped via passive health check (max_fails=1 fail_timeout=5s)
    # Certbot manages SSL below — DO NOT REMOVE last comment line

    upstream {upstream_http} {{
        least_conn;
        {http_servers}
        keepalive 128;
        keepalive_requests 10000;
        keepalive_timeout 60s;
    }}

    upstream {upstream_ws} {{
        hash $request_uri consistent;
        {ws_servers}
    }}

    server {{
        listen {listen_port};
        listen [::]:{listen_port};
        server_name {server_name};

        access_log /var/log/nginx/toolbox_access.log;
        error_log  /var/log/nginx/toolbox_error.log warn;

        proxy_hide_header Server;
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        proxy_connect_timeout 60s;
        proxy_send_timeout    300s;
        proxy_read_timeout    300s;

        location / {{
            root {static_root};
            try_files $uri $uri/ /index.html;

            location ~* \\.(js|css|png|jpg|ico|svg|woff2|ttf|eot)$ {{
                expires 1h;
                add_header Cache-Control "public, immutable";
                access_log off;
            }}
            location ~* \\.html$ {{
                expires -1;
                add_header Cache-Control "no-store, no-cache, must-revalidate";
            }}
        }}

        location /api/ {{
            proxy_pass http://{upstream_http};
            {self._PROXY_COMMON}
            proxy_set_header Cookie $http_cookie;
            proxy_set_header Authorization $http_authorization;
            proxy_pass_header Set-Cookie;
            limit_req zone={rate_zone} burst={rate_burst} nodelay;
        }}

        location /sse/ {{
            proxy_pass http://{upstream_http};
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding on;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }}

        location /ws {{
            proxy_pass http://{upstream_ws};
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Cookie $http_cookie;
            proxy_buffering off;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }}

        location ~ ^/ws/([^/]+)/([^/]+)$ {{
            proxy_pass http://{upstream_ws};
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header Cookie $http_cookie;
            proxy_buffering off;
            proxy_read_timeout 3600s;
        }}

        location = /validateSession {{
            limit_except POST {{ deny all; }}
            limit_req zone=tb_auth_limit burst={auth_burst} nodelay;
            proxy_pass http://{upstream_http}/validateSession;
            {self._PROXY_COMMON}
            proxy_connect_timeout 10s;
            proxy_read_timeout    30s;
        }}

        location = /IsValidSession {{
            limit_except GET {{ deny all; }}
            proxy_pass http://{upstream_http}/IsValidSession;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header Cookie $http_cookie;
            proxy_connect_timeout 5s;
            proxy_read_timeout    10s;
        }}

        location = /web/logoutS {{
            limit_except POST {{ deny all; }}
            limit_req zone=tb_auth_limit burst={auth_burst} nodelay;
            proxy_pass http://{upstream_http}/web/logoutS;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header Cookie $http_cookie;
        }}

        location ~ ^/auth/(discord|google)/(url|callback)$ {{
            limit_req zone=tb_auth_limit burst={auth_burst} nodelay;
            proxy_pass http://{upstream_http};
            {self._PROXY_COMMON}
            proxy_set_header Cookie $http_cookie;
            proxy_pass_header Set-Cookie;
            proxy_connect_timeout 15s;
            proxy_read_timeout    30s;
        }}

        location = /auth/magic/verify {{
            limit_except GET {{ deny all; }}
            limit_req zone=tb_auth_limit burst={auth_burst} nodelay;
            proxy_pass http://{upstream_http}/auth/magic/verify;
            {self._PROXY_COMMON}
            proxy_connect_timeout 10s;
            proxy_read_timeout    30s;
        }}

        location /health {{
            proxy_pass http://{upstream_http}/health;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            access_log off;
        }}
        {admin_block}

        error_page 429 @rate_limited;
        location @rate_limited {{
            default_type application/json;
            return 429 '{{"error":"TooManyRequests","message":"Rate limit exceeded"}}';
        }}

        error_page 500 502 503 504 /50x.html;
        location = /50x.html {{
            root {static_root}/web/assets;
            internal;
        }}
    }}
    # certbot managed SSL block appears below — DO NOT REMOVE THIS LINE
    """

    def _generate_admin_ui_block(self, web_port: int) -> str:
        """Nur Config-Text — kein I/O, kein Seiteneffekt."""
        if IS_WINDOWS:
            return f"""
            location ^~ /admin/manager/ {{
                proxy_pass http://127.0.0.1:{web_port}/;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_hide_header Server;
            }}
            location ^~ /admin/minio/ {{
                proxy_pass http://127.0.0.1:9001/;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_hide_header Server;
            }}"""

        htpasswd_path = "/etc/nginx/admin_htpasswd"
        if not os.path.exists(htpasswd_path):
            logger.warning(f"htpasswd missing at {htpasswd_path} — call write_htpasswd() first")
            return f"# Admin UI disabled — htpasswd missing (run: tb workers nginx-config --write-htpasswd)"

        return f"""
            location ^~ /admin/manager/ {{
                auth_basic "Restricted Admin";
                auth_basic_user_file {htpasswd_path};
                proxy_pass http://127.0.0.1:{web_port}/;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_hide_header Server;
            }}
            location ^~ /admin/minio/ {{
                auth_basic "Restricted Admin";
                auth_basic_user_file {htpasswd_path};
                proxy_pass http://127.0.0.1:9001/;
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_hide_header Server;
            }}"""

    def _check_nginx_includes(self) -> Dict[str, bool]:
        """
        Liest nginx.conf readonly — prüft ob box-enabled include vorhanden.
        Schreibt NICHTS. Gibt zurück was fehlt.
        """
        conf_path = Path(DEFAULT_CONF_PATH)
        if not conf_path.exists():
            return {"conf_exists": False, "box_enabled": False}
        try:
            content = conf_path.read_text()
            return {
                "conf_exists": True,
                "box_enabled": "box-enabled" in content,
            }
        except PermissionError:
            logger.warning(f"Cannot read {conf_path} — permission denied")
            return {"conf_exists": True, "box_enabled": False}

    def patch_nginx_conf_include(self) -> bool:
        """
        NON-INVASIV: Fügt NUR die fehlende include-Zeile in den http-Block ein.
        Schreibt nie die gesamte nginx.conf neu.
        Bricht ab wenn http-Block nicht eindeutig erkennbar.
        """
        if IS_WINDOWS:
            logger.warning("patch_nginx_conf_include not supported on Windows")
            return False

        conf_path = Path(DEFAULT_CONF_PATH)
        if not conf_path.exists():
            logger.error(f"nginx.conf not found at {conf_path}")
            return False

        try:
            content = conf_path.read_text()
        except PermissionError:
            logger.error(f"Cannot read {conf_path} — run with sudo")
            return False

        if "box-enabled" in content:
            logger.info("include box-enabled already present — nothing to do")
            return True

        # Suche das letzte '}' im http-Block — hänge include davor
        # Strategie: letztes '}' das auf eigener Zeile steht
        lines = content.splitlines()
        insert_at = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "}":
                insert_at = i
                break

        if insert_at is None:
            logger.error("Could not locate closing brace of http block — aborting patch")
            return False

        include_lines = [
            "",
            "    # ToolBoxV2 box configs — managed by cli_worker_manager",
            "    include /etc/nginx/box-enabled/*;",
        ]
        for j, line in enumerate(include_lines):
            lines.insert(insert_at + j, line)

        new_content = "\n".join(lines) + "\n"

        # Backup vor dem Schreiben
        backup_path = conf_path.with_suffix(".conf.bak")
        try:
            backup_path.write_text(content)
            logger.info(f"nginx.conf backup → {backup_path}")
        except Exception as e:
            logger.warning(f"Could not write backup: {e}")

        try:
            conf_path.write_text(new_content)
            logger.info(f"Patched {conf_path} — include box-enabled added")
            return True
        except PermissionError:
            logger.error("Cannot write nginx.conf — run with sudo")
            return False
        except Exception as e:
            logger.error(f"Patch failed: {e}")
            return False

    def ensure_nginx_ready(self) -> bool:
        """
        Einmalig beim Start: prüft Include, warnt oder patcht.
        Aufgerufen von start_all() und nginx-init CLI.
        Gibt False zurück wenn nginx nicht bereit ist.
        """
        checks = self._check_nginx_includes()

        if not checks["conf_exists"]:
            logger.error(
                f"nginx.conf not found at {DEFAULT_CONF_PATH}. "
                "Install nginx first: sudo apt install nginx"
            )
            return False

        if not checks["box_enabled"]:
            logger.warning(
                "nginx.conf missing: include /etc/nginx/box-enabled/*\n"
                "  Auto-patching... (backup saved as nginx.conf.bak)\n"
                "  To do this manually run: tb workers nginx-init"
            )
            if not self.patch_nginx_conf_include():
                logger.error(
                    "Auto-patch failed. Add manually to http block in nginx.conf:\n"
                    "    include /etc/nginx/box-enabled/*;"
                )
                return False

        return True

    # =========================================================================
    # Write methods
    # =========================================================================


    def write_htpasswd(self) -> Optional[str]:
        pwd = os.environ.get("ADMIN_UI_PASSWORD", "")
        if not pwd:
            logger.warning("ADMIN_UI_PASSWORD not set — admin UI disabled")
            return None
        if IS_WINDOWS:
            logger.warning("htpasswd not supported on Windows")
            return None

        htpasswd_path = "/etc/nginx/admin_htpasswd"
        hashed = _hash_password_for_nginx(pwd)
        if not hashed:
            logger.error("No password hashing backend available — pip install passlib")
            return None

        try:
            os.makedirs(os.path.dirname(htpasswd_path), exist_ok=True)
            with open(htpasswd_path, "w") as f:
                f.write(f"admin:{hashed}\n")
            os.chmod(htpasswd_path, 0o640)  # nginx-readable, nicht world-readable
            logger.info(f"htpasswd written → {htpasswd_path}")
            return htpasswd_path
        except PermissionError:
            logger.error("Cannot write htpasswd — run with sudo")
            return None
        except Exception as e:
            logger.error(f"write_htpasswd failed: {e}")
            return None

    def write_site_config(
        self,
        max_http_workers: int,
        max_ws_workers: int,
        base_http_port: int,
        base_ws_port: int,
        remote_nodes: List[Tuple[str, int]] = None,
        site_name: str = "toolbox",
        force: bool = False,
        write_htpasswd: bool = True,
    ) -> bool:
        available_dir = Path(DEFAULT_BOX_AVAILABLE)
        enabled_dir = Path(DEFAULT_BOX_ENABLED)
        config_path = available_dir / site_name
        symlink_path = enabled_dir / site_name

        if config_path.exists() and not force:
            logger.info(f"Site config exists at {config_path} — skipping (use force=True)")
            self._ensure_symlink(config_path, symlink_path)
            return True

        if write_htpasswd and not IS_WINDOWS:
            self.write_htpasswd()

        content = self.generate_site_config(
            max_http_workers, max_ws_workers,
            base_http_port, base_ws_port, remote_nodes,
        )
        try:
            available_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(content)
            self._ensure_symlink(config_path, symlink_path)
            logger.info(f"Site config written → {config_path}")
            return True
        except PermissionError:
            logger.error("Permission denied — run with sudo")
            return False
        except Exception as e:
            logger.error(f"Failed to write site config: {e}")
            return False

    def _ensure_symlink(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            return
        os.symlink(src, dst)
        logger.info(f"Symlink created: {dst} → {src}")

    def test_config(self) -> bool:
        if not self._nginx_path:
            return False
        try:
            result = subprocess.run([self._nginx_path, "-t"], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Nginx config test failed: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Config test error: {e}")
            return False

    def reload(self) -> bool:
        if not self._nginx_path:
            return False
        try:
            subprocess.run([self._nginx_path, "-s", "reload"], check=True, capture_output=True)
            logger.info("Nginx reloaded")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Nginx reload failed: {e.stderr}")
            return False

    def start(self) -> bool:
        if not self._nginx_path:
            return False
        try:
            subprocess.run([self._nginx_path], check=True, capture_output=True)
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
        worker.run(port=port)
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
        # Unix sockets disabled on all platforms - use TCP ports only
        # This ensures consistent behavior across Windows, Linux, and macOS
        return None

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

    def _write_initial_nginx_config(self, force: bool = False) -> None:
        """Write nginx site config ONCE. Certbot owns the file after that."""
        cfg = self.config
        self._nginx.write_site_config(
            max_http_workers=cfg.http_worker.workers,
            max_ws_workers=getattr(cfg.ws_worker, 'max_workers', 4),
            base_http_port=cfg.http_worker.port,
            base_ws_port=cfg.ws_worker.port,
            remote_nodes=self._cluster.get_remote_addresses(),
            site_name="toolbox",
            force=force,
        )

    def _update_nginx_config(self) -> None:
        """Reload nginx — never rewrites the config file."""
        if self._nginx.test_config():
            self._nginx.reload()
        else:
            logger.error("Nginx config test failed — reload skipped")

    def start_all(self) -> bool:
        logger.info("Starting all services...")

        if not self.start_broker():
            return False

        for _ in range(self.config.http_worker.workers):
            self.start_http_worker()
        self.start_ws_worker()

        self._metrics_collector.start(self._workers)
        self._health_checker.start(self._workers)
        self._cluster.start()

        if self.config.nginx.enabled:
            self._write_initial_nginx_config(force=False)
            if not self._nginx.is_installed():
                self._nginx.install()
            # NEU: include-Check vor reload
            if not self._nginx.ensure_nginx_ready():
                logger.warning("nginx not fully configured — workers running but nginx may not route traffic")
            elif self._nginx.test_config():
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
                info = (self.start_http_worker() if wtype == WorkerType.HTTP
                        else self.start_ws_worker())
                if info:
                    started.append(info.worker_id)
        elif target < len(current):
            for w in current[:len(current) - target]:
                self.stop_worker(w.worker_id)
                stopped.append(w.worker_id)

        if started or stopped:
            self._health_checker.update_workers(self._workers)
            self._metrics_collector.update_workers(self._workers)
            self._update_nginx_config()  # enthält reload — kein zweites reload mehr

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
        path = urlparse(self.path).path
        if path.startswith("/admin/manager"):
            path = path[len("/admin/manager"):]  # exakt 14 chars
        path = path.rstrip("/") or "/"
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
        path = urlparse(self.path).path
        if path.startswith("/admin/manager"):
            path = path[len("/admin/manager"):]  # exakt 14 chars
        path = path.rstrip("/") or "/"
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
        origin = self.headers.get("Origin", "")
        allowed = {
            "http://localhost",
            f"http://127.0.0.1",
            f"https://simplecore.app",
            f"http://127.0.0.1:{self.server.server_address[1]}",
        }
        if origin in allowed:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
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

    parser = argparse.ArgumentParser(
        description="ToolBoxV2 Worker Manager",
        prog="tb workers",
        epilog=textwrap.dedent("""tb workers start             Startet alle Services + Web UI
tb workers stop              Stoppt alles
tb workers restart           stop + start
tb workers status            JSON Status-Dump
tb workers update            Rolling Update aller HTTP Worker

tb workers nginx-init        Patcht nginx.conf include (einmalig, non-invasiv)
tb workers nginx-check       Readonly Status aller nginx Pfade
tb workers nginx-check --patch   ... + patcht include wenn fehlend
tb workers nginx-config      Schreibt box-available/toolbox
tb workers nginx-config --force          Überschreibt existierende Config
tb workers nginx-config --write-htpasswd Schreibt htpasswd neu (ADMIN_UI_PASSWORD)
tb workers nginx-reload      test + reload

tb workers worker-start -t http   Startet einzelnen HTTP Worker
tb workers worker-start -t ws     Startet einzelnen WS Worker
tb workers worker-stop -w <id>    Stoppt Worker by ID

tb workers cluster-join --host H --port P --secret S

tb workers debug             Startet Debug-Server auf dist/""")
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
            "update",
            "nginx-init",       # NEU: prüft/patcht nginx.conf include
            "nginx-config",
            "nginx-reload",
            "nginx-check",      # NEU: readonly status-check, kein schreiben
            "worker-start",
            "worker-stop",
            "cluster-join",
            "debug",
        ],
    )
    parser.add_argument("-c", "--config",     help="Config file path")
    parser.add_argument("--force",            action="store_true",
                        help="nginx-config: Overwrite existing site config")
    parser.add_argument("--write-htpasswd",   action="store_true",
                        help="nginx-config: (Re)write htpasswd from ADMIN_UI_PASSWORD")
    parser.add_argument("--patch",            action="store_true",
                        help="nginx-check: Patch nginx.conf include if missing")
    parser.add_argument("-w", "--worker-id",  help="worker-start/stop: Worker ID")
    parser.add_argument("-t", "--type",       choices=["http", "ws"], default="http",
                        help="worker-start: Worker type")
    parser.add_argument("--host",             help="cluster-join: Remote host")
    parser.add_argument("--port",             type=int,
                        help="cluster-join: Remote port")
    parser.add_argument("--secret",           help="cluster-join: Cluster secret")
    parser.add_argument("--no-ui",            action="store_true",
                        help="start: Disable web UI")
    parser.add_argument("-v", "--verbose",    action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from toolboxv2.utils.workers.config import load_config
    config  = load_config(args.config)
    manager = WorkerManager(config)

    # -------------------------------------------------------------------------
    if args.command == "start":
        if not manager.start_all():
            sys.exit(1)
        if config.manager.web_ui_enabled and not args.no_ui:
            Thread(
                target=run_web_ui,
                args=(manager, config.manager.web_ui_host, config.manager.web_ui_port),
                daemon=True,
            ).start()
        try:
            while manager._running:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_all()

    # -------------------------------------------------------------------------
    elif args.command == "stop":
        manager.stop_all()

    # -------------------------------------------------------------------------
    elif args.command == "restart":
        manager.stop_all()
        time.sleep(2)
        manager.start_all()

    # -------------------------------------------------------------------------
    elif args.command == "status":
        print(json.dumps(manager.get_status(), indent=2))

    # -------------------------------------------------------------------------
    elif args.command == "update":
        manager.rolling_update()

    # -------------------------------------------------------------------------
    elif args.command == "nginx-init":
        """
        Prüft ob nginx.conf das box-enabled include enthält.
        Patcht minimal wenn nicht — schreibt NIEMALS die gesamte nginx.conf neu.
        """
        checks = manager._nginx._check_nginx_includes()
        if not checks["conf_exists"]:
            print(f"✗ nginx.conf not found at {DEFAULT_CONF_PATH}")
            sys.exit(1)
        if checks["box_enabled"]:
            print("✓ nginx.conf already contains include box-enabled — nothing to do")
            sys.exit(0)
        print(f"  nginx.conf at {DEFAULT_CONF_PATH} missing include box-enabled")
        print("  Patching (backup → nginx.conf.bak)...")
        if manager._nginx.patch_nginx_conf_include():
            print("✓ Patched successfully")
            if manager._nginx.test_config():
                manager._nginx.reload()
                print("✓ nginx reloaded")
            else:
                print("✗ nginx config test failed after patch — check manually")
                sys.exit(1)
        else:
            print("✗ Patch failed — add manually to http block in nginx.conf:")
            print("      include /etc/nginx/box-enabled/*;")
            sys.exit(1)

    # -------------------------------------------------------------------------
    elif args.command == "nginx-check":
        """
        Readonly status-check — schreibt nichts.
        Mit --patch: patcht include falls fehlend.
        """
        checks = manager._nginx._check_nginx_includes()
        print(f"nginx.conf path:     {DEFAULT_CONF_PATH}")
        print(f"conf exists:         {'✓' if checks['conf_exists'] else '✗'}")
        print(f"box-enabled include: {'✓' if checks['box_enabled'] else '✗ MISSING'}")
        print(f"box-available dir:   {'✓' if Path(DEFAULT_BOX_AVAILABLE).exists() else '✗'}")
        print(f"box-enabled dir:     {'✓' if Path(DEFAULT_BOX_ENABLED).exists() else '✗'}")
        site_path = Path(DEFAULT_BOX_AVAILABLE) / "toolbox"
        symlink_path = Path(DEFAULT_BOX_ENABLED) / "toolbox"
        print(f"toolbox config:      {'✓' if site_path.exists() else '✗'}")
        print(f"toolbox symlink:     {'✓' if symlink_path.exists() else '✗'}")
        print(f"nginx installed:     {'✓' if manager._nginx.is_installed() else '✗'}")
        print(f"nginx version:       {manager._nginx.get_version() or 'n/a'}")
        htpasswd = Path("/etc/nginx/admin_htpasswd")
        print(f"htpasswd:            {'✓' if htpasswd.exists() else '✗ (run nginx-config --write-htpasswd)'}")

        if not checks["box_enabled"] and getattr(args, "patch", False):
            print("\n  --patch set: patching nginx.conf...")
            if manager._nginx.patch_nginx_conf_include():
                print("✓ Done")
            else:
                sys.exit(1)
        elif not checks["box_enabled"]:
            print("\n  Run: tb workers nginx-init   to patch automatically")
            sys.exit(1)

    # -------------------------------------------------------------------------
    elif args.command == "nginx-config":
        if getattr(args, "write_htpasswd", False):
            path = manager._nginx.write_htpasswd()
            if path:
                print(f"✓ htpasswd written to {path}")
            else:
                print("✗ htpasswd failed — is ADMIN_UI_PASSWORD set?")
                sys.exit(1)
        ok = manager._nginx.write_site_config(
            max_http_workers=config.http_worker.workers,
            max_ws_workers=getattr(config.ws_worker, "max_workers", 4),
            base_http_port=config.http_worker.port,
            base_ws_port=config.ws_worker.port,
            site_name="toolbox",
            force=getattr(args, "force", False),
            write_htpasswd=False,  # bereits oben erledigt wenn --write-htpasswd
        )
        if ok:
            print(f"✓ Site config written (force={args.force})")
            if manager._nginx.test_config():
                print("✓ nginx config test passed")
            else:
                print("✗ nginx config test failed")
                sys.exit(1)
        else:
            print("✗ Failed — check logs")
            sys.exit(1)

    # -------------------------------------------------------------------------
    elif args.command == "nginx-reload":
        if manager._nginx.test_config():
            if manager._nginx.reload():
                print("✓ nginx reloaded")
            else:
                print("✗ reload failed")
                sys.exit(1)
        else:
            print("✗ config test failed — reload aborted")
            sys.exit(1)

    # -------------------------------------------------------------------------
    elif args.command == "worker-start":
        info = (
            manager.start_http_worker(args.worker_id)
            if args.type == "http"
            else manager.start_ws_worker(args.worker_id)
        )
        print(json.dumps(info.to_dict() if info else {"error": "failed"}, indent=2))

    # -------------------------------------------------------------------------
    elif args.command == "worker-stop":
        manager.stop_worker(args.worker_id)

    # -------------------------------------------------------------------------
    elif args.command == "cluster-join":
        if not args.host:
            print("✗ --host required")
            sys.exit(1)
        if manager.add_cluster_node(args.host, args.port or 9000, args.secret):
            print(f"✓ Joined {args.host}:{args.port or 9000}")
        else:
            print("✗ Join failed")
            sys.exit(1)

    # -------------------------------------------------------------------------
    elif args.command == "debug":
        path = tb_root_dir / "dist"
        if not path.exists():
            os.system("npm run build")
        if path.exists():
            run_debug_server(path, args.port or 8080)
        else:
            print(f"✗ dist not found at {path}")
            sys.exit(1)


if __name__ == "__main__":
    main()

"""
tb workers start             Startet alle Services + Web UI
tb workers stop              Stoppt alles
tb workers restart           stop + start
tb workers status            JSON Status-Dump
tb workers update            Rolling Update aller HTTP Worker

tb workers nginx-init        Patcht nginx.conf include (einmalig, non-invasiv)
tb workers nginx-check       Readonly Status aller nginx Pfade
tb workers nginx-check --patch   ... + patcht include wenn fehlend
tb workers nginx-config      Schreibt box-available/toolbox
tb workers nginx-config --force          Überschreibt existierende Config
tb workers nginx-config --write-htpasswd Schreibt htpasswd neu (ADMIN_UI_PASSWORD)
tb workers nginx-reload      test + reload

tb workers worker-start -t http   Startet einzelnen HTTP Worker
tb workers worker-start -t ws     Startet einzelnen WS Worker
tb workers worker-stop -w <id>    Stoppt Worker by ID

tb workers cluster-join --host H --port P --secret S

tb workers debug             Startet Debug-Server auf dist/

certbot install --cert-name simplecore.app

"""


if __name__ == "__main__":
    main()
