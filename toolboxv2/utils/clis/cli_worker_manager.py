#!/usr/bin/env python3
"""
cli_worker_manager.py - Complete Worker Manager for ToolBoxV2

Orchestrates:
- Nginx installation and configuration
- HTTP and WebSocket worker processes
- ZeroMQ event broker
- Zero-downtime rolling updates
- Health monitoring
- Minimal web UI
- CLI interface
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


# ============================================================================
# Worker State
# ============================================================================

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
class WorkerInfo:
    """Worker process information."""
    worker_id: str
    worker_type: WorkerType
    pid: int
    port: int
    state: WorkerState = WorkerState.STOPPED
    started_at: float = 0.0
    restart_count: int = 0
    last_health_check: float = 0.0
    healthy: bool = False

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type.value,
            "pid": self.pid,
            "port": self.port,
            "state": self.state.value,
            "started_at": self.started_at,
            "restart_count": self.restart_count,
            "last_health_check": self.last_health_check,
            "healthy": self.healthy,
            "uptime": time.time() - self.started_at if self.started_at > 0 else 0,
        }


# ============================================================================
# Nginx Manager
# ============================================================================

class NginxManager:
    """Manage Nginx installation and configuration."""

    def __init__(self, config):
        self.config = config.nginx
        self._nginx_path = self._find_nginx()

    def _find_nginx(self) -> str | None:
        """Find nginx binary."""
        paths = [
            "/usr/sbin/nginx",
            "/usr/local/sbin/nginx",
            "/opt/nginx/sbin/nginx",
            shutil.which("nginx"),
        ]

        for path in paths:
            if path and os.path.exists(path):
                return path

        return None

    def is_installed(self) -> bool:
        """Check if nginx is installed."""
        return self._nginx_path is not None

    def install(self) -> bool:
        """Install nginx (Linux only)."""
        system = platform.system().lower()

        if system != "linux":
            logger.error("Auto-install only supported on Linux")
            return False

        try:
            # Detect package manager
            if shutil.which("apt-get"):
                subprocess.run(
                    ["sudo", "apt-get", "update"],
                    check=True, capture_output=True
                )
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "nginx"],
                    check=True, capture_output=True
                )
            elif shutil.which("yum"):
                subprocess.run(
                    ["sudo", "yum", "install", "-y", "nginx"],
                    check=True, capture_output=True
                )
            elif shutil.which("dnf"):
                subprocess.run(
                    ["sudo", "dnf", "install", "-y", "nginx"],
                    check=True, capture_output=True
                )
            else:
                logger.error("No supported package manager found")
                return False

            self._nginx_path = self._find_nginx()
            return self._nginx_path is not None

        except subprocess.CalledProcessError as e:
            logger.error(f"Nginx install failed: {e}")
            return False

    def generate_config(self, http_ports: List[int], ws_ports: List[int]) -> str:
        """Generate nginx configuration."""
        cfg = self.config

        # HTTP upstream
        http_servers = "\n        ".join(
            f"server 127.0.0.1:{port} weight=1 max_fails=3 fail_timeout=30s;"
            for port in http_ports
        )

        # WS upstream
        ws_servers = "\n        ".join(
            f"server 127.0.0.1:{port};"
            for port in ws_ports
        )

        # Rate limiting
        rate_limit_zone = ""
        rate_limit_directive = ""
        if cfg.rate_limit_enabled:
            rate_limit_zone = f"""
    # Rate limiting
    limit_req_zone $binary_remote_addr zone={cfg.rate_limit_zone}:10m rate={cfg.rate_limit_rate};
"""
            rate_limit_directive = f"""
        limit_req zone={cfg.rate_limit_zone} burst={cfg.rate_limit_burst} nodelay;
"""

        # SSL configuration
        ssl_config = ""
        listen_directive = f"listen {cfg.listen_port};"
        if cfg.ssl_enabled and cfg.ssl_certificate:
            listen_directive = f"""listen {cfg.listen_ssl_port} ssl http2;
        ssl_certificate {cfg.ssl_certificate};
        ssl_certificate_key {cfg.ssl_certificate_key};
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_prefer_server_ciphers on;"""

        # Static files
        static_location = ""
        if cfg.static_enabled:
            static_root = os.path.abspath(cfg.static_root)
            static_location = f"""
        # Static files
        location / {{
            root {static_root};
            try_files $uri $uri/ /index.html;
            expires 1h;
            add_header Cache-Control "public, immutable";
        }}
"""

        config_content = f"""# ToolBoxV2 Nginx Configuration
# Generated by cli_worker_manager.py

worker_processes auto;
error_log {cfg.error_log} warn;
pid {cfg.pid_file};

events {{
    worker_connections 10240;
    use epoll;
    multi_accept on;
}}

http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log {cfg.access_log} main;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml application/json application/javascript
               application/xml application/xml+rss text/javascript;
{rate_limit_zone}
    # HTTP Backend
    upstream {cfg.upstream_http} {{
        least_conn;
        {http_servers}
        keepalive 32;
    }}

    # WebSocket Backend
    upstream {cfg.upstream_ws} {{
        hash $request_uri consistent;
        {ws_servers}
    }}

    server {{
        {listen_directive}
        server_name {cfg.server_name};
{static_location}
        # API endpoints
        location /api/ {{
{rate_limit_directive}
            proxy_pass http://{cfg.upstream_http};
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 10s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            proxy_buffering off;
        }}

        # SSE endpoints
        location /api/stream/ {{
            proxy_pass http://{cfg.upstream_http};
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding off;
        }}

        # WebSocket endpoint
        location /ws {{
            proxy_pass http://{cfg.upstream_ws};
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
        }}

        # Health check
        location /health {{
            proxy_pass http://{cfg.upstream_http}/health;
            proxy_connect_timeout 2s;
            proxy_read_timeout 2s;
        }}

        # Manager UI (internal only)
        location /manager/ {{
            allow 127.0.0.1;
            deny all;
            proxy_pass http://127.0.0.1:9000/;
        }}
    }}
}}
"""
        return config_content

    def write_config(self, http_ports: List[int], ws_ports: List[int]) -> bool:
        """Write nginx configuration file."""
        config_content = self.generate_config(http_ports, ws_ports)

        try:
            # Write to config path
            config_path = Path(self.config.config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, "w") as f:
                f.write(config_content)

            # Create symlink
            symlink_path = Path(self.config.symlink_path)
            if symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(config_path)

            logger.info(f"Nginx config written to {config_path}")
            return True

        except PermissionError:
            logger.error("Permission denied writing nginx config. Run with sudo?")
            return False
        except Exception as e:
            logger.error(f"Failed to write nginx config: {e}")
            return False

    def test_config(self) -> bool:
        """Test nginx configuration."""
        if not self._nginx_path:
            return False

        try:
            result = subprocess.run(
                [self._nginx_path, "-t"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Nginx test failed: {e}")
            return False

    def reload(self) -> bool:
        """Reload nginx configuration."""
        if not self._nginx_path:
            return False

        try:
            subprocess.run(
                [self._nginx_path, "-s", "reload"],
                check=True, capture_output=True
            )
            logger.info("Nginx reloaded")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Nginx reload failed: {e}")
            return False

    def start(self) -> bool:
        """Start nginx."""
        if not self._nginx_path:
            return False

        try:
            subprocess.run(
                [self._nginx_path],
                check=True, capture_output=True
            )
            logger.info("Nginx started")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Nginx start failed: {e}")
            return False

    def stop(self) -> bool:
        """Stop nginx."""
        if not self._nginx_path:
            return False

        try:
            subprocess.run(
                [self._nginx_path, "-s", "stop"],
                check=True, capture_output=True
            )
            logger.info("Nginx stopped")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Nginx stop failed: {e}")
            return False


# ============================================================================
# Worker Process Functions (Module-level for Windows multiprocessing)
# ============================================================================

def _run_broker_process(config_dict: Dict):
    """Run ZMQ broker in separate process. Must be module-level for Windows."""
    import asyncio
    import sys

    # Windows: Use SelectorEventLoop for ZMQ compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Try relative import first, then absolute
    from toolboxv2.utils.workers.config import Config
    from toolboxv2.utils.workers.event_manager import run_broker

    # Reconstruct config from dict
    config = Config.from_dict(config_dict)
    asyncio.run(run_broker(config))


def _run_http_worker_process(worker_id: str, config_dict: Dict, port: int):
    """Run HTTP worker in separate process. Must be module-level for Windows."""
    # Try relative import first, then absolute
    from toolboxv2.utils.workers.config import Config
    from toolboxv2.utils.workers.server_worker import HTTPWorker
    config = Config.from_dict(config_dict)
    worker = HTTPWorker(worker_id, config)
    worker.run(port=port)


def _run_ws_worker_process(worker_id: str, config_dict: Dict, port: int):
    """Run WS worker in separate process. Must be module-level for Windows."""
    import asyncio
    import sys

    # Windows: Use SelectorEventLoop for ZMQ compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Try relative import first, then absolute
    from toolboxv2.utils.workers.config import Config
    from toolboxv2.utils.workers.ws_worker import WSWorker

    import asyncio
    config = Config.from_dict(config_dict)
    # Override port in config
    config.ws_worker.port = port
    worker = WSWorker(worker_id, config)
    asyncio.run(worker.run())


# ============================================================================
# Worker Manager
# ============================================================================

class WorkerManager:
    """Manages all worker processes."""

    def __init__(self, config):
        self.config = config
        self._workers: Dict[str, WorkerInfo] = {}
        self._processes: Dict[str, Process] = {}
        self._nginx = NginxManager(config)
        self._broker_process: Process | None = None
        self._running = False
        self._health_task: asyncio.Task | None = None

        # Port allocation - ensure no overlap
        # HTTP: 8000, 8001, 8002, ...
        # WS: 8100, 8101, 8102, ... (separate range)
        self._next_http_port = config.http_worker.port
        # WS port should be well separated from HTTP range
        ws_base = config.ws_worker.port
        if ws_base < config.http_worker.port + 100:
            ws_base = config.http_worker.port + 100  # Ensure 100 port gap
        self._next_ws_port = ws_base

    def _allocate_http_port(self) -> int:
        """Allocate next HTTP port."""
        port = self._next_http_port
        self._next_http_port += 1
        return port

    def _allocate_ws_port(self) -> int:
        """Allocate next WS port."""
        port = self._next_ws_port
        self._next_ws_port += 1
        return port

    def start_broker(self) -> bool:
        """Start ZMQ broker process."""
        if self._broker_process and self._broker_process.is_alive():
            return True

        # Serialize config to dict for cross-process transfer
        config_dict = self.config.to_dict()

        self._broker_process = Process(
            target=_run_broker_process,
            args=(config_dict,),
            name="zmq_broker"
        )
        self._broker_process.start()

        time.sleep(0.5)  # Wait for broker to start

        if self._broker_process.is_alive():
            logger.info(f"ZMQ broker started (PID: {self._broker_process.pid})")
            return True

        logger.error("ZMQ broker failed to start")
        return False

    def stop_broker(self):
        """Stop ZMQ broker."""
        if self._broker_process and self._broker_process.is_alive():
            self._broker_process.terminate()
            self._broker_process.join(timeout=5)
            if self._broker_process.is_alive():
                self._broker_process.kill()
            logger.info("ZMQ broker stopped")

    def start_http_worker(self, worker_id: str = None, port: int = None) -> WorkerInfo | None:
        """Start an HTTP worker."""
        if worker_id is None:
            worker_id = f"http_{uuid.uuid4().hex[:8]}"

        if port is None:
            port = self._allocate_http_port()

        # Serialize config to dict for cross-process transfer
        config_dict = self.config.to_dict()

        process = Process(
            target=_run_http_worker_process,
            args=(worker_id, config_dict, port),
            name=worker_id
        )
        process.start()

        info = WorkerInfo(
            worker_id=worker_id,
            worker_type=WorkerType.HTTP,
            pid=process.pid,
            port=port,
            state=WorkerState.STARTING,
            started_at=time.time(),
        )

        self._workers[worker_id] = info
        self._processes[worker_id] = process

        # Wait for startup
        time.sleep(0.5)
        if process.is_alive():
            info.state = WorkerState.RUNNING
            logger.info(f"HTTP worker started: {worker_id} (port {port})")
            return info

        info.state = WorkerState.FAILED
        logger.error(f"HTTP worker failed to start: {worker_id}")
        return None

    def start_ws_worker(self, worker_id: str = None, port: int = None) -> WorkerInfo | None:
        """Start a WebSocket worker."""
        if worker_id is None:
            worker_id = f"ws_{uuid.uuid4().hex[:8]}"

        if port is None:
            port = self._allocate_ws_port()

        # Serialize config to dict for cross-process transfer
        config_dict = self.config.to_dict()

        process = Process(
            target=_run_ws_worker_process,
            args=(worker_id, config_dict, port),
            name=worker_id
        )
        process.start()

        info = WorkerInfo(
            worker_id=worker_id,
            worker_type=WorkerType.WS,
            pid=process.pid,
            port=port,
            state=WorkerState.STARTING,
            started_at=time.time(),
        )

        self._workers[worker_id] = info
        self._processes[worker_id] = process

        time.sleep(0.5)
        if process.is_alive():
            info.state = WorkerState.RUNNING
            logger.info(f"WS worker started: {worker_id} (port {port})")
            return info

        info.state = WorkerState.FAILED
        logger.error(f"WS worker failed to start: {worker_id}")
        return None

    def stop_worker(self, worker_id: str, graceful: bool = True) -> bool:
        """Stop a worker."""
        if worker_id not in self._processes:
            return False

        info = self._workers.get(worker_id)
        process = self._processes[worker_id]

        if info:
            info.state = WorkerState.STOPPING

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
        """Restart a worker."""
        if worker_id not in self._workers:
            return None

        info = self._workers[worker_id]
        port = info.port
        worker_type = info.worker_type

        self.stop_worker(worker_id)
        del self._workers[worker_id]

        if worker_type == WorkerType.HTTP:
            return self.start_http_worker(worker_id, port)
        elif worker_type == WorkerType.WS:
            return self.start_ws_worker(worker_id, port)

        return None

    def _update_nginx_config(self):
        """Update nginx configuration with current workers."""
        http_ports = [
            w.port for w in self._workers.values()
            if w.worker_type == WorkerType.HTTP and w.state == WorkerState.RUNNING
        ]
        ws_ports = [
            w.port for w in self._workers.values()
            if w.worker_type == WorkerType.WS and w.state == WorkerState.RUNNING
        ]

        self._nginx.write_config(http_ports, ws_ports)

    def start_all(self):
        """Start all workers and nginx."""
        logger.info("Starting all services...")

        # Start broker first
        if not self.start_broker():
            logger.error("Failed to start broker")
            return False

        # Start HTTP workers
        num_http = self.config.http_worker.workers
        for i in range(num_http):
            self.start_http_worker()

        # Start WS worker (single instance for now)
        self.start_ws_worker()

        # Configure and start nginx
        if self.config.nginx.enabled:
            self._update_nginx_config()

            if not self._nginx.is_installed():
                logger.warning("Nginx not installed. Attempting to install...")
                if not self._nginx.install():
                    logger.error("Nginx installation failed")

            if self._nginx.test_config():
                self._nginx.reload()
            else:
                logger.error("Nginx config test failed")

        self._running = True
        logger.info("All services started")
        return True

    def stop_all(self, graceful: bool = True):
        """Stop all workers and nginx."""
        logger.info("Stopping all services...")
        self._running = False

        # Stop workers
        for worker_id in list(self._processes.keys()):
            self.stop_worker(worker_id, graceful=graceful)

        # Stop broker
        self.stop_broker()

        logger.info("All services stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "running": self._running,
            "broker_alive": self._broker_process.is_alive() if self._broker_process else False,
            "workers": {wid: w.to_dict() for wid, w in self._workers.items()},
            "nginx": {
                "installed": self._nginx.is_installed(),
                "enabled": self.config.nginx.enabled,
            },
        }

    def get_workers(self) -> List[Dict]:
        """Get all workers."""
        return [w.to_dict() for w in self._workers.values()]

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all workers."""
        total_requests = 0
        total_connections = 0
        total_errors = 0
        latencies = []

        for worker in self._workers.values():
            # In real implementation, fetch from workers via ZMQ
            # For now, return placeholder data
            pass

        http_count = sum(1 for w in self._workers.values() if w.worker_type == WorkerType.HTTP)
        ws_count = sum(1 for w in self._workers.values() if w.worker_type == WorkerType.WS)

        return {
            "total_workers": len(self._workers),
            "http_workers": http_count,
            "ws_workers": ws_count,
            "total_requests": total_requests,
            "total_connections": total_connections,
            "total_errors": total_errors,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        }

    def get_health(self) -> Dict[str, Any]:
        """Get health status of all components."""
        workers_healthy = all(
            w.state == WorkerState.RUNNING and self._processes.get(w.worker_id, None) and self._processes[w.worker_id].is_alive()
            for w in self._workers.values()
        )

        broker_healthy = self._broker_process and self._broker_process.is_alive()

        return {
            "healthy": workers_healthy and broker_healthy,
            "broker": {
                "alive": broker_healthy,
                "pid": self._broker_process.pid if self._broker_process else None,
            },
            "workers": {
                wid: {
                    "healthy": self._processes.get(wid) and self._processes[wid].is_alive(),
                    "state": w.state.value,
                }
                for wid, w in self._workers.items()
            },
            "nginx": {
                "installed": self._nginx.is_installed(),
            },
        }

    def scale_workers(self, worker_type: str, target_count: int) -> Dict[str, Any]:
        """Scale workers to target count."""
        wtype = WorkerType.HTTP if worker_type == "http" else WorkerType.WS
        current_workers = [w for w in self._workers.values() if w.worker_type == wtype]
        current_count = len(current_workers)

        started = []
        stopped = []

        if target_count > current_count:
            # Scale up
            for _ in range(target_count - current_count):
                if wtype == WorkerType.HTTP:
                    info = self.start_http_worker()
                else:
                    info = self.start_ws_worker()
                if info:
                    started.append(info.worker_id)

        elif target_count < current_count:
            # Scale down - stop oldest first
            to_stop = current_workers[:current_count - target_count]
            for worker in to_stop:
                self.stop_worker(worker.worker_id)
                stopped.append(worker.worker_id)

        # Update nginx config
        if started or stopped:
            self._update_nginx_config()
            if self._nginx.is_installed():
                self._nginx.reload()

        return {
            "status": "ok",
            "previous_count": current_count,
            "current_count": target_count,
            "started": started,
            "stopped": stopped,
            "message": f"Scaled {worker_type} from {current_count} to {len([w for w in self._workers.values() if w.worker_type == wtype])}",
        }

    def rolling_update(self, delay: float = None, validate: bool = True):
        """Perform zero-downtime rolling update with validation."""
        if delay is None:
            delay = self.config.manager.rolling_update_delay

        logger.info("Starting rolling update...")

        http_workers = [w for w in self._workers.values() if w.worker_type == WorkerType.HTTP]
        ws_workers = [w for w in self._workers.values() if w.worker_type == WorkerType.WS]

        # Update HTTP workers one by one
        for info in list(http_workers):
            logger.info(f"Rolling update: replacing {info.worker_id}")

            # Start new worker
            new_info = self.start_http_worker()
            if not new_info:
                logger.error(f"Failed to start replacement for {info.worker_id}")
                continue

            # Wait for startup
            time.sleep(delay)

            # Validate new worker
            if validate and not self._validate_worker(new_info):
                logger.error(f"Validation failed for {new_info.worker_id}, rolling back")
                self.stop_worker(new_info.worker_id)
                continue

            logger.info(f"New worker {new_info.worker_id} validated successfully")

            # Update nginx
            self._update_nginx_config()
            if self._nginx.is_installed():
                self._nginx.reload()

            # Drain old worker
            info.state = WorkerState.DRAINING
            time.sleep(delay)

            # Stop old worker
            self.stop_worker(info.worker_id)
            logger.info(f"Replaced {info.worker_id} with {new_info.worker_id}")

        # Update WS workers
        for info in list(ws_workers):
            logger.info(f"Rolling update: replacing WS {info.worker_id}")

            new_info = self.start_ws_worker()
            if not new_info:
                logger.error(f"Failed to start replacement for {info.worker_id}")
                continue

            time.sleep(delay)

            if validate and not self._validate_worker(new_info):
                logger.error(f"Validation failed for {new_info.worker_id}")
                self.stop_worker(new_info.worker_id)
                continue

            self._update_nginx_config()
            if self._nginx.is_installed():
                self._nginx.reload()

            info.state = WorkerState.DRAINING
            time.sleep(delay * 2)  # Longer drain for WS

            self.stop_worker(info.worker_id)
            logger.info(f"Replaced WS {info.worker_id} with {new_info.worker_id}")

        logger.info("Rolling update complete")

    def _validate_worker(self, worker_info: WorkerInfo) -> bool:
        """Validate that a worker is healthy and responding."""
        import urllib.error
        import urllib.request

        if worker_info.worker_type == WorkerType.HTTP:
            url = f"http://127.0.0.1:{worker_info.port}/health"
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    req = urllib.request.Request(url, method='GET')
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        if resp.status == 200:
                            return True
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                    pass
                time.sleep(0.5)

            return False

        elif worker_info.worker_type == WorkerType.WS:
            # For WS, just check if process is alive
            process = self._processes.get(worker_info.worker_id)
            return process is not None and process.is_alive()

        return False


# ============================================================================
# Web UI Handler
# ============================================================================

class ManagerWebUI(BaseHTTPRequestHandler):
    """Minimal web UI for worker manager."""

    manager: 'WorkerManager' = None

    def log_message(self, format, *args):
        logger.debug(f"WebUI: {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/":
            self._serve_dashboard()
        elif path == "/api/status":
            self._json_response(self.manager.get_status())
        elif path == "/api/workers":
            self._json_response(self.manager.get_workers())
        elif path == "/api/metrics":
            self._json_response(self.manager.get_metrics())
        elif path == "/api/health":
            self._json_response(self.manager.get_health())
        else:
            self._not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        if path == "/api/workers/start":
            worker_type = data.get("type", "http")
            count = data.get("count", 1)
            results = []
            for _ in range(count):
                if worker_type == "http":
                    info = self.manager.start_http_worker()
                else:
                    info = self.manager.start_ws_worker()
                if info:
                    results.append(info.to_dict())
            self._json_response({"status": "ok", "workers": results})

        elif path == "/api/workers/stop":
            worker_id = data.get("worker_id")
            graceful = data.get("graceful", True)
            if worker_id:
                self.manager.stop_worker(worker_id, graceful=graceful)
                self._json_response({"status": "ok"})
            else:
                self._json_response({"status": "error", "message": "worker_id required"}, 400)

        elif path == "/api/workers/restart":
            worker_id = data.get("worker_id")
            if worker_id:
                info = self.manager.restart_worker(worker_id)
                self._json_response({"status": "ok", "worker": info.to_dict() if info else None})
            else:
                self._json_response({"status": "error"}, 400)

        elif path == "/api/rolling-update":
            # Start rolling update in background
            delay = data.get("delay", self.manager.config.manager.rolling_update_delay)
            validate = data.get("validate", True)
            thread = Thread(target=self.manager.rolling_update, args=(delay, validate), daemon=True)
            thread.start()
            self._json_response({"status": "ok", "message": "Rolling update started"})

        elif path == "/api/scale":
            worker_type = data.get("type", "http")
            target_count = data.get("count", 1)
            result = self.manager.scale_workers(worker_type, target_count)
            self._json_response(result)

        elif path == "/api/shutdown":
            graceful = data.get("graceful", True)
            thread = Thread(target=self.manager.stop_all, args=(graceful,), daemon=True)
            thread.start()
            self._json_response({"status": "ok", "message": "Shutdown initiated"})

        elif path == "/api/nginx/reload":
            success = self.manager._nginx.reload()
            self._json_response({"status": "ok" if success else "error"})

        else:
            self._not_found()

    def _json_response(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def _not_found(self):
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": "Not Found"}')

    def _serve_dashboard(self):
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToolBoxV2 Worker Manager</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #475569;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }

        .header h1 {
            font-size: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header h1::before {
            content: "⚡";
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-badge.running { background: rgba(34, 197, 94, 0.2); color: var(--success); }
        .status-badge.stopped { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
        .status-badge.warning { background: rgba(245, 158, 11, 0.2); color: var(--warning); }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-dot.running { background: var(--success); }
        .status-dot.stopped { background: var(--danger); }
        .status-dot.starting { background: var(--warning); }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }

        .card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }

        .metric {
            background: var(--bg-card);
            padding: 16px;
            border-radius: 8px;
        }

        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--accent);
        }

        .metric-value.success { color: var(--success); }
        .metric-value.warning { color: var(--warning); }
        .metric-value.danger { color: var(--danger); }

        .workers-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 400px;
            overflow-y: auto;
        }

        .worker-item {
            background: var(--bg-card);
            padding: 16px;
            border-radius: 8px;
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 12px;
            align-items: center;
        }

        .worker-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .worker-id {
            font-family: 'Consolas', monospace;
            font-size: 0.875rem;
            color: var(--accent);
        }

        .worker-meta {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .worker-stats {
            display: flex;
            gap: 16px;
            font-size: 0.75rem;
        }

        .worker-stat {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .worker-stat-value {
            font-weight: 600;
            color: var(--text-primary);
        }

        .worker-stat-label {
            color: var(--text-secondary);
        }

        .worker-actions {
            display: flex;
            gap: 8px;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary {
            background: var(--accent);
            color: white;
        }

        .btn-primary:hover {
            background: var(--accent-hover);
        }

        .btn-secondary {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--border);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
        }

        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.3);
        }

        .btn-success {
            background: rgba(34, 197, 94, 0.2);
            color: var(--success);
        }

        .btn-success:hover {
            background: rgba(34, 197, 94, 0.3);
        }

        .btn-sm {
            padding: 4px 8px;
            font-size: 0.75rem;
        }

        .actions-bar {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .scale-control {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-card);
            padding: 8px 12px;
            border-radius: 8px;
        }

        .scale-control input {
            width: 60px;
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--bg-secondary);
            color: var(--text-primary);
            text-align: center;
        }

        .scale-control label {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .log-panel {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 12px;
            font-family: 'Consolas', monospace;
            font-size: 0.75rem;
            max-height: 200px;
            overflow-y: auto;
        }

        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid var(--border);
        }

        .log-entry:last-child {
            border-bottom: none;
        }

        .log-time {
            color: var(--text-secondary);
            margin-right: 8px;
        }

        .log-info { color: var(--accent); }
        .log-success { color: var(--success); }
        .log-warning { color: var(--warning); }
        .log-error { color: var(--danger); }

        .progress-bar {
            height: 4px;
            background: var(--bg-card);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s;
        }

        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 8px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transform: translateY(100px);
            opacity: 0;
            transition: all 0.3s;
        }

        .toast.show {
            transform: translateY(0);
            opacity: 1;
        }

        .toast.success { border-color: var(--success); }
        .toast.error { border-color: var(--danger); }
    </style>
</head>
<body>
    <div class="header">
        <h1>ToolBoxV2 Worker Manager</h1>
        <div id="system-status">
            <span class="status-badge stopped">
                <span class="status-dot stopped"></span>
                Loading...
            </span>
        </div>
    </div>

    <div class="grid">
        <!-- System Metrics -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">📊 System Metrics</span>
                <span id="last-update" style="font-size: 0.75rem; color: var(--text-secondary);">--</span>
            </div>
            <div class="metrics-grid" id="metrics">
                <div class="metric">
                    <div class="metric-label">Total Workers</div>
                    <div class="metric-value" id="metric-workers">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Active Connections</div>
                    <div class="metric-value" id="metric-connections">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Requests/min</div>
                    <div class="metric-value" id="metric-requests">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Latency</div>
                    <div class="metric-value" id="metric-latency">0ms</div>
                </div>
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="card">
            <div class="card-header">
                <span class="card-title">⚡ Quick Actions</span>
            </div>
            <div class="actions-bar">
                <button class="btn btn-primary" onclick="startWorker('http')">+ HTTP Worker</button>
                <button class="btn btn-primary" onclick="startWorker('ws')">+ WS Worker</button>
                <button class="btn btn-success" onclick="rollingUpdate()">🔄 Rolling Update</button>
                <button class="btn btn-secondary" onclick="reloadNginx()">🔃 Reload Nginx</button>
                <button class="btn btn-danger" onclick="shutdownAll()">⏹️ Shutdown All</button>
            </div>
            <div style="margin-top: 16px;">
                <div class="scale-control">
                    <label>Scale HTTP to:</label>
                    <input type="number" id="http-scale" value="2" min="0" max="16">
                    <button class="btn btn-sm btn-secondary" onclick="scaleWorkers('http')">Apply</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Workers List -->
    <div class="card">
        <div class="card-header">
            <span class="card-title">👷 Workers</span>
            <span id="worker-count" style="font-size: 0.875rem;">0 workers</span>
        </div>
        <div class="workers-list" id="workers-list">
            <div class="empty-state">No workers running</div>
        </div>
    </div>

    <!-- Activity Log -->
    <div class="card" style="margin-top: 20px;">
        <div class="card-header">
            <span class="card-title">📜 Activity Log</span>
            <button class="btn btn-sm btn-secondary" onclick="clearLogs()">Clear</button>
        </div>
        <div class="log-panel" id="log-panel"></div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        let logs = [];
        let previousWorkers = {};

        function addLog(message, type = 'info') {
            const time = new Date().toLocaleTimeString();
            logs.unshift({ time, message, type });
            if (logs.length > 50) logs.pop();
            renderLogs();
        }

        function renderLogs() {
            const panel = document.getElementById('log-panel');
            panel.innerHTML = logs.map(log => `
                <div class="log-entry">
                    <span class="log-time">${log.time}</span>
                    <span class="log-${log.type}">${log.message}</span>
                </div>
            `).join('');
        }

        function clearLogs() {
            logs = [];
            renderLogs();
        }

        function showToast(message, type = 'info') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast show ' + type;
            setTimeout(() => toast.className = 'toast', 3000);
        }

        function formatUptime(seconds) {
            if (seconds < 60) return Math.round(seconds) + 's';
            if (seconds < 3600) return Math.round(seconds / 60) + 'm';
            return Math.round(seconds / 3600) + 'h ' + Math.round((seconds % 3600) / 60) + 'm';
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                // Update system status
                const statusEl = document.getElementById('system-status');
                const isRunning = data.running && data.broker_alive;
                statusEl.innerHTML = `
                    <span class="status-badge ${isRunning ? 'running' : 'stopped'}">
                        <span class="status-dot ${isRunning ? 'running' : 'stopped'}"></span>
                        ${isRunning ? 'Running' : 'Stopped'}
                    </span>
                `;

                // Update metrics
                const workers = Object.values(data.workers || {});
                const httpWorkers = workers.filter(w => w.worker_type === 'http');
                const wsWorkers = workers.filter(w => w.worker_type === 'ws');

                document.getElementById('metric-workers').textContent = workers.length;
                document.getElementById('worker-count').textContent = `${workers.length} workers`;
                document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();

                // Render workers
                renderWorkers(workers);

                // Check for new/removed workers
                const currentIds = new Set(workers.map(w => w.worker_id));
                const prevIds = new Set(Object.keys(previousWorkers));

                for (const w of workers) {
                    if (!prevIds.has(w.worker_id)) {
                        addLog(`Worker started: ${w.worker_id} (${w.worker_type})`, 'success');
                    }
                }
                for (const id of prevIds) {
                    if (!currentIds.has(id)) {
                        addLog(`Worker stopped: ${id}`, 'warning');
                    }
                }

                previousWorkers = Object.fromEntries(workers.map(w => [w.worker_id, w]));

            } catch (err) {
                console.error('Fetch error:', err);
            }
        }

        function renderWorkers(workers) {
            const list = document.getElementById('workers-list');

            if (workers.length === 0) {
                list.innerHTML = '<div class="empty-state">No workers running</div>';
                return;
            }

            list.innerHTML = workers.map(w => `
                <div class="worker-item">
                    <span class="status-dot ${w.state}"></span>
                    <div class="worker-info">
                        <span class="worker-id">${w.worker_id}</span>
                        <span class="worker-meta">
                            ${w.worker_type.toUpperCase()} • Port ${w.port} • PID ${w.pid} •
                            Uptime: ${formatUptime(w.uptime || 0)}
                        </span>
                    </div>
                    <div class="worker-actions">
                        <button class="btn btn-sm btn-secondary" onclick="restartWorker('${w.worker_id}')">
                            🔄 Restart
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="stopWorker('${w.worker_id}')">
                            ⏹️ Stop
                        </button>
                    </div>
                </div>
            `).join('');
        }

        async function startWorker(type, count = 1) {
            addLog(`Starting ${count} ${type} worker(s)...`, 'info');
            const res = await fetch('/api/workers/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ type, count })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                showToast(`Started ${data.workers.length} worker(s)`, 'success');
            }
            fetchStatus();
        }

        async function stopWorker(id) {
            if (!confirm(`Stop worker ${id}?`)) return;
            addLog(`Stopping worker: ${id}`, 'warning');
            await fetch('/api/workers/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ worker_id: id, graceful: true })
            });
            fetchStatus();
        }

        async function restartWorker(id) {
            addLog(`Restarting worker: ${id}`, 'info');
            await fetch('/api/workers/restart', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ worker_id: id })
            });
            fetchStatus();
        }

        async function rollingUpdate() {
            if (!confirm('Start rolling update? This will restart all workers one by one.')) return;
            addLog('Starting rolling update...', 'info');
            showToast('Rolling update started', 'success');
            await fetch('/api/rolling-update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ validate: true })
            });
        }

        async function scaleWorkers(type) {
            const count = parseInt(document.getElementById(type + '-scale').value);
            addLog(`Scaling ${type} workers to ${count}`, 'info');
            const res = await fetch('/api/scale', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ type, count })
            });
            const data = await res.json();
            showToast(data.message || 'Scaling complete', data.status === 'ok' ? 'success' : 'error');
            fetchStatus();
        }

        async function shutdownAll() {
            if (!confirm('Shutdown all workers?')) return;
            addLog('Initiating shutdown...', 'error');
            await fetch('/api/shutdown', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ graceful: true })
            });
        }

        async function reloadNginx() {
            addLog('Reloading nginx...', 'info');
            const res = await fetch('/api/nginx/reload', { method: 'POST' });
            const data = await res.json();
            showToast(data.status === 'ok' ? 'Nginx reloaded' : 'Nginx reload failed',
                      data.status === 'ok' ? 'success' : 'error');
        }

        // Initial load
        fetchStatus();
        addLog('Dashboard loaded', 'success');

        // Auto-refresh every 2 seconds
        setInterval(fetchStatus, 2000);
    </script>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


def run_web_ui(manager: WorkerManager, host: str, port: int):
    """Run the web UI server."""

    ManagerWebUI.manager = manager

    # Try to bind, fallback to alternative port if needed
    max_retries = 5
    current_port = port

    for attempt in range(max_retries):
        try:
            server = HTTPServer((host, current_port), ManagerWebUI)
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logger.info(f"Web UI running on http://{host}:{current_port}")
            server.serve_forever()
            break
        except PermissionError as e:
            logger.warning(f"Port {current_port} permission denied, trying {current_port + 1}")
            current_port += 1
        except OSError as e:
            if e.errno == 10048 or "Address already in use" in str(e):  # Windows / Unix
                logger.warning(f"Port {current_port} in use, trying {current_port + 1}")
                current_port += 1
            else:
                logger.error(f"Web UI failed to start: {e}")
                break
    else:
        logger.error(f"Web UI failed to start after {max_retries} attempts")


# ============================================================================
# CLI
# ============================================================================

def main():
    from platform import system
    if system() == "Windows":
        from multiprocessing import freeze_support
        freeze_support()

    parser = argparse.ArgumentParser(
        description="ToolBoxV2 Worker Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  start         Start all workers and nginx
  stop          Stop all workers
  restart       Restart all workers
  status        Show status
  update        Zero-downtime rolling update
  nginx-config  Generate and write nginx config
  nginx-reload  Reload nginx configuration
  worker-start  Start a single worker
  worker-stop   Stop a single worker
"""
    )

    parser.add_argument("command", nargs="?", default="start",
                        choices=["start", "stop", "restart", "status", "update",
                                 "nginx-config", "nginx-reload", "worker-start", "worker-stop"])
    parser.add_argument("-c", "--config", help="Config file path")
    parser.add_argument("-w", "--worker-id", help="Worker ID for worker commands")
    parser.add_argument("-t", "--type", choices=["http", "ws"], default="http",
                        help="Worker type for worker-start")
    parser.add_argument("--no-ui", action="store_true", help="Disable web UI")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load config
    from toolboxv2.utils.workers.config import load_config
    config = load_config(args.config)

    # Create manager
    manager = WorkerManager(config)

    # Execute command
    if args.command == "start":
        if not manager.start_all():
            sys.exit(1)

        # Start web UI
        if config.manager.web_ui_enabled and not args.no_ui:
            ui_thread = Thread(
                target=run_web_ui,
                args=(manager, config.manager.web_ui_host, config.manager.web_ui_port),
                daemon=True
            )
            ui_thread.start()

        # Wait for shutdown
        try:
            while manager._running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            manager.stop_all()

    elif args.command == "stop":
        manager.stop_all()

    elif args.command == "restart":
        manager.stop_all()
        time.sleep(2)
        manager.start_all()

    elif args.command == "status":
        status = manager.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "update":
        manager.rolling_update()

    elif args.command == "nginx-config":
        http_ports = [config.http_worker.port + i for i in range(config.http_worker.workers)]
        ws_ports = [config.ws_worker.port]

        nginx = NginxManager(config)
        content = nginx.generate_config(http_ports, ws_ports)
        print(content)

    elif args.command == "nginx-reload":
        nginx = NginxManager(config)
        nginx.reload()

    elif args.command == "worker-start":
        if args.type == "http":
            info = manager.start_http_worker(args.worker_id)
        else:
            info = manager.start_ws_worker(args.worker_id)

        if info:
            print(json.dumps(info.to_dict(), indent=2))
        else:
            print("Failed to start worker")
            sys.exit(1)

    elif args.command == "worker-stop":
        if not args.worker_id:
            print("--worker-id required")
            sys.exit(1)
        manager.stop_worker(args.worker_id)


if __name__ == "__main__":
    main()
