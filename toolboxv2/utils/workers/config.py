#!/usr/bin/env python3
"""
config.py - Configuration Management for ToolBoxV2 Worker System

Handles YAML configuration with environment variable overrides.
Supports: local development, production server, Tauri desktop app.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ============================================================================
# Environment Detection
# ============================================================================

class Environment:
    """Detect runtime environment."""

    @staticmethod
    def is_tauri() -> bool:
        """Check if running inside Tauri."""
        return os.environ.get("TAURI_ENV", "").lower() == "true" or \
            "tauri" in sys.executable.lower()

    @staticmethod
    def is_production() -> bool:
        """Check if production mode."""
        return os.environ.get("TB_ENV", "development").lower() == "production"

    @staticmethod
    def is_development() -> bool:
        """Check if development mode."""
        return not Environment.is_production()

    @staticmethod
    def get_mode() -> str:
        """Get current mode string."""
        if Environment.is_tauri():
            return "tauri"
        elif Environment.is_production():
            return "production"
        return "development"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ZMQConfig:
    """ZeroMQ configuration."""
    # Broker XPUB -> Workers connect with SUB
    pub_endpoint: str = "tcp://127.0.0.1:5555"
    # Workers connect with PUB -> Broker XSUB
    sub_endpoint: str = "tcp://127.0.0.1:5556"
    # RPC: Broker ROUTER, Workers DEALER
    req_endpoint: str = "tcp://127.0.0.1:5557"
    rep_endpoint: str = "tcp://127.0.0.1:5557"
    # HTTP->WS forwarding: Broker PULL, HTTP Workers PUSH
    http_to_ws_endpoint: str = "tcp://127.0.0.1:5558"
    hwm_send: int = 10000
    hwm_recv: int = 10000
    reconnect_interval: int = 1000  # ms
    heartbeat_interval: int = 5000  # ms


@dataclass
class SessionConfig:
    """Session/Cookie configuration."""
    cookie_name: str = "tb_session"
    cookie_secret: str = ""  # Must be set in production!
    cookie_max_age: int = 86400 * 7  # 7 days
    cookie_secure: bool = True
    cookie_httponly: bool = True
    cookie_samesite: str = "Lax"
    # Signed cookie payload fields
    payload_fields: List[str] = field(default_factory=lambda: [
        "user_id", "session_id", "level", "spec", "user_name", "exp"
    ])


@dataclass
class AuthConfig:
    """Authentication configuration."""
    clerk_enabled: bool = True
    clerk_secret_key: str = ""  # From environment
    clerk_publishable_key: str = ""  # From environment
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600  # 1 hour
    api_key_header: str = "X-API-Key"
    bearer_header: str = "Authorization"


@dataclass
class HTTPWorkerConfig:
    """HTTP worker configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 4  # Number of worker processes
    max_concurrent: int = 100  # Per worker
    timeout: int = 30
    keepalive: int = 65
    backlog: int = 2048
    # Instance ID prefix
    instance_prefix: str = "http"


@dataclass
class WSWorkerConfig:
    """WebSocket worker configuration."""
    host: str = "127.0.0.1"
    port: int = 8100  # Separated from HTTP range (8000-8099)
    max_connections: int = 10000
    ping_interval: int = 30
    ping_timeout: int = 10
    max_message_size: int = 1048576  # 1MB
    compression: bool = True
    instance_prefix: str = "ws"


@dataclass
class NginxConfig:
    """Nginx configuration."""
    enabled: bool = True
    config_path: str = "/etc/nginx/sites-available/toolboxv2"
    symlink_path: str = "/etc/nginx/sites-enabled/toolboxv2"
    pid_file: str = "/run/nginx.pid"
    access_log: str = "/var/log/nginx/toolboxv2_access.log"
    error_log: str = "/var/log/nginx/toolboxv2_error.log"
    # Server settings
    server_name: str = "localhost"
    listen_port: int = 80
    listen_ssl_port: int = 443
    ssl_enabled: bool = False
    ssl_certificate: str = ""
    ssl_certificate_key: str = ""
    # Static files
    static_root: str = "./dist"
    static_enabled: bool = True
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_zone: str = "tb_limit"
    rate_limit_rate: str = "10r/s"
    rate_limit_burst: int = 20
    # Load balancing
    upstream_http: str = "tb_http_backend"
    upstream_ws: str = "tb_ws_backend"


@dataclass
class ManagerConfig:
    """Worker manager configuration."""
    web_ui_enabled: bool = True
    web_ui_host: str = "127.0.0.1"
    web_ui_port: int = 9000
    control_socket: str = ""  # Unix socket path
    pid_file: str = ""
    log_file: str = ""
    health_check_interval: int = 10
    restart_delay: int = 2
    max_restart_attempts: int = 5
    rolling_update_delay: int = 5


@dataclass
class ToolBoxV2Config:
    """ToolBoxV2 integration configuration."""
    instance_id: str = "tbv2_worker"
    modules_preload: List[str] = field(default_factory=list)
    api_prefix: str = "/api"
    api_allowed_mods: List[str] = field(default_factory=list)
    # CloudM Auth
    auth_module: str = "CloudM.AuthClerk"
    verify_session_func: str = "verify_session"


@dataclass
class Config:
    """Main configuration container."""
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    http_worker: HTTPWorkerConfig = field(default_factory=HTTPWorkerConfig)
    ws_worker: WSWorkerConfig = field(default_factory=WSWorkerConfig)
    nginx: NginxConfig = field(default_factory=NginxConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    toolbox: ToolBoxV2Config = field(default_factory=ToolBoxV2Config)

    # Runtime
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization (Windows multiprocessing)."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Reconstruct config from dictionary."""
        return _dict_to_dataclass(cls, data)


# ============================================================================
# Configuration Loading
# ============================================================================

def _deep_update(base: dict, updates: dict) -> dict:
    """Deep merge dictionaries."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_env_vars(obj: Any) -> Any:
    """Resolve ${ENV_VAR} patterns in configuration."""
    if isinstance(obj, str):
        import re
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            env_var = match.group(1)
            default = ""
            if ":" in env_var:
                env_var, default = env_var.split(":", 1)
            return os.environ.get(env_var, default)

        return re.sub(pattern, replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _dict_to_dataclass(cls, data: dict) -> Any:
    """Convert dict to dataclass recursively."""
    if not data:
        return cls()

    from dataclasses import fields, is_dataclass

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            # Handle nested dataclasses
            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[field_name] = _dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)


def load_config(
    config_path: str | None = None,
    env_overrides: bool = True
) -> Config:
    """
    Load configuration from YAML file with environment overrides.

    Search order:
    1. Explicit config_path
    2. TB_CONFIG environment variable
    3. ./config.yaml
    4. ~/.toolboxv2/worker_config.yaml
    5. /etc/toolboxv2/worker_config.yaml

    Args:
        config_path: Explicit path to config file
        env_overrides: Apply environment variable overrides

    Returns:
        Config dataclass
    """
    # Find config file
    search_paths = []

    if config_path:
        search_paths.append(config_path)

    if os.environ.get("TB_CONFIG"):
        search_paths.append(os.environ["TB_CONFIG"])

    search_paths.extend([
        "./config.yaml",
        "./worker_config.yaml",
        str(Path.home() / ".toolboxv2" / "worker_config.yaml"),
        "/etc/toolboxv2/worker_config.yaml",
    ])

    config_data = {}

    for path in search_paths:
        if os.path.exists(path):
            with open(path) as f:
                config_data = yaml.safe_load(f) or {}
            break

    # Resolve environment variables in config
    config_data = _resolve_env_vars(config_data)

    # Apply direct environment overrides
    if env_overrides:
        env_mapping = {
            "TB_ENV": ("environment",),
            "TB_DEBUG": ("debug",),
            "TB_LOG_LEVEL": ("log_level",),
            "TB_DATA_DIR": ("data_dir",),
            # Session
            "TB_COOKIE_SECRET": ("session", "cookie_secret"),
            "TB_COOKIE_NAME": ("session", "cookie_name"),
            # Auth
            "CLERK_SECRET_KEY": ("auth", "clerk_secret_key"),
            "CLERK_PUBLISHABLE_KEY": ("auth", "clerk_publishable_key"),
            # HTTP Worker
            "TB_HTTP_HOST": ("http_worker", "host"),
            "TB_HTTP_PORT": ("http_worker", "port"),
            "TB_HTTP_WORKERS": ("http_worker", "workers"),
            # WS Worker
            "TB_WS_HOST": ("ws_worker", "host"),
            "TB_WS_PORT": ("ws_worker", "port"),
            # Nginx
            "TB_NGINX_SERVER_NAME": ("nginx", "server_name"),
            "TB_STATIC_ROOT": ("nginx", "static_root"),
            # ZMQ
            "TB_ZMQ_PUB": ("zmq", "pub_endpoint"),
            "TB_ZMQ_SUB": ("zmq", "sub_endpoint"),
        }

        for env_var, path in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Navigate to nested dict
                current = config_data
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                # Set value with type conversion
                final_key = path[-1]
                if final_key in ["port", "workers", "max_concurrent", "timeout"]:
                    value = int(value)
                elif final_key in ["debug", "ssl_enabled", "rate_limit_enabled"]:
                    value = value.lower() in ("true", "1", "yes")
                current[final_key] = value

    # Set environment-specific defaults
    env_mode = Environment.get_mode()

    if env_mode == "development":
        config_data.setdefault("debug", True)
        config_data.setdefault("log_level", "DEBUG")
        config_data.setdefault("nginx", {}).setdefault("enabled", False)
        config_data.setdefault("session", {}).setdefault("cookie_secure", False)

    elif env_mode == "tauri":
        # Tauri desktop app - single-user, local
        config_data.setdefault("debug", False)
        config_data.setdefault("nginx", {}).setdefault("enabled", False)
        config_data.setdefault("http_worker", {}).setdefault("workers", 1)
        config_data.setdefault("http_worker", {}).setdefault("host", "127.0.0.1")
        config_data.setdefault("ws_worker", {}).setdefault("host", "127.0.0.1")
        config_data.setdefault("manager", {}).setdefault("web_ui_enabled", False)

    elif env_mode == "production":
        config_data.setdefault("debug", False)
        config_data.setdefault("log_level", "INFO")
        config_data.setdefault("session", {}).setdefault("cookie_secure", True)
        # Ensure cookie secret is set
        if not config_data.get("session", {}).get("cookie_secret"):
            raise ValueError("TB_COOKIE_SECRET must be set in production!")

    # Set data directory
    if not config_data.get("data_dir"):
        if env_mode == "tauri":
            config_data["data_dir"] = str(Path.home() / ".toolboxv2")
        else:
            config_data["data_dir"] = os.environ.get(
                "TB_DATA_DIR",
                str(Path.home() / ".toolboxv2")
            )

    # Create data directory
    data_dir = Path(config_data["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Set derived paths
    if not config_data.get("manager", {}).get("control_socket"):
        config_data.setdefault("manager", {})["control_socket"] = str(
            data_dir / "manager.sock"
        )

    if not config_data.get("manager", {}).get("pid_file"):
        config_data.setdefault("manager", {})["pid_file"] = str(
            data_dir / "manager.pid"
        )

    if not config_data.get("manager", {}).get("log_file"):
        config_data.setdefault("manager", {})["log_file"] = str(
            data_dir / "logs" / "manager.log"
        )

    # Convert to dataclass
    return _dict_to_dataclass(Config, config_data)


def get_default_config_yaml() -> str:
    """Generate default configuration YAML with comments."""
    return '''# ToolBoxV2 Worker System Configuration
# Environment variables can be used: ${VAR_NAME} or ${VAR_NAME:default}

# Runtime environment: development, production, tauri
environment: "${TB_ENV:development}"
debug: false
log_level: "INFO"
data_dir: "${TB_DATA_DIR:}"

# ZeroMQ IPC Configuration
# WICHTIG: Jeder Socket braucht einen eigenen Port!
zmq:
  pub_endpoint: "tcp://127.0.0.1:5555"   # Broker XPUB -> Worker SUB
  sub_endpoint: "tcp://127.0.0.1:5556"   # Worker PUB -> Broker XSUB
  req_endpoint: "tcp://127.0.0.1:5557"   # RPC endpoint
  rep_endpoint: "tcp://127.0.0.1:5557"   # Same as req (ROUTER/DEALER)
  http_to_ws_endpoint: "tcp://127.0.0.1:5558"  # HTTP->WS forwarding
  hwm_send: 10000
  hwm_recv: 10000
  reconnect_interval: 1000
  heartbeat_interval: 5000

# Session Configuration (Signed Cookies - Stateless)
session:
  cookie_name: "tb_session"
  cookie_secret: "${TB_COOKIE_SECRET:}"  # Required in production!
  cookie_max_age: 604800  # 7 days
  cookie_secure: true
  cookie_httponly: true
  cookie_samesite: "Lax"
  payload_fields:
    - "user_id"
    - "session_id"
    - "level"
    - "spec"
    - "user_name"
    - "exp"

# Authentication
auth:
  clerk_enabled: true
  clerk_secret_key: "${CLERK_SECRET_KEY:}"
  clerk_publishable_key: "${CLERK_PUBLISHABLE_KEY:}"
  jwt_algorithm: "HS256"
  jwt_expiry: 3600
  api_key_header: "X-API-Key"
  bearer_header: "Authorization"

# HTTP Worker Configuration
http_worker:
  host: "127.0.0.1"
  port: 8000
  workers: 4  # Number of worker processes
  max_concurrent: 100  # Max concurrent requests per worker
  timeout: 30
  keepalive: 65
  backlog: 2048
  instance_prefix: "http"

# WebSocket Worker Configuration
ws_worker:
  host: "127.0.0.1"
  port: 8001
  max_connections: 10000
  ping_interval: 30
  ping_timeout: 10
  max_message_size: 1048576  # 1MB
  compression: true
  instance_prefix: "ws"

# Nginx Configuration
nginx:
  enabled: true
  config_path: "/etc/nginx/sites-available/toolboxv2"
  symlink_path: "/etc/nginx/sites-enabled/toolboxv2"
  server_name: "${TB_NGINX_SERVER_NAME:localhost}"
  listen_port: 80
  listen_ssl_port: 443
  ssl_enabled: false
  ssl_certificate: ""
  ssl_certificate_key: ""
  # Static files
  static_root: "${TB_STATIC_ROOT:./dist}"
  static_enabled: true
  # Rate limiting
  rate_limit_enabled: true
  rate_limit_zone: "tb_limit"
  rate_limit_rate: "10r/s"
  rate_limit_burst: 20
  # Upstreams
  upstream_http: "tb_http_backend"
  upstream_ws: "tb_ws_backend"

# Worker Manager Configuration
manager:
  web_ui_enabled: true
  web_ui_host: "127.0.0.1"
  web_ui_port: 9000
  control_socket: "${TB_DATA_DIR:~/.toolboxv2}/manager.sock"
  pid_file: "${TB_DATA_DIR:~/.toolboxv2}/manager.pid"
  log_file: "${TB_DATA_DIR:~/.toolboxv2}/logs/manager.log"
  health_check_interval: 10
  restart_delay: 2
  max_restart_attempts: 5
  rolling_update_delay: 5

# ToolBoxV2 Integration
toolbox:
  instance_id: "tbv2_worker"
  modules_preload: []
  api_prefix: "/api"
  api_allowed_mods: []
  auth_module: "CloudM.AuthClerk"
  verify_session_func: "verify_session"
'''


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description="ToolBoxV2 Config Manager")
    subparsers = parser.add_subparsers(dest="command")

    # Generate default config
    gen_parser = subparsers.add_parser("generate", help="Generate default config")
    gen_parser.add_argument("-o", "--output", default="config.yaml")

    # Validate config
    val_parser = subparsers.add_parser("validate", help="Validate config")
    val_parser.add_argument("-c", "--config", help="Config file path")

    # Show config
    show_parser = subparsers.add_parser("show", help="Show loaded config")
    show_parser.add_argument("-c", "--config", help="Config file path")

    args = parser.parse_args()

    if args.command == "generate":
        with open(args.output, "w") as f:
            f.write(get_default_config_yaml())
        print(f"Generated config: {args.output}")

    elif args.command == "validate":
        try:
            config = load_config(args.config)
            print("✓ Configuration valid")
            print(f"  Environment: {config.environment}")
            print(f"  HTTP Workers: {config.http_worker.workers}")
            print(f"  WS Max Connections: {config.ws_worker.max_connections}")
        except Exception as e:
            print(f"✗ Configuration error: {e}")
            sys.exit(1)

    elif args.command == "show":
        config = load_config(args.config)
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(config), indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
