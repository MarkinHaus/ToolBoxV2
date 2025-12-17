#!/usr/bin/env python3
"""
config.py - Configuration Management for ToolBoxV2 Worker System

Handles YAML configuration with environment variable overrides.
Supports: local development, production server, Tauri desktop app.

Enhanced with:
- open_modules: List of publicly accessible modules (no auth required)
- Level system configuration
- WebSocket authentication options
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
# Access Level Constants
# ============================================================================


class AccessLevel:
    """User access levels for authorization."""
    ADMIN = -1           # Full access to everything
    NOT_LOGGED_IN = 0    # Anonymous user, only public endpoints
    LOGGED_IN = 1        # Authenticated user
    TRUSTED = 2          # Trusted/verified user


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ZMQConfig:
    """ZeroMQ configuration."""
    pub_endpoint: str = "tcp://127.0.0.1:5555"
    sub_endpoint: str = "tcp://127.0.0.1:5556"
    req_endpoint: str = "tcp://127.0.0.1:5557"
    rep_endpoint: str = "tcp://127.0.0.1:5557"
    http_to_ws_endpoint: str = "tcp://127.0.0.1:5558"
    hwm_send: int = 10000
    hwm_recv: int = 10000
    reconnect_interval: int = 1000
    heartbeat_interval: int = 5000


@dataclass
class SessionConfig:
    """Session/Cookie configuration."""
    cookie_name: str = "tb_session"
    cookie_secret: str = ""
    cookie_max_age: int = 86400 * 7
    cookie_secure: bool = True
    cookie_httponly: bool = True
    cookie_samesite: str = "Lax"
    payload_fields: List[str] = field(default_factory=lambda: [
        "user_id", "session_id", "level", "spec", "user_name", "exp"
    ])


@dataclass
class AuthConfig:
    """Authentication configuration."""
    clerk_enabled: bool = True
    clerk_secret_key: str = ""
    clerk_publishable_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600
    api_key_header: str = "X-API-Key"
    bearer_header: str = "Authorization"
    # WebSocket auth requirement
    ws_require_auth: bool = False
    ws_allow_anonymous: bool = True


@dataclass
class HTTPWorkerConfig:
    """HTTP worker configuration."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 4
    max_concurrent: int = 100
    timeout: int = 30
    keepalive: int = 65
    backlog: int = 2048
    instance_prefix: str = "http"


@dataclass
class WSWorkerConfig:
    """WebSocket worker configuration."""
    host: str = "localhost"
    port: int = 8100
    max_connections: int = 10000
    ping_interval: int = 30
    ping_timeout: int = 10
    max_message_size: int = 1048576
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
    server_name: str = "localhost"
    listen_port: int = 80
    listen_ssl_port: int = 443
    ssl_enabled: bool = False
    ssl_certificate: str = ""
    ssl_certificate_key: str = ""
    static_root: str = "./dist"
    static_enabled: bool = True
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_zone: str = "tb_limit"
    rate_limit_rate: str = "10r/s"
    rate_limit_burst: int = 20
    # Auth endpoint rate limiting (stricter)
    auth_rate_limit_rate: str = "5r/s"
    auth_rate_limit_burst: int = 10
    # Upstreams
    upstream_http: str = "tb_http_backend"
    upstream_ws: str = "tb_ws_backend"


@dataclass
class ManagerConfig:
    """Worker manager configuration."""
    web_ui_enabled: bool = True
    web_ui_host: str = "127.0.0.1"
    web_ui_port: int = 9005
    control_socket: str = ""
    pid_file: str = ""
    log_file: str = ""
    health_check_interval: int = 10
    restart_delay: int = 2
    max_restart_attempts: int = 5
    rolling_update_delay: int = 5


@dataclass
class ToolBoxV2Config:
    """ToolBoxV2 integration configuration with access control."""
    instance_id: str = "tbv2_worker"
    modules_preload: List[str] = field(default_factory=list)
    api_prefix: str = "/api"
    api_allowed_mods: List[str] = field(default_factory=list)
    # CloudM Auth
    auth_module: str = "CloudM.AuthClerk"
    verify_session_func: str = "verify_session"

    # === Access Control ===
    # Modules that are publicly accessible (no auth required)
    open_modules: List[str] = field(default_factory=list)

    # Default required level for non-open modules/functions
    default_required_level: int = AccessLevel.LOGGED_IN

    # Level requirements per module (optional override)
    level_requirements: Dict[str, int] = field(default_factory=dict)

    # Admin-only modules (require level -1)
    admin_modules: List[str] = field(default_factory=lambda: [
        "CloudM.AuthClerk",
        "ToolBox",
    ])


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

    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
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

    if not is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]

            if is_dataclass(field_type):
                kwargs[key] = _dict_to_dataclass(field_type, value or {})
            elif hasattr(field_type, '__origin__'):
                kwargs[key] = value
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with environment overrides.
    """
    config_data = {}

    search_paths = [
        config_path,
        os.environ.get("TB_CONFIG"),
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.cwd() / "toolbox.yaml",
        Path.home() / ".toolboxv2" / "config.yaml",
        Path("/etc/toolboxv2/config.yaml"),
    ]

    config_file = None
    for path in search_paths:
        if path and Path(path).exists():
            config_file = Path(path)
            break

    if config_file:
        with open(config_file) as f:
            loaded = yaml.safe_load(f) or {}
            config_data = _resolve_env_vars(loaded)

    env_mapping = {
        "TB_ENV": ["environment"],
        "TB_DEBUG": ["debug"],
        "TB_LOG_LEVEL": ["log_level"],
        "TB_COOKIE_SECRET": ["session", "cookie_secret"],
        "CLERK_SECRET_KEY": ["auth", "clerk_secret_key"],
        "CLERK_PUBLISHABLE_KEY": ["auth", "clerk_publishable_key"],
        "TB_HTTP_HOST": ["http_worker", "host"],
        "TB_HTTP_PORT": ["http_worker", "port"],
        "TB_HTTP_WORKERS": ["http_worker", "workers"],
        "TB_WS_HOST": ["ws_worker", "host"],
        "TB_WS_PORT": ["ws_worker", "port"],
        "TB_NGINX_SERVER_NAME": ["nginx", "server_name"],
        "TB_STATIC_ROOT": ["nginx", "static_root"],
        "TB_OPEN_MODULES": ["toolbox", "open_modules"],
    }

    for env_var, path in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            current = config_data
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            final_key = path[-1]

            if final_key in ["port", "workers", "max_concurrent", "timeout"]:
                value = int(value)
            elif final_key in ["debug", "ssl_enabled", "rate_limit_enabled", "ws_require_auth"]:
                value = value.lower() in ("true", "1", "yes")
            elif final_key in ["open_modules", "admin_modules", "modules_preload"]:
                value = [v.strip() for v in value.split(",") if v.strip()]

            current[final_key] = value

    env_mode = Environment.get_mode()

    if env_mode == "development":
        config_data.setdefault("debug", True)
        config_data.setdefault("log_level", "DEBUG")
        config_data.setdefault("nginx", {}).setdefault("enabled", False)
        config_data.setdefault("session", {}).setdefault("cookie_secure", False)
        config_data.setdefault("auth", {}).setdefault("ws_allow_anonymous", True)

    elif env_mode == "tauri":
        config_data.setdefault("debug", False)
        config_data.setdefault("nginx", {}).setdefault("enabled", False)
        config_data.setdefault("http_worker", {}).setdefault("workers", 1)
        config_data.setdefault("http_worker", {}).setdefault("host", "localhost")
        config_data.setdefault("ws_worker", {}).setdefault("host", "localhost")
        config_data.setdefault("manager", {}).setdefault("web_ui_enabled", False)
        config_data.setdefault("auth", {}).setdefault("ws_allow_anonymous", True)

    elif env_mode == "production":
        config_data.setdefault("debug", False)
        config_data.setdefault("log_level", "INFO")
        config_data.setdefault("session", {}).setdefault("cookie_secure", True)
        config_data.setdefault("auth", {}).setdefault("ws_require_auth", True)
        if not config_data.get("session", {}).get("cookie_secret"):
            raise ValueError("TB_COOKIE_SECRET must be set in production!")

    if not config_data.get("data_dir"):
        if env_mode == "tauri":
            config_data["data_dir"] = str(Path.home() / ".toolboxv2")
        else:
            config_data["data_dir"] = os.environ.get(
                "TB_DATA_DIR",
                str(Path.home() / ".toolboxv2")
            )

    data_dir = Path(config_data["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

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
zmq:
  pub_endpoint: "tcp://127.0.0.1:5555"
  sub_endpoint: "tcp://127.0.0.1:5556"
  req_endpoint: "tcp://127.0.0.1:5557"
  rep_endpoint: "tcp://127.0.0.1:5557"
  http_to_ws_endpoint: "tcp://127.0.0.1:5558"
  hwm_send: 10000
  hwm_recv: 10000
  reconnect_interval: 1000
  heartbeat_interval: 5000

# Session Configuration (Signed Cookies)
session:
  cookie_name: "tb_session"
  cookie_secret: "${TB_COOKIE_SECRET:}"
  cookie_max_age: 604800
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
  # WebSocket auth settings
  ws_require_auth: false
  ws_allow_anonymous: true

# HTTP Worker Configuration
http_worker:
  host: "127.0.0.1"
  port: 8000
  workers: 4
  max_concurrent: 100
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
  max_message_size: 1048576
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
  static_root: "${TB_STATIC_ROOT:./dist}"
  static_enabled: true
  # Rate limiting
  rate_limit_enabled: true
  rate_limit_zone: "tb_limit"
  rate_limit_rate: "10r/s"
  rate_limit_burst: 20
  # Auth endpoint rate limiting (stricter to prevent brute force)
  auth_rate_limit_rate: "5r/s"
  auth_rate_limit_burst: 10
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

# ToolBoxV2 Integration with Access Control
toolbox:
  instance_id: "tbv2_worker"
  modules_preload: []
  api_prefix: "/api"
  api_allowed_mods: []
  auth_module: "CloudM.AuthClerk"
  verify_session_func: "verify_session"

  # === Access Control Configuration ===
  #
  # Level System:
  #   -1 = Admin (full access)
  #    0 = Not logged in (anonymous)
  #    1 = Logged in (authenticated user)
  #    2 = Trusted user (verified/premium)
  #
  # Access Rules:
  #   1. Modules in open_modules are fully public
  #   2. Functions starting with 'open' are always public
  #   3. Admin modules require level -1
  #   4. All other endpoints require at least level 1 (logged in)

  # Publicly accessible modules (no auth required)
  # Example: ["PublicAPI", "WebContent", "Assets"]
  open_modules: []

  # Default required level for protected endpoints
  default_required_level: 1

  # Per-module/function level requirements (optional)
  # Format: "Module": level or "Module.function": level
  # level_requirements:
  #   "UserSettings": 1
  #   "AdminPanel": -1
  #   "Premium.export": 2

  # Admin-only modules (require level -1)
  admin_modules:
    - "CloudM.AuthClerk"
    - "ToolBox"
'''


# ============================================================================
# CLI
# ============================================================================


def main():
    """CLI for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description="ToolBoxV2 Config Manager")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate", help="Generate default config")
    gen_parser.add_argument("-o", "--output", default="config.yaml")

    val_parser = subparsers.add_parser("validate", help="Validate config")
    val_parser.add_argument("-c", "--config", help="Config file path")

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
            print(f"  Open Modules: {config.toolbox.open_modules}")
            print(f"  Admin Modules: {config.toolbox.admin_modules}")
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
