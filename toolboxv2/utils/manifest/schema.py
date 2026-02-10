"""
tb-manifest.yaml Schema Definition
===================================

Pydantic models for validating and working with the unified ToolBoxV2 configuration.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================


class Environment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    STAGING = "staging"
    TAURI = "tauri"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DatabaseMode(str, Enum):
    """
    Database mode abbreviations.

    LC = LOCAL_DICT     - Local dictionary (JSON file), no config needed
    LR = LOCAL_REDIS    - Local Redis server, needs redis.url
    RR = REMOTE_REDIS   - Remote Redis server, needs redis.url or credentials
    CB = CLUSTER_BLOB   - Encrypted blob storage (MinIO), needs minio.* config
    """
    LC = "LC"  # LOCAL_DICT
    LR = "LR"  # LOCAL_REDIS
    RR = "RR"  # REMOTE_REDIS
    CB = "CB"  # CLUSTER_BLOB


class AuthProvider(str, Enum):
    """Authentication provider."""
    CUSTOM = "custom"
    CLERK = "clerk"  # deprecated, use CUSTOM
    LOCAL = "local"
    NONE = "none"


# =============================================================================
# Sub-Config Models
# =============================================================================


class AppConfig(BaseModel):
    """Application identity configuration."""
    name: str = Field(default="ToolBoxV2", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    instance_id: str = Field(default="tbv2_main", description="Unique instance identifier")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)


class AutostartConfig(BaseModel):
    """
    Autostart configuration for system boot.

    Supports:
    - Windows: Task Scheduler
    - Linux: systemd
    - macOS: launchd
    """
    enabled: bool = Field(default=False, description="Enable autostart on system boot")
    services: List[str] = Field(
        default_factory=list,
        description="Services to start (workers, db, minio, etc.)"
    )
    commands: List[str] = Field(
        default_factory=list,
        description="Custom commands to execute on start (e.g., 'tb -v', 'tb status')"
    )


class ModDependency(BaseModel):
    """Module dependency specification."""
    name: str
    version_spec: str = "*"  # e.g., ">=0.1.0", "^1.0.0"


class ModsConfig(BaseModel):
    """Module management configuration."""
    installed: Dict[str, str] = Field(
        default_factory=dict,
        description="Installed modules with version specs (e.g., {'CloudM': '^0.1.0'})"
    )
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mod-to-mod dependencies (e.g., {'isaa': ['CloudM>=0.1.0']})"
    )
    init_modules: List[str] = Field(
        default_factory=list,
        description="Modules to load on app start"
    )
    open_modules: List[str] = Field(
        default_factory=list,
        description="Modules accessible via API without auth"
    )
    watch_modules: List[str] = Field(
        default_factory=list,
        description="Modules to reload on file change (dev mode)"
    )


class LocalDBConfig(BaseModel):
    """LOCAL_DICT (LC) configuration."""
    path: str = Field(default=".data/MiniDictDB.json")


class RedisConfig(BaseModel):
    """Redis configuration for LR/RR modes."""
    url: str = Field(default="${DB_CONNECTION_URI:redis://localhost:6379}")
    username: str = Field(default="${DB_USERNAME:}")
    password: str = Field(default="${DB_PASSWORD:}")
    db_index: int = Field(default=0)
    max_connections: int = Field(default=10)


class MinioConfig(BaseModel):
    """MinIO configuration for CB mode."""
    endpoint: str = Field(default="${MINIO_ENDPOINT:localhost:9000}")
    access_key: str = Field(default="${MINIO_ACCESS_KEY:minioadmin}")
    secret_key: str = Field(default="${MINIO_SECRET_KEY:minioadmin}")
    bucket: str = Field(default="toolbox-data")
    use_ssl: bool = Field(default=False)
    # Cloud sync (optional)
    cloud_endpoint: str = Field(default="${MINIO_CLOUD_ENDPOINT:}")
    cloud_access_key: str = Field(default="${MINIO_CLOUD_ACCESS_KEY:}")
    cloud_secret_key: str = Field(default="${MINIO_CLOUD_SECRET_KEY:}")


class DatabaseConfig(BaseModel):
    """
    Database configuration.

    Modes:
    - LC (LOCAL_DICT): Local JSON file storage
    - LR (LOCAL_REDIS): Local Redis server
    - RR (REMOTE_REDIS): Remote Redis server
    - CB (CLUSTER_BLOB): MinIO blob storage
    """
    mode: DatabaseMode = Field(default=DatabaseMode.LC)
    local: LocalDBConfig = Field(default_factory=LocalDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)


class ZMQConfig(BaseModel):
    """ZeroMQ IPC configuration."""
    pub_endpoint: str = Field(default="tcp://127.0.0.1:5555")
    sub_endpoint: str = Field(default="tcp://127.0.0.1:5556")
    req_endpoint: str = Field(default="tcp://127.0.0.1:5557")
    http_to_ws_endpoint: str = Field(default="tcp://127.0.0.1:5558")
    hwm_send: int = Field(default=10000)
    hwm_recv: int = Field(default=10000)


class ManagerConfig(BaseModel):
    """Worker manager configuration."""
    web_ui_enabled: bool = Field(default=True)
    web_ui_host: str = Field(default="127.0.0.1")
    web_ui_port: int = Field(default=9000)
    health_check_interval: int = Field(default=10)
    restart_delay: int = Field(default=2)
    max_restart_attempts: int = Field(default=5)


class ServicesConfig(BaseModel):
    """Managed services configuration."""
    enabled: List[str] = Field(
        default_factory=lambda: ["workers", "db"],
        description="Services to run (workers, db, minio, registry)"
    )
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    manager: ManagerConfig = Field(default_factory=ManagerConfig)


class HTTPWorkerInstance(BaseModel):
    """Single HTTP worker instance configuration."""
    name: str = Field(default="http_main")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    workers: int = Field(default=4, description="Number of worker processes")
    max_concurrent: int = Field(default=100)
    timeout: int = Field(default=30)
    ssl: bool = Field(default=False)
    ssl_cert: Optional[str] = Field(default=None)
    ssl_key: Optional[str] = Field(default=None)


class WSWorkerInstance(BaseModel):
    """Single WebSocket worker instance configuration."""
    name: str = Field(default="ws_main")
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8100)
    max_connections: int = Field(default=10000)
    ping_interval: int = Field(default=30)
    ping_timeout: int = Field(default=10)
    compression: bool = Field(default=True)


class WorkersConfig(BaseModel):
    """Worker instances configuration - supports multiple HTTP/HTTPS and WS workers."""
    http: List[HTTPWorkerInstance] = Field(
        default_factory=lambda: [HTTPWorkerInstance()],
        description="HTTP/HTTPS worker instances"
    )
    websocket: List[WSWorkerInstance] = Field(
        default_factory=lambda: [WSWorkerInstance()],
        description="WebSocket worker instances"
    )


class NginxConfig(BaseModel):
    """Nginx reverse proxy configuration."""
    enabled: bool = Field(default=True)
    server_name: str = Field(default="${TB_NGINX_SERVER_NAME:localhost}")
    listen_port: int = Field(default=80)
    listen_ssl_port: int = Field(default=443)
    ssl_enabled: bool = Field(default=False)
    ssl_certificate: str = Field(default="")
    ssl_certificate_key: str = Field(default="")
    static_enabled: bool = Field(default=True)
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_zone: str = Field(default="tb_limit")
    rate_limit_rate: str = Field(default="10r/s")
    rate_limit_burst: int = Field(default=20)


class ClerkConfig(BaseModel):
    """Clerk authentication configuration."""
    secret_key: str = Field(default="${CLERK_SECRET_KEY:}")
    publishable_key: str = Field(default="${CLERK_PUBLISHABLE_KEY:}")


class SessionConfig(BaseModel):
    """Session/cookie configuration."""
    cookie_name: str = Field(default="tb_session")
    cookie_secret: str = Field(default="${TB_COOKIE_SECRET:}")
    cookie_max_age: int = Field(default=604800)  # 7 days
    cookie_secure: bool = Field(default=False)
    cookie_httponly: bool = Field(default=True)
    cookie_samesite: str = Field(default="Lax")


class AuthConfig(BaseModel):
    """Authentication configuration."""
    provider: AuthProvider = Field(default=AuthProvider.CUSTOM)
    clerk: ClerkConfig = Field(default_factory=ClerkConfig)  # deprecated, kept for backwards compat
    session: SessionConfig = Field(default_factory=SessionConfig)
    ws_require_auth: bool = Field(default=False)
    ws_allow_anonymous: bool = Field(default=True)


class PathsConfig(BaseModel):
    """Directory paths configuration."""
    # Base directories
    data_dir: str = Field(default="${TB_DATA_DIR:./.data}")
    config_dir: str = Field(default="./.config")
    logs_dir: str = Field(default="./logs")
    # Module directories
    mods_dir: str = Field(default="./mods")
    mods_dev_dir: str = Field(default="./mods_dev")
    mods_storage_dir: str = Field(default="./mods_sto")
    # Build/Distribution - IMPORTANT for nginx static_root!
    dist_dir: str = Field(default="${TB_DIST_DIR:./dist}")
    web_dir: str = Field(default="./web")
    # Registry cache
    registry_cache_dir: str = Field(default="./.tb-registry/cache")


class RegistryConfig(BaseModel):
    """Package registry configuration."""
    url: str = Field(default="https://registry.simplecore.app")
    auto_update: bool = Field(default=False)
    check_interval: int = Field(default=3600, description="Seconds between update checks")


class ToolboxConfig(BaseModel):
    """ToolBox-specific settings."""
    client_prefix: str = Field(default="api-client")
    timeout_seconds: int = Field(default=60)
    max_instances: int = Field(default=2)
    api_prefix: str = Field(default="/api")


class UtilitiesConfig(BaseModel):
    """ToolBox utilities configuration (for future use)."""
    enabled: List[str] = Field(
        default_factory=list,
        description="Enabled utilities (password_manager, file_widget, scheduler, etc.)"
    )


# =============================================================================
# Feature System Configuration
# =============================================================================


class FeatureSpec(BaseModel):
    """Feature specification from features/*/feature.yaml"""
    name: str = Field(description="Feature identifier")
    version: str = Field(default="0.1.0", description="Feature version")
    enabled: bool = Field(default=False, description="Whether feature is active")
    immutable: bool = Field(default=False, description="If true, warns on disable")
    description: str = Field(default="", description="Human-readable description")
    files: List[str] = Field(default_factory=list, description="File patterns belonging to feature")
    imports: List[str] = Field(default_factory=list, description="Python imports for feature")
    dependencies: List[str] = Field(default_factory=list, description="pip/uv packages to install")
    commands: List[str] = Field(default_factory=list, description="CLI commands provided")
    requires: List[str] = Field(default_factory=list, description="Other features this depends on")


class FeaturesConfig(BaseModel):
    """Features configuration in manifest"""
    core: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="core", version="0.1.25", enabled=True, immutable=True,
            description="Core ToolBox functionality"
        )
    )
    cli: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="cli", version="0.1.25", enabled=True,
            description="Command line interface"
        )
    )
    web: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="web", version="0.1.25", enabled=False,
            description="Web workers and API"
        )
    )
    desktop: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="desktop", version="0.1.25", enabled=False,
            description="Desktop UI with PyQt6"
        )
    )
    isaa: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="isaa", version="0.1.25", enabled=False,
            description="AI/LLM integration"
        )
    )
    exotic: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="exotic", version="0.1.25", enabled=False,
            description="Scientific computing extras"
        )
    )


# =============================================================================
# ISAA Configuration (conditional - only when isaa is installed)
# =============================================================================


class IsaaModelsConfig(BaseModel):
    """ISAA LLM model configuration."""
    fast: str = Field(default="${FASTMODEL:ollama/llama3.1}")
    complex: str = Field(default="${COMPLEXMODEL:ollama/llama3.1}")
    blitz: str = Field(default="${BLITZMODEL:ollama/llama3.1}")
    summary: str = Field(default="${SUMMARYMODEL:ollama/llama3.1}")
    image: str = Field(default="${IMAGEMODEL:ollama/llava}")
    audio: str = Field(default="${AUDIOMODEL:groq/whisper-large-v3-turbo}")
    embedding: str = Field(default="${DEFAULTMODELEMBEDDING:gemini/text-embedding-004}")


class IsaaCheckpointConfig(BaseModel):
    """ISAA checkpoint configuration."""
    enabled: bool = Field(default=True)
    interval_seconds: int = Field(default=300)
    max_checkpoints: int = Field(default=10)
    checkpoint_dir: str = Field(default="./checkpoints")
    auto_save_on_exit: bool = Field(default=True)
    auto_load_on_start: bool = Field(default=True)
    max_age_hours: int = Field(default=24)


class IsaaRateLimiterConfig(BaseModel):
    """ISAA rate limiter configuration."""
    enable_rate_limiting: bool = Field(default=True)
    enable_model_fallback: bool = Field(default=True)
    enable_key_rotation: bool = Field(default=True)
    key_rotation_mode: str = Field(default="balance")
    max_retries: int = Field(default=3)
    wait_if_all_exhausted: bool = Field(default=True)


class IsaaSelfAgentConfig(BaseModel):
    """ISAA self-agent configuration - matches FlowAgentBuilder.AgentConfig."""
    name: str = Field(default="self")
    description: str = Field(default="Production-ready FlowAgent")
    version: str = Field(default="2.0.0")
    system_message: str = Field(default="You are a production-ready autonomous agent.")
    temperature: float = Field(default=0.7)
    max_tokens_output: int = Field(default=2048)
    max_tokens_input: int = Field(default=32768)
    vfs_max_window_lines: int = Field(default=250)
    api_key_env_var: str = Field(default="OPENROUTER_API_KEY")
    use_fast_response: bool = Field(default=True)
    max_parallel_tasks: int = Field(default=3)
    verbose_logging: bool = Field(default=False)
    stream: bool = Field(default=True)
    checkpoint: IsaaCheckpointConfig = Field(default_factory=IsaaCheckpointConfig)
    rate_limiter: IsaaRateLimiterConfig = Field(default_factory=IsaaRateLimiterConfig)


class IsaaCodeExecutorConfig(BaseModel):
    """ISAA code executor configuration."""
    type: Literal["restricted", "docker", "none"] = Field(default="restricted")
    docker_image: str = Field(default="python:3.12-slim")
    timeout: int = Field(default=30)


class IsaaMCPConfig(BaseModel):
    """ISAA MCP integration configuration."""
    enabled: bool = Field(default=True)
    servers: List[Dict[str, Any]] = Field(default_factory=list)


class IsaaA2AConfig(BaseModel):
    """ISAA A2A integration configuration."""
    enabled: bool = Field(default=False)
    known_agents: Dict[str, str] = Field(default_factory=dict)
    default_task_timeout: int = Field(default=120)


class IsaaObservabilityConfig(BaseModel):
    """ISAA observability configuration."""
    langfuse_enabled: bool = Field(default=False)
    langfuse_host: str = Field(default="")
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")


class IsaaConfig(BaseModel):
    """ISAA self-agent configuration (only loaded when isaa is installed)."""
    enabled: bool = Field(default=True)
    models: IsaaModelsConfig = Field(default_factory=IsaaModelsConfig)
    self_agent: IsaaSelfAgentConfig = Field(default_factory=IsaaSelfAgentConfig)
    code_executor: IsaaCodeExecutorConfig = Field(default_factory=IsaaCodeExecutorConfig)
    mcp: IsaaMCPConfig = Field(default_factory=IsaaMCPConfig)
    a2a: IsaaA2AConfig = Field(default_factory=IsaaA2AConfig)
    observability: IsaaObservabilityConfig = Field(default_factory=IsaaObservabilityConfig)


# =============================================================================
# Main Manifest Model
# =============================================================================


class TBManifest(BaseModel):
    """
    ToolBoxV2 Unified Configuration Manifest.

    This is the single source of truth for the ToolBox installation.
    Changes here are applied to all sub-configs when saved.
    """
    manifest_version: str = Field(default="1.0.0")

    # Core configuration sections
    app: AppConfig = Field(default_factory=AppConfig)
    autostart: AutostartConfig = Field(default_factory=AutostartConfig)
    mods: ModsConfig = Field(default_factory=ModsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    nginx: NginxConfig = Field(default_factory=NginxConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    toolbox: ToolboxConfig = Field(default_factory=ToolboxConfig)
    utilities: UtilitiesConfig = Field(default_factory=UtilitiesConfig)

    # Feature system
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    # Conditional section - only used when isaa is installed
    isaa: Optional[IsaaConfig] = Field(default=None)

    # Environment-specific overrides
    environments: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "development": {
                "app.debug": True,
                "app.log_level": "DEBUG",
                "database.mode": "LC",
                "nginx.enabled": False,
                "auth.session.cookie_secure": False,
            },
            "production": {
                "app.debug": False,
                "app.log_level": "INFO",
                "database.mode": "CB",
                "nginx.enabled": True,
                "nginx.ssl_enabled": True,
                "auth.session.cookie_secure": True,
            },
            "staging": {
                "app.debug": False,
                "app.log_level": "DEBUG",
                "database.mode": "CB",
                "nginx.enabled": True,
            },
        },
        description="Environment-specific configuration overrides"
    )

    @model_validator(mode="after")
    def validate_isaa_config(self) -> "TBManifest":
        """Enable isaa config if isaa is in installed mods."""
        if "isaa" in self.mods.installed and self.isaa is None:
            self.isaa = IsaaConfig()
        return self

    @model_validator(mode="after")
    def validate_database_config(self) -> "TBManifest":
        """Validate database configuration based on mode."""
        mode = self.database.mode
        if mode in (DatabaseMode.LR, DatabaseMode.RR):
            # Redis modes need URL or credentials
            redis = self.database.redis
            if not redis.url and not (redis.username and redis.password):
                pass  # Will use defaults, which is fine for development
        return self

    def is_isaa_installed(self) -> bool:
        """Check if isaa module is installed."""
        return "isaa" in self.mods.installed

    def get_effective_config(self) -> "TBManifest":
        """Apply environment overrides and return effective configuration."""
        env = self.app.environment.value
        overrides = self.environments.get(env, {})

        if not overrides:
            return self

        # Create a copy and apply overrides
        data = self.model_dump()

        for key_path, value in overrides.items():
            _apply_nested_override(data, key_path, value)

        return TBManifest.model_validate(data)


def _apply_nested_override(data: dict, key_path: str, value: Any) -> None:
    """Apply a nested override like 'app.debug' = True."""
    keys = key_path.split(".")
    current = data

    for key in keys[:-1]:
        # Handle array indexing like 'workers.http[0].workers'
        if "[" in key:
            base_key, idx = key.split("[")
            idx = int(idx.rstrip("]"))
            current = current.get(base_key, [])[idx]
        else:
            if key not in current:
                current[key] = {}
            current = current[key]

    final_key = keys[-1]
    if "[" in final_key:
        base_key, idx = final_key.split("[")
        idx = int(idx.rstrip("]"))
        current[base_key][idx] = value
    else:
        current[final_key] = value

