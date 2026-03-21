# tb-manifest.yaml — Schema Referenz

Vollständige Pydantic-Modelle aus `schema.py`.

<!-- verified: schema.py::TBManifest -->

---

## Top-Level: TBManifest

```python
class TBManifest(BaseModel):
    manifest_version: str = "1.0.0"
    app: AppConfig
    autostart: AutostartConfig
    mods: ModsConfig
    database: DatabaseConfig
    services: ServicesConfig
    workers: WorkersConfig
    nginx: NginxConfig
    auth: AuthConfig
    paths: PathsConfig
    registry: RegistryConfig
    toolbox: ToolboxConfig
    utilities: UtilitiesConfig
    observability: ObservabilityConfig
    features: FeaturesConfig
    isaa: Optional[IsaaConfig]  # Nur bei installiertem isaa-Modul
    environments: Dict[str, Dict[str, Any]]
```

<!-- verified: schema.py::AppConfig -->

## app: AppConfig

```python
class AppConfig(BaseModel):
    name: str = "ToolBoxV2"
    version: str = "0.1.0"
    instance_id: str = "tbv2_main"
    environment: Environment  # development | production | staging | tauri
    debug: bool = False
    log_level: LogLevel  # DEBUG | INFO | WARNING | ERROR
    ping_interval: int = 0
    profile: Optional[ProfileType]  # consumer | homelab | server | business | developer
```

<!-- verified: schema.py::DatabaseMode -->

## database: DatabaseConfig

```python
class DatabaseConfig(BaseModel):
    mode: DatabaseMode  # LC | LR | RR | CB
    local: LocalDBConfig
    redis: RedisConfig
    minio: MinioConfig

class LocalDBConfig:
    path: str = ".data/MiniDictDB.json"

class RedisConfig:
    url: str = "${DB_CONNECTION_URI:redis://localhost:6379}"
    username: str
    password: str
    db_index: int = 0
    max_connections: int = 10

class MinioConfig:
    endpoint: str = "${MINIO_ENDPOINT:localhost:9000}"
    access_key: str
    secret_key: str
    bucket: str = "toolbox-data"
    use_ssl: bool = False
    cloud_endpoint: str
    cloud_access_key: str
    cloud_secret_key: str
```

<!-- verified: schema.py::WorkersConfig -->

## workers: WorkersConfig

```python
class WorkersConfig(BaseModel):
    http: List[HTTPWorkerInstance]
    websocket: List[WSWorkerInstance]

class HTTPWorkerInstance:
    name: str = "http_main"
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 4
    max_concurrent: int = 100
    timeout: int = 30
    ssl: bool = False
    ssl_cert: Optional[str]
    ssl_key: Optional[str]

class WSWorkerInstance:
    name: str = "ws_main"
    host: str = "127.0.0.1"
    port: int = 8100
    max_connections: int = 10000
    ping_interval: int = 30
    ping_timeout: int = 10
    compression: bool = True
```

<!-- verified: schema.py::NginxConfig -->

## nginx: NginxConfig

```python
class NginxConfig(BaseModel):
    enabled: bool = True
    server_name: str = "${TB_NGINX_SERVER_NAME:localhost}"
    listen_port: int = 80
    listen_ssl_port: int = 443
    ssl_enabled: bool = False
    ssl_certificate: str
    ssl_certificate_key: str
    static_enabled: bool = True
    rate_limit_enabled: bool = True
    rate_limit_zone: str = "tb_limit"
    rate_limit_rate: str = "10r/s"
    rate_limit_burst: int = 20
```

<!-- verified: schema.py::AuthConfig -->

## auth: AuthConfig

```python
class AuthConfig(BaseModel):
    provider: AuthProvider  # custom | clerk | local | none
    session: SessionConfig
    ws_require_auth: bool = False
    ws_allow_anonymous: bool = True

class SessionConfig:
    cookie_name: str = "tb_session"
    cookie_secret: str = "${TB_COOKIE_SECRET:}"
    cookie_max_age: int = 604800  # 7 days
    cookie_secure: bool = False
    cookie_httponly: bool = True
    cookie_samesite: str = "Lax"
```

<!-- verified: schema.py::PathsConfig -->

## paths: PathsConfig

```python
class PathsConfig(BaseModel):
    data_dir: str = "${TB_DATA_DIR:./.data}"
    config_dir: str = "./.config"
    logs_dir: str = "./logs"
    mods_dir: str = "./mods"
    mods_dev_dir: str = "./mods_dev"
    mods_storage_dir: str = "./mods_sto"
    dist_dir: str = "${TB_DIST_DIR:./dist}"
    web_dir: str = "./web"
    registry_cache_dir: str = "./.tb-registry/cache"
```

<!-- verified: schema.py::ObservabilityConfig -->

## observability: ObservabilityConfig

```python
class ObservabilityConfig(BaseModel):
    sync: ObservabilitySyncConfig
    dashboard: ObservabilityDashboardConfig
    cleanup: LogCleanupConfig
    slow_on_init: bool = False

class ObservabilitySyncConfig:
    enabled: bool = False
    target: Literal["minio", "remote_minio"]
    remote_endpoint: str
    remote_access_key: str
    remote_secret_key: str
    remote_bucket: str = "system-audit-logs"
    interval_seconds: int = 300

class ObservabilityDashboardConfig:
    enabled: bool = False
    endpoint: str = "${OPENOBSERVE_ENDPOINT:http://localhost:5080}"
    user: str
    password: str
    org: str = "default"
    system_stream: str = "system_logs"
    audit_stream: str = "audit_logs"
    flush_interval: float = 5.0

class LogCleanupConfig:
    enabled: bool = False
    max_age_days: int = 30
    max_size_mb: int = 500
    keep_levels: List[str] = ["ERROR", "WARNING"]
    keep_audit: bool = True
```

---

## Converter: Generierte Dateien

`ConfigConverter.apply_all()` erzeugt 3 Dateien:

<!-- verified: converter.py::ConfigConverter._generate_worker_config -->

### 1. .config.yaml (Python Worker)

```yaml
environment: development
debug: false
data_dir: ./data

zmq:
  pub_endpoint: tcp://127.0.0.1:5555
  sub_endpoint: tcp://127.0.0.1:5556

session:
  cookie_name: tb_session
  cookie_max_age: 604800

http_worker:
  host: 127.0.0.1
  port: 8000
  workers: 4

nginx:
  enabled: true
  static_root: ./dist
```

<!-- verified: converter.py::ConfigConverter._generate_rust_config -->

### 2. bin/config.toml (Rust Server)

```toml
[server]
ip = "0.0.0.0"
port = 8080
dist_path = "./dist"
open_modules = ["CloudM.AuthManager", "CloudM.Auth"]
init_modules = ["CloudM", "DB"]

[toolbox]
client_prefix = "api-client"
timeout_seconds = 60
max_instances = 2

[session]
secret_key = "${TB_COOKIE_SECRET:}"
duration_minutes = 10080
```

<!-- verified: converter.py::ConfigConverter._generate_services_json -->

### 3. services.json (Auto-Start)

```json
{
  "version": "1.0.0",
  "autostart": {
    "enabled": false,
    "services": ["workers", "db"]
  },
  "workers": {
    "http": [{"name": "http_main", "host": "127.0.0.1", "port": 8000}],
    "websocket": [{"name": "ws_main", "host": "127.0.0.1", "port": 8100}]
  }
}
```
