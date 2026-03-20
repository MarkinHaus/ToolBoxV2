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


class ProfileType(str, Enum):
    """User profile — controls bare 'tb' default behavior."""
    CONSUMER  = "consumer"   # tb → tb gui
    HOMELAB   = "homelab"    # tb → interactive dashboard
    SERVER    = "server"     # tb → ASCII status overview + exit
    BUSINESS  = "business"   # tb → 3-line health summary + exit
    DEVELOPER = "developer"  # tb → interactive dashboard + dev hints


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
    profiling: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    ping_interval: int = Field(default=0)
    profile: Optional[ProfileType] = Field(
        default=None,
        description="User profile — set during first-run. None = first-run not completed."
    )


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

    default_name: str  = Field(default="my-db")


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
    auto_install: bool = Field(
            default=False,
            description=(
                "If True AND a guarded import fails: auto-install missing pip/uv deps, "
                "then retry. If still failing: abort with clear error. "
                "Overrides the global FeaturesConfig.auto_install_deps."
            )
        )

class FeaturesConfig(BaseModel):
    """Features configuration in manifest"""

    # Globaler Auto-Install Flag
    auto_install_deps: bool = Field(
        default=False,
        description=(
            "Global: auto-install missing pip/uv deps when a guarded import fails. "
            "Per-feature override: FeatureSpec.auto_install. "
            "False = clear error message + abort (no silent installs)."
        )
    )

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
    mini: FeatureSpec = Field(
        default_factory=lambda: FeatureSpec(
            name="mini", version="0.1.25", enabled=True, immutable=True,
            description="Absolute minimal install — App + types, no CLI/web"
        )
    )

    def get_active_features(self) -> List[str]:
        """
        Iteriert dynamisch über alle Felder und gibt die Namen derer zurück,
        die aktiv (enabled=True) sind.
        """
        # 'self' gibt beim Iterieren (field_name, field_value) zurück
        return [
            name for name, spec in self
            if isinstance(spec, FeatureSpec) and spec.enabled
        ]

    def should_auto_install(self, feature_name: str) -> bool:
        """
        Prüfe ob Auto-Install für dieses Feature aktiv ist.

        Auflösung:
          1. FeatureSpec.auto_install (per-feature override)
          2. FeaturesConfig.auto_install_deps (global)
        """
        spec = getattr(self, feature_name, None)
        if spec is None:
            return self.auto_install_deps
        return spec.auto_install or self.auto_install_deps

# =============================================================================
# Observability Configuration
# =============================================================================


class LogCleanupConfig(BaseModel):
    """Automatic log cleanup rules."""
    enabled: bool = Field(default=False, description="Enable automatic cleanup")
    max_age_days: int = Field(default=30, description="Delete logs older than N days")
    max_size_mb: int = Field(default=500, description="Max total log size before cleanup")
    keep_levels: List[str] = Field(
        default_factory=lambda: ["ERROR", "WARNING"],
        description="Always keep these levels (even past max_age)"
    )
    keep_audit: bool = Field(default=True, description="Never delete audit logs automatically")


class ObservabilitySyncConfig(BaseModel):
    """Log sync target configuration."""
    enabled: bool = Field(default=False)
    target: Literal["minio", "remote_minio"] = Field(
        default="minio",
        description="minio = local MinIO from database.minio, remote_minio = separate endpoint"
    )
    remote_endpoint: str = Field(default="${MINIO_REMOTE_ENDPOINT:}")
    remote_access_key: str = Field(default="${MINIO_REMOTE_ACCESS_KEY:}")
    remote_secret_key: str = Field(default="${MINIO_REMOTE_SECRET_KEY:}")
    remote_bucket: str = Field(default="system-audit-logs")
    remote_secure: bool = Field(default=False)
    interval_seconds: int = Field(default=300, description="Auto-sync interval (0 = manual only)")


class ObservabilityDashboardConfig(BaseModel):
    """OpenObserve / dashboard connection."""
    enabled: bool = Field(default=False)
    endpoint: str = Field(default="${OPENOBSERVE_ENDPOINT:http://localhost:5080}")
    user: str = Field(default="${OPENOBSERVE_USER:root@example.com}")
    password: str = Field(default="${OPENOBSERVE_PASSWORD:}")
    org: str = Field(default="${OPENOBSERVE_ORG:default}")
    system_stream: str = Field(default="system_logs")
    audit_stream: str = Field(default="audit_logs")
    flush_interval: float = Field(default=5.0, description="Seconds between batch pushes")
    verify_ssl: bool = Field(default=False)


class ObservabilityConfig(BaseModel):
    """
    Observability configuration — log analysis, sync, and dashboard.

    Controls:
    - Live log streaming to OpenObserve dashboard
    - Log sync to local/remote MinIO
    - Automatic log cleanup
    """
    sync: ObservabilitySyncConfig = Field(default_factory=ObservabilitySyncConfig)
    dashboard: ObservabilityDashboardConfig = Field(default_factory=ObservabilityDashboardConfig)
    cleanup: LogCleanupConfig = Field(default_factory=LogCleanupConfig)
    slow_on_init: bool = Field(default=False)

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
    system_message: str = Field(default="""# ISAA Agent System Prompt v2.1

---

## IDENTITY

You are an autonomous agent in the ISAA system.
Your job: complete the user's task using available tools, then call `final_answer`.
You are NOT a chatbot by default — **you act**.
But you can also think out loud, brainstorm, and have a real conversation when that is what is needed.
Read the MODE section below to know which applies.

---

## MODE — ACT vs. TALK

**ACT mode** (default when a concrete task is given):
- Tool calls, verification, `final_answer`
- Minimal prose, maximum action
- Output: results, file references, status

**TALK mode** (when the user is brainstorming, exploring, or in audio/voice):
- Conversational, warm, exploratory
- No forced tool use — respond like a knowledgeable partner.
- all information must be in context or from tool results!
- Ask clarifying questions, offer options, think out loud together
- Output: natural language, ideas, questions back

**How to detect TALK mode:**
- "what do you think about…", "let's brainstorm…", "help me understand…"
- Audio/voice context (short sentences, no markdown lists)
- No concrete deliverable requested
- User explicitly says "just talk to me" or "let's think through this"

**Switching:** You can switch mid-conversation. If the user shifts from brainstorming to "okay, build it", "deep dive" — switch to ACT immediately.

---

## YOUR TOOLS

### Always available (static)
| Tool | When to use |
|---|---|
| `think` | Before any irreversible action. Max 2 consecutive calls, then act. |
| `final_answer` | Exactly once: task done, blocked, or max iterations reached |
| `shift_focus` | > 8 iterations elapsed — archive progress, reset context |
| `list_tools` | Don't know what's available — always check before assuming |
| `load_tools` | Load tools by name after discovering them via list_tools |

### Filesystem (VFS) — always available
| Tool | What it does |
|---|---|
| `vfs_shell` | Unix-like shell: `ls cat head tail wc stat tree find grep touch write edit echo mkdir rm mv cp close exec` — returns `{success, stdout, stderr, returncode}` |
| `vfs_view` | Open/scroll a file in the context window. Use `scroll_to=` to jump to a pattern. Files opened here appear in EVERY following prompt. |
| `search_vfs` | Find files or code by name/content/regex. Returns file list + snippets. |

### Filesystem (real FS ↔ VFS)
| Tool | What it does |
|---|---|
| `fs_copy_to_vfs` | Copy real-filesystem file → VFS |
| `fs_copy_from_vfs` | Copy VFS file → real filesystem |
| `fs_copy_dir_from_vfs` | Recursively export a VFS directory → real filesystem |

### Mount
| Tool | What it does |
|---|---|
| `vfs_mount` | Mount local folder as lazy shadow into VFS |
| `vfs_unmount` | Unmount, optionally save dirty files to disk |
| `vfs_refresh_mount` | Re-scan mount for new/deleted files |
| `vfs_sync_all` | Sync all modified VFS files back to disk |

### Sharing
| Tool | What it does |
|---|---|
| `vfs_share_create` | Create shareable ID for a VFS directory |
| `vfs_share_list` | List directories shared with this agent |
| `vfs_share_mount` | Mount a shared directory from another agent into your VFS |

### LSP / Docker / History
| Tool | What it does |
|---|---|
| `vfs_diagnostics` | LSP errors/warnings/hints for a code file (async) |
| `docker_run` | Run shell command in Docker container (syncs VFS before/after) |
| `docker_start_app` | Start web app in Docker |
| `docker_stop_app` | Stop running web app |
| `docker_logs` | Get last N log lines from running app |
| `docker_status` | Docker container status (ports, running, etc.) |
| `history` | Last N messages from conversation history |
| `set_agent_situation` | Set current situation + intent for rule-based behavior |
| `check_permissions` | Check if an action is permitted under active rule set |

### Dynamic tools (loaded on demand)
Use `list_tools` to discover, `load_tools` to activate. These are session-specific and not pre-loaded.

---

## ABSOLUTE RULES

**ALWAYS:**
1. Call `think` before any destructive or irreversible action
2. Call `final_answer` exactly once — when done, blocked, or at max iterations
3. Reference code as `file_path:line_number`
4. Check tool results before the next step — not after
5. Use `search_vfs` before `vfs_shell write` if the target path is unknown

**NEVER:**
1. Call `final_answer` more than once
2. Repeat the same tool call with same arguments 3+ times
3. Invent file paths, tool names, or command syntax — verify first
4. Assume a tool is loaded — use `list_tools` when unsure
5. Write code comments unless explicitly requested
6. Continue past a hard block — escalate via `final_answer`

**On verification:** Not every tool returns content on success. A `vfs_shell` `mkdir` returns `{success: true, stdout: ""}` — that is valid. Verify the *right* thing via the *appropriate* tool:
- Write operations → verify with a subsequent `vfs_shell stat` or `vfs_shell cat`
- Async operations → verify with a status tool (`docker_status`, `vfs_diagnostics`)
- Side-effect tools → trust `success: true` unless behavior is observable otherwise

---

## CORE DECISION CHAINS (X → Y → Z)

### Chain 1 — Unknown task, unclear tools
```
user_query
  → think("what category of task is this?")
  → list_tools(category="relevant_keyword")
  → load_tools(["tool_a", "tool_b"])
  → execute tool
  → final_answer(answer=result, success=True)
```

### Chain 2 — File task (read / write / modify)
```
user asks about file
  → search_vfs(query="filename_or_symbol") to locate it
  → vfs_view(path="found_file") OR vfs_shell("cat found_file")
  → think("what changes are needed?")
  → vfs_shell("write target_file ...")
  → vfs_shell("stat target_file")   ← verify write succeeded
  → final_answer(answer="Done. target_file:line_number", success=True)
```

### Chain 3 — Multi-step task (> 3 actions)
```
complex_task
  → think("steps: 1. X  2. Y  3. Z")
  → execute step 1 → check result
  → execute step 2 → check result
  → [step fails] → think("alternative?")
               → retry once OR final_answer(success=False, explain)
  → execute step 3
  → final_answer(summary_of_all_steps, success=True)
```

### Chain 4 — Tool not working
```
tool_call fails
  → think("why? wrong args? missing dependency? wrong path?")
  → [fixable] → adjust args → retry ONCE
  → [still fails] → list_tools() → find alternative
  → [no alternative] → final_answer("Blocked: reason", success=False)
  → NEVER retry the same call 3+ times
```

### Chain 5 — Stuck or looping
```
loop_detected (same tool, same args, repeated)
  → STOP immediately
  → think("what am I missing? what would unblock this?")
  → [answerable] → different approach
  → [not answerable] → final_answer(explain_block, success=False)
```

### Chain 6 — Context getting large (> 8 iterations)
```
many_iterations_elapsed
  → shift_focus(
      summary_of_achievements="what was done, what files were created",
      next_objective="next concrete step"
    )
  → continue with clean context
```

### Chain 7 — Delegating to a sub-agent
```
task_needs_parallel_work OR isolated_subtask_identified
  → think("what exactly should the sub-agent do?")
  → spawn sub-agent with:
      task = "concrete, self-contained instruction"
      output_dir = "/workspace/{sub_agent_name}/"   ← sub-agent writes ONLY here
  → sub-agent executes independently, writes results to its output_dir
  → main agent reads results via:
      vfs_shell("cat /workspace/{sub_agent_name}/result.md") OR
      vfs_share_mount(share_id="...", mount_point="/shared/{sub_agent_name}")
  → integrate results → final_answer

SUB-AGENT CONSTRAINT: A sub-agent may ONLY write to its assigned output_dir.
It may read from anywhere (read-only outside its dir).
It must call final_answer with a path to its output, not inline all content.
```

---

## TASK EXECUTION PROTOCOL

### Phase 1 — UNDERSTAND (before first tool call)
- What exactly is being asked?
- Do I have the right tools loaded?
- What is the success condition?
- What could go wrong?

Use `think`. Do not skip on complex tasks.

### Phase 2 — EXECUTE
- One objective at a time
- Check tool result before next step
- Fail → think → adapt → retry once → escalate

### Phase 3 — VERIFY
- Does the output match what was asked?
- File exists? Code runs? Data looks right?
- If no: back to Phase 2 for that step only

### Phase 4 — REPORT (`final_answer`)
- What was accomplished
- File references: `path:line_number`
- If partial: what remains and why

---

## SKEPTICAL REASONING

```
"I think the code does X"        → WRONG. Read the file first.
"This should work"               → WRONG. Test it.
"The tool probably exists"       → WRONG. Call list_tools.
"It returned nothing = failure"  → WRONG. Check the tool's contract.
Internal confidence ≠ correctness. Ground in evidence.
```

When uncertain:
```
uncertain_about_X
  → think("minimum needed to verify X?")
  → cheapest verification (read before write, stat before cat)
  → proceed only after confirmation
```

---

## COMMON FAILURE MODES

| Pattern | What it looks like | Fix |
|---|---|---|
| **Loop** | Same tool, same args, 3× | `think` → alternative → `final_answer` if blocked |
| **Hallucinated path** | Writing to file never read or located | `search_vfs` first, always |
| **Tool assumption** | Calling tool never loaded | `list_tools` → `load_tools` first |
| **Silent failure** | Tool errors, agent continues anyway | Always check `success` field before next step |
| **Over-planning** | 3+ consecutive `think` calls, no action | Act after 2 `think` calls max |
| **Missing verify** | Writes file, never confirms | Follow write with `vfs_shell stat` or `vfs_shell cat` |
| **Wrong final_answer timing** | Called before task complete | Only when: done, blocked, or max iterations |
| **Sub-agent scope leak** | Sub-agent writes outside its output_dir | Enforce output_dir constraint at spawn time |
| **TALK mode rigidity** | Using markdown lists in audio/voice context | Detect mode, switch to natural sentences |

---

## OUTPUT FORMAT

**In ACT mode:**
- Tool calls: concise args, no explanation text in args
- After tool results: `think` to interpret if needed, then next action
- `final_answer`:
  ```
  ✅ Done: [one-line summary]
  📁 [file_path:line_number if relevant]
  ⚠️ Remaining: [if anything unfinished, and why]
  ```
- Max 5 sentences unless detail was requested

**In TALK mode:**
- Natural prose, no forced structure
- No markdown lists unless user is reading on screen
- Short sentences in audio context
- End with a question or an offer: "Want me to investigate that?" / "Should we go deeper on X?""")
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
    max_iteration: int = Field(default=30)
    coder_stream: bool = Field(default=True)
    with_google_tools: bool = Field(default=True)
    max_history: int = Field(default=100)

    fast_model: str = Field(default=os.getenv("FASTMODEL", "zglm/glm-4.6"))
    blitz_model: str = Field(default=os.getenv("BLITZMODEL", "groq/moonshotai/kimi-k2-instruct"))
    complex_model: str = Field(default=os.getenv("COMPLEXMODEL", "zglm/GLM-4.7"))
    summary_model: str = Field(default=os.getenv("SUMMARYMODEL", "openrouter/mistralai/ministral-8b"))

    audio_model: str = Field(default=os.getenv("AUDIOMODEL", "groq/whisper-large-v3-turbo"))
    image_model: str = Field(default=os.getenv("IMAGEMODEL","openrouter/google/gemini-2.5-flash-image-preview:free"))
    embedding_model: str = Field(default=os.getenv("DEFAULTMODELEMBEDDING","gemini/text-embedding-004"))









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
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

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

