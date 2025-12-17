"""
ToolBoxV2 MCP Server - Data Types
=================================
Memory-efficient data containers using __slots__
Following ToolBox V2 Architecture Guidelines
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from datetime import datetime
import time


# =============================================================================
# ENUMS
# =============================================================================


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"
    STRUCTURED = "structured"


class ServerMode(str, Enum):
    """Server transport mode."""

    STDIO = "stdio"
    HTTP = "http"


class PermissionLevel(str, Enum):
    """API key permission levels."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class FlowState(str, Enum):
    """Flow session state."""

    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


# =============================================================================
# DATA CLASSES WITH __slots__ (Memory Efficient)
# =============================================================================


@dataclass(slots=True)
class APIKeyInfo:
    """API Key metadata - frequently created, must be memory efficient."""

    name: str
    permissions: List[str]
    created: float
    last_used: Optional[float] = None
    usage_count: int = 0

    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "permissions": self.permissions,
            "created": self.created,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
        }


@dataclass(slots=True)
class FlowSession:
    """Flow session data - many instances possible."""

    session_id: str
    flow_name: str
    created: float
    last_activity: float
    state: FlowState
    context: Dict[str, Any]
    history: List[str]

    def update_activity(self):
        self.last_activity = time.time()

    def is_expired(self, timeout: int = 3600) -> bool:
        return time.time() - self.last_activity > timeout


@dataclass(slots=True)
class ToolResult:
    """Result from tool execution."""

    success: bool
    content: str
    execution_time: float
    cached: bool = False
    error: Optional[str] = None


@dataclass(slots=True)
class CacheEntry:
    """Cache entry for query results."""

    key: str
    value: Any
    timestamp: float
    ttl: int = 300  # 5 minutes default

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


@dataclass(slots=True)
class PerformanceMetrics:
    """Server performance metrics."""

    requests_handled: int = 0
    total_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    init_time: float = 0.0

    @property
    def avg_response_time(self) -> float:
        if self.requests_handled == 0:
            return 0.0
        return self.total_response_time / self.requests_handled

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def record_request(
        self, response_time: float, cached: bool = False, error: bool = False
    ):
        self.requests_handled += 1
        self.total_response_time += response_time
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        if error:
            self.errors += 1


# =============================================================================
# CONFIGURATION (Not slots - needs flexibility)
# =============================================================================


@dataclass
class MCPConfig:
    """MCP Server configuration with smart defaults."""

    # Server Identity
    server_name: str = "toolboxv2_mcp"
    server_version: str = "3.0.0"

    # Transport
    server_mode: ServerMode = ServerMode.STDIO
    http_host: str = "127.0.0.1"  # Secure default (not 0.0.0.0)
    http_port: int = 8765

    # Security
    api_keys_file: str = "MCPConfig/mcp_api_keys.json"
    require_auth: bool = True

    # Features
    enable_python: bool = True
    enable_docs: bool = True
    enable_flows: bool = True
    enable_system: bool = True

    # Performance
    lazy_load: bool = True
    use_cache: bool = True
    cache_ttl: int = 300
    max_cache_size: int = 100

    # Sessions
    session_timeout: int = 3600
    max_sessions: int = 100

    # Logging
    silent_mode: bool = False
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_name": self.server_name,
            "server_version": self.server_version,
            "mode": self.server_mode.value,
            "features": {
                "python": self.enable_python,
                "docs": self.enable_docs,
                "flows": self.enable_flows,
                "system": self.enable_system,
            },
            "performance": {
                "lazy_load": self.lazy_load,
                "cache_enabled": self.use_cache,
                "cache_ttl": self.cache_ttl,
            },
        }


# =============================================================================
# RESOURCE TEMPLATES
# =============================================================================

FLOWAGENTS_DISCOVERY_TEMPLATE = """# ToolBoxV2 MCP Server - Resource Discovery

## 🚀 Server Capabilities

### Core Tools
1. **toolbox_execute** - Execute any ToolBox module function
2. **python_execute** - Run Python code with persistent state
3. **docs_reader** - Search documentation (Inverted Index)
4. **docs_writer** - Create/update documentation
5. **source_code_lookup** - Find code elements
6. **toolbox_status** - System health check
7. **toolbox_info** - Module discovery

### Flow Tools
- **flow_start** - Initialize workflow
- **flow_continue** - Continue workflow with input
- **flow_status** - Check workflow state
- **flow_list** - List available flows

## ⚡ Performance Tips
- Use `section_id` for direct docs access (fastest)
- Enable caching with `use_cache=true`
- Set appropriate `max_results` limits
- Use `format_type="json"` for programmatic processing

## 🔗 Quick Start
```
1. toolbox_status() → System overview
2. toolbox_info(info_type="modules") → Discover modules
3. toolbox_execute(module="target", function="action") → Execute
```
"""

PYTHON_EXECUTION_TEMPLATE = """# Python Execution Environment

## Available Variables
- `app` / `tb` - ToolBoxV2 App instance
- `result` - Last execution result

## Persistent State
Variables defined in one call persist to the next.

## Example
```python
# Define a function
def greet(name):
    return f"Hello, {name}!"

# Use it later
result = greet("World")
print(result)
```

## Safety Notes
- Execution has timeout protection (default: 30s)
- stdout/stderr are captured and returned
- Full ToolBox access via `app` variable
"""

PERFORMANCE_GUIDE_TEMPLATE = """# Performance Optimization Guide

## Query Optimization
| Method | Speed | Use Case |
|--------|-------|----------|
| `section_id` | ⚡ Fastest | Direct access |
| `file_path` | 🚀 Fast | File-scoped search |
| `tags` | 🔄 Medium | Tag filtering |
| `query` | 🐢 Slower | Full-text search |

## Caching Strategy
- Cache TTL: {cache_ttl}s
- Max entries: {max_cache_size}
- Auto-cleanup on limit

## Current Metrics
- Requests: {requests}
- Avg Response: {avg_time:.3f}s
- Cache Hit Rate: {hit_rate:.1%}
"""
