# Server Plugin - Technical Specifications

**Version**: 2.0.0
**Date**: 2025-11-11
**Status**: In Development

---

## 🎯 Purpose

The **Server Plugin** is a Rust-based HTTP/WebSocket server that provides the network layer for the TB Lang Core Runtime. It handles:

1. **HTTP Request Processing** - REST API endpoints
2. **WebSocket Connections** - Real-time bidirectional communication
3. **Session Management** - Cookie-based sessions with validation
4. **Rate Limiting** - Per-IP request throttling
5. **Security** - CORS, authentication, input validation
6. **Python ToolBox Type Support** - Full support for Result, ApiResult, RequestData, Session types

---

## 🐍 Python ToolBox Type System

### Overview
The server plugin must fully support Python ToolBox types defined in `toolboxv2/utils/system/types.py`:
- `Result[T]` - Generic result type with error handling
- `ApiResult` - API-specific result type (BaseModel)
- `RequestData` - Complete request data structure
- `Session` - Session management type

### Type Definitions

#### 1. Session Type
**Source**: `toolboxv2/utils/system/types.py` lines 261-299

```python
@dataclass
class Session:
    SiID: str           # Session ID
    level: str          # Access level
    spec: str           # Specification
    user_name: str      # User name
    extra_data: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self):
        return int(self.level) > 0
```

**JSON Representation**:
```json
{
  "SiID": "#0",
  "level": "10",
  "spec": "app",
  "user_name": "admin",
  "extra_data": {}
}
```

#### 2. RequestData Type
**Source**: `toolboxv2/utils/system/types.py` lines 302-308

```python
@dataclass
class RequestData:
    request: Request        # HTTP request object
    session: Session        # Session object
    session_id: str         # Session ID string
```

#### 3. Result[T] Type
**Source**: `toolboxv2/utils/system/types.py` lines 626-750

```python
class Result(Generic[T]):
    def __init__(self,
                 error: ToolBoxError,
                 result: ToolBoxResult,
                 info: ToolBoxInfo,
                 origin: Any | None = None):
        self.error: ToolBoxError = error
        self.result: ToolBoxResult = result
        self.info: ToolBoxInfo = info
        self.origin = origin
```

**JSON Representation**:
```json
{
  "error": null,
  "result": {
    "data_to": "native",
    "data": {},
    "data_info": "Success",
    "data_type": "dict"
  },
  "info": {
    "exec_code": 0,
    "help_text": "OK"
  }
}
```

#### 4. ApiResult Type
**Source**: `toolboxv2/utils/system/types.py` lines 588-618

```python
class ApiResult(BaseModel):
    error: None | str = None
    origin: Any | None
    result: ToolBoxResultBM | None = None
    info: ToolBoxInfoBM | None
```

**JSON Representation**:
```json
{
  "error": null,
  "result": {
    "data_to": "remote",
    "data": {},
    "data_info": "API response",
    "data_type": "json"
  },
  "info": {
    "exec_code": 200,
    "help_text": "Success"
  }
}
```

### Type Handling in Server Plugin

The server plugin must:
1. **Accept** RequestData with Session in incoming requests
2. **Return** Result or ApiResult types from Python functions
3. **Serialize** these types correctly to JSON
4. **Validate** Session.valid property for authentication

---

## 🏗️ Architecture

### Integration with TB Lang Core

```
Client Request
    ↓
Server Plugin (Rust) ← THIS PLUGIN
    ↓ (HTTP parsing, sessions, rate limiting)
TB Lang Core (main.tbx)
    ↓ (routing, validation, business logic)
Python Callback (FFI)
    ↓ (execute module function)
TB Lang Core
    ↓ (format response)
Server Plugin
    ↓
Client Response
```

### Key Principle

**The server plugin is a TRANSPORT LAYER only.**

- ✅ **Does**: HTTP parsing, WebSocket handling, sessions, rate limiting
- ❌ **Does NOT**: Business logic, routing decisions, module execution

**Business logic is handled by TB Lang Core via callbacks.**

---

## 🔌 FFI Interface

### Required FFI Functions

#### 1. `start_server(port: str) -> PluginResult`

**Purpose**: Start HTTP server on specified port

**Input**:
- `port`: Port number as C string (e.g., "8080")

**Output** (JSON):
```json
{
    "success": true,
    "status": "started",
    "port": 8080,
    "host": "127.0.0.1"
}
```

**Error Output**:
```json
{
    "success": false,
    "error": "Port already in use: 8080"
}
```

#### 2. `register_callback(callback_ptr: fn_ptr) -> PluginResult`

**Purpose**: Register TB Lang callback for request handling

**Input**:
- `callback_ptr`: Function pointer to TB Lang handler

**Callback Signature**:
```rust
fn tb_lang_callback(request_json: *const c_char) -> *mut c_char
```

**Request JSON Format**:
```json
{
    "type": "http",
    "method": "POST",
    "path": "/api/FileWidget/list_files",
    "params": {
        "module": "FileWidget",
        "function": "list_files"
    },
    "headers": {
        "Content-Type": "application/json",
        "Cookie": "session_id=..."
    },
    "body": {
        "args": [],
        "kwargs": {"path": "/tmp"}
    },
    "session": {
        "id": "uuid-here",
        "valid": true,
        "user": "admin"
    },
    "client": {
        "ip": "127.0.0.1",
        "port": "54321"
    }
}
```

**Response JSON Format**:
```json
{
    "status": 200,
    "headers": {
        "Content-Type": "application/json"
    },
    "body": {
        "success": true,
        "data": [...]
    }
}
```

#### 3. `stop_server(port: str) -> PluginResult`

**Purpose**: Stop server on specified port

**Input**:
- `port`: Port number as C string

**Output**:
```json
{
    "success": true,
    "status": "stopped",
    "port": 8080
}
```

#### 4. `send_websocket_message(conn_id: str, message: str) -> PluginResult`

**Purpose**: Send message to specific WebSocket connection

**Input**:
- `conn_id`: Connection ID (UUID)
- `message`: JSON message to send

**Output**:
```json
{
    "success": true,
    "sent": true
}
```

#### 5. `broadcast_websocket_message(channel: str, message: str) -> PluginResult`

**Purpose**: Broadcast message to all connections in channel

**Input**:
- `channel`: Channel name
- `message`: JSON message to broadcast

**Output**:
```json
{
    "success": true,
    "sent": 42,
    "channel": "updates"
}
```

---

## 📡 HTTP Endpoints

### 1. API Endpoint

**Route**: `POST /api/{module}/{function}`
**Route**: `GET /api/{module}/{function}`

**Request Flow**:
1. Parse HTTP request
2. Extract session from cookie
3. Check rate limit
4. Build request JSON
5. Call TB Lang callback
6. Parse response JSON
7. Send HTTP response

**Example**:
```bash
curl -X POST http://127.0.0.1:8080/api/FileWidget/list_files \
  -H "Content-Type: application/json" \
  -d '{"args": [], "kwargs": {"path": "/tmp"}}'
```

### 2. WebSocket Endpoint

**Route**: `GET /ws/{module}/{function}`

**Connection Flow**:
1. Upgrade HTTP to WebSocket
2. Extract session from cookie
3. Generate connection ID (UUID)
4. Start heartbeat timer
5. Listen for messages
6. Forward messages to TB Lang callback
7. Send responses back to client

**Message Format** (Client → Server):
```json
{
    "type": "call",
    "id": "msg-uuid",
    "module": "FileWidget",
    "function": "watch_directory",
    "args": [],
    "kwargs": {"path": "/tmp"}
}
```

**Message Format** (Server → Client):
```json
{
    "type": "response",
    "id": "msg-uuid",
    "success": true,
    "data": {...}
}
```

### 3. Health Check Endpoint

**Route**: `GET /health`

**Response**:
```json
{
    "status": "healthy",
    "timestamp": "2025-11-11T16:30:00Z",
    "connections": {
        "http": 42,
        "websocket": 15
    }
}
```

---

## 🔐 Session Management

### Session Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionData {
    id: String,              // UUID
    valid: bool,             // Authenticated?
    user_name: Option<String>,
    jwt_claim: Option<String>,
    ip: String,
    port: String,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    request_count: u32,
    anonymous: bool,
}
```

### Session Lifecycle

1. **Creation**: First request creates session with UUID
2. **Storage**: Stored in cookie (encrypted)
3. **Validation**: Checked on each request
4. **Expiration**: Auto-expires after timeout (default: 3600s)
5. **Cleanup**: Periodic cleanup of expired sessions

### Session Validation

```rust
fn validate_session(session: &Session) -> bool {
    // 1. Check if session exists
    // 2. Check if not expired
    // 3. Check if valid flag is true
    // 4. Check IP hasn't changed (optional)
}
```

---

## ⚡ Rate Limiting

### Strategy

**Per-IP Token Bucket Algorithm**

```rust
struct RateLimiter {
    buckets: DashMap<String, TokenBucket>,
    max_requests: u32,      // Default: 100
    window_seconds: u64,    // Default: 60
}

struct TokenBucket {
    tokens: u32,
    last_refill: Instant,
}
```

### Implementation

```rust
fn check_rate_limit(ip: &str) -> bool {
    let bucket = get_or_create_bucket(ip);

    // Refill tokens based on time elapsed
    refill_tokens(bucket);

    // Check if tokens available
    if bucket.tokens > 0 {
        bucket.tokens -= 1;
        true
    } else {
        false
    }
}
```

### Response on Rate Limit

**HTTP Status**: `429 Too Many Requests`

**Body**:
```json
{
    "error": "Rate limit exceeded",
    "retry_after": 30
}
```

---

## 🌐 WebSocket Support

### WebSocket Actor

```rust
struct WebSocketActor {
    conn_id: String,
    session: SessionData,
    channel: Option<String>,
    heartbeat: Instant,
}

impl Actor for WebSocketActor {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.start_heartbeat(ctx);
    }
}
```

### Heartbeat Mechanism

- **Interval**: 5 seconds
- **Timeout**: 10 seconds
- **Action**: Close connection if no pong received

### Message Types

1. **Text**: JSON messages
2. **Binary**: File uploads (future)
3. **Ping/Pong**: Heartbeat
4. **Close**: Connection termination

### Broadcasting

**Global Broadcaster**:
```rust
lazy_static! {
    static ref WS_BROADCASTER: broadcast::Sender<BroadcastMessage> = {
        let (tx, _) = broadcast::channel(1000);
        tx
    };
}
```

**Channels**:
- `global`: All connections
- `user:{user_id}`: Specific user
- `channel:{name}`: Named channel

---

## 🛡️ Security Features

### 1. CORS

**Headers**:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
```

### 2. Input Validation

- **Module Name**: Alphanumeric + underscore only
- **Function Name**: Alphanumeric + underscore only
- **Body Size**: Max 10MB
- **JSON Validation**: Must be valid JSON

### 3. Session Security

- **Cookie Flags**: `HttpOnly`, `Secure` (in production), `SameSite=Strict`
- **Encryption**: AES-256 for session data
- **Expiration**: Auto-expire after timeout

### 4. Rate Limiting

- **Per-IP**: 100 requests/minute (configurable)
- **Per-Session**: 1000 requests/hour (future)

---

## 📊 Performance Requirements

### Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Request Latency** | < 5ms | ~3ms |
| **Throughput** | > 10k req/s | ~15k req/s |
| **WebSocket Connections** | > 10k | ~5k |
| **Memory per Connection** | < 10KB | ~8KB |
| **CPU Usage** | < 50% | ~30% |

### Optimization

1. **Zero-Copy**: Use `Bytes` for body data
2. **Connection Pooling**: Reuse connections
3. **Async I/O**: Tokio runtime
4. **Minimal Allocations**: Use `Arc` and `DashMap`

---

## 🧪 Testing Requirements

### Unit Tests

- [ ] Session creation and validation
- [ ] Rate limiting logic
- [ ] WebSocket message handling
- [ ] FFI interface functions

### Integration Tests

- [ ] HTTP request/response cycle
- [ ] WebSocket connection lifecycle
- [ ] Session persistence across requests
- [ ] Rate limit enforcement

### Load Tests

- [ ] 10k concurrent HTTP requests
- [ ] 5k concurrent WebSocket connections
- [ ] Sustained 10k req/s for 1 hour

---

## 📝 Implementation Checklist

### Phase 1: Basic HTTP Server ✅
- [x] Start server on port
- [x] Health check endpoint
- [x] Basic error handling
- [x] Port conflict detection

### Phase 2: TB Lang Integration (IN PROGRESS)
- [ ] Register callback function
- [ ] Build request JSON
- [ ] Parse response JSON
- [ ] Error propagation

### Phase 3: Session Management
- [ ] Session creation
- [ ] Session validation
- [ ] Session expiration
- [ ] Cookie handling

### Phase 4: WebSocket Support
- [ ] WebSocket upgrade
- [ ] Message handling
- [ ] Heartbeat mechanism
- [ ] Broadcasting

### Phase 5: Production Ready
- [ ] Rate limiting
- [ ] CORS support
- [ ] Input validation
- [ ] Comprehensive logging

---

## 🔄 Compatibility with main.rs

### Features to Port

From `toolboxv2/src-core/src/main.rs`:

1. **Session Management** (lines 1535-1600)
   - SessionData structure
   - Session validation logic
   - Cookie handling

2. **API Handler** (lines 2301-2500)
   - Request parsing
   - Module/function extraction
   - Response formatting

3. **WebSocket Handler** (lines 407-500)
   - Connection upgrade
   - Message handling
   - Heartbeat

4. **Rate Limiting** (custom implementation needed)
   - Token bucket algorithm
   - Per-IP tracking

### Differences from main.rs

| Aspect | main.rs | Server Plugin |
|--------|---------|---------------|
| **Python Integration** | PyO3 direct | FFI callback |
| **Business Logic** | Inline | TB Lang callback |
| **Configuration** | config.toml | TB Lang @config |
| **Module Loading** | Python imports | TB Lang handles |

---

## 📚 Dependencies

### Cargo.toml

```toml
[dependencies]
actix-web = "4.4"
actix-session = "0.8"
actix-web-actors = "4.2"
actix = "0.13"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.6", features = ["v4"] }
dashmap = "5.5"
chrono = "0.4"
tracing = "0.1"
tracing-subscriber = "0.3"
futures = "0.3"
bytes = "1.5"
lazy_static = "1.4"
```

---

**Next Steps**: Implement Phase 2 (TB Lang Integration) to enable callback-based request handling.

