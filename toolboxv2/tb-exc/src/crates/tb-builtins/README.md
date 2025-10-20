# TB Built-in Functions

High-performance, non-blocking built-in functions for TB Language.

## Features

### 🚀 **High Performance**
- **Async/Non-blocking I/O**: All I/O operations use Tokio for maximum concurrency
- **Zero-copy where possible**: Efficient memory usage with Arc and shared references
- **Multi-threaded runtime**: 4-worker Tokio runtime for parallel execution

### 📁 **File I/O**
- Real file operations (async)
- Blob storage support (distributed, content-addressable)
- Encryption support for sensitive data
- Automatic caching for blob files

### 🌐 **Networking**
- **HTTP/HTTPS**: Session management, cookies, custom headers
- **TCP**: Client and server with callbacks
- **UDP**: Client and server support
- **WebSocket**: (Planned)

### 🛠️ **Utilities**
- **JSON**: Parse and stringify with full type conversion
- **YAML**: Parse and stringify support
- **Time**: Timezone-aware time operations
- **Base64**: Encoding and decoding
- **Hex**: Byte conversion utilities

## Architecture

```
tb-builtins/
├── src/
│   ├── lib.rs              # Main entry point, function registration
│   ├── error.rs            # Error types and conversions
│   ├── file_io.rs          # File I/O operations
│   ├── blob.rs             # Distributed blob storage
│   ├── networking.rs       # HTTP, TCP, UDP networking
│   ├── utils.rs            # JSON, YAML, time utilities
│   └── builtins_impl.rs    # Built-in function implementations
└── Cargo.toml
```

## Usage in TB Language

### File I/O

```tb
// Read and write files
write_file("data.txt", "Hello, World!")
let content = read_file("data.txt")
print(content)  // "Hello, World!"

// Check file existence
if file_exists("config.json") {
    let config = read_file("config.json")
}

// Blob storage
let storage = blob_init(["http://server1:8080", "http://server2:8080"])
let blob_id = blob_create(storage, "Important data")
let data = blob_read(storage, blob_id)
```

### Networking

```tb
// HTTP requests
let session = http_session("https://api.example.com")
let response = http_request(session, "/users", "GET")
print(response.status)  // 200
print(response.body)

// POST with JSON
let data = {"name": "Alice", "age": 30}
let response = http_request(session, "/users", "POST", data)

// TCP server
let on_connect = fn(addr, msg) { print("Client connected: " + addr) }
let on_disconnect = fn(addr) { print("Client disconnected: " + addr) }
let on_message = fn(addr, msg) { print("Received: " + msg) }

let server = create_server(
    on_connect,
    on_disconnect,
    on_message,
    "0.0.0.0",
    8080,
    "tcp"
)

// TCP client
let conn = connect_to(on_connect, on_disconnect, on_message, "localhost", 8080, "tcp")
send_to(conn, "Hello, Server!")
```

### JSON/YAML

```tb
// JSON
let json_str = '{"name": "Bob", "age": 25}'
let data = json_parse(json_str)
print(data["name"])  // "Bob"

let obj = {"users": ["Alice", "Bob"], "count": 2}
let json = json_stringify(obj, true)  // Pretty-printed

// YAML
let yaml_str = "name: Alice\nage: 30"
let data = yaml_parse(yaml_str)

let config = {"port": 8080, "debug": true}
let yaml = yaml_stringify(config)
```

### Time

```tb
// Current time
let now = time()
print(now["year"])       // 2024
print(now["month"])      // 10
print(now["hour"])       // 14
print(now["timezone"])   // "Local"
print(now["iso8601"])    // "2024-10-19T14:30:45+00:00"

// Specific timezone
let ny_time = time("America/New_York")
let tokyo_time = time("Asia/Tokyo")
```

## Implementation Details

### Async Runtime

All I/O operations run on a global Tokio runtime:

```rust
pub static RUNTIME: Lazy<tokio::runtime::Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime")
});
```

### Blob Storage

Based on the Python `blobs.py` implementation with:
- **Consistent hashing** for distribution across servers
- **Local caching** for performance
- **Content-addressable storage** (SHA-256 hashing)
- **Encryption support** (AES-GCM)
- **Automatic failover** to backup servers

### Error Handling

All errors are converted to TB's `TBError` type:

```rust
pub enum BuiltinError {
    Io(std::io::Error),
    Network(reqwest::Error),
    Serialization(String),
    BlobStorage(String),
    Encryption(String),
    InvalidArgument(String),
    NotFound(String),
    Runtime(String),
}

impl From<BuiltinError> for TBError {
    fn from(err: BuiltinError) -> Self {
        TBError::RuntimeError {
            message: err.to_string(),
        }
    }
}
```

## Performance Characteristics

### File I/O
- **Async operations**: Non-blocking, concurrent file access
- **Blob caching**: O(1) cache lookup, reduces network calls
- **Streaming**: Large files handled efficiently

### Networking
- **Connection pooling**: HTTP sessions reuse connections
- **Persistent cookies**: Automatic cookie management
- **Timeout handling**: 30-second default timeout
- **Retry logic**: Automatic retry with exponential backoff (blob storage)

### JSON/YAML
- **Zero-copy parsing**: Uses `serde` for efficient deserialization
- **Type conversion**: Automatic conversion between TB and JSON/YAML types
- **Pretty printing**: Optional formatting for readability

### Time
- **Timezone support**: Full IANA timezone database
- **Multiple formats**: ISO8601, RFC3339, RFC2822
- **Efficient parsing**: Uses `chrono` for fast date/time operations

## Dependencies

```toml
tokio = { version = "1.40", features = ["full"] }
reqwest = { version = "0.12", features = ["json", "cookies", "stream"] }
serde_json = "1.0"
serde_yaml = "0.9"
chrono = { version = "0.4", features = ["serde"] }
chrono-tz = "0.9"
sha2 = "0.10"
aes-gcm = "0.10"
dashmap = "6.1"
```

## Testing

Run tests with:

```bash
# Unit tests
cargo test -p tb-builtins

# Integration tests (TB Language)
python toolboxv2/utils/tbx/test/test_tb_lang2.py --filter "Built-in Functions"
```

## Future Enhancements

- [ ] WebSocket support
- [ ] gRPC client/server
- [ ] GraphQL client
- [ ] Database connections (PostgreSQL, MySQL, MongoDB)
- [ ] Message queues (RabbitMQ, Kafka)
- [ ] Caching (Redis, Memcached)
- [ ] File compression (gzip, zstd)
- [ ] Image processing
- [ ] CSV/Excel parsing
- [ ] Regular expressions
- [ ] Cryptographic functions (signing, verification)

## License

MIT License - See LICENSE file for details

---

**TB Language v0.1.0** - *Fast, Safe, Interoperable* 🚀

