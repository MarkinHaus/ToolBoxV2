# ToolBoxV2 Worker System

High-performance Python worker system for ToolBoxV2 - replacing Rust server with raw WSGI.

## Architecture

```
                    ┌─────────────┐
                    │    Nginx    │
                    │ (Load Bal., │
                    │ Rate Limit) │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ HTTP Worker │   │ HTTP Worker │   │ WS Worker   │
  │  (WSGI)     │   │  (WSGI)     │   │ (asyncio)   │
  │  Port 8000  │   │  Port 8001  │   │  Port 8010  │
  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
         │                 │                 │
         └────────────┬────┴─────────────────┘
                      │
               ┌──────▼──────┐
               │ ZeroMQ      │
               │ Event Broker│
               │ (Pub/Sub)   │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │ ToolBoxV2   │
               │ App Instance│
               └─────────────┘
```

## Features

- **Raw WSGI**: No framework overhead
- **Signed Cookies**: Stateless session management for horizontal scaling
- **ZeroMQ IPC**: High-performance inter-worker communication
- **Nginx Integration**: Rate limiting, load balancing, SSL termination
- **Zero-Downtime Updates**: Rolling updates without service interruption
- **Multi-Environment**: Development, Production, Tauri desktop

## Quick Start

### Installation

```bash
cd tbv2_workers
pip install -r requirements.txt
```

### Generate Config

```bash
python config.py generate -o config.yaml
```

### Start All Workers

```bash
python cli_worker_manager.py start
```

### Development Mode (without Nginx)

```bash
# Terminal 1: Start broker
python event_manager.py

# Terminal 2: Start HTTP worker
python server_worker.py -v

# Terminal 3: Start WS worker
python ws_worker.py -v
```

## Configuration

Environment variables:
- `TB_ENV`: Environment (development/production/tauri)
- `TB_COOKIE_SECRET`: Session cookie secret (required in production!)
- `TB_HTTP_PORT`: HTTP worker port
- `TB_WS_PORT`: WebSocket worker port
- `CLERK_SECRET_KEY`: Clerk auth secret

See `config.yaml` for all options.

## Components

### server_worker.py - HTTP Worker

Raw WSGI application handling:
- API requests (`/api/Module/function`)
- Session management via signed cookies
- Clerk authentication integration
- ToolBoxV2 module function routing

### ws_worker.py - WebSocket Worker

Minimal processing WebSocket handler:
- Maximum connection capacity (~10,000 connections)
- Channel/room subscriptions
- Message forwarding via ZeroMQ
- NO business logic (handled by HTTP workers)

### event_manager.py - ZeroMQ Event System

Inter-worker communication:
- PUB/SUB for broadcasts
- PUSH/PULL for HTTP→WS message forwarding
- REQ/REP for RPC calls

### session.py - Stateless Sessions

Signed cookie implementation:
- HMAC-SHA256 signatures
- No server-side storage needed
- Perfect for horizontal scaling

### cli_worker_manager.py - Orchestration

Complete management:
- Start/stop/restart workers
- Zero-downtime rolling updates
- Nginx configuration generation
- Health monitoring
- Web UI at http://localhost:9000

## API Routing

```
GET/POST /api/{ModuleName}/{FunctionName}
```

Maps to:
```python
app.a_run_any((ModuleName, FunctionName), **kwargs)
```

## Session/Auth Flow

1. Client sends Bearer token or Cookie
2. Worker verifies via `CloudM.AuthClerk.verify_session`
3. Session data stored in signed cookie
4. Subsequent requests use cookie (stateless)

## WebSocket Protocol

```json
// Server → Client (on connect)
{"type": "connected", "conn_id": "uuid"}

// Client → Server (any message)
// Forwarded via ZMQ to HTTP workers for processing

// Server → Client (response)
// Via ZMQ WS_SEND event
```

## Nginx Configuration

Generated automatically with:
- Rate limiting (10r/s default)
- Load balancing (least_conn)
- WebSocket upgrade handling
- Static file serving
- SSL termination (optional)

## Tauri Integration

For desktop apps:

```python
from tauri_integration import tauri_start_workers

result = tauri_start_workers()
# {"status": "ok", "http_url": "http://127.0.0.1:8000", "ws_url": "ws://127.0.0.1:8001"}
```

## File Structure

```
tbv2_workers/
├── __init__.py           # Package exports
├── config.py             # YAML configuration
├── config.yaml           # Default config
├── event_manager.py      # ZeroMQ event system
├── session.py            # Signed cookie sessions
├── server_worker.py      # HTTP WSGI worker
├── ws_worker.py          # WebSocket worker
├── cli_worker_manager.py # Orchestration CLI
├── toolbox_integration.py # ToolBoxV2 integration
├── tauri_integration.py  # Tauri desktop support
└── requirements.txt      # Dependencies
```

## Usage with ToolBoxV2

Place in `toolboxv2/utils/workers/` or keep separate.

Integration in `toolboxv2/__main__.py`:

```python
def server_helper(instance_id="default", **kwargs):
    # Your existing implementation
    app = App(prefix=instance_id, args=AppArgs())
    # ... setup ...
    return app
```

Workers call `server_helper()` to get App instance.

## Performance Notes

- HTTP Worker: ~5000 req/s per worker (Waitress)
- WS Worker: ~10,000 concurrent connections per instance
- ZeroMQ: ~100,000 msg/s throughput

## CLI Commands

```bash
# Start all
python cli_worker_manager.py start

# Stop all
python cli_worker_manager.py stop

# Rolling update
python cli_worker_manager.py update

# Status
python cli_worker_manager.py status

# Generate nginx config
python cli_worker_manager.py nginx-config

# Reload nginx
python cli_worker_manager.py nginx-reload

# Start single worker
python cli_worker_manager.py worker-start -t http
python cli_worker_manager.py worker-start -t ws

# Stop single worker
python cli_worker_manager.py worker-stop -w worker_id
```

## Production Checklist

- [ ] Set `TB_COOKIE_SECRET` (64+ random chars)
- [ ] Set `TB_ENV=production`
- [ ] Configure SSL in nginx
- [ ] Set `CLERK_SECRET_KEY` if using Clerk
- [ ] Adjust `http_worker.workers` based on CPU cores
- [ ] Adjust `ws_worker.max_connections` based on RAM
- [ ] Enable rate limiting in nginx config
- [ ] Set up log rotation

## License

Part of ToolBoxV2
