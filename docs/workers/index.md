# Worker System

> **Location**: `toolboxv2/utils/workers/`
> **Replaces**: Legacy Rust/Actix server for all web/server deployments

High-performance Python worker system using raw WSGI + asyncio + ZeroMQ. Rust (Actix/Tauri) is retained only for the desktop app binary.

## Architecture

```
Nginx (port 80/443)
│   Load balancing · Rate limiting · SSL termination
│
├── HTTP Workers × N    (WSGI, port 8000, 8001, ...)
│   Raw WSGI via Waitress — ~5,000 req/s per worker
│   Signed cookie sessions · Module function routing
│
├── WebSocket Workers × M    (asyncio, port 8010)
│   ~10,000 concurrent connections per instance
│   NO business logic — forwards via ZeroMQ
│
└── ZeroMQ Event Broker
    PUB/SUB broadcasts · PUSH/PULL HTTP→WS forwarding · REQ/REP RPC
```

## Quick Start

### Start All Workers

```bash
# Via tb CLI
tb workers start

# Via Python CLI directly
cd toolboxv2/utils/workers
python cli_worker_manager.py start
```

[USAGE INFOS](../core/workers.md)

### Development Mode (no Nginx)

```bash
# Terminal 1
python event_manager.py

# Terminal 2
python server_worker.py -v

# Terminal 3
python ws_worker.py -v
```

### Generate Config

```bash
python config.py generate -o config.yaml
```

## CLI Reference

```bash
python cli_worker_manager.py start           # Start all workers
python cli_worker_manager.py stop            # Stop all workers
python cli_worker_manager.py update          # Zero-downtime rolling update
python cli_worker_manager.py status          # Show worker status
python cli_worker_manager.py nginx-config    # Generate nginx config
python cli_worker_manager.py nginx-reload    # Reload nginx
python cli_worker_manager.py worker-start -t http   # Start single HTTP worker
python cli_worker_manager.py worker-start -t ws     # Start single WS worker
python cli_worker_manager.py worker-stop -w <id>    # Stop specific worker
```

Web UI available at `http://localhost:9000` while workers are running.

## Components

### `server_worker.py` — HTTP Worker

Raw WSGI application:

- API requests via `GET/POST /api/{ModuleName}/{FunctionName}`
- Maps to `app.a_run_any((ModuleName, FunctionName), **kwargs)`
- Session management via signed cookies (HMAC-SHA256, stateless)
- CloudM.Auth JWT validation on protected routes

### `ws_worker.py` — WebSocket Worker

Minimal processing handler:

- Accepts connections, assigns `conn_id`
- Forwards all messages to HTTP workers via ZeroMQ
- Receives `WS_SEND` events from ZeroMQ → pushes to clients
- No business logic in this layer

### `event_manager.py` — ZeroMQ Event System

```
HTTP Worker  →  PUSH  →  Broker  →  PULL  →  App Instance
App Instance →  PUB   →  Broker  →  SUB   →  All Workers
HTTP Worker  →  REQ   →  Broker  →  REP   →  App Instance (RPC)
```

Throughput: ~100,000 msg/s.

### `session.py` — Stateless Sessions

HMAC-SHA256 signed cookies — no server-side session store needed, perfect for horizontal scaling.

### `tauri_integration.py` — Desktop Support

```python
from tauri_integration import tauri_start_workers

result = tauri_start_workers()
# {"status": "ok", "http_url": "http://127.0.0.1:8000", "ws_url": "ws://127.0.0.1:8001"}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TB_ENV` | `development` | `development` / `production` / `tauri` |
| `TB_COOKIE_SECRET` | — | **Required in production** (64+ chars) |
| `TB_HTTP_PORT` | `8000` | HTTP worker base port |
| `TB_WS_PORT` | `8010` | WebSocket worker port |
| `CLERK_SECRET_KEY` | — | Legacy Clerk auth (if still used) |

### `tb-manifest.yaml`

```yaml
workers:
  http:
    enabled: true
    instances: 2        # Scale to CPU core count in production
    port: 8000
  ws:
    enabled: true
    instances: 1
    port: 8010

environments:
  production:
    workers:
      http:
        instances: 4
```

## WebSocket Protocol

```json
// Server → Client on connect
{"type": "connected", "conn_id": "uuid"}

// Client → Server (any message)
// Forwarded via ZeroMQ to HTTP workers

// Server → Client (response via ZMQ WS_SEND event)
{"type": "response", "data": {...}}
```

## Nginx Configuration

Generated automatically with `cli_worker_manager.py nginx-config`:

```nginx
upstream http_workers {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

upstream ws_workers {
    server 127.0.0.1:8010;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location /api/ {
        limit_req zone=api burst=20;
        proxy_pass http://http_workers;
    }

    location /ws/ {
        proxy_pass http://ws_workers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Performance

| Component | Throughput |
|-----------|-----------|
| HTTP Worker (Waitress/WSGI) | ~5,000 req/s per worker |
| WS Worker | ~10,000 concurrent connections |
| ZeroMQ Broker | ~100,000 msg/s |

## Production Checklist

- [ ] Set `TB_COOKIE_SECRET` (64+ random chars)
- [ ] Set `TB_ENV=production`
- [ ] Configure SSL in nginx
- [ ] Scale `http_worker.instances` to CPU core count
- [ ] Adjust `ws_worker.max_connections` based on available RAM
- [ ] Enable rate limiting in nginx config
- [ ] Set up log rotation
- [ ] Set `CLOUDM_JWT_SECRET` for auth (see [CloudM Auth](../mods/CloudM/auth.md))

## File Structure

```
toolboxv2/utils/workers/
├── __init__.py
├── config.py               ← YAML configuration loader
├── config.yaml             ← Default config
├── event_manager.py        ← ZeroMQ event broker
├── session.py              ← Signed cookie sessions
├── server_worker.py        ← HTTP WSGI worker
├── ws_worker.py            ← WebSocket worker
├── cli_worker_manager.py   ← Orchestration CLI + Web UI
├── toolbox_integration.py  ← ToolBoxV2 App instance bridge
├── tauri_integration.py    ← Tauri desktop support
└── requirements.txt
```

## Related

- [Server Guide](../guides/howto_server.md) — Full production setup walkthrough
- [Manifest Reference](../manifest/ref_manifest.md) — Worker configuration schema
- [System Stack](../new/analysis/installation/stack.md) — Where workers fit in the full architecture
- [Tauri App Architecture](../new/old_docs/TauriApp_Architecture.md) — Desktop app using `tauri_integration.py`
