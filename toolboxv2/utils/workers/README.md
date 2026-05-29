# ToolBoxV2 Worker System

A high-performance, self-healing Python worker system. Raw WSGI for HTTP, asyncio
for WebSockets, and a leaderless ZeroMQ bus that elects its own broker at runtime —
no single point of failure, no dedicated broker process required.

It also implements a **One-Port Collective**: multiple FastTB UIs can share a single
HTTP port. The first app to bind owns `/`; later apps either merge into it (loaded
fresh from source) or run as independent specialists on their own port. Nothing
blocks, nothing freezes.

---

## Architecture

```
                          ┌──────────────┐
                          │    Nginx     │   load balancing, rate limit, SSL
                          └──────┬───────┘
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
     ┌────────────┐      ┌────────────┐       ┌────────────┐
     │ HTTP Worker│      │ HTTP Worker│       │ WS Worker  │
     │  (WSGI)    │      │  (WSGI)    │       │ (asyncio)  │
     └─────┬──────┘      └─────┬──────┘       └─────┬──────┘
           └───────────────────┼────────────────────┘
                               ▼
                    ┌────────────────────┐
                    │  ZeroMQ Event Bus  │  PUB/SUB · REQ/REP · PUSH/PULL
                    │  (P2P, self-elect) │  one worker == leader at a time
                    └────────────────────┘
```

The **leader** owns only the internal bus (event routing, RPC, heartbeat,
topology, HTTP→WS forwarding). It does **not** sit in the HTTP request path —
request load balancing is nginx's job. A leader dying therefore never interrupts
HTTP routing; the bus is restored within a heartbeat timeout.

### One-Port Collective (FastTB)

```
   App A starts first  ──►  binds port, owns "/"  + "/customA"
   App B starts later  ──►  port busy → ask the owner to mount App B from src
        compatible?         (no collision on App B's non-root routes)
          yes ──► merge: App B "/" relocates to "/appB", "/customB" stays top-level
          no  ──► run as SPECIALIST on the next free port (own process)
```

Mounted code is always re-imported from source at mount time, so it reflects the
newest version on disk regardless of when the owner started.

---

## Self-Healing

| Failure | Detection | Recovery |
|---|---|---|
| Leader dies (crash) | follower watchdog: no heartbeat | a follower re-binds the bus and becomes leader (`_takeover`) |
| Leader stops on request | `/live` → `SYS_RELINQUISH` RPC | leader demotes to follower; another takes over |
| Live-UI owner dies | standby worker on same port | standby binds the freed port and serves the UI |
| HTTP/WS worker dies | manager health check | manager respawns it (new process = fresh source) |

Re-election uses randomized jitter to avoid thundering herd. A worker that
relinquishes leadership resets its own watchdog, biasing the election toward
*other* candidates — but if it is the only node, it re-elects itself (the system
never runs with zero brokers).

---

## Components

| File | Role |
|---|---|
| `server_worker.py` | HTTP worker (Waitress/WSGI): API routing, auth, sessions, `/live`, standby bind |
| `ws_worker.py` | WebSocket worker (asyncio): up to ~10k connections, channels, ZMQ forwarding |
| `event_manager.py` | ZeroMQ bus + P2P leader election, takeover, graceful relinquish |
| `session.py` | Stateless signed-cookie sessions (HMAC-SHA256) + JWT bearer fallback |
| `fast_tb.py` | FastTB routing, `serve()`, One-Port-Collective mount/unmount |
| `fast_tb_handler.py` | Dispatch engine + `as_wsgi_app()` (built-ins + WS infra) |
| `fast/tray_api.py` | Tray/desktop bridge + collective `mount_app`/`unmount_app` commands |
| `fast/local_ui.py` | Local admin UI (the default `/` owner) |
| `cli_worker_manager.py` | Orchestration: spawn/stop/restart, health, nginx, cluster |
| `tauri_integration.py` | Desktop sidecar: FastTB UI + bus + WS in one process |
| `ws_bridge.py` | `app.ws_send` / `ws_broadcast` over ZMQ |
| `config.py` | Dataclass config + YAML/env loading |

---

## Quick Start

```bash
pip install -r requirements.txt

# generate a config (optional; sane defaults otherwise)
python config.py generate -o config.yaml

# start everything: broker-capable workers, WS worker, live-UI replicas, nginx
tb workers start
```

When `tb workers start` runs, the live-UI replicas register the collective mount
commands, so other FastTB UIs can attach to the same port automatically. The first
to bind owns `/`; the rest stand by for failover.

### Serving a FastTB app

```python
from toolboxv2.utils.workers.fast_tb import FastTB

app = FastTB(title="My App")

@app.get("/")
async def index(request):
    return "<h1>hello</h1>"

# Owner if the port is free; joins the owner from src if busy and compatible;
# otherwise binds the next free port as a specialist.
app.serve(
    host="127.0.0.1", port=8080,
    module_path="my.package.app_module",   # owner re-imports this from src
    fallback_prefix="myapp",               # own "/" -> "/myapp" when merging
)
```

To make an app step aside voluntarily (e.g. keep `/` free for another UI):

```python
app.relocate_root("myapp")   # this app's "/" now lives at "/myapp"
app.serve(host="127.0.0.1", port=8080)
```

### Development mode (no nginx)

```bash
python server_worker.py -v      # HTTP worker (also self-elects broker if free)
python ws_worker.py -v          # WS worker
```

The bus needs no separate broker process: whichever worker binds the ZMQ
endpoints first becomes the leader; the rest connect as followers.

---

## CLI

```
tb workers start            start all services + live UI
tb workers stop             stop everything
tb workers restart          stop + start
tb workers status           show worker/cluster status
tb workers update           rolling, zero-downtime update
tb workers live             open the /live dashboard

tb workers worker-start -t http|ws    start a single worker
tb workers worker-stop -w <id>        stop a single worker

tb workers nginx-init       write initial nginx site config
tb workers nginx-check      validate nginx wiring
tb workers nginx-config     print generated nginx config
tb workers nginx-reload     reload nginx

tb workers cluster-join     join a remote node (P2P cluster)
tb workers debug            run the debug server
```

---

## `/live` Dashboard

The live dashboard exposes the manager surface (status, workers, metrics, scale,
rolling update, nginx reload) and direct bus control. It is intentionally **not**
the public `/` — it sits behind its own gated path and is protected by a dashboard
key.

Direct broker control (not a manager-API proxy):

```
POST /live/mgr/broker/relinquish   ask the current leader to step down
```

This issues a `SYS_RELINQUISH` RPC over the bus. Because only the leader binds the
REP socket, the call always reaches the right worker; a follower then takes over.

Enable it with a dashboard key:

```bash
export LIVE_DASHBOARD_KEY="<random>"
# or set manager.live_dashboard_key in config
```

---

## API Routing

```
GET|POST /api/{Module}/{Function}   →   app.a_run_any((Module, Function), **kwargs)
```

Built-in endpoints served by every HTTP worker: `/health`, `/metrics`, `/api/ip`,
`/api/ping`, `/api/geo`, `/api/client-logs`, plus the auth endpoints
(`/validateSession`, `/auth/discord/*`, `/auth/google/*`, `/auth/magic/verify`, …).

---

## Sessions & Auth

1. Client sends a session cookie or a `Bearer` token.
2. The worker validates it (cookie first; JWT bearer fallback for cross-origin /
   Tauri where cookies are not sent).
3. Session state lives in a **signed cookie** (HMAC-SHA256) — no server-side
   store, so workers scale horizontally with no shared session backend.

OAuth providers: Discord, Google, and passwordless magic-link.

---

## WebSocket Protocol

```jsonc
// server → client on connect
{"type": "connected", "conn_id": "uuid"}

// client → server: any message is forwarded over ZMQ to an HTTP worker
// server → client: delivered via a WS_SEND event through the bus
```

WS workers carry **no** business logic — they manage connections and channels and
forward messages. Handlers run in HTTP workers via `app.websocket()` routes.

---

## Configuration

Key environment variables:

| Var | Meaning |
|---|---|
| `TB_ENV` | `development` \| `production` \| `tauri` |
| `TB_COOKIE_SECRET` | session signing secret (set 64+ chars in production) |
| `TB_HTTP_PORT` / `TB_WS_PORT` | worker ports |
| `TB_TRAY_URL` | tray/collective owner base URL (set automatically by the owner) |
| `LIVE_DASHBOARD_KEY` | `/live` auth key (empty disables the route) |

Notable config sections (`config.py`): `zmq` (endpoints, heartbeat/takeover timing),
`http_worker`, `ws_worker`, `nginx`, `manager` (live-UI port, replica count,
restart policy), `session`, `auth`, `toolbox` (`api_prefix`).

---

## Production Checklist

- [ ] `TB_ENV=production`
- [ ] `TB_COOKIE_SECRET` set (64+ random chars)
- [ ] `LIVE_DASHBOARD_KEY` set
- [ ] SSL configured in nginx
- [ ] `http_worker.workers` tuned to CPU cores
- [ ] `ws_worker.max_connections` tuned to RAM
- [ ] `manager.live_ui_replicas >= 2` for live-UI failover
- [ ] rate limiting enabled in nginx
- [ ] log rotation set up

---

## Performance (indicative)

- HTTP worker: ~5,000 req/s each (Waitress)
- WS worker: ~10,000 concurrent connections per instance
- ZeroMQ bus: ~100,000 msg/s throughput

---

Part of ToolBoxV2.
