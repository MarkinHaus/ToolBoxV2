# FastTB + Minu Bridge

> Serve MinuView classes as FastTB endpoints — one decorator, three routes.

---

## Architecture

```
    @minu.view("/dashboard")           ┌──── GET /dashboard ──────── HTML/JSON
    class Dashboard(MinuView):    ──▶  ├──── POST /dashboard/event ── Event dispatch
        ...                            └──── WS  /ws/dashboard ────── Live updates
```

The bridge generates all three endpoints from one class. No manual route
registration, no boilerplate. Auth is opt-in via `require_auth=True`.

---

## Quick start

```python
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import MinuView, State, Column, Heading, Text, Button, Input, Row

app = FastTB(title="MyApp")
minu = MinuBridge(app)

@minu.view("/dashboard")
class Dashboard(MinuView):
    counter = State(0)

    def render(self):
        return Column(
            Heading("Dashboard"),
            Text(f"Count: {self.counter.value}"),
            Button("Increment", on_click="increment"),
        )

    async def increment(self, event):
        self.counter.value += 1
```

Run with:
```bash
uvicorn myapp:app --port 8000         # Standalone ASGI
# or
worker.run(fast_tb_app=app)           # Inside HTTPWorker
```

Browse `http://localhost:8000/dashboard` → rendered HTML with live state.

---

## Auth-gated views

```python
@minu.view("/admin", require_auth=True)
class AdminPanel(MinuView):
    def render(self):
        return Column(
            Heading(f"Welcome, {self.user.name}"),
            Text(f"Level: {self.user.level}"),
        )
```

Anonymous users get `401`. Authenticated users get `self.user` populated
from `SessionData` automatically.

---

## User system integration

The bridge wires `SessionData` → `MinuView.request_data` → `self.user`:

```python
@minu.view("/profile")
class Profile(MinuView):
    def render(self):
        if self.user.is_authenticated:
            return Column(
                Text(f"Name: {self.user.name}"),
                Text(f"Email: {self.user.email}"),
                Text(f"UID: {self.user.uid}"),
            )
        return Column(
            Text("Not logged in"),
            Button("Login", on_click="login"),
        )

    async def on_mount(self):
        user = await self.ensure_user()
        if user.is_authenticated:
            data = await user.get_mod_data("Profile")
            # ... use persisted data
```

---

## Event handling

### Via HTTP (POST)

```javascript
// Client-side
fetch('/dashboard/event', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        viewId: 'view-abc123',
        handler: 'increment',
        data: {}
    })
})
```

Response:
```json
{
    "ok": true,
    "patches": [
        {"type": "state_update", "viewId": "view-abc123", "path": "view-abc123.counter", "value": 1}
    ],
    "result": null
}
```

### Via WebSocket (real-time)

```javascript
ws.send(JSON.stringify({
    type: 'event',
    sessionId: 'session-xyz',
    viewId: 'view-abc123',
    handler: 'increment',
    data: {}
}))
```

---

## Generated endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `{path}` | Render view — HTML for browsers, JSON for API clients |
| GET | `{path}?format=json` | Force JSON output |
| GET | `{path}?format=html` | Force HTML output |
| POST | `{path}/event` | Dispatch event to handler method |
| WS | `/ws{path}` | WebSocket for live state sync |

---

## Shared sections (multiplayer)

Works out of the box — SharedManager is accessed via `self.shared_manager`:

```python
@minu.view("/game")
class GameView(MinuView):
    players = State([])

    async def on_mount(self):
        self.game = await self.join_shared("game_lobby")
        if self.game:
            self.players.value = self.game.get("players", [])
            self.game.on_change("players", self._on_players)

    def _on_players(self, change):
        self.players.value = self.game.get("players", [])

    def render(self):
        return Column(
            Heading("Game Lobby"),
            *[Text(p["name"]) for p in self.players.value],
            Button("Join", on_click="join_game"),
        )

    async def join_game(self, event):
        await self.game.append("players", {
            "id": self.user.uid,
            "name": self.user.name,
        }, author_id=self.user.uid)
```

---

## File placement

```
toolboxv2/mods/Minu/
├── minu_bridge.py       # ← new file (this bridge)
├── __init__.py
├── core.py
├── user.py
├── shared.py
└── ...
```

---

## API reference

### `MinuBridge(fast_tb_app, app_instance=None)`

| Method | Signature | Description |
|--------|-----------|-------------|
| `view` | `(path, require_auth=False)` | Decorator — registers MinuView as FastTB routes |
| `list_views` | `() -> List[dict]` | List all registered Minu views |

### Injected into each view

| What | How | Source |
|------|-----|--------|
| `self.user` | Auto-populated from session | `SessionData` → `MinuUser` |
| `self.request_data` | Built from `ParsedRequest` | Bridge constructs minimal wrapper |
| `self._app` | ToolBoxV2 App | Passed from `MinuBridge(app_instance=...)` |


# FastTB + MinuBridge — Docs

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │           Your App Code              │
                     │  @app.get() @app.websocket()         │
                     │  @minu.view() Result.sse()           │
                     └──────────────┬────────────────────────┘
                                    │
                     ┌──────────────▼────────────────────────┐
                     │  FastTB + FastTBHandler                │
                     │  Route matching · DI · Static mount    │
                     └──────────────┬────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────────┐
              ▼                     ▼                         ▼
   ┌───────────────────┐  ┌─────────────────┐  ┌────────────────────┐
   │  HTTPWorker        │  │  Standalone      │  │  Uvicorn (ASGI)    │
   │  (3 lines added)   │  │  as_wsgi_app()   │  │  via a2wsgi        │
   │                    │  │  reuses           │  │                    │
   │  Session, Auth,    │  │  HTTPWorker       │  │  Same infra        │
   │  CORS, client-logs │  │  infrastructure   │  │  via as_wsgi_app   │
   └───────────────────┘  └─────────────────┘  └────────────────────┘
```

**Key principle:** `as_wsgi_app()` creates an HTTPWorker internally and delegates
to its `wsgi_app`. All built-in endpoints (auth, session, CORS, client-logs,
health, metrics, ToolBox API) are available automatically. FastTB routes are
the fallback before 404 — not a replacement.

---

## Static file serving

FastTB can serve local directories with path traversal protection:

```python
app = FastTB(title="MyApp")
app.mount_static("/dist", "/path/to/dist")
app.mount_static("/web", "/path/to/web")
```

Hashed filenames (`main-5d3f7ed2.js`) get `Cache-Control: immutable`.
Path traversal attempts (`../../../etc/passwd`) are blocked via `os.path.normpath` + prefix check.

MinuBridge auto-mounts the dist directory from `$DIST_DIR` or `app.data_dir/../dist`.

---

## WebSocket handlers

### FastTB native

```python
@app.websocket("/ws/echo")
class EchoHandler:
    async def on_connect(self, conn_id, session):
        return {"type": "connected", "conn_id": conn_id}

    async def on_message(self, payload, conn_id, session, request=None):
        return {"type": "echo", "message": payload.get("message", "")}

    async def on_disconnect(self, conn_id, session):
        pass
```

WS handlers are exported in ToolBoxV2 format via `get_websocket_handlers()`.
The existing `WebSocketMessageHandler` + ZMQ routing picks them up — no custom WS server needed.

### MinuBridge WS

MinuBridge views use the standard `/ws/Minu/ui` WebSocket path.
The HTML output calls `MinuRenderer.mount()` which opens the WS connection
and subscribes to the view. State changes from button clicks are sent over WS
and patched live in the browser.

```python
@minu.view("/counter")
class Counter(MinuView):
    count = State(0)

    def render(self):
        return Column(
            Text(str(self.count.value)),
            Button("+", on_click="increment"),  # triggers via WS
        )

    async def increment(self, event):
        self.count.value += 1  # patched live in browser via WS
```

---

## SSE (Server-Sent Events)

Use `Result.sse()` or `Result.stream()` from any FastTB endpoint:

```python
@app.get("/sse/clock")
async def sse_clock(request: ParsedRequest):
    async def clock_gen():
        for i in range(30):
            yield {"event": "tick", "data": {"time": time.strftime("%H:%M:%S")}}
            await asyncio.sleep(1)

    return Result.sse(stream_generator=clock_gen())
```

Client-side:

```javascript
const es = new EventSource('/sse/clock');
es.addEventListener('tick', e => console.log(JSON.parse(e.data)));
es.addEventListener('stream_end', () => es.close());
```

`Result.sse()` wraps via `SSEGenerator.create_sse_stream()` which handles:
- Async/sync generators, iterables, single items
- `stream_start` / `stream_end` lifecycle events
- Error formatting
- Optional cleanup function

For non-SSE streaming (binary, file download):

```python
return Result.stream(stream_generator=my_gen(), content_type="application/octet-stream")
return Result.file_stream(generator=chunks(), filename="export.csv")
return Result.file_path("/path/to/report.pdf")
```

---

## Standalone mode vs HTTPWorker

### Standalone (uvicorn)

```python
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler

app = FastTB(title="MyApp")
# ... register routes ...

handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app()  # reuses HTTPWorker internally

# uvicorn myapp:wsgi_app --port 8000
```

`as_wsgi_app()` creates an HTTPWorker with full infrastructure:
session management, auth endpoints, CORS, client-logs, ToolBox API.
Your FastTB routes are the fallback before 404.

### Inside HTTPWorker (production)

Three lines in `server_worker.py`:

```python
# 1. __init__
self._fast_tb_handler = None

# 2. run()
if fast_tb_app:
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    self._fast_tb_handler = FastTBHandler(fast_tb_app, self._session_manager)
    ws_handlers = fast_tb_app.get_websocket_handlers()
    if ws_handlers:
        self._app.websocket_handlers.update(ws_handlers)

# 3. wsgi_app() — after toolbox_handler check, before 404
elif self._fast_tb_handler and self._fast_tb_handler.has_route(request.path, request.method):
    status, headers, body = self._run_async(
        self._fast_tb_handler.handle_request(request)
    )
```

---

## MinuBridge — view decorator

```python
app = FastTB(title="MyApp")
minu = MinuBridge(app)

@minu.view("/dashboard")
class Dashboard(MinuView):
    ...
```

Generates:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/dashboard` | Render (HTML for browsers, JSON for API) |
| POST | `/dashboard/event` | Event dispatch (HTTP fallback) |
| WS | `/ws/dashboard` | Live state updates |

### Auth gate

```python
@minu.view("/admin", require_auth=True)
class AdminPanel(MinuView):
    ...
```

### HTML output

The GET handler returns HTML that:
1. Loads TBJS via `get_app().web_context()` (Webpack chunks)
2. Boots TB with full config (theme, nav, etc.)
3. Calls `MinuRenderer.mount('#minu-root', viewName)` which opens WS
4. Falls back to static `_renderView()` if WS fails

---

## Parameter injection

FastTB inspects handler signatures and injects automatically:

| Priority | Source | Match rule |
|----------|--------|-----------|
| 1 | `ParsedRequest` | Param named `request` or type-hinted |
| 2 | `SessionData` | Param named `session` or type-hinted |
| 3 | Path params | `{param}` in URL |
| 4 | Query params | `?key=value` |
| 5 | JSON body | `request.json_data` fields |
| 6 | Form data | `request.form_data` fields |
| 7 | Default | Python default value |

`*args` and `**kwargs` are skipped. Types are coerced (`int`, `float`, `bool`).

---

## File layout

```
toolboxv2/utils/workers/
├── fast_tb.py           # FastTB class, decorators, mount_static
├── fast_tb_handler.py   # Dispatch, DI, static serving, as_wsgi_app
├── server_worker.py     # HTTPWorker — 3 lines added for integration
└── session.py           # SessionData

toolboxv2/mods/Minu/
├── minu_bridge.py       # MinuBridge: @minu.view() decorator
├── core.py              # MinuView, components, MinuSession
├── user.py              # AnonymousUser, AuthenticatedUserWrapper
└── shared.py            # SharedSection, SharedManager

toolboxv2/tbjs/src/ui/
├── index.js             # Exports MinuRenderer (already done)
└── components/Minu/
    └── MinuRenderer.js  # The one and only renderer
```
