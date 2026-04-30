# ToolBoxV2 → FastTB → Minu — Unified Stack Documentation

## 1. System overview

ToolBoxV2 is a modular Python platform. Its web layer consists of three tiers that build on each other — each tier reuses the one below it, nothing is reimplemented.

```
┌────────────────────────────────────────────────────────────┐
│  Minu UI Framework                                         │
│  Reactive Python views → JSON → MinuRenderer.js            │
│  @minu.view("/dashboard") generates GET + POST + WS        │
├────────────────────────────────────────────────────────────┤
│  FastTB                                                     │
│  @app.get/post/put/delete/websocket decorators             │
│  Parameter injection · mount_static · Result.sse/stream    │
├────────────────────────────────────────────────────────────┤
│  HTTPWorker (server_worker.py)                              │
│  WSGI · Session · Auth · CORS · /api/* · /health · ZMQ    │
├────────────────────────────────────────────────────────────┤
│  Infrastructure                                             │
│  ws_worker (WS on port 5001) · ZMQ broker · nginx          │
└────────────────────────────────────────────────────────────┘
```

FastTB does not replace HTTPWorker — it extends it. In standalone mode (`as_wsgi_app()`), FastTB creates an HTTPWorker internally and checks its routes first, then falls through to the HTTPWorker's built-in endpoints. In production mode (3-line integration), FastTB routes are checked after HTTPWorker's built-ins, before the 404 fallback.

MinuBridge does not replace FastTB — it uses FastTB's decorator system to generate routes from MinuView classes, and uses the existing MinuRenderer.js from the TBJS bundle for client-side rendering.

---

## 2. Request lifecycle

### Standalone mode (uvicorn / waitress)

```
Browser request
  │
  ▼
patched_wsgi (FastTBHandler.as_wsgi_app)
  │
  ├── FastTB route match? ──yes──▶ parse_request → DI → handler → response
  │                                (body read once, session, CORS, cookies)
  │
  └── no match ──▶ HTTPWorker.wsgi_app
                    ├── OPTIONS → CORS preflight
                    ├── /auth/* → AuthHandler
                    ├── /health → health check
                    ├── /api/client-logs → client log sink
                    ├── /api/* → ToolBoxHandler (module calls)
                    └── else → 404
```

### Production mode (cli_worker_manager)

```
nginx → Waitress → HTTPWorker.wsgi_app
  │
  ├── (built-in endpoints as above)
  │
  ├── /api/* → ToolBoxHandler
  │
  ├── FastTB route match? ──yes──▶ FastTBHandler.handle_request
  │   (3-line integration)
  │
  └── else → 404
```

### Minu view lifecycle (HTML mode)

```
1. Browser navigates to /counter (via TBJS Router)
2. Router fetches /counter as AJAX
3. FastTB GET handler:
   - Creates MinuView instance
   - Registers in MinuSession
   - Calls view.to_dict() → JSON
   - Returns HTML fragment (not full page)
4. Router injects fragment into #MainContent
5. Inline <script> runs immediately (TBJS already loaded):
   - new MinuRenderer()._renderView(viewData) → DOM
   - Patches renderer._send to use HTTP POST
6. User clicks button:
   - MinuRenderer._bindEvent fires
   - _triggerEvent → patched _send
   - POST /counter/event {viewId, handler, data}
7. Server:
   - Finds view by viewId (searches all sessions)
   - Calls view.increment(event_data)
   - Returns {ok, patches, view}
8. Client:
   - _applyPatches for state changes
   - _renderView for full re-render
```

---

## 3. FastTB

### Route decorators

```python
from toolboxv2.utils.workers.fast_tb import FastTB

app = FastTB(title="MyApp")

@app.get("/users/{user_id}")
async def get_user(user_id: int, verbose: bool = False):
    return {"id": user_id, "verbose": verbose}

@app.post("/items")
async def create_item(name: str, price: float = 0.0):
    return (201, {"name": name, "price": price})

@app.route("/echo", methods=["GET", "POST"])
async def echo(request: ParsedRequest):
    return {"method": request.method, "body": request.json_data}
```

### Parameter injection

The handler's signature is inspected. Parameters are resolved in this order:

| Priority | Source | Match rule |
|----------|--------|-----------|
| 1 | ParsedRequest | Named `request` or type-hinted `ParsedRequest` |
| 2 | SessionData | Named `session` or type-hinted `SessionData` |
| 3 | Path params | `{param}` in URL template |
| 4 | Query params | `?key=value` |
| 5 | JSON body | Fields in `request.json_data` |
| 6 | Form data | Fields in `request.form_data` |
| 7 | Default | Python default value |

`*args` and `**kwargs` are skipped. Types `int`, `float`, `bool` are auto-coerced from strings.

### Response formatting

Return whatever is natural — auto-converted:

| Return type | HTTP response |
|-------------|---------------|
| `dict` / `list` | JSON 200 |
| `None` | `{"status": "ok"}` 200 |
| `str` starting with `<` | HTML 200 |
| `str` (other) | `{"result": "..."}` 200 |
| `bytes` | octet-stream 200 |
| `(status, dict)` | JSON with custom status |
| `(status, headers, body)` | Full passthrough |
| `Result.sse(...)` | SSE stream |
| `Result.stream(...)` | Generic stream |

### Static file mounting

```python
app.mount_static("/dist", "/path/to/dist")
app.mount_static("/web", "/path/to/web")
```

Path traversal blocked via `os.path.normpath` + prefix check. Hashed filenames (`main-5d3f7ed2.js`) get `Cache-Control: immutable`.

### WebSocket handlers

```python
@app.websocket("/ws/echo")
class EchoHandler:
    async def on_connect(self, conn_id, session):
        return {"type": "connected"}

    async def on_message(self, payload, conn_id, session, request=None):
        return {"type": "echo", "message": payload.get("message")}

    async def on_disconnect(self, conn_id, session):
        pass
```

Handlers are exported via `app.get_websocket_handlers()` into `app.websocket_handlers`. The existing `ws_worker` + ZMQ routing picks them up in production. In standalone mode (no ws_worker), WebSocket is not available — use HTTP POST event dispatch instead.

### SSE (Server-Sent Events)

```python
@app.get("/sse/clock")
async def clock(request: ParsedRequest):
    async def ticks():
        for i in range(30):
            yield {"event": "tick", "data": {"time": time.strftime("%H:%M:%S")}}
            await asyncio.sleep(1)

    return Result.sse(stream_generator=ticks())
```

Client:
```javascript
const es = new EventSource('/sse/clock');
es.addEventListener('tick', e => console.log(JSON.parse(e.data)));
es.addEventListener('stream_end', () => es.close());
```

`Result.sse()` wraps via `SSEGenerator.create_sse_stream()` — handles async/sync generators, lifecycle events (`stream_start`/`stream_end`), errors, cleanup functions.

---

## 4. MinuBridge

### View decorator

```python
from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import MinuView, State, Column, Text, Button

app = FastTB(title="MyApp")
minu = MinuBridge(app)

@minu.view("/counter")
class CounterView(MinuView):
    count = State(0)

    def render(self):
        return Column(
            Text(str(self.count.value)),
            Button("+", on_click="increment"),
        )

    async def increment(self, event):
        self.count.value += 1
```

This registers:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/counter` | Render view (HTML fragment or JSON) |
| POST | `/counter/event` | Event dispatch |
| WS | `/ws/counter` | Live updates (production only) |

### Auth-gated views

```python
@minu.view("/admin", require_auth=True)
class AdminPanel(MinuView):
    def render(self):
        return Text(f"Welcome {self.user.name}, level {self.user.level}")
```

Anonymous requests get 401. Authenticated users get `self.user` populated from SessionData.

### HTML output

The GET handler returns an **HTML fragment** (not a full page). This is critical — TBJS Router fetches it via AJAX and injects it into `#MainContent`. The fragment contains:

1. `<div id="minu-root"></div>` — mount target
2. `<script>` block that:
   - Stores the serialized view as `window.__MINU_BRIDGE__`
   - Creates `new TB.ui.MinuRenderer()` (already loaded by TBJS)
   - Calls `_renderView()` for initial render
   - Patches `_send()` to dispatch events via HTTP POST

No `<!DOCTYPE>`, no `<html>`, no `web_context()` — TBJS is already running in the parent page.

### Event dispatch

In standalone mode (no ws_worker), events are dispatched via HTTP POST:

```
MinuRenderer._bindEvent        (unchanged — from MinuRenderer.js)
  → _triggerEvent              (unchanged)
    → _send                    (PATCHED to use fetch())
      → POST /counter/event
        {viewId, handler, data}
      ← {ok, patches, view}
        → _applyPatches()      (unchanged — from MinuRenderer.js)
        → _renderView()        (unchanged — full re-render)
```

Only `_send` is replaced (one method, transport layer). All rendering, event binding, patching, and component logic comes from the existing `MinuRenderer.js` — single source of truth.

In production mode (ws_worker running), `MinuRenderer.mount()` opens a WebSocket to `/ws/Minu/ui` and events flow over WS with live push. The HTTP POST path is the fallback.

### Docs-Ergänzung für `minu_bridge_docs.md`

Füge diese Section nach dem **"Standalone mode with WebSocket"** Block ein, vor **"User system"**:

---

---

## Styling: Automatic Paper CSS Injection

FastTB automatically injects the Paper design system (neo-brutalist style from
`nbpaper_style.md`) into any HTML response that doesn't already have TBJS loaded.

### How it works

When a handler returns raw HTML (string starting with `<`), FastTBHandler checks:

1. Does the HTML contain TBJS markers (`tbjs`, `TB.init`, `web_context`)? → **skip** (TBJS has its own styles)
2. Does the HTML already have substantial `<style>` content (>200 chars)? → **skip** (user styled it)
3. Neither → **inject** IBM Plex fonts + Paper CSS before `</head>`

This means unstyled HTML handlers look identical to the `/docs` page and
welcome screen — without any manual CSS work.

### Opting out

```python
app = FastTB(title="MyApp")
app.inject_style = False   # No auto-injection on any route
```

### Mixing styled and unstyled routes

Routes that use `get_app().web_context()` (Minu views, TBJS-powered pages)
are never affected — the injection detects TBJS and skips automatically.

```python
# Auto-styled (Paper CSS injected)
@app.get("/about")
async def about():
    return """<html><head><title>About</title></head>
    <body><div class="container">
        <h1>About Us</h1>
        <p>This gets Paper styling automatically.</p>
    </div></body></html>"""

# TBJS-styled (no injection — web_context detected)
@app.get("/dashboard")
async def dashboard():
    return get_app().web_context() + """
    <div id="app">TBJS handles styling here</div>
    """ + app.hot_reload_script()

# Self-styled (no injection — <style> block detected)
@app.get("/custom")
async def custom():
    return """<html><head><style>
        body { background: black; color: lime; font-family: monospace; }
    </style></head><body><h1>Custom</h1></body></html>"""
```

### Available CSS classes (Paper)

The injected CSS provides these classes matching `nbpaper_style.md`:

| Class | Usage |
|-------|-------|
| `.container` | Centered content column (max 740px) |
| `.card` | Bordered surface with offset shadow, hover lift |
| `.btn` | Neo-brutalist button (uppercase mono, offset shadow) |
| `.btn-primary` | Accent-colored button |
| `.badge` | Uppercase mono label |
| `.grid` | 2-column grid (stacks on mobile) |
| `.subtitle` | Uppercase mono secondary text |
| `.footer` | Top-ruled footer area |
| `code` | Inline code with ink border |
| `pre` | Code block with offset shadow |

All styles use CSS custom properties (`--ink`, `--paper-bg`, `--paper-surface`,
`--font-display`, `--font-body`) and support dark mode via
`prefers-color-scheme: dark` automatically.

---

## User system


The bridge wires `SessionData` → `request_data` → `MinuUser.from_request()` → `self.user`:

```python
@minu.view("/profile")
class Profile(MinuView):
    def render(self):
        if self.user.is_authenticated:
            return Column(
                Text(f"Name: {self.user.name}"),
                Text(f"UID: {self.user.uid}"),
            )
        return Text("Not logged in")

    async def on_mount(self):
        user = await self.ensure_user()
        if user.is_authenticated:
            data = await user.get_mod_data("Profile")
```

### Shared sections (multiplayer)

```python
@minu.view("/game")
class GameView(MinuView):
    players = State([])

    async def on_mount(self):
        self.game = await self.join_shared("lobby")
        if self.game:
            self.players.value = self.game.get("players", [])
            self.game.on_change("players", self._sync)

    def _sync(self, change):
        self.players.value = self.game.get("players", [])

    async def join(self, event):
        await self.game.append("players", {
            "id": self.user.uid, "name": self.user.name
        }, author_id=self.user.uid)
```

---

## 5. Running

### Standalone (development)

```python
# myapp.py
app = FastTB(title="MyApp")
minu = MinuBridge(app)
# ... register views ...

handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app()
```

```bash
pip install uvicorn a2wsgi
uvicorn myapp:wsgi_app --port 8000
```

`as_wsgi_app()` creates an HTTPWorker internally. All built-in endpoints (auth, session, CORS, health, client-logs, ToolBox API) are available. FastTB routes are checked first.

### Production (HTTPWorker integration)

Three lines in `server_worker.py`:

```python
# 1. In HTTPWorker.__init__:
self._fast_tb_handler = None

# 2. In HTTPWorker.run():
if fast_tb_app:
    from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
    self._fast_tb_handler = FastTBHandler(fast_tb_app, self._session_manager)
    ws_handlers = fast_tb_app.get_websocket_handlers()
    if ws_handlers:
        self._app.websocket_handlers.update(ws_handlers)

# 3. In HTTPWorker.wsgi_app(), after ToolBox API check, before 404:
elif self._fast_tb_handler and self._fast_tb_handler.has_route(request.path, request.method):
    status, headers, body = self._run_async(
        self._fast_tb_handler.handle_request(request)
    )
```

### WebSocket (production with ws_worker)

The `ws_worker` runs on a separate port (default 5001). It forwards all messages via ZMQ to the HTTP worker. FastTB WebSocket handlers register in `app.websocket_handlers` and are dispatched by `WebSocketMessageHandler`. The `ws_bridge` provides the return channel (`app.ws_send`, `app.ws_broadcast`).

---

## 6. File layout

```
toolboxv2/
├── utils/workers/
│   ├── fast_tb.py              FastTB class, decorators, mount_static, resolve_static
│   ├── fast_tb_handler.py      DI, dispatch, static serving, as_wsgi_app
│   ├── server_worker.py        HTTPWorker (3 lines added for FastTB)
│   ├── ws_worker.py            WebSocket server (ZMQ forwarding)
│   ├── ws_bridge.py            WS return channel (app.ws_send/broadcast)
│   ├── session.py              SessionData, SessionManager
│   └── config.py               WorkerConfig, load_config
│
├── mods/Minu/
│   ├── minu_bridge.py          @minu.view() decorator, HTTP event dispatch
│   ├── __init__.py             Module entry, register_view, WS handlers, SSE
│   ├── core.py                 MinuView, ReactiveState, Components, MinuSession
│   ├── user.py                 AnonymousUser, AuthenticatedUserWrapper, MinuUser
│   └── shared.py               SharedSection, SharedManager, SharedMixin
│
├── tbjs/src/
│   ├── index.js                TB framework core, TB.init()
│   └── ui/
│       ├── index.js            UI exports (includes MinuRenderer)
│       └── components/Minu/
│           └── MinuRenderer.js The one renderer (WS + DOM + patching)
│
└── index.js                    App entry (boots TB, sets window.TB)
```

---

## 7. Key design decisions

**No second renderer.** MinuRenderer.js is the single source of truth for all component rendering. The bridge patches only `_send()` (transport layer) — all rendering, event binding, and patching logic is reused unchanged.

**Fragment, not full page.** The GET handler returns an HTML fragment that TBJS Router injects into `#MainContent`. No `<!DOCTYPE>`, no `web_context()`. This prevents double-rendering when the Router re-fetches the URL.

**Body read once.** In standalone mode, `patched_wsgi` checks FastTB routes by path+method only (no body read). If there's a match, `parse_request` reads the body exactly once. If no match, the body is untouched and HTTPWorker reads it.

**View lookup by viewId.** POST event dispatch finds views by `viewId` across all sessions (not just the cookie-session). This handles standalone mode where cookie persistence between GET and POST is not guaranteed.

**HTTPWorker reuse.** `as_wsgi_app()` creates an HTTPWorker internally. All built-in endpoints (auth, session, CORS, health, client-logs) come for free. FastTB adds routes on top, never beside or below.

---

## 8. Tests

87 unit tests covering:

| Area | Tests | Coverage |
|------|-------|----------|
| Path regex | 8 | Static, params, multi-params, unicode, traversal |
| Route registration | 9 | All HTTP methods, multi-method, names |
| Route resolution | 8 | Match, no-match, case-insensitive, index rebuild |
| WebSocket | 6 | Class registration, export format, partial handlers |
| Parameter injection | 12 | All 6 sources, combined, missing, defaults |
| Type coercion | 8 | int, float, bool, str, invalid fallback |
| Response formatting | 10 | dict, list, None, HTML, bytes, tuples |
| Execution | 4 | Async, sync, 404, exception→500 |
| Static mount | 11 | Mount, resolve, traversal, content-type, cache |
| `*args`/`**kwargs` | 2 | Skipped by DI |
| Edge cases | 5 | Duplicates, special chars, multi-value query |

```bash
python -m unittest test_fast_tb -v
```
