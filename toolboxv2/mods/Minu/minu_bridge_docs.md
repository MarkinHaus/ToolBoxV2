# FastTB + Minu Bridge

> Serve MinuView classes as FastTB endpoints — one decorator, three routes, live WebSocket updates.

---

## Architecture

```
                         @minu.view("/dashboard")
                         class Dashboard(MinuView):
                             ...
                                │
                ┌───────────────┼───────────────────┐
                ▼               ▼                   ▼
      GET /dashboard    POST /dashboard/event   WS /ws/openDashboard
      HTML or JSON      HTTP event fallback     Live state push
                                                (auto-flush on every
                                                 state change)
```

The bridge generates all three endpoints from one class. The client connects
via WebSocket for real-time patches and falls back to HTTP POST automatically
if WS is unavailable.

---

## Quick start

```python
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import MinuView, State, Column, Heading, Text, Button

app = FastTB(title="MyApp")
minu = MinuBridge(app)

@minu.view("/counter")
class Counter(MinuView):
    count = State(0)

    def render(self):
        return Column(
            Heading("Counter"),
            Text(f"Count: {self.count.value}"),
            Button("+", on_click="increment"),
        )

    async def increment(self, event):
        self.count.value += 1  # pushed live via WS

# Start with WS infrastructure
handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app(enable_ws=True)

# waitress or uvicorn
from waitress import serve
serve(wsgi_app, host="0.0.0.0", port=8000)
```

Browse `http://localhost:8000/counter` — state changes are pushed live over WebSocket.

---

## Access control

MinuBridge provides two levels of access control via bridge-level defaults
and per-view overrides.

### Bridge-level defaults

```python
minu = MinuBridge(app)
minu.public = True       # Views are public by default (no auth required)
minu.ws_public = True    # WebSocket connections allowed without auth
```

For a private app where everything requires login:

```python
minu.public = False      # All views require auth by default
minu.ws_public = False   # WS requires auth by default
```

### Per-view overrides

```python
@minu.view("/counter")                                    # uses bridge defaults
class Counter(MinuView): ...

@minu.view("/admin", require_auth=True)                   # HTTP requires auth, WS uses bridge default
class Admin(MinuView): ...

@minu.view("/dashboard", ws_public=False)                 # HTTP public, WS requires auth
class Dashboard(MinuView): ...

@minu.view("/monitor", require_auth=True, ws_public=True) # HTTP requires auth, WS is public
class Monitor(MinuView): ...
```

### The `open` path convention

The ToolBoxV2 `AccessController` grants anonymous access to handlers whose
function name starts with `open`. MinuBridge uses this convention for the
WebSocket route:

- `ws_public=True` → WS path becomes `/ws/openDashboard` → `AccessController` sees function `openDashboard` → allowed
- `ws_public=False` → WS path stays `/ws/dashboard` → `AccessController` requires auth

This is handled automatically by MinuBridge. When `ws_public=True` and
the path doesn't already contain `open`, the WS path is rewritten:

```
/dashboard  →  /ws/openDashboard   (public)
/admin      →  /ws/admin           (auth required)
```

If the user is not authenticated and WS is private, the client receives
an `ACCESS_DENIED` error, shows a warning toast ("Live updates require login"),
and falls back to HTTP POST silently.

---

## Event transport: WS + HTTP fallback

### How it works

1. Page loads → client renders view from initial JSON
2. Client opens WebSocket to `/ws/open{Path}` (port from config, default 8100)
3. Client sends `{type: "init", sessionId: "..."}` → server binds `_send_callback`
4. Button clicks → sent as WS events → server runs handler → patches pushed live
5. If WS fails → client falls back to HTTP POST → patches returned in response

### Live push during async handlers

State changes inside `await` loops are pushed immediately:

```python
async def auto_increment(self, event):
    for i in range(10):
        self.count.value += 1   # each change pushed live via WS
        await asyncio.sleep(0.2)
```

This works because `ReactiveState.value` setter → `_on_state_change` →
`session._schedule_flush()` → `create_task(force_flush())`. Each value change
creates a flush task on the event loop, so patches are sent between `await` points.

### WS message format

```javascript
// Client → Server
ws.send(JSON.stringify({
    type: "event",
    viewId: "view-abc123",
    handler: "increment",
    payload: { type: "click", value: "", timestamp: 1234567890 }
}))

// Server → Client (state patches)
{ type: "patches", sessionId: "anon-...", patches: [
    { type: "state_update", viewId: "view-abc123", path: "view-abc123.count", value: 1 }
]}

// Server → Client (full re-render)
{ type: "render", sessionId: "anon-...", view: { viewId: "...", component: {...}, state: {...} } }

// Server → Client (event result with patches)
{ type: "event_result", patches: [...], view: {...} }
```

### HTTP fallback format

```javascript
// POST /counter/event
fetch(eventUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ viewId: "view-abc123", handler: "increment", data: {} })
})

// Response (success)
{ ok: true, patches: [...], view: {...}, result: null }

// Response (error)
{ ok: false, error: "AttributeError: ...", handler: "reset", hint: "Did you mean 'count'?", source: "demo.py:42 in reset", traceback: "..." }
```

---

## Error reporting

Handler errors produce detailed diagnostics in both the terminal and the browser.

### Terminal output

```
[MinuBridge] Handler 'reset' on LiveCounter failed:
  Error: AttributeError: 'LiveCounter' has no attribute 'counte'
  Location: demo_minu_ws_sse.py:87 in reset
  Hint: Did you mean 'count'? (typo: 'counte')
  Traceback:
    ...
    File "demo_minu_ws_sse.py", line 87, in reset
      self.counte.value = 0
    AttributeError: 'LiveCounter' has no attribute 'counte'
```

### Browser toast

```
⚠ reset()
AttributeError: 'LiveCounter' has no attribute 'counte'
💡 Did you mean 'count'?
```

### Browser console (styled)

```
[Minu] Handler Error  reset
AttributeError: 'LiveCounter' has no attribute 'counte'
💡 Did you mean 'count'? (typo: 'counte')
📍 demo_minu_ws_sse.py:87 in reset

Traceback (most recent call last):
  ...
```

### Supported error hints

| Error type | Hint |
|-----------|------|
| `AttributeError: has no attribute 'xyz'` | Suggests closest matching state attribute via `difflib` |
| `TypeError: unsupported operand` | Warns about type mismatch (e.g. `+= "1"` instead of `+= 1`) |
| `TypeError` (generic) | Points to handler name for argument check |

Error reporting is identical for both WS and HTTP transport — same error
object, same toast, same console output.

---

## Generated endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `{path}` | Render view — HTML for browsers, JSON for API clients |
| GET | `{path}?format=json` | Force JSON output |
| GET | `{path}?format=html` | Force HTML output |
| POST | `{path}/event` | Event dispatch (HTTP fallback) |
| WS | `/ws/open{Path}` | Live state sync (public, if `ws_public=True`) |
| WS | `/ws{path}` | Live state sync (auth required, if `ws_public=False`) |

---

## Standalone mode with WebSocket

```python
handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app(enable_ws=True)
```

`enable_ws=True` starts three background threads automatically:

1. **ZMQ Broker** — PUB/SUB proxy for inter-worker communication
2. **WS Worker** — WebSocket server on `config.ws_worker.port` (default 8100)
3. **Event Bridge** — connects HTTP worker to WS worker via ZMQ, installs `app.ws_send()`

The client JS auto-connects to the WS port. If `enable_ws` is not set,
it defaults to `True` when any `@minu.view()` or `@app.websocket()` routes exist.

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

## User system

The bridge wires `SessionData` → `MinuView.request_data` → `self.user`:

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

---

## SSE (Server-Sent Events)

Use `Result.sse()` from any FastTB endpoint, or use the `@app.sse()` decorator:

```python
# Via Result.sse()
@app.get("/sse/clock")
async def sse_clock(request: ParsedRequest):
    async def clock_gen():
        for i in range(30):
            yield {"event": "tick", "data": {"time": time.strftime("%H:%M:%S")}}
            await asyncio.sleep(1)
    return Result.sse(stream_generator=clock_gen())

# Via @app.sse() decorator (sets headers automatically)
@app.sse("/sse/events")
async def stream_events(request: ParsedRequest):
    for i in range(10):
        yield {"event": "tick", "data": {"count": i}}
        await asyncio.sleep(1)
```

---

## Shared sections (multiplayer)

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

    async def join_game(self, event):
        await self.game.append("players", {
            "id": self.user.uid, "name": self.user.name,
        }, author_id=self.user.uid)
```

# Automatisch an in development (TB_ENV != "production")
app = FastTB(title="MyApp")
# app.hot_reload ist True

# Manuell ausschalten
app.hot_reload = False

# Extra Verzeichnisse watchen
app.watch("./my_templates", "./my_assets")

# MinuBridge setzt es automatisch für das Minu-Modul-Verzeichnis
minu = MinuBridge(app)
# → watchdog beobachtet toolboxv2/mods/Minu/ + Hauptskript-Verzeichnis

---

## API reference

### `MinuBridge(fast_tb_app, app_instance=None)`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `public` | `bool` | `True` | Default: views are public. `False` = all views require auth |
| `ws_public` | `bool` | `True` | Default: WS open for anonymous. `False` = WS requires auth |
| `with_3d` | `bool` | `False` | Enable 3D background |
| `style_toggle` | `bool` | `True` | Show Glass↔Paper toggle |
| `default_style` | `str` | `"paper"` | Initial style (`"glass"` or `"paper"`) |

| Method | Signature | Description |
|--------|-----------|-------------|
| `view` | `(path, require_auth=None, ws_public=None, icon=None, label=None, nav=True)` | Decorator — registers MinuView |
| `add_nav_item` | `(path, label, icon="link", bottom=False)` | Add static nav link |
| `list_views` | `() -> List[dict]` | List all registered views |

### `@minu.view()` parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | required | URL path (e.g. `"/dashboard"`) |
| `require_auth` | `bool\|None` | `None` | HTTP auth gate. `None` = use `not minu.public` |
| `ws_public` | `bool\|None` | `None` | WS access. `None` = use `minu.ws_public` |
| `icon` | `str\|None` | `None` | Material Symbols icon for NavMenu |
| `label` | `str\|None` | `None` | NavMenu label (default: class name) |
| `nav` | `bool` | `True` | Include in auto-generated NavMenu |

### Injected into each view

| What | How | Source |
|------|-----|--------|
| `self.user` | Auto-populated from session | `SessionData` → `MinuUser` |
| `self.request_data` | Built from `ParsedRequest` | Bridge constructs minimal wrapper |
| `self._app` | ToolBoxV2 App | Passed from `MinuBridge(app_instance=...)` |

---

## Parameter injection (FastTB)

| Priority | Source | Match rule |
|----------|--------|-----------|
| 1 | `ParsedRequest` | Param named `request` or type-hinted |
| 2 | `SessionData` | Param named `session` or type-hinted |
| 3 | Path params | `{param}` in URL |
| 4 | Query params | `?key=value` |
| 5 | JSON body | `request.json_data` fields |
| 6 | Form data | `request.form_data` fields |
| 7 | Default | Python default value |

---

## File layout

```
toolboxv2/utils/workers/
├── fast_tb.py           # FastTB class, decorators, WebSocketContext, mount_static
├── fast_tb_handler.py   # Dispatch, DI, Result handling, as_wsgi_app, WS infra startup
├── server_worker.py     # HTTPWorker, WebSocketMessageHandler, AccessController
├── ws_worker.py         # WebSocket server (websockets lib)
├── ws_bridge.py         # ZMQ→WS bridge (app.ws_send, app.ws_broadcast)
├── event_manager.py     # ZMQ PUB/SUB + PUSH/PULL
├── session.py           # SessionData, SessionManager, SignedCookieSession
└── config.py            # Config dataclasses (WSWorkerConfig etc.)

toolboxv2/mods/Minu/
├── minu_bridge.py       # MinuBridge: @minu.view() decorator, WS+HTTP dispatch
├── core.py              # MinuView, MinuSession, ReactiveState, Components
├── user.py              # AnonymousUser, AuthenticatedUserWrapper, MinuUser
├── shared.py            # SharedSection, SharedManager
└── shared_api.py        # REST + WS endpoints for shared sections

toolboxv2/tbjs/src/ui/
└── components/Minu/
    └── MinuRenderer.js  # Client-side renderer, _applyPatches, _renderView
```
