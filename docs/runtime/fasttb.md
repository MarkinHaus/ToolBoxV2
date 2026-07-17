# FastTB — API Reference & Integration Guide

> **Also see:** [FastTBHandler](fast_tb_handler.md) — internal dispatch engine (DI, routing, WSGI bridge, SSE streaming).

> FastAPI-like routing for the ToolBoxV2 Worker System.
> Zero migration, dual-mode: WSGI (Waitress) or ASGI (Uvicorn).

---

## Quick Start

```python
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.workers.session import SessionData

app = FastTB(title="MyService")

@app.get("/hello/{name}")
async def hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/items")
async def create_item(request: ParsedRequest, session: SessionData):
    data = request.json_data
    return {"created": True, "user": session.user_name}
```

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              FastTB                      │
                    │  @app.get() / @app.post() / @app.ws()   │
                    └────────────┬────────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────────┐
                    │          FastTBHandler                   │
                    │  Route matching · DI · Response format   │
                    └────────────┬────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────────┐
              ▼                  ▼                      ▼
     ┌────────────────┐  ┌──────────────┐  ┌───────────────────┐
     │  HTTPWorker     │  │  Standalone   │  │  Uvicorn (ASGI)   │
     │  (WSGI/Waitress)│  │  WSGI app     │  │  via a2wsgi       │
     │  ← 3 lines mod │  │  .as_wsgi_app │  │  FastTB.__call__  │
     └────────────────┘  └──────────────┘  └───────────────────┘
```

Two files, zero dependencies beyond what's already in the project.

---

## API Reference

### `FastTB`

```python
class FastTB(title="FastTB", app_instance=None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `get` | `(path: str, name="")` | Register GET endpoint |
| `post` | `(path: str, name="")` | Register POST endpoint |
| `put` | `(path: str, name="")` | Register PUT endpoint |
| `delete` | `(path: str, name="")` | Register DELETE endpoint |
| `patch` | `(path: str, name="")` | Register PATCH endpoint |
| `route` | `(path, methods=["GET"], name="")` | Register for multiple methods |
| `websocket` | `(path: str)` | Register WebSocket handler class |
| `has_route` | `(path, method) -> bool` | Check if route exists |
| `resolve_route` | `(path, method) -> (Route, params) \| None` | Match and extract params |
| `list_routes` | `() -> List[dict]` | Debug: list all registered routes |
| `get_websocket_handlers` | `() -> Dict` | Export WS handlers for ToolBoxV2 |
| `__call__` | `(scope, receive, send)` | ASGI entrypoint for Uvicorn |

### `FastTBHandler`

```python
class FastTBHandler(fast_tb_app, session_manager=None)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `has_route` | `(path, method) -> bool` | Delegates to FastTB |
| `handle_request` | `async (request: ParsedRequest) -> (status, headers, body)` | Full dispatch pipeline |
| `as_wsgi_app` | `() -> Callable` | Returns standalone WSGI callable |

---

## Path Parameters

FastAPI-style `{param}` syntax, converted to named regex groups internally.

```python
@app.get("/users/{user_id}/posts/{post_id}")
async def get_post(user_id: str, post_id: int):
    # user_id: str from URL
    # post_id: auto-coerced to int
    return {"user": user_id, "post": post_id}
```

Supported type coercions: `str` (default), `int`, `float`, `bool`.

---

## Parameter Injection

The handler inspects your function signature and injects parameters automatically:

| Priority | Source | Match Rule |
|----------|--------|------------|
| 1 | `ParsedRequest` | Parameter named `request` OR type-hinted `ParsedRequest` |
| 2 | `SessionData` | Parameter named `session` OR type-hinted `SessionData` |
| 3 | Path params | Parameter name matches `{param}` in URL |
| 4 | Query params | Parameter name found in `?key=value` |
| 5 | JSON body | Parameter name found in `request.json_data` dict |
| 6 | Form data | Parameter name found in `request.form_data` dict |
| 7 | Default value | Uses Python default if nothing matched |

Parameters without a match and without a default raise a `ValueError` → HTTP 500.

```python
@app.get("/search/{category}")
async def search(
    request: ParsedRequest,   # ← injected: full request
    session: SessionData,     # ← injected: session from cookie
    category: str,            # ← injected: from URL path
    q: str,                   # ← injected: from ?q=...
    limit: int = 20,          # ← injected: from ?limit=... or default 20
):
    ...
```

---

## Response Formatting

Return whatever is natural — the handler auto-converts:

| Return Type | HTTP Response |
|-------------|---------------|
| `dict` / `list` | JSON 200 |
| `None` | `{"status": "ok"}` 200 |
| `str` starting with `<` | HTML 200 |
| `str` (other) | `{"result": "..."}` 200 |
| `bytes` | `application/octet-stream` 200 |
| `(status, dict)` | JSON with custom status |
| `(status, headers, body)` | Full passthrough |

```python
@app.get("/health")
async def health():
    return {"status": "ok"}           # → JSON 200

@app.get("/page")
async def page():
    return "<h1>Hello</h1>"           # → HTML 200

@app.post("/create")
async def create(name: str):
    return (201, {"id": "abc"})       # → JSON 201

@app.get("/raw")
async def raw():
    return (200, {"Content-Type": "text/csv"}, b"a,b\n1,2")  # passthrough
```

---

## WebSocket Handlers

Register a class — methods map to ToolBoxV2's ZMQ-based WebSocket lifecycle:

```python
@app.websocket("/ws/chat")
class ChatHandler:
    async def on_connect(self, conn_id, session):
        print(f"{conn_id} connected")

    async def on_message(self, payload, conn_id, session, request):
        # payload: parsed dict from client
        return {"echo": payload}  # auto-sent back via ws_send

    async def on_disconnect(self, conn_id, session):
        print(f"{conn_id} left")
```

The `get_websocket_handlers()` method exports these in ToolBoxV2 format so
`WebSocketMessageHandler` picks them up via ZMQ without any changes.

---

## Integration Mode A: Inside HTTPWorker (Production)

Three insertions in `server_worker.py`, zero rewrites:

### 1. `__init__` — add field

```python
self._fast_tb_handler: "FastTBHandler | None" = None
```

### 2. `run()` — accept and init FastTB

```python
def run(self, host=None, port=None, do_run=True, fast_tb_app=None):
    ...
    # After self._toolbox_handler = ToolBoxHandler(...)
    if fast_tb_app:
        from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
        self._fast_tb_handler = FastTBHandler(fast_tb_app, self._session_manager)
        ws_handlers = fast_tb_app.get_websocket_handlers()
        if ws_handlers:
            self._app.websocket_handlers.update(ws_handlers)
```

### 3. `wsgi_app()` — add route check before 404

```python
# After the toolbox_handler.is_api_request check, before else: 404
elif self._fast_tb_handler and self._fast_tb_handler.has_route(request.path, request.method):
    status, headers, body = self._run_async(
        self._fast_tb_handler.handle_request(request)
    )
```

### Caller side

```python
from toolboxv2.utils.workers.fast_tb import FastTB

my_app = FastTB(title="MyExtension")

@my_app.get("/ext/hello")
async def hello():
    return {"msg": "hello from extension"}

worker = HTTPWorker("http_1", config)
worker.run(fast_tb_app=my_app)
```

---

## Integration Mode B: Standalone ASGI (Uvicorn)

```python
# myapp.py
from toolboxv2.utils.workers.fast_tb import FastTB

app = FastTB(title="Standalone")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

```bash
pip install uvicorn a2wsgi
uvicorn myapp:app --port 8000
```

`FastTB.__call__` → `a2wsgi.WSGIMiddleware` → `FastTBHandler.as_wsgi_app()`.

---

## Integration Mode C: Standalone WSGI (Waitress)

```python
from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler

app = FastTB(title="StandaloneWSGI")

@app.get("/health")
async def health():
    return {"status": "ok"}

handler = FastTBHandler(app)
wsgi_app = handler.as_wsgi_app()

# With waitress:
from waitress import serve
serve(wsgi_app, host="0.0.0.0", port=8000)
```

---

## File Structure

```
toolboxv2/utils/workers/
├── fast_tb.py           # User-facing API (FastTB class, decorators)
├── fast_tb_handler.py   # Dispatch engine (DI, routing, response formatting)
├── server_worker.py     # Existing — 3 lines added for integration
├── session.py           # Existing — SessionData used by DI
└── ...
```

---

## Testing

```bash
python -m unittest test_fast_tb -v
```

74 unit tests covering: path regex, route registration, resolution, WS export,
parameter injection (all 6 sources), type coercion, response formatting,
async/sync execution, edge cases.

Test files: `test_fast_tb.py` + `test_stubs.py` (minimal dep stubs).
