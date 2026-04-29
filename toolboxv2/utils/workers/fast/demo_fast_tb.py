#!/usr/bin/env python3
"""
demo_fast_tb.py - Runnable Demo for FastTB

Shows all features:
  - GET/POST/PUT/DELETE with path params
  - Parameter injection (request, session, path, query, JSON body)
  - Type coercion (int, float, bool)
  - Response types (dict, html, tuple with status, None)
  - WebSocket handler registration
  - Sync + async handlers
  - Route listing

Run standalone (no ToolBoxV2 required):
    pip install uvicorn a2wsgi
    uvicorn demo_fast_tb:app --port 8000 --reload

Or integrate into HTTPWorker:
    worker.run(fast_tb_app=app)

Test with curl:
    curl http://localhost:8000/health
    curl http://localhost:8000/hello/Markin
    curl http://localhost:8000/users/42?verbose=true
    curl -X POST http://localhost:8000/items -H 'Content-Type: application/json' -d '{"name":"Widget","price":9.99}'
    curl http://localhost:8000/search?q=toolbox&limit=5
    curl http://localhost:8000/page
    curl http://localhost:8000/routes
"""

from utils.workers.fast_tb import FastTB
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.workers.session import SessionData
from utils.workers.fast_tb_handler import FastTBHandler

# =============================================================================
# App Instance
# =============================================================================

app = FastTB(title="FastTB Demo")

# =============================================================================
# In-Memory "Database" (for demo purposes)
# =============================================================================

_items_db: dict = {}
_next_id = 1

# =============================================================================
# Endpoints
# =============================================================================


# --- Health / Meta ---

@app.get("/health")
async def health():
    """Return None → auto-converted to {"status": "ok"} 200."""
    return None


@app.get("/routes")
def list_routes():
    """Sync handler — lists all registered routes (for debugging)."""
    return app.list_routes()


# --- Path Parameters + Type Coercion ---

@app.get("/hello/{name}")
async def hello(name: str):
    """Simple path param injection."""
    return {"message": f"Hello {name}!"}


@app.get("/users/{user_id}")
async def get_user(user_id: int, verbose: bool = False):
    """Path param coerced to int, query param coerced to bool."""
    user = {"id": user_id, "name": f"User #{user_id}"}
    if verbose:
        user["meta"] = {"registered": "2024-01-01", "tier": "premium"}
    return user


@app.get("/compute/{a}/{b}")
async def compute(a: float, b: float, op: str = "add"):
    """Multiple path params + query param with default."""
    ops = {
        "add": a + b,
        "sub": a - b,
        "mul": a * b,
        "div": a / b if b != 0 else "infinity",
    }
    result = ops.get(op, f"unknown op: {op}")
    return {"a": a, "b": b, "op": op, "result": result}


# --- Full DI: Request + Session + Path + Query ---

@app.get("/whoami")
async def whoami(request: ParsedRequest, session: SessionData):
    """Inject both request and session objects."""
    return {
        "client_ip": request.client_ip,
        "method": request.method,
        "session_user": session.user_name,
        "authenticated": session.is_authenticated,
    }


# --- CRUD with JSON Body ---

@app.post("/items")
async def create_item(name: str, price: float = 0.0):
    """JSON body fields injected by parameter name."""
    global _next_id
    item_id = _next_id
    _next_id += 1
    item = {"id": item_id, "name": name, "price": price}
    _items_db[item_id] = item
    return (201, item)  # Tuple → JSON with custom status


@app.get("/items")
async def list_items():
    """List all items."""
    return list(_items_db.values())


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    """Get single item by ID."""
    if item_id not in _items_db:
        return (404, {"error": f"Item {item_id} not found"})
    return _items_db[item_id]


@app.put("/items/{item_id}")
async def update_item(item_id: int, name: str = None, price: float = None):
    """Update item — partial update via defaults."""
    if item_id not in _items_db:
        return (404, {"error": f"Item {item_id} not found"})
    item = _items_db[item_id]
    if name is not None:
        item["name"] = name
    if price is not None:
        item["price"] = price
    return item


@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """Delete item."""
    if item_id not in _items_db:
        return (404, {"error": f"Item {item_id} not found"})
    del _items_db[item_id]
    return (200, {"deleted": item_id})


# --- Search with Query Params ---

@app.get("/search")
async def search(q: str, limit: int = 10, offset: int = 0):
    """Query params with type coercion and defaults."""
    # Simulate search
    results = [
        {"title": f"Result {i} for '{q}'", "score": 1.0 - (i * 0.1)}
        for i in range(min(limit, 5))
    ]
    return {
        "query": q,
        "limit": limit,
        "offset": offset,
        "total": len(results),
        "results": results,
    }


# --- HTML Response ---

@app.get("/page")
async def page():
    """Return HTML string → auto-detected and served as text/html."""
    return """<!DOCTYPE html>
<html>
<head><title>FastTB Demo</title></head>
<body style="font-family: monospace; background: #1a1a2e; color: #eee; padding: 2rem;">
    <h1 style="color: #e94560;">FastTB Demo</h1>
    <p>This page was served by a FastTB endpoint.</p>
    <h2>Try these:</h2>
    <ul>
        <li><a href="/health" style="color: #0f3460;">/health</a></li>
        <li><a href="/hello/World" style="color: #0f3460;">/hello/World</a></li>
        <li><a href="/users/42?verbose=true" style="color: #0f3460;">/users/42?verbose=true</a></li>
        <li><a href="/compute/3.14/2.0?op=mul" style="color: #0f3460;">/compute/3.14/2.0?op=mul</a></li>
        <li><a href="/search?q=toolbox&limit=3" style="color: #0f3460;">/search?q=toolbox&limit=3</a></li>
        <li><a href="/routes" style="color: #0f3460;">/routes</a></li>
    </ul>
</body>
</html>"""


# --- Raw Bytes Response ---

@app.get("/binary")
async def binary_endpoint():
    """Return raw bytes."""
    return b"\x89PNG\r\n\x1a\n"  # Fake PNG header


# --- Multi-Method Route ---

@app.route("/echo", methods=["GET", "POST"])
async def echo(request: ParsedRequest):
    """Same handler for GET and POST."""
    return {
        "method": request.method,
        "path": request.path,
        "query": dict(request.query_params),
        "body": request.json_data,
    }


# --- WebSocket Handler ---

@app.websocket("/ws/chat")
class ChatHandler:
    """WebSocket handler — registered in ToolBoxV2 format via get_websocket_handlers()."""

    async def on_connect(self, conn_id, session):
        print(f"[WS] {conn_id} connected")

    async def on_message(self, payload, conn_id, session, request):
        msg = payload.get("message", "")
        print(f"[WS] {conn_id}: {msg}")
        return {"type": "echo", "message": msg, "from": conn_id}

    async def on_disconnect(self, conn_id, session):
        print(f"[WS] {conn_id} disconnected")


@app.websocket("/ws/room/{room_id}")
class RoomHandler:
    """WebSocket with path parameters."""

    async def on_connect(self, conn_id, session):
        print(f"[WS Room] {conn_id} joined")

    async def on_message(self, payload, conn_id, session, request):
        return {"type": "room_msg", "data": payload}


# =============================================================================
# Main — Print route table when run directly
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  FastTB Demo — {app.title}")
    print(f"{'='*60}\n")
    print(f"  Registered Routes:\n")

    for route in app.list_routes():
        method = route["method"].ljust(6)
        path = route["path"].ljust(30)
        handler = route["handler"]
        print(f"    {method} {path} → {handler}")

    handler = FastTBHandler(app)
    wsgi_app = handler.as_wsgi_app()

    # With waitress:
    from waitress import serve

    serve(wsgi_app, host="0.0.0.0", port=8000)
    print(f"\n{'='*60}")
    print(f"  Run with: uvicorn demo_fast_tb:app --port 8000 --reload")
    print(f"{'='*60}\n")
