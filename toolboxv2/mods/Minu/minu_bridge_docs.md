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
