#!/usr/bin/env python3
"""
demo_minu_bridge.py - FastTB + Minu Integration Demo

Shows:
  1. Public view (no auth) with reactive state + button events
  2. Auth-gated view with user info
  3. Form with input binding
  4. Shared section (multiplayer-ready)
  5. All routes auto-generated from @minu.view() decorator

Run:
    uvicorn demo_minu_bridge:app --port 8000

    # Then browse:
    #   http://localhost:8000/            → route index
    #   http://localhost:8000/counter     → public counter view
    #   http://localhost:8000/profile     → auth-gated profile
    #   http://localhost:8000/todo        → todo list with form
    #   http://localhost:8000/lobby       → multiplayer lobby
"""

from toolboxv2.utils.workers.fast_tb import FastTB
from toolboxv2.mods.Minu.minu_bridge import MinuBridge
from toolboxv2.mods.Minu.core import (
    MinuView, State, Column, Row, Heading, Text,
    Button, Input, Card, Badge, List as MinuList, ListItem, Alert, Divider,
)
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler

# =============================================================================
# App + Bridge Setup
# =============================================================================

app = FastTB(title="Minu Bridge Demo")
minu = MinuBridge(app)


# =============================================================================
# 1. Counter — Public view, reactive state
# =============================================================================

@minu.view("/counter")
class CounterView(MinuView):
    count = State(0)
    label = State("Click the button!")

    def render(self):
        return Card(
            Heading("Counter Demo"),
            Text(self.label.value),
            Row(
                Button("-", on_click="decrement"),
                Text(str(self.count.value), className="text-3xl"),
                Button("+", on_click="increment"),
            ),
            Button("Reset", on_click="reset", variant="ghost"),
            title="FastTB + Minu",
            subtitle="Public — no auth required",
        )

    async def increment(self, event):
        self.count.value += 1
        self.label.value = f"Count is {self.count.value}"

    async def decrement(self, event):
        self.count.value -= 1
        self.label.value = f"Count is {self.count.value}"

    async def reset(self, event):
        self.count.value = 0
        self.label.value = "Reset!"


# =============================================================================
# 2. Profile — Auth-gated, user info
# =============================================================================

@minu.view("/profile", require_auth=True)
class ProfileView(MinuView):

    def render(self):
        if self.user.is_authenticated:
            return Card(
                Heading(f"Welcome, {self.user.name}!"),
                Text(f"UID: {self.user.uid}"),
                Text(f"Level: {self.user.level}"),
                Text(f"Session: {self.user.is_authenticated}"),
                Button("Logout", on_click="logout", variant="ghost"),
                title="Profile",
            )
        # Shouldn't reach here due to require_auth, but safety fallback
        return Alert("Not authenticated", variant="warning")

    async def logout(self, event):
        # Would invalidate session in real app
        pass


# =============================================================================
# 3. Todo — Form with input binding
# =============================================================================

@minu.view("/todo")
class TodoView(MinuView):
    todos = State([])
    input_text = State("")
    status = State("idle")

    def render(self):
        todo_items = []
        for i, todo in enumerate(self.todos.value):
            done = todo.get("done", False)
            todo_items.append(
                ListItem(
                    Row(
                        Text(
                            todo["text"],
                            className="text-line-through" if done else "",
                        ),
                        Badge("done" if done else "pending",
                              variant="success" if done else "default"),
                    )
                )
            )

        return Column(
            Card(
                Heading("Todo List"),
                Text(f"{len(self.todos.value)} items"),

                # Input row
                Row(
                    Input(
                        placeholder="What needs to be done?",
                        bind="input_text",
                        on_submit="add_todo",
                    ),
                    Button("Add", on_click="add_todo"),
                ),

                Divider(),

                # Todo list
                MinuList(*todo_items) if todo_items else Text("No todos yet"),

                # Actions
                Row(
                    Button("Clear done", on_click="clear_done", variant="ghost"),
                    Button("Clear all", on_click="clear_all", variant="ghost"),
                ) if self.todos.value else None,
            ),
        )

    async def add_todo(self, event):
        text = self.input_text.value.strip()
        if not text:
            return

        current = self.todos.value.copy()
        current.append({"text": text, "done": False})
        self.todos.value = current
        self.input_text.value = ""

    async def toggle_todo(self, event):
        idx = event.get("index", 0)
        current = self.todos.value.copy()
        if 0 <= idx < len(current):
            current[idx]["done"] = not current[idx]["done"]
            self.todos.value = current

    async def clear_done(self, event):
        self.todos.value = [t for t in self.todos.value if not t.get("done")]

    async def clear_all(self, event):
        self.todos.value = []


# =============================================================================
# 4. Lobby — Shared section (multiplayer-ready)
# =============================================================================

@minu.view("/lobby")
class LobbyView(MinuView):
    players = State([])
    game_state = State("waiting")
    my_name = State("")

    def render(self):
        player_items = [
            ListItem(
                Row(
                    Text(p["name"]),
                    Badge(
                        "ready" if p.get("ready") else "waiting",
                        variant="success" if p.get("ready") else "default",
                    ),
                )
            )
            for p in self.players.value
        ]

        return Column(
            Card(
                Heading("Game Lobby"),
                Badge(self.game_state.value, variant="info"),

                Divider(),

                Text(f"{len(self.players.value)} player(s)"),
                MinuList(*player_items) if player_items else Text("No players yet"),

                Divider(),

                Row(
                    Input(placeholder="Your name", bind="my_name"),
                    Button("Join", on_click="join_game"),
                ),

                Row(
                    Button("Ready", on_click="toggle_ready", variant="primary"),
                    Button("Leave", on_click="leave_game", variant="ghost"),
                ),
            ),
        )

    async def join_game(self, event):
        name = self.my_name.value.strip() or self.user.name
        current = self.players.value.copy()

        # Don't add duplicate
        if any(p["name"] == name for p in current):
            return

        current.append({"name": name, "ready": False})
        self.players.value = current

    async def toggle_ready(self, event):
        name = self.my_name.value.strip() or self.user.name
        current = self.players.value.copy()
        for p in current:
            if p["name"] == name:
                p["ready"] = not p.get("ready", False)
        self.players.value = current

        # Check if all ready
        if current and all(p.get("ready") for p in current):
            self.game_state.value = "starting"

    async def leave_game(self, event):
        name = self.my_name.value.strip() or self.user.name
        self.players.value = [p for p in self.players.value if p["name"] != name]


# =============================================================================
# Route Index
# =============================================================================

@app.get("/")
async def index():
    """Landing page with links to all Minu views."""
    views = minu.list_views()
    links = "".join(
        f'<li><a href="{v["path"]}">{v["view"]}</a> — <code>{v["path"]}</code></li>'
        for v in views
    )
    return f"""<!DOCTYPE html>
<html>
<head><title>Minu Bridge Demo</title></head>
<body style="font-family: system-ui; max-width: 640px; margin: 2rem auto; padding: 0 1rem;">
    <h1>FastTB + Minu Bridge Demo</h1>
    <p>Registered views:</p>
    <ul>{links}</ul>
    <hr>
    <p><a href="/routes">/routes</a> — all FastTB routes (JSON)</p>
</body>
</html>"""


@app.get("/routes")
def routes():
    return app.list_routes()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  FastTB + Minu Bridge Demo")
    print("=" * 50 + "\n")
    print("  Views:")
    for v in minu.list_views():
        print(f"    {v['path'].ljust(20)} → {v['view']}")
    print(f"\n  All routes:")
    for r in app.list_routes():
        print(f"    {r['method'].ljust(6)} {r['path'].ljust(30)} → {r['handler']}")
    print("\n" + "=" * 50)
    print("  Run: uvicorn demo_minu_bridge:app --port 8000")
    print("=" * 50 + "\n")

    handler = FastTBHandler(app)
    wsgi_app = handler.as_wsgi_app()

    # With waitress:
    from waitress import serve

    serve(wsgi_app, host="0.0.0.0", port=8000)
