"""
HUD Test Widget - Number Guessing Game
======================================

Tests all DomEngine features:
- swap (innerHTML, outerHTML)
- append/prepend
- delete
- run_js
- batch_update
- data-ws-action with data-ws-include
"""

import html
import random
from typing import Any, Dict

from openai.types.image_edit_params import ImageEditParams

from toolboxv2.utils.extras.hud_widget import HudWidget, action, register_widget

Name = "CloudM.HUD_GAMES"


class GuessGameWidget(HudWidget):
    """
    Number Guessing Mini-Game.
    Tests: swap, append, delete, run_js, batch updates
    """

    def __init__(self):
        super().__init__(widget_id=Name, title="🎯 Number Game", icon="🎮")
        # Game state per connection (in production: use session/db)
        self.games: Dict[str, Dict] = {}

    def _get_game(self, conn_id: str) -> Dict:
        if conn_id not in self.games:
            self.games[conn_id] = {
                "target": random.randint(1, 100),
                "attempts": 0,
                "max_attempts": 7,
                "history": [],
                "won": False,
            }
        return self.games[conn_id]

    async def render(self, app, request=None) -> str:
        return f"""
        <div id="game-root" style="padding:8px;">
            <div style="text-align:center;margin-bottom:12px;">
                <div style="font-size:24px;margin-bottom:4px;">🎯</div>
                <div style="color:#94a3b8;font-size:11px;">Guess 1-100</div>
            </div>

            <div id="game-status" style="text-align:center;margin-bottom:12px;padding:8px;background:rgba(99,102,241,0.2);border-radius:6px;">
                <span id="attempts-left">7</span> attempts left
            </div>

            <div style="display:flex;gap:6px;margin-bottom:12px;">
                {self.input_ws("guess-input", placeholder="1-100", input_type="number", style="flex:1;text-align:center;")}
                {self.button_ws("Guess", "guess", include="#guess-input", style="primary", icon="🎯")}
            </div>

            <div id="hint" style="text-align:center;min-height:24px;font-size:12px;color:#94a3b8;"></div>

            <div id="history" style="margin-top:12px;max-height:100px;overflow-y:auto;"></div>

            <div style="margin-top:12px;display:flex;gap:6px;">
                {self.button_ws("New Game", "reset", style="secondary", icon="🔄")}
                {self.button_ws("Test Batch", "test_batch", style="secondary", icon="⚡")}
            </div>
        </div>
        """

    def _render_history_item(self, guess: int, hint: str, idx: int) -> str:
        color = "#22c55e" if "🎉" in hint else ("#3b82f6" if "⬆" in hint else "#f59e0b")
        return f"""<div id="history-{idx}" style="display:flex;justify-content:space-between;padding:4px 8px;background:rgba(255,255,255,0.05);border-radius:4px;margin-bottom:4px;font-size:11px;">
            <span style="color:{color};font-weight:600;">{guess}</span>
            <span style="color:#64748b;">{hint}</span>
        </div>"""

    # =========================================================================
    # Actions
    # =========================================================================

    @action("guess")
    async def on_guess(self, app, payload: Dict, conn_id: str, request) -> Dict:
        """Process a guess - tests swap, append, run_js"""
        game = self._get_game(conn_id)

        if game["won"]:
            return (
                self.dom().swap("#hint", "🏆 Already won! Start new game.").to_response()
            )

        # Parse input
        try:
            guess = int(payload.get("guess-input", 0))
        except (ValueError, TypeError):
            return (
                self.dom()
                .swap("#hint", "❌ Enter a number 1-100")
                .run_js("document.getElementById('guess-input').select()")
                .to_response()
            )

        if guess < 1 or guess > 100:
            return self.dom().swap("#hint", "❌ Must be 1-100").to_response()

        # Process guess
        game["attempts"] += 1
        remaining = game["max_attempts"] - game["attempts"]

        if guess == game["target"]:
            game["won"] = True
            hint = f"🎉 Correct in {game['attempts']} tries!"
            game["history"].append((guess, hint))

            return (
                self.dom()
                .swap(
                    "#hint", f'<span style="color:#22c55e;font-weight:600;">{hint}</span>'
                )
                .swap("#game-status", "🏆 YOU WIN!")
                .append(
                    "#history",
                    self._render_history_item(guess, hint, len(game["history"])),
                )
                .run_js("document.getElementById('guess-input').disabled = true")
                .to_response()
            )

        elif remaining <= 0:
            hint = f"💀 Game Over! Number was {game['target']}"
            game["history"].append((guess, "❌"))

            return (
                self.dom()
                .swap("#hint", f'<span style="color:#ef4444;">{hint}</span>')
                .swap("#game-status", "💀 GAME OVER")
                .append(
                    "#history",
                    self._render_history_item(guess, "❌", len(game["history"])),
                )
                .run_js("document.getElementById('guess-input').disabled = true")
                .to_response()
            )

        else:
            if guess < game["target"]:
                hint = "⬆️ Higher"
            else:
                hint = "⬇️ Lower"

            game["history"].append((guess, hint))

            return (
                self.dom()
                .swap("#hint", hint)
                .swap("#attempts-left", str(remaining))
                .append(
                    "#history",
                    self._render_history_item(guess, hint, len(game["history"])),
                )
                .run_js(
                    "document.getElementById('guess-input').value=''; document.getElementById('guess-input').focus()"
                )
                .to_response()
            )

    @action("reset")
    async def on_reset(self, app, payload: Dict, conn_id: str, request) -> Dict:
        """Reset game - tests multiple swaps + run_js"""
        # Clear game state
        self.games[conn_id] = {
            "target": random.randint(1, 100),
            "attempts": 0,
            "max_attempts": 7,
            "history": [],
            "won": False,
        }

        return (
            self.dom()
            .swap("#hint", "")
            .swap("#history", "")
            .swap("#game-status", '<span id="attempts-left">7</span> attempts left')
            .run_js("""
                const input = document.getElementById('guess-input');
                input.disabled = false;
                input.value = '';
                input.focus();
            """)
            .to_response()
        )

    @action("test_batch")
    async def on_test_batch(self, app, payload: Dict, conn_id: str, request) -> Dict:
        """Test batch update with multiple operations"""
        return (
            self.dom()
            .swap("#hint", "⚡ Batch test executed!")
            .prepend(
                "#history",
                '<div style="color:#f59e0b;font-size:10px;padding:4px;">--- Batch Test ---</div>',
            )
            .run_js(
                "console.log('[DomEngine] Batch test OK'); HUD.notify('Batch update works!', 'success')"
            )
            .to_response()
        )


class CounterWidget(HudWidget):
    """
    Simple Counter Widget.
    Tests: basic swap, outerHTML replace
    """

    def __init__(self):
        super().__init__(widget_id="counter_test", title="🔢 Counter", icon="🔢")
        self.counts: Dict[str, int] = {}

    def _get_count(self, conn_id: str) -> int:
        return self.counts.get(conn_id, 0)

    async def render(self, app, request=None) -> str:
        return (
            """
        <div id="counter-root" style="padding:8px;text-align:center;">
            <div id="counter-value" style="font-size:48px;font-weight:bold;color:#6366f1;margin:12px 0;">0</div>
            <div style="display:flex;gap:8px;justify-content:center;">
            """
            + self.button_ws("-10", "add", payload={"delta": -10}, style="secondary")
            + self.button_ws("-1", "add", payload={"delta": -1}, style="secondary")
            + self.button_ws("+1", "add", payload={"delta": 1}, style="primary")
            + self.button_ws("+10", "add", payload={"delta": 10}, style="primary")
            + """
            </div>
            <div style="margin-top:12px;">"""
            + self.button_ws("Reset", "reset", style="danger", icon="🔄")
            + self.button_ws("Double", "double", style="warning", icon="✖️")
            + """
            </div>
        </div>
        """
        )

    @action("add")
    async def on_add(self, app, payload: Dict, conn_id: str, request) -> Dict:
        delta = payload.get("delta", 1)
        self.counts[conn_id] = self._get_count(conn_id) + delta
        count = self.counts[conn_id]

        color = "#22c55e" if count > 0 else ("#ef4444" if count < 0 else "#6366f1")
        return (
            self.dom()
            .swap("#counter-value", f'<span style="color:{color}">{count}</span>')
            .to_response()
        )

    @action("reset")
    async def on_reset(self, app, payload: Dict, conn_id: str, request) -> Dict:
        self.counts[conn_id] = 0
        return (
            self.dom()
            .swap("#counter-value", "0")
            .run_js("HUD.notify('Counter reset!', 'info')")
            .to_response()
        )

    @action("double")
    async def on_double(self, app, payload: Dict, conn_id: str, request) -> Dict:
        self.counts[conn_id] = self._get_count(conn_id) * 2
        count = self.counts[conn_id]
        return self.dom().swap("#counter-value", str(count)).to_response()


class TodoWidget(HudWidget):
    """
    Todo List Widget.
    Tests: append, delete, outerHTML replace
    """

    def __init__(self):
        super().__init__(widget_id="todo_test", title="✅ Todo List", icon="📝")
        self.todos: Dict[str, list] = {}
        self.id_counter = 0

    def _get_todos(self, conn_id: str) -> list:
        if conn_id not in self.todos:
            self.todos[conn_id] = []
        return self.todos[conn_id]

    async def render(self, app, request=None) -> str:
        return f"""
        <div id="todo-root" style="padding:8px;">
            <div style="display:flex;gap:6px;margin-bottom:12px;">
                {self.input_ws("todo-input", placeholder="New task...", style="flex:1;")}
                {self.button_ws("Add", "add", include="#todo-input", style="success", icon="➕")}
            </div>
            <div id="todo-count" style="font-size:10px;color:#64748b;margin-bottom:8px;">0 tasks</div>
            <div id="todo-list"></div>
        </div>
        """

    def _render_todo(self, todo: Dict) -> str:
        done_style = "text-decoration:line-through;opacity:0.5;" if todo["done"] else ""
        icon = "☑️" if todo["done"] else "⬜"
        return f"""<div id="todo-{todo["id"]}" style="display:flex;align-items:center;justify-content:space-between;padding:6px 8px;background:rgba(255,255,255,0.05);border-radius:4px;margin-bottom:4px;{done_style}">
            <span style="flex:1;font-size:12px;">{html.escape(todo["text"])}</span>
            <div style="display:flex;gap:4px;">
                <button data-ws-action="toggle" data-ws-payload='{{"id":{todo["id"]}}}' style="background:none;border:none;cursor:pointer;font-size:14px;">{icon}</button>
                <button data-ws-action="delete" data-ws-payload='{{"id":{todo["id"]}}}' style="background:none;border:none;cursor:pointer;font-size:14px;">🗑️</button>
            </div>
        </div>"""

    @action("add")
    async def on_add(self, app, payload: Dict, conn_id: str, request) -> Dict:
        text = payload.get("todo-input", "").strip()
        if not text:
            return {}

        self.id_counter += 1
        todo = {"id": self.id_counter, "text": text, "done": False}
        self._get_todos(conn_id).append(todo)

        count = len(self._get_todos(conn_id))
        return (
            self.dom()
            .append("#todo-list", self._render_todo(todo))
            .swap("#todo-count", f"{count} task{'s' if count != 1 else ''}")
            .run_js(
                "document.getElementById('todo-input').value=''; document.getElementById('todo-input').focus()"
            )
            .to_response()
        )

    @action("toggle")
    async def on_toggle(self, app, payload: Dict, conn_id: str, request) -> Dict:
        todo_id = payload.get("id")
        todos = self._get_todos(conn_id)

        for todo in todos:
            if todo["id"] == todo_id:
                todo["done"] = not todo["done"]
                return (
                    self.dom()
                    .replace(f"#todo-{todo_id}", self._render_todo(todo))
                    .to_response()
                )
        return {}

    @action("delete")
    async def on_delete(self, app, payload: Dict, conn_id: str, request) -> Dict:
        todo_id = payload.get("id")
        todos = self._get_todos(conn_id)
        self.todos[conn_id] = [t for t in todos if t["id"] != todo_id]

        count = len(self._get_todos(conn_id))
        return (
            self.dom()
            .delete(f"#todo-{todo_id}")
            .swap("#todo-count", f"{count} task{'s' if count != 1 else ''}")
            .to_response()
        )


# =========================================================================
# Register all test widgets
# =========================================================================

guess_game = GuessGameWidget()
counter_widget = CounterWidget()
todo_widget = TodoWidget()

register_widget(guess_game)
register_widget(counter_widget)
register_widget(todo_widget)


# =========================================================================
# Export functions (für ToolBoxV2 Integration)
# =========================================================================
#
from toolboxv2 import get_app

version = "0.1.0"
export = get_app(f"{Name}.EXPORT").tb


# Beispiel export decorator (auskommentiert da standalone)
@export(mod_name=Name, api=True, version=version)
async def hud_guess_game(app, request=None):
    return await guess_game.render(app, request)


@export(mod_name=Name, api=True, version=version)
async def hud_counter_test(app, request=None):
    return await counter_widget.render(app, request)


@export(mod_name=Name, api=True, version=version)
async def hud_todo_test(app, request=None):
    return await todo_widget.render(app, request)


@export(mod_name=Name, api=True, version=version)
async def hud_action(app, action: str, payload: dict, conn_id: str, request=None):
    """Route action to correct widget based on widget_id in payload or conn context."""
    # In real implementation: determine widget from request context
    # For test: try all widgets
    from toolboxv2.utils.extras.hud_widget import get_widget

    widget_id = payload.get("_widget_id")
    if widget_id:
        widget = get_widget(widget_id)
        if widget:
            return await widget.handle_action(app, action, payload, conn_id, request)

    # Fallback: try guess_game
    return await guess_game.handle_action(app, action, payload, conn_id, request)


# =========================================================================
# Standalone Test
# =========================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=== DomBuilder Test ===\n")

        # Test 1: Single swap
        result = guess_game.dom().swap("#test", "Hello").to_response()
        print(f"Single swap: {result}")
        assert result["type"] == "dom_update"
        assert result["selector"] == "#test"

        # Test 2: Batch update
        result = (
            guess_game.dom()
            .swap("#a", "A")
            .append("#b", "B")
            .delete("#c")
            .run_js("console.log('test')")
            .to_response()
        )
        print(f"Batch update: {result}")
        assert result["type"] == "batch_update"
        assert len(result["updates"]) == 4

        # Test 3: Render
        html = await guess_game.render(None, None)
        print(f"\nRender output (first 200 chars):\n{html[:200]}...")
        assert "game-root" in html
        assert "data-ws-action" in html

        # Test 4: Action handling
        result = await guess_game.on_reset(None, {}, "test-conn", None)
        print(f"\nReset action result: {result}")
        assert "type" in result

        print("\n✅ All tests passed!")

    asyncio.run(test())
