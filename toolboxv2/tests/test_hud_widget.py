"""
Tests for HUD Widget Base Class
===============================

Tests the HudWidget class functionality including:
- Widget initialization
- Action registration and handling
- HTML helper methods
- Session/user helpers
- Widget registry
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from toolboxv2.utils.extras.hud_widget import (
    HudWidget,
    HudWidgetConfig,
    register_widget,
    get_widget,
    get_all_widgets,
    _widget_registry,
)


class TestHudWidgetConfig:
    """Tests for HudWidgetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HudWidgetConfig(widget_id="test", title="Test Widget")
        assert config.widget_id == "test"
        assert config.title == "Test Widget"
        assert config.icon == "📦"
        assert config.description == ""
        assert config.requires_auth is False
        assert config.min_level == 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HudWidgetConfig(
            widget_id="custom",
            title="Custom Widget",
            icon="🔧",
            description="A custom widget",
            requires_auth=True,
            min_level=5
        )
        assert config.icon == "🔧"
        assert config.requires_auth is True
        assert config.min_level == 5


class TestHudWidgetInit:
    """Tests for HudWidget initialization."""

    def test_basic_init(self):
        """Test basic widget initialization."""
        widget = HudWidget("my_widget", "My Widget")
        assert widget.widget_id == "my_widget"
        assert widget.title == "My Widget"
        assert widget.icon == "📦"
        assert widget.actions == {}

    def test_init_with_icon(self):
        """Test widget initialization with custom icon."""
        widget = HudWidget("test", "Test", icon="🚀")
        assert widget.icon == "🚀"

    def test_config_created(self):
        """Test that config is created on init."""
        widget = HudWidget("test", "Test Widget", "🎯")
        assert widget._config.widget_id == "test"
        assert widget._config.title == "Test Widget"
        assert widget._config.icon == "🎯"


class TestHudWidgetActions:
    """Tests for action registration and handling."""

    def test_action_decorator(self):
        """Test action decorator registers handler."""
        widget = HudWidget("test", "Test")

        @widget.action("do_something")
        async def handler(app, payload, conn_id, request):
            return {"done": True}

        assert "do_something" in widget.actions
        assert widget.actions["do_something"] == handler

    @pytest.mark.asyncio
    async def test_handle_registered_action(self):
        """Test handling a registered action."""
        widget = HudWidget("test", "Test")

        @widget.action("greet")
        async def greet_handler(app, payload, conn_id, request):
            name = payload.get("name", "World")
            return {"message": f"Hello, {name}!"}

        result = await widget.handle_action(
            app=None,
            action="greet",
            payload={"name": "User"},
            conn_id="conn123",
            request=None
        )

        assert result == {"message": "Hello, User!"}

    @pytest.mark.asyncio
    async def test_handle_unknown_action(self):
        """Test handling an unknown action returns error."""
        widget = HudWidget("test", "Test")

        result = await widget.handle_action(
            app=None,
            action="unknown",
            payload={},
            conn_id="conn123",
            request=None
        )

        assert "error" in result
        assert "Unknown action" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_action_exception(self):
        """Test action handler exception is caught."""
        widget = HudWidget("test", "Test")

        @widget.action("fail")
        async def fail_handler(app, payload, conn_id, request):
            raise ValueError("Something went wrong")

        result = await widget.handle_action(
            app=None,
            action="fail",
            payload={},
            conn_id="conn123",
            request=None
        )

        assert "error" in result
        assert "Something went wrong" in result["error"]

    @pytest.mark.asyncio
    async def test_handle_action_non_dict_result(self):
        """Test action returning non-dict is wrapped."""
        widget = HudWidget("test", "Test")


class TestHudWidgetHtmlHelpers:
    """Tests for HTML helper methods."""

    def test_button_primary(self):
        """Test primary button generation."""
        widget = HudWidget("test", "Test")
        html = widget.button("Click Me", "do_click")

        assert "Click Me" in html
        assert "HUD.action('test', 'do_click'" in html
        assert "linear-gradient" in html  # Primary style

    def test_button_with_payload(self):
        """Test button with payload."""
        widget = HudWidget("test", "Test")
        html = widget.button("Save", "save", payload={"id": 123})

        # Payload is HTML-escaped in the onclick attribute
        assert '&quot;id&quot;: 123' in html or '&quot;id&quot;:123' in html.replace(" ", "")

    def test_button_with_icon(self):
        """Test button with icon."""
        widget = HudWidget("test", "Test")
        html = widget.button("Delete", "delete", icon="🗑️", style="danger")

        assert "🗑️" in html
        assert "#ef4444" in html  # Danger color

    def test_button_styles(self):
        """Test different button styles."""
        widget = HudWidget("test", "Test")

        primary = widget.button("P", "a", style="primary")
        secondary = widget.button("S", "a", style="secondary")
        danger = widget.button("D", "a", style="danger")
        success = widget.button("G", "a", style="success")

        assert "linear-gradient" in primary
        assert "rgba(255,255,255,0.1)" in secondary
        assert "#ef4444" in danger
        assert "#22c55e" in success

    def test_input_field(self):
        """Test input field generation."""
        widget = HudWidget("test", "Test")
        html = widget.input_field("search", "do_search", placeholder="Search...")

        assert 'id="hud-input-search"' in html
        assert 'placeholder="Search..."' in html
        assert "HUD.action('test','do_search'" in html
        assert "event.key==='Enter'" in html

    def test_input_field_password(self):
        """Test password input field."""
        widget = HudWidget("test", "Test")
        html = widget.input_field("pwd", "submit", input_type="password")

        assert 'type="password"' in html

    def test_select_field(self):
        """Test select field generation."""
        widget = HudWidget("test", "Test")
        options = [
            {"value": "a", "label": "Option A"},
            {"value": "b", "label": "Option B"},
        ]
        html = widget.select_field("choice", "select", options, selected="b")

        assert 'id="hud-select-choice"' in html
        assert "Option A" in html
        assert "Option B" in html
        assert 'value="b" selected' in html

    def test_card(self):
        """Test card generation."""
        widget = HudWidget("test", "Test")
        html = widget.card("Card Title", "Card content here")

        assert "Card Title" in html
        assert "Card content here" in html
        assert "border-radius:8px" in html

    def test_card_with_actions(self):
        """Test card with actions."""
        widget = HudWidget("test", "Test")
        actions = widget.button("Action", "do_action")
        html = widget.card("Title", "Content", actions=actions)

        assert "Action" in html
        assert "HUD.action" in html

    def test_list_item(self):
        """Test list item generation."""
        widget = HudWidget("test", "Test")
        html = widget.list_item("Item Title", subtitle="Subtitle", icon="📄")

        assert "Item Title" in html
        assert "Subtitle" in html
        assert "📄" in html

    def test_html_escaping(self):
        """Test that HTML is properly escaped."""
        widget = HudWidget("test", "Test")
        html = widget.button("<script>alert('xss')</script>", "action")

        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestHudWidgetSessionHelpers:
    """Tests for session/user helper methods."""

    def test_get_user_id_none_request(self):
        """Test get_user_id with None request."""
        assert HudWidget.get_user_id(None) == ""

    def test_get_user_id_with_session(self):
        """Test get_user_id with session data."""
        request = MagicMock()
        request.session = {"user_id": "user123", "user_name": "testuser"}

        assert HudWidget.get_user_id(request) == "user123"

    def test_get_user_id_fallback_to_name(self):
        """Test get_user_id falls back to user_name."""
        request = MagicMock()
        request.session = {"user_id": "", "user_name": "testuser"}

        assert HudWidget.get_user_id(request) == "testuser"

    def test_get_user_level_none_request(self):
        """Test get_user_level with None request."""
        assert HudWidget.get_user_level(None) == 0

    def test_get_user_level_with_session(self):
        """Test get_user_level with session data."""
        request = MagicMock()
        request.session = {"level": 5}

        assert HudWidget.get_user_level(request) == 5

    def test_is_authenticated_none_request(self):
        """Test is_authenticated with None request."""
        assert HudWidget.is_authenticated(None) is False

    def test_is_authenticated_true(self):
        """Test is_authenticated returns True."""
        request = MagicMock()
        request.session = {"validated": True, "anonymous": False}

        assert HudWidget.is_authenticated(request) is True

    def test_is_authenticated_anonymous(self):
        """Test is_authenticated returns False for anonymous."""
        request = MagicMock()
        request.session = {"validated": True, "anonymous": True}

        assert HudWidget.is_authenticated(request) is False


class TestWidgetRegistry:
    """Tests for widget registry functions."""

    def setup_method(self):
        """Clear registry before each test."""
        _widget_registry.clear()

    def test_register_widget(self):
        """Test registering a widget."""
        widget = HudWidget("test", "Test Widget")
        register_widget(widget)

        assert "test" in _widget_registry
        assert _widget_registry["test"] == widget

    def test_get_widget(self):
        """Test getting a registered widget."""
        widget = HudWidget("my_widget", "My Widget")
        register_widget(widget)

        retrieved = get_widget("my_widget")
        assert retrieved == widget

    def test_get_widget_not_found(self):
        """Test getting non-existent widget returns None."""
        assert get_widget("nonexistent") is None

    def test_get_all_widgets(self):
        """Test getting all widgets."""
        widget1 = HudWidget("w1", "Widget 1")
        widget2 = HudWidget("w2", "Widget 2")
        register_widget(widget1)
        register_widget(widget2)

        all_widgets = get_all_widgets()
        assert len(all_widgets) == 2
        assert "w1" in all_widgets
        assert "w2" in all_widgets

    def test_get_all_widgets_returns_copy(self):
        """Test get_all_widgets returns a copy."""
        widget = HudWidget("test", "Test")
        register_widget(widget)

        all_widgets = get_all_widgets()
        all_widgets["new"] = HudWidget("new", "New")

        # Original registry should not be modified
        assert "new" not in _widget_registry


class TestHudWidgetPushMethods:
    """Tests for push update methods."""

    @pytest.mark.asyncio
    async def test_push_update(self):
        """Test push_update sends correct message."""
        widget = HudWidget("test", "Test")
        app = MagicMock()
        app.ws_send = AsyncMock()

        await widget.push_update(app, "conn123", "<div>New content</div>")

        app.ws_send.assert_called_once_with("conn123", {
            "type": "single_widget_update",
            "widget_id": "test",
            "html": "<div>New content</div>",
        })

    @pytest.mark.asyncio
    async def test_push_notification(self):
        """Test push_notification sends correct message."""
        widget = HudWidget("test", "Test")
        app = MagicMock()
        app.ws_send = AsyncMock()

        await widget.push_notification(app, "conn123", "Hello!", "success", 5000)

        app.ws_send.assert_called_once_with("conn123", {
            "type": "hud_notification",
            "message": "Hello!",
            "level": "success",
            "duration": 5000,
        })

    @pytest.mark.asyncio
    async def test_push_clipboard(self):
        """Test push_clipboard sends correct message."""
        widget = HudWidget("test", "Test")
        app = MagicMock()
        app.ws_send = AsyncMock()

        await widget.push_clipboard(app, "conn123", "copied text", "Copied!")

        app.ws_send.assert_called_once_with("conn123", {
            "type": "hud_clipboard",
            "text": "copied text",
            "notification": "Copied!",
        })

