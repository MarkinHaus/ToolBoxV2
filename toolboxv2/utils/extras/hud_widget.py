"""
HUD Widget Base Class
=====================

Base class for creating interactive HUD widgets that can:
- Render HTML content
- Handle user actions (button clicks, form submissions)
- Push updates to the frontend
- Access user session data

Usage:
    from toolboxv2.utils.extras.hud_widget import HudWidget

    class PasswordWidget(HudWidget):
        def __init__(self):
            super().__init__("password_manager", "🔐 Passwords")

        async def render(self, app, request=None):
            return '<div>Password Manager Content</div>'

        async def handle_action(self, app, action, payload, conn_id, request=None):
            if action == "copy":
                return {"copied": True}
            return {"error": "Unknown action"}

    # Register the widget
    password_widget = PasswordWidget()

    @export(mod_name="MyMod", api=True, version="0.1.0")
    async def hud_password_manager(app, request=None):
        return await password_widget.render(app, request)

    @export(mod_name="MyMod", api=True, version="0.1.0")
    async def hud_action(app, action, payload, conn_id, request=None):
        return await password_widget.handle_action(app, action, payload, conn_id, request)
"""

import json
import html
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class HudWidgetConfig:
    """Configuration for a HUD widget."""
    widget_id: str
    title: str
    icon: str = "📦"
    description: str = ""
    requires_auth: bool = False
    min_level: int = 0


class HudWidget:
    """
    Base class for interactive HUD widgets.

    Provides:
    - Action registration via decorators
    - HTML helper methods for buttons, inputs, etc.
    - Session/request access for user-specific content
    - Push update capabilities
    """

    def __init__(self, widget_id: str, title: str, icon: str = "📦"):
        self.widget_id = widget_id
        self.title = title
        self.icon = icon
        self.actions: Dict[str, Callable] = {}
        self._config = HudWidgetConfig(
            widget_id=widget_id,
            title=title,
            icon=icon
        )

    def action(self, name: str):
        """
        Decorator to register an action handler.

        Usage:
            @widget.action("copy")
            async def copy_password(app, payload, conn_id, request):
                return {"copied": True}
        """
        def decorator(func: Callable):
            self.actions[name] = func
            return func
        return decorator

    async def handle_action(
        self,
        app,
        action: str,
        payload: Dict[str, Any],
        conn_id: str,
        request=None
    ) -> Dict[str, Any]:
        """
        Handle an incoming action from the HUD frontend.

        Args:
            app: ToolBoxV2 App instance
            action: The action name
            payload: Action payload data
            conn_id: WebSocket connection ID
            request: RequestData object with session info

        Returns:
            Dict with action result
        """
        if action in self.actions:
            handler = self.actions[action]
            try:
                result = await handler(app, payload, conn_id, request)
                return result if isinstance(result, dict) else {"data": result}
            except Exception as e:
                return {"error": str(e)}

        return {"error": f"Unknown action: {action}"}

    async def render(self, app, request=None) -> str:
        """
        Render the widget HTML content.
        Override this method in subclasses.

        Args:
            app: ToolBoxV2 App instance
            request: RequestData object with session info

        Returns:
            HTML string
        """
        return f'<div class="hud-widget-placeholder">Widget: {self.title}</div>'

    # =========================================================================
    # HTML Helper Methods
    # =========================================================================

    def button(
        self,
        label: str,
        action: str,
        payload: Optional[Dict] = None,
        style: str = "primary",
        icon: str = ""
    ) -> str:
        """
        Generate an action button HTML.

        Args:
            label: Button text
            action: Action name to trigger
            payload: Optional payload data
            style: Button style (primary, secondary, danger, icon)
            icon: Optional icon emoji

        Returns:
            HTML string for the button
        """
        payload_json = html.escape(json.dumps(payload or {}))
        icon_html = f'<span class="btn-icon">{icon}</span>' if icon else ''

        styles = {
            "primary": "background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border:none;",
            "secondary": "background:rgba(255,255,255,0.1);color:white;border:1px solid rgba(255,255,255,0.2);",
            "danger": "background:#ef4444;color:white;border:none;",
            "success": "background:#22c55e;color:white;border:none;",
            "icon": "background:transparent;color:white;border:none;padding:4px;font-size:16px;",
        }

        btn_style = styles.get(style, styles["secondary"])

        return f'''<button
            onclick="HUD.action('{self.widget_id}', '{action}', {payload_json})"
            style="{btn_style}padding:8px 12px;border-radius:6px;font-size:11px;cursor:pointer;display:inline-flex;align-items:center;gap:4px;"
        >{icon_html}{html.escape(label)}</button>'''

    def input_field(
        self,
        name: str,
        action: str,
        placeholder: str = "",
        input_type: str = "text"
    ) -> str:
        """
        Generate an input field that triggers action on Enter.

        Args:
            name: Input name/id
            action: Action to trigger on Enter
            placeholder: Placeholder text
            input_type: Input type (text, password, number, etc.)

        Returns:
            HTML string for the input
        """
        return f'''<input
            type="{input_type}"
            id="hud-input-{name}"
            placeholder="{html.escape(placeholder)}"
            onkeypress="if(event.key==='Enter')HUD.action('{self.widget_id}','{action}',{{value:this.value}})"
            style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:white;padding:8px 12px;border-radius:6px;font-size:12px;width:100%;box-sizing:border-box;"
        />'''

    def select_field(
        self,
        name: str,
        action: str,
        options: list,
        selected: str = ""
    ) -> str:
        """
        Generate a select dropdown that triggers action on change.

        Args:
            name: Select name/id
            action: Action to trigger on change
            options: List of options [{"value": "...", "label": "..."}]
            selected: Currently selected value

        Returns:
            HTML string for the select
        """
        options_html = ""
        for opt in options:
            value = opt.get("value", opt.get("label", ""))
            label = opt.get("label", value)
            sel = "selected" if value == selected else ""
            options_html += f'<option value="{html.escape(str(value))}" {sel}>{html.escape(str(label))}</option>'

        return f'''<select
            id="hud-select-{name}"
            onchange="HUD.action('{self.widget_id}','{action}',{{value:this.value}})"
            style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:white;padding:8px 12px;border-radius:6px;font-size:12px;width:100%;box-sizing:border-box;"
        >{options_html}</select>'''

    def card(self, title: str, content: str, actions: str = "") -> str:
        """
        Generate a card container.

        Args:
            title: Card title
            content: Card content HTML
            actions: Optional actions HTML (buttons, etc.)

        Returns:
            HTML string for the card
        """
        actions_html = f'<div style="margin-top:12px;display:flex;gap:8px;">{actions}</div>' if actions else ''

        return f'''<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:8px;color:#e2e8f0;">{html.escape(title)}</div>
            <div style="font-size:12px;color:#94a3b8;">{content}</div>
            {actions_html}
        </div>'''

    def list_item(
        self,
        title: str,
        subtitle: str = "",
        actions: str = "",
        icon: str = ""
    ) -> str:
        """
        Generate a list item with optional actions.

        Args:
            title: Item title
            subtitle: Optional subtitle
            actions: Optional actions HTML
            icon: Optional icon emoji

        Returns:
            HTML string for the list item
        """
        icon_html = f'<span style="font-size:18px;margin-right:10px;">{icon}</span>' if icon else ''
        subtitle_html = f'<div style="font-size:10px;color:#64748b;margin-top:2px;">{html.escape(subtitle)}</div>' if subtitle else ''

        return f'''<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
            <div style="display:flex;align-items:center;">
                {icon_html}
                <div>
                    <div style="font-size:12px;color:#e2e8f0;">{html.escape(title)}</div>
                    {subtitle_html}
                </div>
            </div>
            <div style="display:flex;gap:4px;">{actions}</div>
        </div>'''

    # =========================================================================
    # Push Update Methods
    # =========================================================================

    async def push_update(self, app, conn_id: str, html_content: str):
        """
        Push an HTML update to a specific connection.

        Args:
            app: ToolBoxV2 App instance
            conn_id: Target WebSocket connection ID
            html_content: New HTML content for the widget
        """
        await app.ws_send(conn_id, {
            "type": "single_widget_update",
            "widget_id": self.widget_id,
            "html": html_content,
        })

    async def push_notification(
        self,
        app,
        conn_id: str,
        message: str,
        level: str = "info",
        duration: int = 3000
    ):
        """
        Push a notification to a specific connection.

        Args:
            app: ToolBoxV2 App instance
            conn_id: Target WebSocket connection ID
            message: Notification message
            level: 'info', 'success', 'warning', 'error'
            duration: Duration in milliseconds
        """
        await app.ws_send(conn_id, {
            "type": "hud_notification",
            "message": message,
            "level": level,
            "duration": duration,
        })

    async def push_clipboard(self, app, conn_id: str, text: str, notification: str = None):
        """
        Push text to clipboard on the frontend.

        Args:
            app: ToolBoxV2 App instance
            conn_id: Target WebSocket connection ID
            text: Text to copy
            notification: Optional notification message after copy
        """
        await app.ws_send(conn_id, {
            "type": "hud_clipboard",
            "text": text,
            "notification": notification,
        })

    # =========================================================================
    # Session/User Helpers
    # =========================================================================

    @staticmethod
    def get_user_id(request) -> str:
        """Extract user ID from request."""
        if request is None:
            return ""
        session = getattr(request, 'session', None)
        if session:
            return session.get('user_id', '') or session.get('user_name', '')
        return ""

    @staticmethod
    def get_user_level(request) -> int:
        """Extract user level from request."""
        if request is None:
            return 0
        session = getattr(request, 'session', None)
        if session:
            return int(session.get('level', 0))
        return 0

    @staticmethod
    def is_authenticated(request) -> bool:
        """Check if user is authenticated."""
        if request is None:
            return False
        session = getattr(request, 'session', None)
        if session:
            return session.get('validated', False) and not session.get('anonymous', True)
        return False


# =========================================================================
# Widget Registry
# =========================================================================

_widget_registry: Dict[str, HudWidget] = {}


def register_widget(widget: HudWidget):
    """Register a widget in the global registry."""
    _widget_registry[widget.widget_id] = widget


def get_widget(widget_id: str) -> Optional[HudWidget]:
    """Get a widget from the registry."""
    return _widget_registry.get(widget_id)


def get_all_widgets() -> Dict[str, HudWidget]:
    """Get all registered widgets."""
    return _widget_registry.copy()

