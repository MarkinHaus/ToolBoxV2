"""
HUD Widget Base Class
=====================

Base class for creating interactive HUD widgets that can:
- Render HTML content
- Handle user actions (button clicks, form submissions)
- Push updates to the frontend via DomEngine (HTMX-style)
- Access user session data

Usage:
    from toolboxv2.utils.extras.hud_widget import HudWidget, register_widget

    class MyWidget(HudWidget):
        def __init__(self):
            super().__init__("my_widget", "My Widget", "🎮")

        async def render(self, app, request=None):
            return '<div>Content</div>'

        @action("do_something")
        async def do_something(self, app, payload, conn_id, request):
            return self.dom().swap("#target", "<p>Updated!</p>").to_response()

    register_widget(MyWidget())
"""

import html
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# =========================================================================
# DomBuilder - HTMX-Style Response Builder
# =========================================================================


class DomBuilder:
    """
    Builder for HTMX-style WebSocket DOM updates.

    Usage:
        return self.dom() \\
            .swap("#counter", "5") \\
            .append("#log", "<div>New entry</div>") \\
            .run_js("document.getElementById('input').focus()") \\
            .to_response()
    """

    def __init__(self):
        self.updates: List[Dict] = []

    def swap(
        self, selector: str, html_content: str, strategy: str = "innerHTML"
    ) -> "DomBuilder":
        """Replace content with specified strategy."""
        self.updates.append(
            {"selector": selector, "html": html_content, "swap": strategy}
        )
        return self

    def append(self, selector: str, html_content: str) -> "DomBuilder":
        """Append HTML at end of element (beforeend)."""
        return self.swap(selector, html_content, "beforeend")

    def prepend(self, selector: str, html_content: str) -> "DomBuilder":
        """Prepend HTML at start of element (afterbegin)."""
        return self.swap(selector, html_content, "afterbegin")

    def replace(self, selector: str, html_content: str) -> "DomBuilder":
        """Replace entire element (outerHTML)."""
        return self.swap(selector, html_content, "outerHTML")

    def delete(self, selector: str) -> "DomBuilder":
        """Remove element from DOM."""
        return self.swap(selector, "", "delete")

    def run_js(
        self, js_code: str, selector: str = "body", data: Dict = None
    ) -> "DomBuilder":
        """Execute JavaScript in frontend sandbox."""
        self.updates.append(
            {
                "selector": selector,
                "html": "",
                "swap": "none",
                "run_js": js_code,
                "data": data or {},
            }
        )
        return self

    def to_response(self) -> Dict:
        """Build final response dict for WebSocket."""
        if len(self.updates) == 0:
            return {}
        if len(self.updates) == 1:
            msg = self.updates[0].copy()
            msg["type"] = "dom_update"
            return msg
        return {"type": "batch_update", "updates": self.updates}


# =========================================================================
# Configuration
# =========================================================================


@dataclass
class HudWidgetConfig:
    """Configuration for a HUD widget."""

    widget_id: str
    title: str
    icon: str = "📦"
    description: str = ""
    requires_auth: bool = False
    min_level: int = 0


# =========================================================================
# Action Decorator
# =========================================================================


def action(name: str):
    """
    Decorator to register an action handler on a HudWidget method.

    Usage:
        class MyWidget(HudWidget):
            @action("click")
            async def on_click(self, app, payload, conn_id, request):
                return {"clicked": True}
    """

    def decorator(func: Callable):
        func._action_name = name
        return func

    return decorator


# =========================================================================
# HudWidget Base Class
# =========================================================================


class HudWidget:
    """
    Base class for interactive HUD widgets.
    """

    def __init__(
        self, widget_id: str, title: str, icon: str = "📦", mod_name: str = None
    ):
        self.widget_id = mod_name or widget_id
        self.title = title
        self.icon = icon
        self.actions: Dict[str, Callable] = {}
        self._config = HudWidgetConfig(widget_id=widget_id, title=title, icon=icon)
        self._register_actions()

    def _register_actions(self):
        """Auto-register methods decorated with @action."""
        for name in dir(self):
            if name.startswith("_"):
                continue
            method = getattr(self, name, None)
            if callable(method) and hasattr(method, "_action_name"):
                self.actions[method._action_name] = method

    def register_action(self, name: str):
        """
        Decorator to register an action handler (instance method style).
        """

        def decorator(func: Callable):
            self.actions[name] = func
            return func

        return decorator

    # =========================================================================
    # DomBuilder Factory
    # =========================================================================

    def dom(self) -> DomBuilder:
        """Create a new DomBuilder for HTMX-style responses."""
        return DomBuilder()

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def handle_action(
        self, app, action_name: str, payload: Dict[str, Any], conn_id: str, request=None
    ) -> Dict[str, Any]:
        """Handle an incoming action from the HUD frontend."""
        if action_name in self.actions:
            handler = self.actions[action_name]
            try:
                result = await handler(app, payload, conn_id, request)
                return result if isinstance(result, dict) else {"data": result}
            except Exception as e:
                return {"error": str(e)}
        return {"error": f"Unknown action: {action_name}"}

    async def render(self, app, request=None) -> str:
        """Render the widget HTML content. Override in subclasses."""
        return f'<div class="hud-widget-placeholder">Widget: {self.title}</div>'

    # =========================================================================
    # HTML Helpers - Classic (onclick)
    # =========================================================================

    def button(
        self,
        label: str,
        action_name: str,
        payload: Optional[Dict] = None,
        style: str = "primary",
        icon: str = "",
    ) -> str:
        """Generate button with onclick handler."""
        payload_json = html.escape(json.dumps(payload or {}))
        icon_html = f'<span class="btn-icon">{icon}</span>' if icon else ""
        btn_style = self._get_button_style(style)

        return f'''<button
            onclick="HUD.action('{self.widget_id}', '{action_name}', {payload_json})"
            style="{btn_style}padding:8px 12px;border-radius:6px;font-size:11px;cursor:pointer;display:inline-flex;align-items:center;gap:4px;"
        >{icon_html}{html.escape(label)}</button>'''

    # =========================================================================
    # HTML Helpers - WS Style (data-ws-action)
    # =========================================================================

    def button_ws(
        self,
        label: str,
        action_name: str,
        payload: Optional[Dict] = None,
        style: str = "primary",
        icon: str = "",
        include: str = "",
    ) -> str:
        """
        Generate button using data-ws-action system (cleaner HTML).

        Args:
            label: Button text
            action_name: Action to trigger
            payload: Static payload data
            style: primary/secondary/danger/success/icon
            icon: Optional icon emoji
            include: CSS selector for inputs to include (e.g. "#my-input")
        """
        payload_str = html.escape(json.dumps(payload or {}))
        icon_html = f'<span style="margin-right:4px;">{icon}</span>' if icon else ""
        btn_style = self._get_button_style(style)
        include_attr = f'data-ws-include="{include}"' if include else ""

        return f'''<button
            data-ws-action="{action_name}"
            data-ws-payload="{payload_str}"
            data-widget-id="{self.widget_id}"
            {include_attr}
            style="{btn_style}padding:8px 12px;border-radius:6px;font-size:11px;cursor:pointer;display:inline-flex;align-items:center;gap:4px;"
        >{icon_html}{html.escape(label)}</button>'''

    def input_ws(
        self,
        input_id: str,
        placeholder: str = "",
        input_type: str = "text",
        style: str = "",
    ) -> str:
        """Generate input field for use with data-ws-include."""
        default_style = "background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:white;padding:8px 12px;border-radius:6px;font-size:12px;"
        return f'''<input

            data-widget-id="{self.widget_id}"
            type="{input_type}"
            id="{input_id}"
            name="{input_id}"
            placeholder="{html.escape(placeholder)}"
            style="{default_style}{style}"
        />'''

    def _get_button_style(self, style: str) -> str:
        styles = {
            "primary": "background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;border:none;",
            "secondary": "background:rgba(255,255,255,0.1);color:white;border:1px solid rgba(255,255,255,0.2);",
            "danger": "background:#ef4444;color:white;border:none;",
            "success": "background:#22c55e;color:white;border:none;",
            "warning": "background:#f59e0b;color:white;border:none;",
            "icon": "background:transparent;color:white;border:none;padding:4px;font-size:16px;",
        }
        return styles.get(style, styles["secondary"])

    # =========================================================================
    # HTML Helpers - Other
    # =========================================================================

    def input_field(
        self, name: str, action_name: str, placeholder: str = "", input_type: str = "text"
    ) -> str:
        """Input field that triggers action on Enter."""
        return f'''<input
            type="{input_type}"
            id="hud-input-{name}"
            placeholder="{html.escape(placeholder)}"
            onkeypress="if(event.key==='Enter')HUD.action('{self.widget_id}','{action_name}',{{value:this.value}})"
            style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:white;padding:8px 12px;border-radius:6px;font-size:12px;width:100%;box-sizing:border-box;"
        />'''

    def select_field(
        self, name: str, action_name: str, options: list, selected: str = ""
    ) -> str:
        """Select dropdown that triggers action on change."""
        options_html = ""
        for opt in options:
            value = opt.get("value", opt.get("label", ""))
            label = opt.get("label", value)
            sel = "selected" if value == selected else ""
            options_html += f'<option value="{html.escape(str(value))}" {sel}>{html.escape(str(label))}</option>'

        return f"""<select
            id="hud-select-{name}"
            onchange="HUD.action('{self.widget_id}','{action_name}',{{value:this.value}})"
            style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:white;padding:8px 12px;border-radius:6px;font-size:12px;width:100%;box-sizing:border-box;"
        >{options_html}</select>"""

    def card(self, title: str, content: str, actions: str = "") -> str:
        """Card container."""
        actions_html = (
            f'<div style="margin-top:12px;display:flex;gap:8px;">{actions}</div>'
            if actions
            else ""
        )
        return f"""<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="font-weight:600;font-size:13px;margin-bottom:8px;color:#e2e8f0;">{html.escape(title)}</div>
            <div style="font-size:12px;color:#94a3b8;">{content}</div>
            {actions_html}
        </div>"""

    def list_item(
        self, title: str, subtitle: str = "", actions: str = "", icon: str = ""
    ) -> str:
        """List item with optional actions."""
        icon_html = (
            f'<span style="font-size:18px;margin-right:10px;">{icon}</span>'
            if icon
            else ""
        )
        subtitle_html = (
            f'<div style="font-size:10px;color:#64748b;margin-top:2px;">{html.escape(subtitle)}</div>'
            if subtitle
            else ""
        )
        return f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1);">
            <div style="display:flex;align-items:center;">
                {icon_html}
                <div>
                    <div style="font-size:12px;color:#e2e8f0;">{html.escape(title)}</div>
                    {subtitle_html}
                </div>
            </div>
            <div style="display:flex;gap:4px;">{actions}</div>
        </div>"""

    # =========================================================================
    # Push Methods
    # =========================================================================

    async def push_update(self, app, conn_id: str, html_content: str):
        """Push HTML update to connection."""
        await app.ws_send(
            conn_id,
            {
                "type": "single_widget_update",
                "widget_id": self.widget_id,
                "html": html_content,
            },
        )

    async def push_dom(self, app, conn_id: str, dom_builder: DomBuilder):
        """Push DomBuilder response to connection."""
        response = dom_builder.to_response()
        if response:
            await app.ws_send(conn_id, response)

    async def push_notification(
        self, app, conn_id: str, message: str, level: str = "info", duration: int = 3000
    ):
        """Push notification toast."""
        await app.ws_send(
            conn_id,
            {
                "type": "hud_notification",
                "message": message,
                "level": level,
                "duration": duration,
            },
        )

    async def push_clipboard(
        self, app, conn_id: str, text: str, notification: str = None
    ):
        """Copy text to clipboard."""
        await app.ws_send(
            conn_id,
            {
                "type": "hud_clipboard",
                "text": text,
                "notification": notification,
            },
        )

    # =========================================================================
    # Session Helpers
    # =========================================================================

    @staticmethod
    def get_user_id(request) -> str:
        if request is None:
            return ""
        session = getattr(request, "session", None)
        if session:
            return session.get("user_id", "") or session.get("user_name", "")
        return ""

    @staticmethod
    def get_user_level(request) -> int:
        if request is None:
            return 0
        session = getattr(request, "session", None)
        if session:
            return int(session.get("level", 0))
        return 0

    @staticmethod
    def is_authenticated(request) -> bool:
        if request is None:
            return False
        session = getattr(request, "session", None)
        if session:
            return session.get("validated", False) and not session.get("anonymous", True)
        return False


# =========================================================================
# Widget Registry
# =========================================================================

_widget_registry: Dict[str, HudWidget] = {}


def register_widget(widget: HudWidget):
    """Register a widget in the global registry."""
    print(f"Registering widget: {widget.widget_id}")
    _widget_registry[widget.widget_id] = widget


def get_widget(widget_id: str) -> Optional[HudWidget]:
    """Get a widget from the registry."""
    return _widget_registry.get(widget_id)


def get_all_widgets() -> Dict[str, HudWidget]:
    """Get all registered widgets."""
    return _widget_registry.copy()
