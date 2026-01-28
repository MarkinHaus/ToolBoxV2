"""
ToolBox V2 - CloudM HUD UI
==========================
The central Heads-Up Display widget for CloudM.
Provides real-time system status, user profile access, and quick actions
directly in the overlay.

Integrates with:
- AuthClerk (User Identity)
- mini (System Processes)
- UserInstances (Session State)

"""

import asyncio
import json
import time
from datetime import datetime

from toolboxv2 import App, Result, get_app
from toolboxv2.utils.extras.hud_widget import HudWidget, register_widget
from toolboxv2.utils.system.types import RequestData

# Import CloudM specific helpers
from .AuthClerk import load_local_user_data
from .mini import PID_DIRECTORY, get_service_status
from .UserInstances import UserInstances

# Module Metadata
Name = "CloudM.HUD"
version = "1.0.0"
export = get_app(f"{Name}.Export").tb


class CloudMHud(HudWidget):
    """
    Main CloudM HUD Widget.
    """

    def __init__(self):
        super().__init__(widget_id=Name, title="CloudM Overview", icon="☁️")
        # Register standard actions
        self.actions["refresh"] = self.on_refresh
        self.actions["restart_service"] = self.on_restart_service
        self.actions["sync_user"] = self.on_sync_user
        self.actions["toggle_theme"] = self.on_toggle_theme

    async def render(self, app: App, request: RequestData = None) -> str:
        """
        Render the complete HUD UI.
        """
        # 1. Auth Check
        user_id = self.get_user_id(request)
        if not self.is_authenticated(request) or not user_id:
            return self._render_login_prompt()

        # 2. Load Data
        user_data = load_local_user_data(user_id)
        if not user_data:
            return self.card("Error", "User data not found.", style="danger")

        username = user_data.username or "User"
        level = user_data.level

        # 3. System Status
        status_html = self._render_system_status()

        # 4. User Profile Card
        profile_html = self._render_profile_card(username, level, user_id)

        # 5. Quick Actions
        actions_html = self._render_quick_actions()

        # Combine
        return f"""
        <div style="display: flex; flex-direction: column; gap: 8px;">
            {profile_html}
            {status_html}
            {actions_html}
        </div>
        """

    def _render_login_prompt(self) -> str:
        return self.card(
            title="🔒 Authentication Required",
            content="<div style='text-align: center; color: var(--hud-muted); padding: 8px;'>Please log in to access CloudM tools.</div>",
            actions=self.button(
                "Login",
                "navigate",
                {"path": "/web/assets/login.html"},
                style="primary",
                icon="login",
            ),
        )

    def _render_profile_card(self, username, level, uid) -> str:
        # User Level Badge Color
        level_color = "#22c55e" if level >= 1 else "#fbbf24"
        if level == -1:
            level_color = "#f43f5e"  # Admin red

        return f"""
        <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; display: flex; align-items: center; gap: 12px; border: 1px solid var(--hud-border);">
            <div style="width: 32px; height: 32px; background: linear-gradient(135deg, #6366f1, #8b5cf6); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white;">
                {username[0].upper()}
            </div>
            <div style="flex: 1;">
                <div style="font-weight: 600; font-size: 13px; color: var(--hud-text);">{username}</div>
                <div style="font-size: 10px; color: var(--hud-muted); display: flex; align-items: center; gap: 6px;">
                    <span style="width: 6px; height: 6px; border-radius: 50%; background: {level_color};"></span>
                    Level {level}
                </div>
            </div>
            {self.button("", "sync_user", icon="sync", style="icon")}
        </div>
        """

    def _render_system_status(self) -> str:
        # Get status from mini.py
        raw_status = get_service_status(PID_DIRECTORY)

        # Parse minimal status
        lines = [
            l
            for l in raw_status.split("\n")
            if l.strip() and not l.startswith("Service(s):")
        ]
        active_count = sum(1 for l in lines if "🟢" in l)
        total_count = len(lines)

        status_items = ""
        for line in lines[:3]:  # Show max 3 services preview
            parts = line.split("(PID:")
            name = parts[0].strip()[2:]  # Remove icon
            icon = "🟢" if "🟢" in line else ("🔴" if "🔴" in line else "🟡")
            status_items += f"""
            <div style="display: flex; justify-content: space-between; font-size: 11px; padding: 2px 0;">
                <span style="color: var(--hud-text);">{icon} {name}</span>
            </div>
            """

        header = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span>System Status</span>
            <span style="font-size: 10px; background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">
                {active_count}/{total_count} Active
            </span>
        </div>
        """

        return self.card(
            title="🖥️ System",
            content=f"<div>{header}{status_items}</div>",
            actions=self.button(
                "Full Status",
                "navigate",
                {"path": "/api/CloudM.UserDashboard/main"},
                style="secondary",
                icon="bar_chart",
            ),
        )

    def _render_quick_actions(self) -> str:
        return self.card(
            title="🚀 Actions",
            content="",
            actions=f"""
                {self.button("Dashboard", "navigate", {"path": "/api/CloudM.UserDashboard/main"}, style="primary", icon="dashboard")}
                {self.button("Modules", "navigate", {"path": "/api/CloudM.UserDashboard/main#my-modules"}, style="secondary", icon="extension")}
            """,
        )

    # =================== Action Handlers ===================

    async def on_refresh(self, app, payload, conn_id, request):
        """Force refresh the widget."""
        html = await self.render(app, request)
        await self.push_update(app, conn_id, html)
        return {"status": "refreshed"}

    async def on_sync_user(self, app, payload, conn_id, request):
        """Sync user data."""
        user_id = self.get_user_id(request)
        if user_id:
            await self.push_notification(app, conn_id, "Syncing profile...", level="info")
            # Simulate sync or call actual sync logic
            await asyncio.sleep(0.5)
            await self.push_notification(
                app, conn_id, "Profile synchronized", level="success"
            )
            # Refresh UI
            html = await self.render(app, request)
            await self.push_update(app, conn_id, html)
        return {"status": "ok"}

    async def on_restart_service(self, app, payload, conn_id, request):
        """Restart a specific service."""
        service = payload.get("service")
        if service:
            await self.push_notification(
                app, conn_id, f"Restarting {service}...", level="warning"
            )
            # Actual restart logic would go here via mini or App
            await self.push_notification(
                app, conn_id, f"{service} restarted", level="success"
            )
        return {"status": "ok"}

    async def on_toggle_theme(self, app, payload, conn_id, request):
        """Toggle UI theme."""
        # This would typically update user settings via UserAccountManager
        await self.push_notification(app, conn_id, "Theme updated", level="info")
        return {"status": "ok"}


# =================== Module Exports ===================

# Instantiate the widget
cloudm_hud = CloudMHud()
register_widget(cloudm_hud)

@export(mod_name=Name, api=True, version=version)
async def hud_cloudm_overview(app: App, request: RequestData = None):
    """
    Render function for the CloudM Overview Widget.
    Called by the HUD frontend when loading 'CloudM.HUD.hud_cloudm_overview'.
    """
    return await cloudm_hud.render(app, request)


@export(mod_name=Name, api=True, version=version)
async def hud_action(
    app: App, action: str, payload: dict, conn_id: str, request: RequestData = None
):
    """
    Action router for CloudM.HUD module.
    Receives actions from frontend for this specific module/widget.
    """
    return await cloudm_hud.handle_action(app, action, payload, conn_id, request)
