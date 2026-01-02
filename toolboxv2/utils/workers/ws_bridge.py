#!/usr/bin/env python3
"""
ws_bridge.py - WebSocket Bridge for ToolBoxV2 HTTP Workers
===========================================================

Provides WebSocket communication methods for App instances that communicate
with WS workers via ZeroMQ.

Features:
    - ws_send(): Send message to specific WebSocket connection
    - ws_broadcast(): Broadcast to all connections in a channel
    - ws_broadcast_all(): Broadcast to ALL connected clients
    - send_notification(): Send Tauri native notifications to clients

Usage Example:
    from toolboxv2.utils.workers.ws_bridge import install_ws_bridge

    # Install bridge on app instance
    bridge = install_ws_bridge(app, event_manager, "my_worker")

    # Send message to specific connection
    await app.ws_send("conn-123", {"type": "update", "data": {...}})

    # Broadcast to channel
    await app.ws_broadcast("general", {"type": "message", "text": "Hello"})

    # Broadcast to all
    await app.ws_broadcast_all({"type": "announcement", "text": "..."})

Notification System:
    For sending notifications to Tauri/Web clients, use the NotificationSystem
    from toolboxv2.utils.extras.notification:

        from toolboxv2.utils.extras.notification import setup_web_notifications

        notifier = setup_web_notifications(bridge)
        notifier.notify_web("Title", "Message")

    See: toolboxv2/utils/extras/notification.py

Frontend Integration:
    The DesktopStatusBar component in tbjs maintains a persistent WebSocket
    connection and listens for notification messages. When received, it
    triggers Tauri's native notification API via tauriAPI.notify().

    See: toolboxv2/tbjs/src/ui/components/Desktop/index.js
"""

import json
import logging
from typing import Any, Dict, List, Optional

from toolboxv2.utils.workers.event_manager import (
    ZMQEventManager,
    Event,
    EventType,
    create_ws_send_event,
    create_ws_broadcast_event,
    create_ws_broadcast_all_event,
)

logger = logging.getLogger(__name__)


class ZMQWSBridge:
    """
    WebSocket bridge that communicates with WS workers via ZeroMQ.

    Provides the same interface as the old Rust bridge:
    - ws_send(conn_id, payload)
    - ws_broadcast(channel_id, payload, source_conn_id)

    Usage:
        bridge = ZMQWSBridge(event_manager, worker_id)
        app._zmq_ws_bridge = bridge  # Set on app instance

        # Then in app methods:
        await app.ws_send(conn_id, {"type": "message", "data": "hello"})
    """

    def __init__(self, event_manager: ZMQEventManager, worker_id: str):
        self._event_manager = event_manager
        self._worker_id = worker_id
        self._logger = logging.getLogger(f"{__name__}.{worker_id}")

    async def send_message(self, conn_id: str, payload: str | dict) -> bool:
        """
        Send message to a specific WebSocket connection.

        Args:
            conn_id: Target connection ID
            payload: JSON string or dict to send

        Returns:
            True if message was sent (doesn't guarantee delivery)
        """
        if not self._event_manager or not self._event_manager._running:
            self._logger.error("Cannot send WS message: event manager not running")
            return False

        try:
            event = create_ws_send_event(
                source=self._worker_id,
                conn_id=conn_id,
                payload=payload,
            )
            await self._event_manager.send_to_ws(event)
            return True

        except Exception as e:
            self._logger.error(f"Failed to send WS message to {conn_id}: {e}")
            return False

    async def broadcast_message(
        self,
        channel_id: str,
        payload: str | dict,
        source_conn_id: str = "",
    ) -> bool:
        """
        Broadcast message to all connections in a channel.

        Args:
            channel_id: Target channel/room ID
            payload: JSON string or dict to send
            source_conn_id: Optional - exclude this connection from broadcast

        Returns:
            True if broadcast was sent (doesn't guarantee delivery)
        """
        if not self._event_manager or not self._event_manager._running:
            self._logger.error("Cannot broadcast WS message: event manager not running")
            return False

        try:
            exclude = [source_conn_id] if source_conn_id else []
            event = create_ws_broadcast_event(
                source=self._worker_id,
                channel=channel_id,
                payload=payload,
                exclude_conn_ids=exclude,
            )
            await self._event_manager.send_to_ws(event)
            return True

        except Exception as e:
            self._logger.error(f"Failed to broadcast WS message to {channel_id}: {e}")
            return False

    async def broadcast_all(
        self,
        payload: str | dict,
        exclude_conn_ids: List[str] | None = None,
    ) -> bool:
        """
        Broadcast message to all connected WebSocket clients.

        Args:
            payload: JSON string or dict to send
            exclude_conn_ids: Optional list of connection IDs to exclude

        Returns:
            True if broadcast was sent
        """
        if not self._event_manager or not self._event_manager._running:
            self._logger.error("Cannot broadcast WS message: event manager not running")
            return False

        try:
            event = create_ws_broadcast_all_event(
                source=self._worker_id,
                payload=payload,
                exclude_conn_ids=exclude_conn_ids,
            )
            await self._event_manager.send_to_ws(event)
            return True

        except Exception as e:
            self._logger.error(f"Failed to broadcast WS message to all: {e}")
            return False

    async def send_notification(
        self,
        title: str,
        content: str,
        conn_id: str | None = None,
        channel: str | None = None,
        icon: str | None = None,
        level: str = "info",
    ) -> bool:
        """
        Send a notification to WebSocket clients that triggers Tauri native notifications.

        This method sends a specially formatted message that the frontend DesktopStatusBar
        component recognizes and displays as a native Tauri notification.

        Args:
            title: Notification title (required)
            content: Notification body/message (required)
            conn_id: Send to specific connection only (optional)
            channel: Broadcast to all connections in channel (optional)
            icon: Notification icon path (optional)
            level: Notification level - "info", "warning", "error", "success" (default: "info")

        Returns:
            True if notification was sent successfully

        Note:
            - If conn_id is provided, sends to that specific connection
            - If channel is provided, broadcasts to all connections in that channel
            - If neither is provided, broadcasts to ALL connected clients

        Example:
            # Send to specific user
            await bridge.send_notification(
                title="Task Complete",
                content="Your export is ready for download",
                conn_id="user-123"
            )

            # Broadcast to all users
            await bridge.send_notification(
                title="System Update",
                content="Server will restart in 5 minutes",
                level="warning"
            )
        """
        payload = {
            "type": "notification",
            "data": {
                "title": title,
                "content": content,
                "level": level,
            }
        }

        if icon:
            payload["data"]["icon"] = icon

        # Route to appropriate send method
        if conn_id:
            return await self.send_message(conn_id, payload)
        elif channel:
            return await self.broadcast_message(channel, payload)
        else:
            return await self.broadcast_all(payload)

    async def join_channel(self, conn_id: str, channel: str) -> bool:
        """Request a connection to join a channel."""
        try:
            event = Event(
                type=EventType.WS_JOIN_CHANNEL,
                source=self._worker_id,
                target="ws_worker",
                payload={"conn_id": conn_id, "channel": channel},
            )
            await self._event_manager.send_to_ws(event)
            return True
        except Exception as e:
            self._logger.error(f"Failed to join channel {channel}: {e}")
            return False

    async def leave_channel(self, conn_id: str, channel: str) -> bool:
        """Request a connection to leave a channel."""
        try:
            event = Event(
                type=EventType.WS_LEAVE_CHANNEL,
                source=self._worker_id,
                target="ws_worker",
                payload={"conn_id": conn_id, "channel": channel},
            )
            await self._event_manager.send_to_ws(event)
            return True
        except Exception as e:
            self._logger.error(f"Failed to leave channel {channel}: {e}")
            return False


def install_ws_bridge(app, event_manager: ZMQEventManager, worker_id: str):
    """
    Install WebSocket bridge methods on a ToolBoxV2 App instance.

    This replaces the old _set_rust_ws_bridge pattern with ZMQ-based communication.

    After calling this function, app.ws_send() and app.ws_broadcast() will work.

    Args:
        app: ToolBoxV2 App instance
        event_manager: Initialized ZMQEventManager
        worker_id: ID of this worker
    """
    bridge = ZMQWSBridge(event_manager, worker_id)
    app._zmq_ws_bridge = bridge

    # Override/add ws_send method
    async def ws_send(conn_id: str, payload: dict):
        """
        Send a message asynchronously to a single WebSocket connection.

        Args:
            conn_id: The unique ID of the target connection.
            payload: A dictionary that will be sent as JSON.
        """
        if app._zmq_ws_bridge is None:
            app.logger.error("Cannot send WebSocket message: ZMQ bridge is not initialized.")
            return False

        try:
            return await app._zmq_ws_bridge.send_message(conn_id, json.dumps(payload))
        except Exception as e:
            app.logger.error(f"Failed to send WebSocket message to {conn_id}: {e}", exc_info=True)
            return False

    # Override/add ws_broadcast method
    async def ws_broadcast(channel_id: str, payload: dict, source_conn_id: str = ""):
        """
        Send a message asynchronously to all clients in a channel/room.

        Args:
            channel_id: The channel to broadcast to.
            payload: A dictionary that will be sent as JSON.
            source_conn_id: Optional - the ID of the original connection to avoid echo.
        """
        if app._zmq_ws_bridge is None:
            app.logger.error("Cannot broadcast WebSocket message: ZMQ bridge is not initialized.")
            return False

        try:
            return await app._zmq_ws_bridge.broadcast_message(
                channel_id, json.dumps(payload), source_conn_id
            )
        except Exception as e:
            app.logger.error(f"Failed to broadcast WebSocket message to channel {channel_id}: {e}", exc_info=True)
            return False

    # Bind methods to app
    app.ws_send = ws_send
    app.ws_broadcast = ws_broadcast

    # Also expose join/leave channel
    app.ws_join_channel = bridge.join_channel
    app.ws_leave_channel = bridge.leave_channel
    app.ws_broadcast_all = bridge.broadcast_all

    # Expose notification method
    async def send_notification(
        title: str,
        content: str,
        conn_id: str | None = None,
        channel: str | None = None,
        icon: str | None = None,
        level: str = "info",
    ) -> bool:
        """
        Send a notification to WebSocket clients that triggers Tauri native notifications.

        Args:
            title: Notification title
            content: Notification body/message
            conn_id: Send to specific connection only (optional)
            channel: Broadcast to all connections in channel (optional)
            icon: Notification icon path (optional)
            level: Notification level - "info", "warning", "error", "success"

        Returns:
            True if notification was sent successfully
        """
        if app._zmq_ws_bridge is None:
            app.logger.error("Cannot send notification: ZMQ bridge is not initialized.")
            return False

        return await app._zmq_ws_bridge.send_notification(
            title=title,
            content=content,
            conn_id=conn_id,
            channel=channel,
            icon=icon,
            level=level,
        )

    app.send_notification = send_notification

    logger.info(f"WebSocket bridge installed for worker {worker_id}")
    return bridge
