#!/usr/bin/env python3
"""
ws_worker.py - High-Performance WebSocket Worker for ToolBoxV2

Designed for maximum connections with minimal processing.
All business logic delegated to HTTP workers via ZeroMQ.

Features:
- Minimal processing overhead
- ZeroMQ integration for message forwarding
- Channel/room subscriptions
- Connection state management
- Heartbeat/ping-pong
- Direct PULL socket for HTTP->WS messages (bypass broker for lower latency)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

try:
    from toolboxv2 import get_logger
except ImportError:
    try:
        from ..system.tb_logger import get_logger
    except ImportError:

        def get_logger():
            import logging

            return logging.getLogger()

try:
    import websockets
    # Try new asyncio API first (websockets >= 13.0)
    try:
        from websockets.asyncio.server import serve as ws_serve
        from websockets.exceptions import ConnectionClosed
        WEBSOCKETS_NEW_API = True
    except ImportError:
        # Fall back to legacy API (websockets < 13.0)
        from websockets.server import serve as ws_serve
        from websockets.exceptions import ConnectionClosed
        WEBSOCKETS_NEW_API = False

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WEBSOCKETS_NEW_API = False

try:
    import zmq
    import zmq.asyncio

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

from toolboxv2.utils.workers.event_manager import (
    ZMQEventManager,
    Event,
    EventType,
    create_ws_send_event,
    create_ws_broadcast_event,
)
from toolboxv2.utils.workers.session import SignedCookieSession, SessionData

logger = get_logger()


# ============================================================================
# Connection Management
# ============================================================================


@dataclass
class WSConnection:
    """WebSocket connection state."""

    conn_id: str
    websocket: Any
    user_id: str = ""
    session_id: str = ""
    level: int = 0  # User access level (0=not logged in, 1=logged in, -1=admin)
    clerk_user_id: str = ""  # Clerk user ID for authentication
    channels: Set[str] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)
    authenticated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        """Check if connection is still open."""
        return self.websocket.open if hasattr(self.websocket, "open") else True


class ConnectionManager:
    """
    Manages WebSocket connections efficiently.

    Uses weak references where possible to avoid memory leaks.
    Optimized for high connection counts.
    """

    def __init__(self, max_connections: int = 10000):
        self.max_connections = max_connections
        self._connections: Dict[str, WSConnection] = {}
        self._user_connections: Dict[str, Set[str]] = {}  # user_id -> conn_ids
        self._channel_connections: Dict[str, Set[str]] = {}  # channel -> conn_ids
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def add(self, conn: WSConnection) -> bool:
        """Add a connection."""
        async with self._lock:
            if len(self._connections) >= self.max_connections:
                logger.warning(f"Max connections reached: {self.max_connections}")
                return False

            self._connections[conn.conn_id] = conn
            return True

    async def remove(self, conn_id: str) -> Optional[WSConnection]:
        """Remove a connection."""
        async with self._lock:
            conn = self._connections.pop(conn_id, None)
            if conn:
                # Clean up user mapping
                if conn.user_id and conn.user_id in self._user_connections:
                    self._user_connections[conn.user_id].discard(conn_id)
                    if not self._user_connections[conn.user_id]:
                        del self._user_connections[conn.user_id]

                # Clean up channel mappings
                for channel in conn.channels:
                    if channel in self._channel_connections:
                        self._channel_connections[channel].discard(conn_id)
                        if not self._channel_connections[channel]:
                            del self._channel_connections[channel]

            return conn

    def get(self, conn_id: str) -> Optional[WSConnection]:
        """Get a connection by ID."""
        return self._connections.get(conn_id)

    async def authenticate(self, conn_id: str, user_id: str, session_id: str):
        """Mark connection as authenticated."""
        async with self._lock:
            conn = self._connections.get(conn_id)
            if conn:
                conn.authenticated = True
                conn.user_id = user_id
                conn.session_id = session_id

                # Add to user mapping
                if user_id not in self._user_connections:
                    self._user_connections[user_id] = set()
                self._user_connections[user_id].add(conn_id)

    async def join_channel(self, conn_id: str, channel: str):
        """Add connection to channel."""
        async with self._lock:
            conn = self._connections.get(conn_id)
            if conn:
                conn.channels.add(channel)

                if channel not in self._channel_connections:
                    self._channel_connections[channel] = set()
                self._channel_connections[channel].add(conn_id)

    async def leave_channel(self, conn_id: str, channel: str):
        """Remove connection from channel."""
        async with self._lock:
            conn = self._connections.get(conn_id)
            if conn:
                conn.channels.discard(channel)

                if channel in self._channel_connections:
                    self._channel_connections[channel].discard(conn_id)
                    if not self._channel_connections[channel]:
                        del self._channel_connections[channel]

    def get_channel_connections(self, channel: str) -> List[WSConnection]:
        """Get all connections in a channel."""
        conn_ids = self._channel_connections.get(channel, set())
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    def get_user_connections(self, user_id: str) -> List[WSConnection]:
        """Get all connections for a user."""
        conn_ids = self._user_connections.get(user_id, set())
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    def get_all_connections(self) -> List[WSConnection]:
        """Get all connections."""
        return list(self._connections.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self._connections),
            "authenticated_connections": sum(
                1 for c in self._connections.values() if c.authenticated
            ),
            "unique_users": len(self._user_connections),
            "active_channels": len(self._channel_connections),
            "max_connections": self.max_connections,
        }


# ============================================================================
# WebSocket Worker
# ============================================================================


class WSWorker:
    """
    High-performance WebSocket worker.

    Minimal processing - forwards messages via ZeroMQ.
    Designed for maximum concurrent connections.
    """

    def __init__(
        self,
        worker_id: str,
        config,
    ):
        self.worker_id = worker_id
        self.config = config
        self._conn_manager = ConnectionManager(config.ws_worker.max_connections)
        self._event_manager: Optional[ZMQEventManager] = None
        self._running = False
        self._server = None

        # Direct PULL socket for HTTP->WS messages (lower latency)
        self._direct_pull_socket = None
        self._direct_ctx = None

        # Metrics
        self._metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "connections_total": 0,
            "errors": 0,
            "direct_messages_received": 0,
        }

    def _process_request_new_api(self, connection, request):
        """Process HTTP request before WebSocket handshake (new API >= 14.0).

        This handles non-WebSocket requests like health checks.
        Returns None to proceed with WebSocket handshake, or a Response to send.

        Note: This is a regular function, not a coroutine, in the new API.
        """
        from http import HTTPStatus
        path = request.path if hasattr(request, 'path') else "/"

        # Handle health check requests (non-WebSocket)
        if path == "/health":
            return connection.respond(HTTPStatus.OK, "OK\n")

        # For all other paths, proceed with WebSocket handshake
        return None

    async def _process_request_legacy(self, path, request_headers):
        """Process HTTP request before WebSocket handshake (legacy API < 13.0).

        This handles non-WebSocket requests like health checks.
        Returns None to proceed with WebSocket handshake, or a tuple
        (status, headers, body) to send an HTTP response instead.

        Note: This is a coroutine in the legacy API.
        """
        from http import HTTPStatus
        # Handle health check requests (non-WebSocket)
        if path == "/health":
            return (
                HTTPStatus.OK,
                [("Content-Type", "text/plain")],
                b"OK",
            )

        # For all other paths, proceed with WebSocket handshake
        return None

    async def start(self):
        """Start the WebSocket worker."""
        logger.info(f"Starting WS worker {self.worker_id}")

        # Initialize ZMQ event manager
        await self._init_event_manager()

        # Initialize direct PULL socket for HTTP->WS messages
        await self._init_direct_pull()

        # Start WebSocket server
        host = self.config.ws_worker.host
        port = self.config.ws_worker.port

        self._running = True

        # Start background tasks
        asyncio.create_task(self._ping_loop())
        asyncio.create_task(self._direct_pull_loop())

        # Build serve kwargs - new API doesn't support 'compression' the same way
        serve_kwargs = {
            "ping_interval": self.config.ws_worker.ping_interval,
            "ping_timeout": self.config.ws_worker.ping_timeout,
            "max_size": self.config.ws_worker.max_message_size,
        }

        # Select handler and process_request based on API version
        if WEBSOCKETS_NEW_API:
            handler = self._handle_connection_new_api
            serve_kwargs["process_request"] = self._process_request_new_api
            logger.info(f"Using new websockets API (>= 13.0)")
        else:
            handler = self._handle_connection_legacy
            serve_kwargs["process_request"] = self._process_request_legacy
            serve_kwargs["compression"] = "deflate" if self.config.ws_worker.compression else None
            logger.info(f"Using legacy websockets API")

        # Start server
        self._server = await ws_serve(
            handler,
            host,
            port,
            **serve_kwargs,
        )

        logger.info(f"WS worker listening on {host}:{port}")

        # Keep running - use serve_forever for new API, wait_closed for legacy
        if WEBSOCKETS_NEW_API:
            await self._server.serve_forever()
        else:
            await self._server.wait_closed()

    async def stop(self):
        """Stop the WebSocket worker."""
        logger.info(f"Stopping WS worker {self.worker_id}")
        self._running = False

        # Close all connections
        for conn in self._conn_manager.get_all_connections():
            try:
                await conn.websocket.close(1001, "Server shutting down")
            except Exception:
                pass

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop event manager
        if self._event_manager:
            await self._event_manager.stop()

        # Close direct PULL socket
        if self._direct_pull_socket:
            self._direct_pull_socket.close()
        if self._direct_ctx:
            self._direct_ctx.term()

        logger.info(f"WS worker {self.worker_id} stopped")

    async def _init_event_manager(self):
        """Initialize ZeroMQ event manager."""
        self._event_manager = ZMQEventManager(
            worker_id=self.worker_id,
            pub_endpoint=self.config.zmq.pub_endpoint,
            sub_endpoint=self.config.zmq.sub_endpoint,
            req_endpoint=self.config.zmq.req_endpoint,
            rep_endpoint=self.config.zmq.rep_endpoint,
            http_to_ws_endpoint=self.config.zmq.http_to_ws_endpoint,
            is_broker=False,
        )
        await self._event_manager.start()

        # Subscribe to ws_worker channel for targeted messages
        self._event_manager.subscribe("ws_worker")

        # Register event handlers
        self._register_event_handlers()

    async def _init_direct_pull(self):
        """Initialize direct PULL socket for HTTP->WS messages."""
        if not ZMQ_AVAILABLE:
            logger.warning("ZMQ not available, direct PULL disabled")
            return

        try:
            self._direct_ctx = zmq.asyncio.Context()
            self._direct_pull_socket = self._direct_ctx.socket(zmq.PULL)
            self._direct_pull_socket.setsockopt(zmq.RCVHWM, 10000)

            # Bind to a worker-specific endpoint
            # This allows HTTP workers to PUSH directly to this WS worker
            direct_endpoint = self.config.zmq.http_to_ws_endpoint.replace(
                "5558", f"555{hash(self.worker_id) % 10 + 8}"
            )
            # Actually, let's connect to the broker's endpoint instead
            # The broker will forward messages from HTTP workers
            self._direct_pull_socket.connect(self.config.zmq.http_to_ws_endpoint)

            logger.info(f"Direct PULL socket connected to {self.config.zmq.http_to_ws_endpoint}")
        except Exception as e:
            logger.error(f"Failed to init direct PULL socket: {e}")
            self._direct_pull_socket = None

    async def _direct_pull_loop(self):
        """Process messages from direct PULL socket."""
        if not self._direct_pull_socket:
            return

        while self._running:
            try:
                # Non-blocking receive with timeout
                if self._direct_pull_socket.poll(100, zmq.POLLIN):
                    msg = await self._direct_pull_socket.recv()
                    self._metrics["direct_messages_received"] += 1

                    try:
                        event = Event.from_bytes(msg)
                        await self._handle_direct_event(event)
                    except Exception as e:
                        logger.error(f"Failed to parse direct event: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Direct PULL loop error: {e}")
                await asyncio.sleep(0.1)

    async def _handle_direct_event(self, event: Event):
        """Handle event received via direct PULL socket."""
        if event.type == EventType.WS_SEND:
            conn_id = event.payload.get("conn_id")
            data = event.payload.get("data")

            if conn_id and data:
                conn = self._conn_manager.get(conn_id)
                if conn and conn.is_alive:
                    try:
                        await conn.websocket.send(data)
                        self._metrics["messages_sent"] += 1
                    except Exception as e:
                        logger.debug(f"Send failed to {conn_id}: {e}")

        elif event.type == EventType.WS_BROADCAST_CHANNEL:
            channel = event.payload.get("channel")
            data = event.payload.get("data")
            exclude = set(event.payload.get("exclude", []))

            if channel and data:
                connections = self._conn_manager.get_channel_connections(channel)
                await self._broadcast_to_connections(connections, data, exclude)

        elif event.type == EventType.WS_BROADCAST_ALL:
            data = event.payload.get("data")
            exclude = set(event.payload.get("exclude", []))

            if data:
                connections = self._conn_manager.get_all_connections()
                await self._broadcast_to_connections(connections, data, exclude)

        elif event.type == EventType.WS_JOIN_CHANNEL:
            conn_id = event.payload.get("conn_id")
            channel = event.payload.get("channel")
            if conn_id and channel:
                await self._conn_manager.join_channel(conn_id, channel)

        elif event.type == EventType.WS_LEAVE_CHANNEL:
            conn_id = event.payload.get("conn_id")
            channel = event.payload.get("channel")
            if conn_id and channel:
                await self._conn_manager.leave_channel(conn_id, channel)

    def _register_event_handlers(self):
        """Register handlers for events from HTTP workers (via PUB/SUB)."""

        @self._event_manager.on(EventType.WS_SEND)
        async def handle_ws_send(event: Event):
            """Send message to specific connection."""
            conn_id = event.payload.get("conn_id")
            data = event.payload.get("data")

            if not conn_id or not data:
                return

            conn = self._conn_manager.get(conn_id)
            if conn and conn.is_alive:
                try:
                    await conn.websocket.send(data)
                    self._metrics["messages_sent"] += 1
                except Exception as e:
                    logger.debug(f"Send failed to {conn_id}: {e}")

        @self._event_manager.on(EventType.WS_BROADCAST_CHANNEL)
        async def handle_ws_broadcast_channel(event: Event):
            """Broadcast to all connections in a channel."""
            channel = event.payload.get("channel")
            data = event.payload.get("data")
            exclude = set(event.payload.get("exclude", []))

            if not channel or not data:
                return

            connections = self._conn_manager.get_channel_connections(channel)
            await self._broadcast_to_connections(connections, data, exclude)

        @self._event_manager.on(EventType.WS_BROADCAST_ALL)
        async def handle_ws_broadcast_all(event: Event):
            """Broadcast to all connections."""
            data = event.payload.get("data")
            exclude = set(event.payload.get("exclude", []))

            if not data:
                return

            connections = self._conn_manager.get_all_connections()
            await self._broadcast_to_connections(connections, data, exclude)

        @self._event_manager.on(EventType.WS_JOIN_CHANNEL)
        async def handle_ws_join_channel(event: Event):
            """Add connection to channel."""
            conn_id = event.payload.get("conn_id")
            channel = event.payload.get("channel")

            if conn_id and channel:
                await self._conn_manager.join_channel(conn_id, channel)

        @self._event_manager.on(EventType.WS_LEAVE_CHANNEL)
        async def handle_ws_leave_channel(event: Event):
            """Remove connection from channel."""
            conn_id = event.payload.get("conn_id")
            channel = event.payload.get("channel")

            if conn_id and channel:
                await self._conn_manager.leave_channel(conn_id, channel)

        @self._event_manager.on(EventType.SHUTDOWN)
        async def handle_shutdown(event: Event):
            """Handle shutdown request."""
            logger.info("Shutdown event received")
            await self.stop()

        @self._event_manager.on(EventType.HEALTH_CHECK)
        async def handle_health_check(event: Event):
            """Respond to health check."""
            await self._event_manager.publish(
                Event(
                    type=EventType.WORKER_HEALTH,
                    source=self.worker_id,
                    target=event.source,
                    payload=self.get_stats(),
                    correlation_id=event.correlation_id,
                )
            )

    async def _broadcast_to_connections(
        self,
        connections: List[WSConnection],
        data: str,
        exclude: Set[str],
    ):
        """Broadcast data to multiple connections efficiently."""
        tasks = []
        for conn in connections:
            if conn.conn_id not in exclude and conn.is_alive:
                tasks.append(self._safe_send(conn, data))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_send(self, conn: WSConnection, data: str):
        """Send data with error handling."""
        try:
            await conn.websocket.send(data)
            self._metrics["messages_sent"] += 1
        except Exception as e:
            logger.debug(f"Send failed to {conn.conn_id}: {e}")

    async def _safe_publish(self, event: Event):
        """Safely publish an event, ignoring errors if event manager is not ready."""
        try:
            if self._event_manager and self._event_manager._running:
                logger.info(f"[WS] Publishing event: type={event.type}, source={event.source}, target={event.target}")
                await self._event_manager.publish(event)
                logger.info(f"[WS] Event published successfully: {event.type}")
            else:
                logger.warning(f"[WS] Event manager not ready: manager={self._event_manager is not None}, running={getattr(self._event_manager, '_running', False) if self._event_manager else False}")
        except Exception as e:
            logger.error(f"[WS] Event publish failed: {e}", exc_info=True)

    def _extract_session_from_websocket(self, websocket) -> Optional[SessionData]:
        """Extract session data from WebSocket connection cookies.

        This allows WebSocket connections to inherit the user's authentication
        state from their HTTP session cookie.
        """
        try:
            # Get cookie header from websocket request
            cookie_header = None

            # New API (websockets >= 13.0)
            if hasattr(websocket, 'request') and websocket.request:
                headers = getattr(websocket.request, 'headers', None)
                if headers:
                    cookie_header = headers.get('Cookie') or headers.get('cookie')

            # Legacy API
            if not cookie_header and hasattr(websocket, 'request_headers'):
                cookie_header = websocket.request_headers.get('Cookie') or websocket.request_headers.get('cookie')

            if not cookie_header:
                logger.debug("[WS] No cookie header found in WebSocket request")
                return None

            # Use the cookie secret from config
            secret = None
            if hasattr(self.config, 'session') and self.config.session:
                secret = getattr(self.config.session, 'cookie_secret', None)

            if not secret:
                # Try environment variable
                secret = os.environ.get('TB_COOKIE_SECRET')

            if not secret or len(secret) < 32:
                logger.debug("[WS] No valid cookie secret configured, cannot verify session")
                return None

            # Parse the session cookie
            session_handler = SignedCookieSession(secret=secret)
            session = session_handler.get_from_cookie_header(cookie_header)

            if session:
                logger.info(f"[WS] Extracted session: user_id={session.user_id}, level={session.level}, authenticated={session.is_authenticated}")
                return session
            else:
                logger.debug("[WS] No valid session found in cookie")
                return None

        except Exception as e:
            logger.warning(f"[WS] Failed to extract session from cookie: {e}")
            return None

    async def _handle_connection_impl(self, websocket, path: str):
        """Internal connection handler implementation."""
        conn_id = str(uuid.uuid4())

        # Extract session from cookie for authentication
        session_data = self._extract_session_from_websocket(websocket)

        conn = WSConnection(
            conn_id=conn_id,
            websocket=websocket,
            user_id=session_data.user_id if session_data else "",
            session_id=session_data.session_id if session_data else "",
            level=session_data.level if session_data else 0,
            clerk_user_id=session_data.clerk_user_id if session_data else "",
            authenticated=session_data.is_authenticated if session_data else False,
            metadata={"path": path},
        )

        logger.info(f"[WS] Connection {conn_id}: user_id={conn.user_id}, clerk_user_id={conn.clerk_user_id}, level={conn.level}, authenticated={conn.authenticated}")

        # Check connection limit
        if not await self._conn_manager.add(conn):
            await websocket.close(1013, "Server overloaded")
            return

        self._metrics["connections_total"] += 1

        logger.debug(
            f"New connection: {conn_id} path={path} (total: {self._conn_manager.connection_count})"
        )

        # Publish connect event (non-blocking, errors ignored)
        await self._safe_publish(
            Event(
                type=EventType.WS_CONNECT,
                source=self.worker_id,
                target="*",
                payload={
                    "conn_id": conn_id,
                    "path": path,
                    "user_id": conn.user_id,
                    "session_id": conn.session_id,
                    "level": conn.level,
                    "clerk_user_id": conn.clerk_user_id,
                    "authenticated": conn.authenticated,
                },
            )
        )

        try:
            # Send connection ID to client
            await websocket.send(
                json.dumps(
                    {
                        "type": "connected",
                        "conn_id": conn_id,
                    }
                )
            )
            logger.info(f"[WS] Sent 'connected' message to {conn_id}")

            # Message loop - MINIMAL PROCESSING
            logger.info(f"[WS] Starting message loop for {conn_id} on path {path}")
            logger.info(f"[WS] WebSocket state: open={getattr(websocket, 'open', 'unknown')}, closed={getattr(websocket, 'closed', 'unknown')}")

            message_count = 0
            async for message in websocket:
                message_count += 1
                self._metrics["messages_received"] += 1
                logger.info(f"[WS] Message #{message_count} received from {conn_id}: {message[:200] if len(message) > 200 else message}")

                # Forward ALL messages to HTTP workers via ZeroMQ
                # NO processing here - just forward
                event = Event(
                    type=EventType.WS_MESSAGE,
                    source=self.worker_id,
                    target="*",
                    payload={
                        "conn_id": conn_id,
                        "user_id": conn.user_id,
                        "session_id": conn.session_id,
                        "level": conn.level,
                        "clerk_user_id": conn.clerk_user_id,
                        "authenticated": conn.authenticated,
                        "data": message,
                        "path": path,
                    },
                )
                logger.info(f"[WS] Publishing WS_MESSAGE event for {conn_id}")
                await self._safe_publish(event)
                logger.info(f"[WS] Message #{message_count} forwarded for {conn_id}")

            logger.info(f"[WS] Message loop ended for {conn_id} after {message_count} messages")

        except ConnectionClosed as e:
            logger.debug(f"Connection closed: {conn_id} ({e.code})")
        except Exception as e:
            logger.error(f"Connection error: {conn_id}: {e}")
            self._metrics["errors"] += 1
        finally:
            # Clean up
            await self._conn_manager.remove(conn_id)

            # Publish disconnect event (non-blocking, errors ignored)
            await self._safe_publish(
                Event(
                    type=EventType.WS_DISCONNECT,
                    source=self.worker_id,
                    target="*",
                    payload={
                        "conn_id": conn_id,
                        "user_id": conn.user_id,
                    },
                )
            )

            logger.debug(
                f"Connection removed: {conn_id} (total: {self._conn_manager.connection_count})"
            )

    async def _handle_connection_new_api(self, websocket):
        """Handler for new websockets API (>= 13.0) - single argument."""
        # Extract path from request
        if hasattr(websocket, 'request') and websocket.request:
            path = websocket.request.path
        elif hasattr(websocket, 'path'):
            path = websocket.path
        else:
            path = "/"
        await self._handle_connection_impl(websocket, path)

    async def _handle_connection_legacy(self, websocket, path: str):
        """Handler for legacy websockets API (< 13.0) - two arguments."""
        await self._handle_connection_impl(websocket, path)

    async def _ping_loop(self):
        """Periodic ping to check dead connections."""
        while self._running:
            await asyncio.sleep(30)

            # Check for dead connections
            dead_connections = []
            for conn in self._conn_manager.get_all_connections():
                if not conn.is_alive:
                    dead_connections.append(conn.conn_id)

            # Remove dead connections
            for conn_id in dead_connections:
                await self._conn_manager.remove(conn_id)

            if dead_connections:
                logger.debug(f"Removed {len(dead_connections)} dead connections")

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self._conn_manager.get_stats()
        stats.update(
            {
                "worker_id": self.worker_id,
                "pid": os.getpid(),
                "messages_received": self._metrics["messages_received"],
                "messages_sent": self._metrics["messages_sent"],
                "connections_total": self._metrics["connections_total"],
                "direct_messages_received": self._metrics["direct_messages_received"],
                "errors": self._metrics["errors"],
            }
        )
        return stats

    async def run(self):
        """Run the WebSocket worker (blocking).

        This method can be called:
        - With asyncio.run() for standalone execution
        - Within an existing event loop as a coroutine
        """
        global logger
        from ..system.getting_and_closing_app import get_app
        print("WS_WORKER:: ",get_app().set_logger(True, self.worker_id))
        get_logger().info("WS_WORKER:: ")
        logger = get_logger()
        # Signal handlers (Unix only)
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()

            def signal_handler():
                loop.create_task(self.stop())

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, signal_handler)
                except NotImplementedError:
                    pass

        try:
            print("Starting WS worker...")
            await self.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            await self.stop()
        except Exception as e:
            logger.error(f"WS worker error: {e}")
            import traceback
            traceback.print_exc()
            await self.stop()

    def run_sync(self):
        """Run the WebSocket worker synchronously (creates new event loop).

        Use this method when calling from a non-async context.
        For async contexts, use `await worker.run()` instead.
        """
        # Windows: Use SelectorEventLoop for ZMQ compatibility
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"WS worker error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# CLI Entry Point
# ============================================================================


async def main():
    if not WEBSOCKETS_AVAILABLE:
        print("ERROR: websockets package required: pip install websockets")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(description="ToolBoxV2 WebSocket Worker", prog="tb ws_worker")
    parser.add_argument("-c", "--config", help="Config file path")
    parser.add_argument("-H", "--host", help="Host to bind")
    parser.add_argument("-p", "--port", type=int, help="Port to bind")
    parser.add_argument("-w", "--worker-id", help="Worker ID")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    from toolboxv2.utils.workers.config import load_config

    config = load_config(args.config)

    # Override from args
    if args.host:
        config.ws_worker.host = args.host
    if args.port:
        config.ws_worker.port = args.port

    # Worker ID
    worker_id = args.worker_id or f"ws_{os.getpid()}"

    # Run worker
    worker = WSWorker(worker_id, config)
    await worker.run()


if __name__ == "__main__":
    # Windows: Use SelectorEventLoop for ZMQ compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
