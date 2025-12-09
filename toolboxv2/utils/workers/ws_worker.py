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
    import websockets
    from websockets.server import serve as ws_serve
    from websockets.exceptions import ConnectionClosed

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from toolboxv2.utils.workers.event_manager import (
    ZMQEventManager,
    Event,
    EventType,
    create_ws_send_event,
    create_ws_broadcast_event,
)

logger = logging.getLogger(__name__)


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

        # Metrics
        self._metrics = {
            "messages_received": 0,
            "messages_sent": 0,
            "connections_total": 0,
            "errors": 0,
        }

    async def start(self):
        """Start the WebSocket worker."""
        logger.info(f"Starting WS worker {self.worker_id}")

        # Initialize ZMQ event manager
        await self._init_event_manager()

        # Start WebSocket server
        host = self.config.ws_worker.host
        port = self.config.ws_worker.port

        self._running = True

        # Start ping task
        asyncio.create_task(self._ping_loop())

        # Start server
        self._server = await ws_serve(
            self._handle_connection,
            host,
            port,
            ping_interval=self.config.ws_worker.ping_interval,
            ping_timeout=self.config.ws_worker.ping_timeout,
            max_size=self.config.ws_worker.max_message_size,
            compression="deflate" if self.config.ws_worker.compression else None,
        )

        logger.info(f"WS worker listening on {host}:{port}")

        # Keep running
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

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self):
        """Register handlers for events from HTTP workers."""

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

    async def _handle_connection(self, websocket, path: str):
        """Handle new WebSocket connection."""
        conn_id = str(uuid.uuid4())
        conn = WSConnection(
            conn_id=conn_id,
            websocket=websocket,
            metadata={"path": path},
        )

        # Check connection limit
        if not await self._conn_manager.add(conn):
            await websocket.close(1013, "Server overloaded")
            return

        self._metrics["connections_total"] += 1

        logger.debug(
            f"New connection: {conn_id} (total: {self._conn_manager.connection_count})"
        )

        # Publish connect event
        await self._event_manager.publish(
            Event(
                type=EventType.WS_CONNECT,
                source=self.worker_id,
                target="*",
                payload={"conn_id": conn_id, "path": path},
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

            # Message loop - MINIMAL PROCESSING
            async for message in websocket:
                self._metrics["messages_received"] += 1

                # Forward ALL messages to HTTP workers via ZeroMQ
                # NO processing here - just forward
                await self._event_manager.publish(
                    Event(
                        type=EventType.WS_MESSAGE,
                        source=self.worker_id,
                        target="*",
                        payload={
                            "conn_id": conn_id,
                            "user_id": conn.user_id,
                            "session_id": conn.session_id,
                            "data": message,
                            "path": path,
                        },
                    )
                )

        except ConnectionClosed as e:
            logger.debug(f"Connection closed: {conn_id} ({e.code})")
        except Exception as e:
            logger.error(f"Connection error: {conn_id}: {e}")
            self._metrics["errors"] += 1
        finally:
            # Clean up
            await self._conn_manager.remove(conn_id)

            # Publish disconnect event
            await self._event_manager.publish(
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
                "errors": self._metrics["errors"],
            }
        )
        return stats

    async def run(self):
        """Run the WebSocket worker (blocking)."""
        import sys

        # Windows: Use SelectorEventLoop for ZMQ compatibility
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Signal handlers (Unix only, Windows uses different mechanism)
        if sys.platform != "win32":

            def signal_handler():
                loop.create_task(self.stop())

            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, signal_handler)
                except NotImplementedError:
                    pass

        try:
            await self.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            await self.stop()
        except Exception as e:
            logger.error(f"WS worker error: {e}")
            await self.stop()
        finally:
            # Proper cleanup
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Wait for cancellation
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )

                # Shutdown async generators
                loop.run_until_complete(loop.shutdown_asyncgens())

            except Exception as e:
                logger.debug(f"Cleanup error: {e}")
            finally:
                if not loop.is_closed():
                    loop.close()


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
    import sys

    # Windows: Use SelectorEventLoop for ZMQ compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
