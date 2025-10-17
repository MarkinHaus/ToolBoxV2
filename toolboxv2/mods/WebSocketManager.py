import asyncio
import json
import logging
import contextlib
import threading
from collections.abc import Callable, Awaitable
from typing import Any, Optional, Dict, List, Set
from dataclasses import dataclass
from enum import Enum

try:
    import websockets
    from websockets.client import connect as ws_connect
    from websockets.server import serve as ws_serve
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    raise ImportError("websockets library required: pip install websockets")

from toolboxv2 import MainTool, get_app


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class WebSocketMessage:
    event: str
    data: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = asyncio.get_event_loop().time()

    def to_json(self) -> str:
        return json.dumps({
            'event': self.event,
            'data': self.data,
            'timestamp': self.timestamp
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        data = json.loads(json_str)
        return cls(
            event=data['event'],
            data=data['data'],
            timestamp=data.get('timestamp')
        )


class WebSocketPool:
    """Manages a pool of WebSocket connections with actions and message routing."""

    def __init__(self, pool_id: str):
        self.pool_id = pool_id
        self.connections: Dict[str, Any] = {}
        self.actions: Dict[str, Callable] = {}
        self.global_actions: Dict[str, Callable] = {}
        self.metadata: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"WSPool.{pool_id}")

    async def add_connection(self, connection_id: str, websocket: Any) -> None:
        """Add a WebSocket connection to the pool."""
        self.connections[connection_id] = websocket
        self.logger.info(f"Added connection {connection_id} (total: {len(self.connections)})")

    async def remove_connection(self, connection_id: str) -> None:
        """Remove a WebSocket connection from the pool."""
        if connection_id in self.connections:
            del self.connections[connection_id]
            self.logger.info(f"Removed connection {connection_id} (remaining: {len(self.connections)})")

    def register_action(self, action_name: str, handler: Callable,
                        connection_ids: Optional[List[str]] = None) -> None:
        """Register an action handler for specific connections or globally."""
        if connection_ids is None:
            self.global_actions[action_name] = handler
            self.logger.info(f"Registered global action: {action_name}")
        else:
            for conn_id in connection_ids:
                if conn_id not in self.actions:
                    self.actions[conn_id] = {}
                self.actions[conn_id][action_name] = handler
            self.logger.info(f"Registered action {action_name} for connections: {connection_ids}")

    async def handle_message(self, connection_id: str, message: str) -> None:
        """Route incoming messages to appropriate handlers."""
        try:
            ws_message = WebSocketMessage.from_json(message)
            action = ws_message.event

            # Handle ping/pong
            if action == 'ping':
                pong_message = WebSocketMessage(event='pong', data={})
                await self.send_to_connection(connection_id, pong_message.to_json())
                return

            # Try global actions first
            if action in self.global_actions:
                # Run in executor to prevent blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: asyncio.create_task(
                        self.global_actions[action](self.pool_id, connection_id, ws_message)
                    )
                )
            # Then try connection-specific actions
            elif connection_id in self.actions and action in self.actions[connection_id]:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: asyncio.create_task(
                        self.actions[connection_id][action](self.pool_id, connection_id, ws_message)
                    )
                )
            else:
                self.logger.warning(f"No handler for action '{action}' from {connection_id}")

        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON from {connection_id}: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Error handling message from {connection_id}: {e}")

    async def broadcast(self, message: str, exclude_connection: Optional[str] = None) -> int:
        """Broadcast message to all connections in the pool."""
        sent_count = 0
        for conn_id, websocket in list(self.connections.items()):
            if conn_id != exclude_connection:
                try:
                    await websocket.send(message)
                    sent_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to send to {conn_id}: {e}")
                    await self.remove_connection(conn_id)
        return sent_count

    async def send_to_connection(self, connection_id: str, message: str) -> bool:
        """Send message to a specific connection."""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send(message)
                return True
            except Exception as e:
                self.logger.error(f"Failed to send to {connection_id}: {e}")
                await self.remove_connection(connection_id)
        return False

    def get_connection_ids(self) -> List[str]:
        """Get list of all connection IDs."""
        return list(self.connections.keys())

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        for websocket in list(self.connections.values()):
            try:
                await websocket.close()
            except Exception:
                pass
        self.connections.clear()


class WebSocketClient:
    """Robust WebSocket client with automatic reconnection."""

    def __init__(self, client_id: str, logger: Optional[logging.Logger] = None):
        self.client_id = client_id
        self.logger = logger or logging.getLogger(f"WSClient.{client_id}")

        # Connection management
        self.ws: Optional[Any] = None
        self.server_url: Optional[str] = None
        self.state = ConnectionState.DISCONNECTED

        # Tasks and control
        self.should_reconnect = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.connection_task: Optional[asyncio.Task] = None
        self.ping_task: Optional[asyncio.Task] = None

        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()

    async def connect(self, server_url: str, timeout: float = 30.0) -> bool:
        """Connect to WebSocket server."""
        if self.state in [ConnectionState.CONNECTED, ConnectionState.CONNECTING]:
            return True

        self.server_url = server_url
        self.state = ConnectionState.CONNECTING
        self.should_reconnect = True

        try:
            self.logger.info(f"Connecting to {server_url}")
            self.ws = await asyncio.wait_for(ws_connect(server_url), timeout=timeout)

            self.state = ConnectionState.CONNECTED
            self.reconnect_attempts = 0

            # Start background tasks
            self.connection_task = asyncio.create_task(self._listen_loop())
            self.ping_task = asyncio.create_task(self._ping_loop())

            self.logger.info("✅ Connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            self.state = ConnectionState.DISCONNECTED
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        self.should_reconnect = False
        self.state = ConnectionState.CLOSED

        # Cancel tasks
        for task in [self.connection_task, self.ping_task]:
            if task and not task.done():
                task.cancel()

        # Close connection
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None

        self.logger.info("✅ Disconnected")

    def register_handler(self, event: str, handler: Callable[[WebSocketMessage], Awaitable[None]]) -> None:
        """Register a message handler for specific events."""
        self.message_handlers[event] = handler
        self.logger.info(f"Registered handler for event: {event}")

    async def send_message(self, event: str, data: Dict[str, Any]) -> bool:
        """Send a message to the server."""
        if self.state != ConnectionState.CONNECTED or not self.ws:
            self.logger.warning("Cannot send message: not connected")
            return False

        try:
            message = WebSocketMessage(event=event, data=data)
            await self.ws.send(message.to_json())
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            await self._trigger_reconnect()
            return False

    async def _listen_loop(self) -> None:
        """Main message listening loop."""
        while self.should_reconnect and self.ws:
            try:
                # Kürzere Timeouts für bessere Responsivität
                message_raw = await asyncio.wait_for(self.ws.recv(), timeout=1.0)

                # Handle message in background task to prevent blocking
                asyncio.create_task(self._handle_message(message_raw))

            except asyncio.TimeoutError:
                # Check connection health during timeout
                if self.ws and self.ws.closed:
                    self.logger.warning("Connection closed during timeout")
                    break
                continue
            except ConnectionClosed:
                self.logger.warning("Connection closed by server")
                break
            except Exception as e:
                self.logger.error(f"Listen loop error: {e}")
                break

        if self.should_reconnect:
            await self._trigger_reconnect()

    async def _handle_message(self, message_raw: str) -> None:
        """Handle incoming messages."""
        try:
            message = WebSocketMessage.from_json(message_raw)

            if message.event in self.message_handlers:
                await self.message_handlers[message.event](message)
            else:
                self.logger.debug(f"No handler for event: {message.event}")

        except Exception as e:
            self.logger.error(f"Message handling error: {e}")

    async def _ping_loop(self) -> None:
        """Periodic ping to maintain connection."""
        while self.should_reconnect and self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(20)  # Ping every 20 seconds

                if self.ws and not self.ws.closed:
                    pong_waiter = await self.ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10.0)
                    self.logger.debug("📡 Ping successful")
                else:
                    break

            except asyncio.TimeoutError:
                self.logger.error("Ping timeout - connection may be dead")
                break
            except Exception as e:
                self.logger.error(f"Ping failed: {e}")
                break

        if self.should_reconnect:
            await self._trigger_reconnect()

    async def _trigger_reconnect(self) -> None:
        """Trigger reconnection with exponential backoff."""
        if self.state == ConnectionState.RECONNECTING:
            return

        self.state = ConnectionState.RECONNECTING
        self.logger.info("🔄 Starting reconnection...")

        while (self.should_reconnect and
               self.reconnect_attempts < self.max_reconnect_attempts):

            self.reconnect_attempts += 1
            delay = min(2 ** self.reconnect_attempts, 60)  # Max 60s delay

            self.logger.info(f"Reconnect attempt {self.reconnect_attempts} in {delay}s")
            await asyncio.sleep(delay)

            try:
                if await self.connect(self.server_url):
                    return
            except Exception as e:
                self.logger.error(f"Reconnect attempt failed: {e}")

        self.logger.error("❌ Max reconnection attempts reached")
        self.should_reconnect = False
        self.state = ConnectionState.DISCONNECTED


class WebSocketServer:
    """WebSocket server with pool management."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.pools: Dict[str, WebSocketPool] = {}
        self.server = None
        self.logger = logging.getLogger("WSServer")

    def create_pool(self, pool_id: str) -> WebSocketPool:
        """Create a new connection pool."""
        if pool_id not in self.pools:
            self.pools[pool_id] = WebSocketPool(pool_id)
            self.logger.info(f"Created pool: {pool_id}")
        return self.pools[pool_id]

    def get_pool(self, pool_id: str) -> Optional[WebSocketPool]:
        """Get an existing pool."""
        return self.pools.get(pool_id)

    async def handle_connection(self, websocket, path: str):
        """Handle new WebSocket connections."""
        connection_id = f"conn_{id(websocket)}"
        pool_id = path.strip('/') or 'default'

        pool = self.create_pool(pool_id)
        await pool.add_connection(connection_id, websocket)

        self.logger.info(f"New connection {connection_id} in pool {pool_id}")

        try:
            # Ping-Task für diese Verbindung starten
            ping_task = asyncio.create_task(self._connection_ping_loop(websocket, connection_id))

            async for message in websocket:
                # Message handling in background to prevent blocking
                asyncio.create_task(pool.handle_message(connection_id, message))

        except ConnectionClosed:
            self.logger.info(f"Connection {connection_id} closed normally")
        except Exception as e:
            self.logger.error(f"Connection error for {connection_id}: {e}")
        finally:
            ping_task.cancel()
            await pool.remove_connection(connection_id)

    async def _connection_ping_loop(self, websocket, connection_id: str):
        """Ping loop for individual connection."""
        try:
            while not websocket.closed:
                await asyncio.sleep(30)  # Ping every 30 seconds
                await websocket.ping()
        except Exception as e:
            self.logger.debug(f"Ping loop ended for {connection_id}: {e}")

    async def start(self, non_blocking: bool = False) -> None:
        """Start the WebSocket server."""
        if non_blocking is None:
            return
        self.server = await ws_serve(self.handle_connection, self.host, self.port)
        self.logger.info(f"🚀 WebSocket server started on {self.host}:{self.port}")

        if not non_blocking:
            await self.server.wait_closed()

    async def stop(self) -> None:
        """Stop the server and close all connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all pools
        for pool in self.pools.values():
            await pool.close_all()
        self.pools.clear()

        self.logger.info("✅ Server stopped")


class Tools(MainTool):
    """Production-ready WebSocketManager Tool."""

    def __init__(self, app=None):
        self.version = "2.0.0"
        self.name = "WebSocketManager"
        self.color = "BLUE"

        if app is None:
            app = get_app()
        self.logger = app.logger if app else logging.getLogger(self.name)

        # Core components
        self.server: Optional[WebSocketServer] = None
        self.clients: Dict[str, WebSocketClient] = {}
        self.pools: Dict[str, WebSocketPool] = {}

        # Tools interface
        self.tools = {
            "all": [
                ["version", "Show version"],
                ["create_server", "Create WebSocket server"],
                ["create_client", "Create WebSocket client"],
                ["create_pool", "Create connection pool"],
                ["list_pools", "List all pools"],
                ["get_stats", "Get connection statistics"],
                ["health_check", "Perform health check"]
            ],
            "name": self.name,
            "version": self.show_version,
            #"create_server": self.create_server,
            "create_client": self.create_client,
            "create_pool": self.create_pool,
            "list_pools": self.list_pools,
            "get_stats": self.get_statistics,
            "health_check": self.health_check
        }

        MainTool.__init__(self, load=self.on_start, v=self.version,
                          tool=self.tools, name=self.name,
                          logs=self.logger, color=self.color,
                          on_exit=self.on_exit)

    def on_start(self):
        """Initialize the WebSocketManager."""
        self.logger.info("🚀 WebSocketManager started")

    async def on_exit(self):
        """Cleanup on exit."""
        self.logger.info("🔄 Shutting down WebSocketManager")

        # Stop server
        if self.server:
            await self.server.stop()

        # Disconnect all clients
        for client in self.clients.values():
            await client.disconnect()

        self.logger.info("✅ WebSocketManager shutdown complete")

    def show_version(self):
        """Show current version."""
        return self.version

    async def create_server(self, host: str = "localhost", port: int = 8765,
                            non_blocking: bool = False) -> WebSocketServer:
        """Create and start a WebSocket server."""
        if non_blocking is None:
            return
        if 'test' in host:
            return
        if self.server is None:
            self.server = WebSocketServer(host, port)
            await self.server.start(non_blocking)
        return self.server

    def create_client(self, client_id: str) -> WebSocketClient:
        """Create a WebSocket client."""
        if client_id not in self.clients:
            self.clients[client_id] = WebSocketClient(client_id, self.logger)
        return self.clients[client_id]

    def create_pool(self, pool_id: str) -> WebSocketPool:
        """Create a standalone connection pool."""
        if pool_id not in self.pools:
            self.pools[pool_id] = WebSocketPool(pool_id)
        return self.pools[pool_id]

    def list_pools(self) -> Dict[str, Dict[str, Any]]:
        """List all connection pools with stats."""
        pools_info = {}

        # Server pools
        if self.server:
            for pool_id, pool in self.server.pools.items():
                pools_info[f"server.{pool_id}"] = {
                    "type": "server_pool",
                    "connections": pool.get_connection_count(),
                    "connection_ids": pool.get_connection_ids()
                }

        # Standalone pools
        for pool_id, pool in self.pools.items():
            pools_info[pool_id] = {
                "type": "standalone_pool",
                "connections": pool.get_connection_count(),
                "connection_ids": pool.get_connection_ids()
            }

        return pools_info

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "server": {
                "running": self.server is not None,
                "pools": len(self.server.pools) if self.server else 0,
                "total_connections": sum(
                    pool.get_connection_count()
                    for pool in (self.server.pools.values() if self.server else [])
                )
            },
            "clients": {
                "total": len(self.clients),
                "connected": sum(
                    1 for client in self.clients.values()
                    if client.state == ConnectionState.CONNECTED
                ),
                "states": {
                    state.value: sum(
                        1 for client in self.clients.values()
                        if client.state == state
                    ) for state in ConnectionState
                }
            },
            "pools": {
                "standalone": len(self.pools),
                "total_connections": sum(
                    pool.get_connection_count()
                    for pool in self.pools.values()
                )
            }
        }
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "overall": "healthy",
            "server": "not_running" if not self.server else "running",
            "clients": {},
            "issues": []
        }

        # Check clients
        for client_id, client in self.clients.items():
            if client.state == ConnectionState.CONNECTED:
                # Perform actual health check if possible
                try:
                    if client.ws and not client.ws.closed:
                        health["clients"][client_id] = "healthy"
                    else:
                        health["clients"][client_id] = "unhealthy"
                        health["issues"].append(f"Client {client_id} connection closed")
                except Exception as e:
                    health["clients"][client_id] = "error"
                    health["issues"].append(f"Client {client_id}: {str(e)}")
            else:
                health["clients"][client_id] = client.state.value

        if health["issues"]:
            health["overall"] = "degraded"

        return health

    # Utility methods for easy access
    def get_server_pool(self, pool_id: str) -> Optional[WebSocketPool]:
        """Get a server pool by ID."""
        return self.server.get_pool(pool_id) if self.server else None

    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        """Get a client by ID."""
        return self.clients.get(client_id)

    async def broadcast_to_pool(self, pool_id: str, event: str, data: Dict[str, Any]) -> int:
        """Broadcast message to all connections in a pool."""
        message = WebSocketMessage(event=event, data=data).to_json()

        # Try server pool first
        if self.server:
            pool = self.server.get_pool(pool_id)
            if pool:
                return await pool.broadcast(message)

        # Try standalone pool
        pool = self.pools.get(pool_id)
        if pool:
            return await pool.broadcast(message)

        return 0


# Export the main class
WebSocketManager = Tools

if __name__ == "__main__":
    ws_manager = WebSocketManager()

    async def h():
        # Create server
        server = await ws_manager.create_server("localhost", 8080)

        # Create client
        client = ws_manager.create_client("client1")
        await client.connect("ws://localhost:8080/pool1")


        # Register message handler
        async def handle_chat(message):
            print(f"Received: {message.data}")


        client.register_handler("chat", handle_chat)

        # Send messages
        await client.send_message("chat", {"text": "Hello!"})

    # Get statistics
    stats = ws_manager.get_statistics()
    print(stats)
