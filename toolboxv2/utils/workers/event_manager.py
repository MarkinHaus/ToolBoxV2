#!/usr/bin/env python3
"""
event_manager.py - ZeroMQ-based Event Manager for ToolBoxV2 Worker System

High-performance pub/sub and request/reply patterns for:
- Inter-worker communication (HTTP -> WS)
- Broadcast events (session invalidation, config reload)
- Direct RPC calls between workers

Patterns:
- PUB/SUB: One-to-many broadcasts
- PUSH/PULL: Load-balanced task distribution
- REQ/REP: Synchronous RPC calls
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import struct
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import zmq
    import zmq.asyncio
except ImportError:
    #raise ImportError("pyzmq required: pip install pyzmq")
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Event Types
# ============================================================================

class EventType(str, Enum):
    """Event types for routing."""
    # Worker lifecycle
    WORKER_START = "worker.start"
    WORKER_STOP = "worker.stop"
    WORKER_HEALTH = "worker.health"
    WORKER_READY = "worker.ready"

    # Session events
    SESSION_CREATE = "session.create"
    SESSION_VALIDATE = "session.validate"
    SESSION_INVALIDATE = "session.invalidate"
    SESSION_SYNC = "session.sync"

    # WebSocket events
    WS_CONNECT = "ws.connect"
    WS_DISCONNECT = "ws.disconnect"
    WS_MESSAGE = "ws.message"
    WS_BROADCAST = "ws.broadcast"
    WS_BROADCAST_CHANNEL = "ws.broadcast_channel"
    WS_BROADCAST_ALL = "ws.broadcast_all"
    WS_SEND = "ws.send"
    WS_JOIN_CHANNEL = "ws.join_channel"
    WS_LEAVE_CHANNEL = "ws.leave_channel"

    # System events
    CONFIG_RELOAD = "system.config_reload"
    SHUTDOWN = "system.shutdown"
    ROLLING_UPDATE = "system.rolling_update"
    HEALTH_CHECK = "system.health_check"

    # Module events
    MODULE_CALL = "module.call"
    MODULE_RESULT = "module.result"

    # Custom events
    CUSTOM = "custom"

    # RPC
    RPC_REQUEST = "rpc.request"
    RPC_RESPONSE = "rpc.response"


# ============================================================================
# Event Data Structures
# ============================================================================

@dataclass
class Event:
    """Event payload for ZeroMQ messages."""
    type: EventType
    source: str  # Worker ID
    target: str  # Worker ID, channel, or "*" for broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    ttl: int = 60  # Time-to-live in seconds

    def to_bytes(self) -> bytes:
        """Serialize event to bytes."""
        data = {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Event":
        """Deserialize event from bytes."""
        obj = json.loads(data.decode("utf-8"))
        return cls(
            type=EventType(obj["type"]),
            source=obj["source"],
            target=obj["target"],
            payload=obj.get("payload", {}),
            correlation_id=obj.get("correlation_id", str(uuid.uuid4())),
            timestamp=obj.get("timestamp", time.time()),
            ttl=obj.get("ttl", 60),
        )

    def is_expired(self) -> bool:
        """Check if event TTL has expired."""
        return time.time() - self.timestamp > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value if isinstance(self.type, Enum) else self.type,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }


@dataclass
class EventHandler:
    """Handler registration for events."""
    callback: Callable
    event_types: Set[EventType]
    filter_func: Callable[[Event], bool] | None = None
    priority: int = 0
    once: bool = False
    _called: bool = False


class EventHandlerRegistry:
    """Registry for event handlers."""

    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._global_handlers: List[EventHandler] = []
        self._lock = threading.Lock()

    def register(
        self,
        event_types: EventType | List[EventType],
        callback: Callable,
        filter_func: Callable | None = None,
        priority: int = 0,
        once: bool = False,
    ) -> EventHandler:
        """Register an event handler."""
        if isinstance(event_types, EventType):
            event_types = [event_types]

        handler = EventHandler(
            callback=callback,
            event_types=set(event_types),
            filter_func=filter_func,
            priority=priority,
            once=once,
        )

        with self._lock:
            for event_type in event_types:
                self._handlers[event_type].append(handler)
                # Sort by priority (higher first)
                self._handlers[event_type].sort(key=lambda h: -h.priority)

        return handler

    def register_global(
        self,
        callback: Callable,
        filter_func: Callable | None = None,
        priority: int = 0,
    ) -> EventHandler:
        """Register a global handler for all events."""
        handler = EventHandler(
            callback=callback,
            event_types=set(),
            filter_func=filter_func,
            priority=priority,
        )

        with self._lock:
            self._global_handlers.append(handler)
            self._global_handlers.sort(key=lambda h: -h.priority)

        return handler

    def unregister(self, handler: EventHandler):
        """Unregister an event handler."""
        with self._lock:
            for event_type in handler.event_types:
                if handler in self._handlers[event_type]:
                    self._handlers[event_type].remove(handler)
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)

    def get_handlers(self, event_type: EventType) -> List[EventHandler]:
        """Get all handlers for an event type."""
        with self._lock:
            handlers = list(self._handlers.get(event_type, []))
            handlers.extend(self._global_handlers)
            return sorted(handlers, key=lambda h: -h.priority)

    def clear(self):
        """Clear all handlers."""
        with self._lock:
            self._handlers.clear()
            self._global_handlers.clear()


# ============================================================================
# ZeroMQ Event Manager
# ============================================================================

class ZMQEventManager:
    """
    ZeroMQ-based event manager for inter-worker communication.

    Supports:
    - PUB/SUB for broadcasts
    - REQ/REP for RPC calls
    - PUSH/PULL for task distribution
    """

    def __init__(
        self,
        worker_id: str,
        pub_endpoint: str = "tcp://127.0.0.1:5555",  # Broker binds XPUB, workers connect SUB
        sub_endpoint: str = "tcp://127.0.0.1:5556",  # Broker binds XSUB, workers connect PUB
        req_endpoint: str = "tcp://127.0.0.1:5557",  # Broker binds ROUTER for RPC
        rep_endpoint: str = "tcp://127.0.0.1:5557",  # Workers connect DEALER (same as req)
        http_to_ws_endpoint: str = "tcp://127.0.0.1:5558",  # HTTP->WS forwarding
        is_broker: bool = False,
        hwm_send: int = 10000,
        hwm_recv: int = 10000,
    ):
        self.worker_id = worker_id
        self.pub_endpoint = pub_endpoint
        self.sub_endpoint = sub_endpoint
        self.req_endpoint = req_endpoint
        self.rep_endpoint = rep_endpoint
        self.http_to_ws_endpoint = http_to_ws_endpoint
        self.is_broker = is_broker
        self.hwm_send = hwm_send
        self.hwm_recv = hwm_recv

        self._ctx: zmq.asyncio.Context | None = None
        self._pub_socket: zmq.asyncio.Socket | None = None
        self._sub_socket: zmq.asyncio.Socket | None = None
        self._req_socket: zmq.asyncio.Socket | None = None
        self._rep_socket: zmq.asyncio.Socket | None = None
        self._push_socket: zmq.asyncio.Socket | None = None
        self._pull_socket: zmq.asyncio.Socket | None = None

        # XPUB/XSUB for broker
        self._xpub_socket: zmq.asyncio.Socket | None = None
        self._xsub_socket: zmq.asyncio.Socket | None = None

        self._registry = EventHandlerRegistry()
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._subscriptions: Set[bytes] = set()

        # Sync context for non-async operations
        self._sync_ctx: zmq.Context | None = None
        self._sync_push: zmq.Socket | None = None

        # Metrics
        self._metrics = {
            "events_sent": 0,
            "events_received": 0,
            "rpc_calls": 0,
            "rpc_timeouts": 0,
            "errors": 0,
        }

        logger.info(
            f"ZMQEventManager initialized: worker_id={worker_id}, is_broker={is_broker}"
        )

    async def start(self):
        """Start the event manager."""
        if self._running:
            return

        self._ctx = zmq.asyncio.Context()
        self._running = True

        if self.is_broker:
            await self._start_broker()
        else:
            await self._start_worker()

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._sub_loop()))

        # Announce worker start
        await self.publish(Event(
            type=EventType.WORKER_START,
            source=self.worker_id,
            target="*",
            payload={"worker_id": self.worker_id, "pid": os.getpid()},
        ))

        logger.info(f"ZMQEventManager started: worker_id={self.worker_id}")

    async def _start_broker(self):
        """Start as central broker (binds to endpoints)."""
        # XPUB for forwarding subscriptions
        self._xpub_socket = self._ctx.socket(zmq.XPUB)
        self._xpub_socket.setsockopt(zmq.SNDHWM, self.hwm_send)
        self._xpub_socket.bind(self.pub_endpoint)

        # XSUB for receiving publications
        self._xsub_socket = self._ctx.socket(zmq.XSUB)
        self._xsub_socket.setsockopt(zmq.RCVHWM, self.hwm_recv)
        self._xsub_socket.bind(self.sub_endpoint)

        # REP for RPC
        self._rep_socket = self._ctx.socket(zmq.ROUTER)
        self._rep_socket.setsockopt(zmq.RCVHWM, self.hwm_recv)
        self._rep_socket.bind(self.req_endpoint)

        # PULL for HTTP->WS forwarding
        self._pull_socket = self._ctx.socket(zmq.PULL)
        self._pull_socket.setsockopt(zmq.RCVHWM, self.hwm_recv)
        self._pull_socket.bind(self.http_to_ws_endpoint)

        # Start proxy task
        self._tasks.append(asyncio.create_task(self._broker_proxy()))
        self._tasks.append(asyncio.create_task(self._rpc_handler_loop()))
        self._tasks.append(asyncio.create_task(self._forward_loop()))

        logger.info("Broker started - XPUB/XSUB proxy running")

    async def _start_worker(self):
        """Start as worker (connects to broker)."""
        # Workers connect SUB to broker's XPUB to receive broadcasts
        self._sub_socket = self._ctx.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.RCVHWM, self.hwm_recv)
        self._sub_socket.connect(self.pub_endpoint)  # Connect to broker's XPUB
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        # Workers connect PUB to broker's XSUB to send events
        self._pub_socket = self._ctx.socket(zmq.PUB)
        self._pub_socket.setsockopt(zmq.SNDHWM, self.hwm_send)
        self._pub_socket.connect(self.sub_endpoint)  # Connect to broker's XSUB

        # REQ/DEALER for RPC calls
        self._req_socket = self._ctx.socket(zmq.DEALER)
        self._req_socket.setsockopt(zmq.IDENTITY, self.worker_id.encode())
        self._req_socket.setsockopt(zmq.RCVHWM, self.hwm_recv)
        self._req_socket.connect(self.req_endpoint)

        # PUSH for HTTP->WS forwarding
        self._push_socket = self._ctx.socket(zmq.PUSH)
        self._push_socket.setsockopt(zmq.SNDHWM, self.hwm_send)
        self._push_socket.connect(self.http_to_ws_endpoint)

        # Start RPC response handler
        self._tasks.append(asyncio.create_task(self._rpc_response_loop()))

        logger.info(f"Worker connected to broker: {self.worker_id}")

    async def _broker_proxy(self):
        """Run XPUB/XSUB proxy for message forwarding."""
        poller = zmq.asyncio.Poller()
        poller.register(self._xpub_socket, zmq.POLLIN)
        poller.register(self._xsub_socket, zmq.POLLIN)

        logger.info("[Broker] Starting XPUB/XSUB proxy loop")
        msg_count = 0

        while self._running:
            try:
                events = dict(await poller.poll(timeout=100))

                # Forward subscriptions from XPUB to XSUB
                if self._xpub_socket in events:
                    msg = await self._xpub_socket.recv()
                    # Log subscription messages (start with \x01 for subscribe, \x00 for unsubscribe)
                    if msg and len(msg) > 0:
                        if msg[0] == 1:
                            logger.info(f"[Broker] New subscription: {msg[1:].decode('utf-8', errors='ignore')[:50]}")
                        elif msg[0] == 0:
                            logger.info(f"[Broker] Unsubscription: {msg[1:].decode('utf-8', errors='ignore')[:50]}")
                    await self._xsub_socket.send(msg)

                # Forward messages from XSUB to XPUB
                if self._xsub_socket in events:
                    msg = await self._xsub_socket.recv()
                    msg_count += 1
                    # Try to parse and log event type
                    try:
                        event = Event.from_bytes(msg)
                        if event.type.startswith("ws."):
                            logger.info(f"[Broker] Forwarding #{msg_count}: {event.type} from {event.source} to {event.target}")
                    except Exception:
                        logger.debug(f"[Broker] Forwarding #{msg_count}: raw message ({len(msg)} bytes)")
                    await self._xpub_socket.send(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broker proxy error: {e}")
                self._metrics["errors"] += 1

    async def _rpc_handler_loop(self):
        """Handle incoming RPC requests (broker only)."""
        while self._running:
            try:
                # Receive multipart: [identity, empty, request]
                frames = await self._rep_socket.recv_multipart()
                if len(frames) < 3:
                    continue

                identity = frames[0]
                request_data = frames[-1]

                event = Event.from_bytes(request_data)

                # Handle RPC request
                response = await self._handle_rpc_request(event)

                # Send response back
                await self._rep_socket.send_multipart([
                    identity,
                    b"",
                    response.to_bytes()
                ])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RPC handler error: {e}")
                self._metrics["errors"] += 1

    async def _handle_rpc_request(self, event: Event) -> Event:
        """Process RPC request and return response."""
        handlers = self._registry.get_handlers(event.type)

        result = {"handled": False}

        for handler in handlers:
            if handler.filter_func and not handler.filter_func(event):
                continue

            try:
                if asyncio.iscoroutinefunction(handler.callback):
                    response = await handler.callback(event)
                else:
                    response = handler.callback(event)

                if response is not None:
                    result = {"handled": True, "response": response}
                    break

            except Exception as e:
                logger.error(f"RPC handler error: {e}")
                result = {"handled": False, "error": str(e)}

        return Event(
            type=EventType.RPC_RESPONSE,
            source=self.worker_id,
            target=event.source,
            payload=result,
            correlation_id=event.correlation_id,
        )

    async def _rpc_response_loop(self):
        """Handle RPC responses (worker only)."""
        while self._running:
            try:
                frames = await self._req_socket.recv_multipart()
                response_data = frames[-1]

                event = Event.from_bytes(response_data)

                # Resolve pending request
                if event.correlation_id in self._pending_requests:
                    future = self._pending_requests.pop(event.correlation_id)
                    if not future.done():
                        future.set_result(event.payload)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RPC response error: {e}")
                self._metrics["errors"] += 1

    async def _forward_loop(self):
        """Forward HTTP->WS messages (broker only)."""
        while self._running:
            try:
                msg = await self._pull_socket.recv()
                event = Event.from_bytes(msg)

                # Broadcast to WS workers
                await self._xpub_socket.send(event.to_bytes())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Forward loop error: {e}")
                self._metrics["errors"] += 1

    async def _sub_loop(self):
        """Process incoming subscription messages."""
        socket = self._sub_socket if not self.is_broker else self._xpub_socket
        logger.info(f"[EventManager] Starting sub loop for worker {self.worker_id}, is_broker={self.is_broker}")

        while self._running:
            try:
                if self.is_broker:
                    # Broker doesn't receive via sub
                    await asyncio.sleep(0.1)
                    continue

                msg = await self._sub_socket.recv()
                self._metrics["events_received"] += 1

                try:
                    event = Event.from_bytes(msg)
                except Exception as e:
                    logger.debug(f"[EventManager] Failed to parse event: {e}")
                    continue

                # Log all WS events for debugging
                if event.type.startswith("ws."):
                    logger.info(f"[EventManager] Received {event.type} from {event.source} to {event.target}")

                # Skip expired events
                if event.is_expired():
                    logger.debug(f"[EventManager] Skipping expired event: {event.type}")
                    continue

                # Skip our own events
                if event.source == self.worker_id:
                    logger.debug(f"[EventManager] Skipping own event: {event.type}")
                    continue

                # Check if event is for us
                if event.target not in ("*", self.worker_id):
                    # Check channel subscriptions
                    if not event.target.encode() in self._subscriptions:
                        logger.debug(f"[EventManager] Skipping event not for us: {event.type} target={event.target}")
                        continue

                # Dispatch to handlers
                logger.debug(f"[EventManager] Dispatching event: {event.type}")
                await self._dispatch_event(event)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sub loop error: {e}")
                self._metrics["errors"] += 1

    async def _dispatch_event(self, event: Event):
        """Dispatch event to registered handlers."""
        handlers = self._registry.get_handlers(event.type)

        if event.type.startswith("ws."):
            logger.info(f"[EventManager] Dispatching {event.type} to {len(handlers)} handlers")

        for handler in handlers:
            if handler.filter_func and not handler.filter_func(event):
                continue

            if handler.once and handler._called:
                continue

            try:
                if asyncio.iscoroutinefunction(handler.callback):
                    await handler.callback(event)
                else:
                    handler.callback(event)

                handler._called = True

            except Exception as e:
                logger.error(f"Event handler error for {event.type}: {e}", exc_info=True)
                self._metrics["errors"] += 1

    # ========================================================================
    # Public API
    # ========================================================================

    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        if not self._running:
            raise RuntimeError("Event manager not started")

        socket = self._pub_socket if not self.is_broker else self._xpub_socket
        await socket.send(event.to_bytes())
        self._metrics["events_sent"] += 1

    async def send_to_ws(self, event: Event):
        """Send event to WS workers via PUSH socket (HTTP workers only)."""
        if not self._push_socket:
            raise RuntimeError("PUSH socket not available")

        await self._push_socket.send(event.to_bytes())
        self._metrics["events_sent"] += 1

    def send_to_ws_sync(self, event: Event):
        """Synchronous version of send_to_ws."""
        if not self._sync_ctx:
            self._sync_ctx = zmq.Context()
            self._sync_push = self._sync_ctx.socket(zmq.PUSH)
            self._sync_push.connect(self.http_to_ws_endpoint)

        self._sync_push.send(event.to_bytes())
        self._metrics["events_sent"] += 1

    async def rpc_call(
        self,
        event: Event,
        timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """Make an RPC call and wait for response."""
        if not self._req_socket:
            raise RuntimeError("REQ socket not available")

        self._metrics["rpc_calls"] += 1

        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[event.correlation_id] = future

        # Send request
        await self._req_socket.send_multipart([b"", event.to_bytes()])

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except TimeoutError:
            self._pending_requests.pop(event.correlation_id, None)
            self._metrics["rpc_timeouts"] += 1
            raise TimeoutError(f"RPC call timed out: {event.type}")

    def subscribe(self, channel: str):
        """Subscribe to a channel."""
        topic = channel.encode()
        self._subscriptions.add(topic)
        if self._sub_socket:
            self._sub_socket.setsockopt(zmq.SUBSCRIBE, topic)

    def unsubscribe(self, channel: str):
        """Unsubscribe from a channel."""
        topic = channel.encode()
        self._subscriptions.discard(topic)
        if self._sub_socket:
            self._sub_socket.setsockopt(zmq.UNSUBSCRIBE, topic)

    def on(
        self,
        event_types: EventType | List[EventType],
        filter_func: Callable | None = None,
        priority: int = 0,
        once: bool = False,
    ):
        """Decorator to register event handlers."""
        def decorator(func: Callable) -> Callable:
            self._registry.register(
                event_types=event_types,
                callback=func,
                filter_func=filter_func,
                priority=priority,
                once=once,
            )
            return func
        return decorator

    def register_handler(
        self,
        event_types: EventType | List[EventType],
        callback: Callable,
        filter_func: Callable | None = None,
        priority: int = 0,
        once: bool = False,
    ) -> EventHandler:
        """Register an event handler."""
        return self._registry.register(
            event_types=event_types,
            callback=callback,
            filter_func=filter_func,
            priority=priority,
            once=once,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get event manager metrics."""
        return dict(self._metrics)

    async def stop(self):
        """Stop the event manager."""
        if not self._running:
            return

        self._running = False

        # Announce worker stop
        try:
            await self.publish(Event(
                type=EventType.WORKER_STOP,
                source=self.worker_id,
                target="*",
                payload={"worker_id": self.worker_id},
            ))
        except Exception:
            pass

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close sockets
        for socket in [
            self._pub_socket, self._sub_socket,
            self._req_socket, self._rep_socket,
            self._push_socket, self._pull_socket,
            self._xpub_socket, self._xsub_socket,
        ]:
            if socket:
                socket.close()

        if self._ctx:
            self._ctx.term()

        if self._sync_push:
            self._sync_push.close()
        if self._sync_ctx:
            self._sync_ctx.term()

        # Clear pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        self._registry.clear()

        logger.info(f"ZMQEventManager stopped: worker_id={self.worker_id}")


# ============================================================================
# Helper Functions
# ============================================================================

def create_ws_send_event(
    source: str,
    conn_id: str,
    payload: str | Dict,
) -> Event:
    """Create WS_SEND event."""
    if isinstance(payload, dict):
        payload = json.dumps(payload)

    return Event(
        type=EventType.WS_SEND,
        source=source,
        target="ws_worker",
        payload={"conn_id": conn_id, "data": payload},
    )


def create_ws_broadcast_event(
    source: str,
    channel: str,
    payload: str | Dict,
    exclude_conn_ids: List[str] | None = None,
) -> Event:
    """Create WS_BROADCAST_CHANNEL event."""
    if isinstance(payload, dict):
        payload = json.dumps(payload)

    return Event(
        type=EventType.WS_BROADCAST_CHANNEL,
        source=source,
        target="ws_worker",
        payload={
            "channel": channel,
            "data": payload,
            "exclude": exclude_conn_ids or [],
        },
    )


def create_ws_broadcast_all_event(
    source: str,
    payload: str | Dict,
    exclude_conn_ids: List[str] | None = None,
) -> Event:
    """Create WS_BROADCAST_ALL event."""
    if isinstance(payload, dict):
        payload = json.dumps(payload)

    return Event(
        type=EventType.WS_BROADCAST_ALL,
        source=source,
        target="*",
        payload={
            "data": payload,
            "exclude": exclude_conn_ids or [],
        },
    )


# ============================================================================
# Broker Process
# ============================================================================

async def run_broker(config):
    """Run ZMQ broker as standalone process."""
    from toolboxv2.utils.workers.config import ZMQConfig

    if isinstance(config, dict):
        zmq_config = ZMQConfig(**config.get("zmq", {}))
    else:
        zmq_config = config.zmq

    broker = ZMQEventManager(
        worker_id="broker",
        pub_endpoint=zmq_config.pub_endpoint,
        sub_endpoint=zmq_config.sub_endpoint,
        req_endpoint=zmq_config.req_endpoint,
        rep_endpoint=zmq_config.rep_endpoint,
        http_to_ws_endpoint=zmq_config.http_to_ws_endpoint,
        is_broker=True,
        hwm_send=zmq_config.hwm_send,
        hwm_recv=zmq_config.hwm_recv,
    )

    await broker.start()

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()

    def signal_handler():
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows
    try:
        await shutdown_event.wait()
    except asyncio.exceptions.CancelledError:
        pass
    await broker.stop()


async def main():
    """CLI entry point for broker."""
    import argparse
    from platform import system
    if system() == "Windows":
        print("Windows detected. Setting event loop policy...")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="ZMQ Event Broker")
    parser.add_argument("-c", "--config", help="Config file path")
    parser.add_argument("--pub", default="tcp://127.0.0.1:5555", help="XPUB endpoint (broker->workers)")
    parser.add_argument("--sub", default="tcp://127.0.0.1:5556", help="XSUB endpoint (workers->broker)")
    parser.add_argument("--req", default="tcp://127.0.0.1:5557", help="ROUTER endpoint (RPC)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    config = {
        "zmq": {
            "pub_endpoint": args.pub,
            "sub_endpoint": args.sub,
            "req_endpoint": args.req,
        }
    }

    await run_broker(config)


if __name__ == "__main__":
    main()
