# ws_bridge

WebSocket bridge that communicates with WS workers via ZeroMQ. Replaces the old Rust-based bridge pattern with a pure-Python ZMQ event system, providing methods for point-to-point messaging, channel broadcasting, and native Tauri notifications.

## Why This Matters

Any ToolBoxV2 module that needs to push real-time data to WebSocket clients — live logs, task progress, system notifications — goes through this bridge. It decouples worker processes from the WebSocket server using ZeroMQ, so workers don't need direct access to the HTTP layer.

## Quick Start

```python
from toolboxv2.utils.workers.ws_bridge import install_ws_bridge

# During app startup — installs ws_send, ws_broadcast, send_notification on app
install_ws_bridge(app, event_manager, worker_id="worker-1")

# Later, anywhere in your async code:
await app.ws_send("conn-42", {"type": "status", "data": "done"})
await app.ws_broadcast("room-alpha", {"type": "update", "value": 42})
await app.send_notification(title="Done", content="Export complete", level="success")
```

## Usage Guide

### Send to a Single Connection

```python
await app.ws_send(
    conn_id="user-123",
    payload={"type": "message", "data": "hello"}
)
```

### Broadcast to a Channel

```python
# source_conn_id excludes the sender to prevent echo
await app.ws_broadcast(
    channel_id="room-alpha",
    payload={"type": "update", "value": 42},
    source_conn_id="conn-42"
)
```

### Trigger a Native Notification

```python
# Send to a specific user
await app.send_notification(
    title="Task Complete",
    content="Your export is ready for download",
    conn_id="user-123"
)

# Broadcast warning to all connected clients
await app.send_notification(
    title="System Update",
    content="Server will restart in 5 minutes",
    level="warning"
)
```

## How It Works

The module has two layers. The core is `ZMQWSBridge`, a thin async wrapper around `ZMQEventManager` that translates high-level calls (`send_message`, `broadcast_message`, etc.) into typed `Event` objects dispatched via ZeroMQ. On top, `install_ws_bridge` creates a `ZMQWSBridge` instance and monkey-patches closure-based async methods (`ws_send`, `ws_broadcast`, `send_notification`) onto the given `App` instance, so any module with access to `app` can push messages without knowing about ZMQ internals. All routing — per-connection, per-channel, global broadcast — is handled by event type (`EventType.WS_SEND`, `WS_BROADCAST`, `WS_BROADCAST_ALL`) and processed by the WS worker on the other end.

## API Reference

### Classes

#### `ZMQWSBridge`

WebSocket bridge that communicates with WS workers via ZeroMQ. Provides the same interface as the old Rust bridge: `ws_send`, `ws_broadcast`, channel management, and notifications.

| Method | Signature | Description |
|--------|-----------|-------------|
| `send_message` | `async def send_message(self, conn_id: str, payload: str \| dict) -> bool` | Send message to a specific WebSocket connection. Returns `True` if sent (doesn't guarantee delivery). |
| `broadcast_message` | `async def broadcast_message(self, channel_id: str, payload: str \| dict, source_conn_id: str = "") -> bool` | Broadcast message to all connections in a channel. Optionally exclude a connection (e.g. the sender). |
| `broadcast_all` | `async def broadcast_all(self, payload: str \| dict, exclude_conn_ids: List[str] \| None = None) -> bool` | Broadcast message to all connected WebSocket clients. Optionally exclude specific connection IDs. |
| `send_notification` | `async def send_notification(self, title: str, content: str, conn_id: str \| None = None, channel: str \| None = None, icon: str \| None = None, level: str = "info") -> bool` | Send a notification that triggers Tauri native notifications. Routes to a specific connection, a channel, or all clients depending on arguments. `level` is one of `"info"`, `"warning"`, `"error"`, `"success"`. |
| `join_channel` | `async def join_channel(self, conn_id: str, channel: str) -> bool` | Request a connection to join a channel. |
| `leave_channel` | `async def leave_channel(self, conn_id: str, channel: str) -> bool` | Request a connection to leave a channel. |

### Functions

#### `install_ws_bridge(app, event_manager: ZMQEventManager, worker_id: str)`

Install WebSocket bridge methods on a ToolBoxV2 App instance. Replaces the old `_set_rust_ws_bridge` pattern with ZMQ-based communication. After calling this function, `app.ws_send()`, `app.ws_broadcast()`, and `app.send_notification()` will work.

**Parameters:**
- `app` — ToolBoxV2 App instance to patch
- `event_manager` — Initialized `ZMQEventManager`
- `worker_id` — ID of this worker

**Side effects:** Sets `app._zmq_ws_bridge` and attaches three async methods to the app instance.

## Dependencies

- `ZMQEventManager` — event manager for ZMQ communication (used in constructor)
- `Event`, `EventType` — event types for routing (`WS_SEND`, `WS_BROADCAST`, `WS_BROADCAST_ALL`, `WS_JOIN_CHANNEL`, `WS_LEAVE_CHANNEL`)
- `create_ws_send_event`, `create_ws_broadcast_event`, `create_ws_broadcast_all_event` — event factory helpers
- `logging` — stdlib logger
- `json` — payload serialization in app-level wrappers

## Used By

- Referenced by [notification](../../utils/extras/notification.md) in `toolboxv2/utils/extras/notification.py` (`setup_web_notifications`)
- Referenced by [adaptive_prompt_system](../../flows/adaptive_prompt_system.md) in `toolboxv2/flows/adaptive_prompt_system.py` (`__init__`)
- Referenced by [chain](../../flows/chain.md) in `toolboxv2/flows/chain.py` (`__init__`)
- Referenced by [icli](../../flows/icli.md) in `toolboxv2/flows/icli.py` (`__init__`)