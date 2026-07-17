# Live Debugger (`utils/extras/live_debugger.py`)

> **File:** `toolboxv2/utils/extras/live_debugger.py` (~138 Zeilen)
> **Typ:** Reference
> Thread + Async Task Stack Dumps für Deadlock/Hang Debugging.

## Why This Matters

Wenn ToolBoxV2 "hängt" (Deadlock, endlose Schleife, geblockter Thread), ist `dump_project_threads()` das Werkzeug um zu sehen: Welche Threads laufen? Wo genau stehen sie? Welche Async-Tasks sind pending?

## API Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `dump_project_threads` | `(reason="manual")` | Dump all thread stacks + async tasks to stderr |

### Output Format

```
======================================================================
DEBUG STACK DUMP (watchdog) - 14:23:05
======================================================================

--- Thread: http-worker-1 (0x1a2b) ---
  ⏸ blockiert in: socket.py:412 → recv()
  📍 Dein Code (aufrufreihenfolge):
     mods/CloudM/Auth.py:145 → validate_session()
       | session = await verifier.verify_session(token)

--- Async Tasks ---
  Task 1: pending  (coro=handle_request, waited 12s)
  Task 2: running  (coro=heartbeat_loop)
```

### Features

- **Project frame filtering**: Only shows `toolboxv2/` frames, not stdlib
- **Blocked detection**: Shows where thread is currently blocked (e.g. `socket.recv()`)
- **Async task dump**: Lists all pending asyncio tasks with wait duration
- **Memory cleanup**: Clears `linecache` after dump (no stale code cached)

## How-to: Use in Code

```python
from toolboxv2.utils.extras.live_debugger import dump_project_threads

# Manual dump when something seems stuck
dump_project_threads(reason="user requested")

# In a watchdog timer
import threading
def watchdog():
    threading.Timer(60.0, watchdog).start()
    dump_project_threads(reason="periodic")

watchdog()
```

## Related

- [Core Types](types.md) — `AppType` threading
- [WSWorker](ws_worker.md) — ping watchdog uses this
