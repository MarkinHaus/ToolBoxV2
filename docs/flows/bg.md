# bg

Background runner that initializes a ToolBoxV2 application in daemon mode, optionally starting the API server before connecting.

## Why This Matters

When you need to run a ToolBoxV2 app as a long-lived background service, this is the entry point. It handles the startup sequence — printing status, starting the API server (in debug or production mode), and connecting the daemon — in a single async call.

## Quick Start

```python
import asyncio
from toolboxv2.flows.bg import run

# Start the app with API server enabled (default)
await run(app)
```

## Live Usage Examples

### Starting with API server (default)

```python
import asyncio
from toolboxv2.flows.bg import run

# Launches API server (debug or production based on app.debug)
# then connects the daemon
await run(app)
```

### Daemon-only mode (no API server)

```python
import asyncio
from toolboxv2.flows.bg import run

# Skips API server startup, connects daemon directly
await run(app, api=False)
```

## How It Works

The function executes a three-step startup sequence:

1. Prints a status message via `app.print("Running...")`.
2. If `api` is `True`, checks `app.debug`: starts the debug handler when debug mode is on, otherwise starts the production API server via `manage_server("start")`. Both utilities are lazily imported from `toolboxv2.utils.clis.api`.
3. Connects the daemon application via `await app.daemon_app.connect(app)`.

The conditional import keeps the module lightweight when API functionality isn't needed.

## API Reference

### Functions

#### `run(app, api=True)` (async)

Starts the background runner for a ToolBoxV2 application, optionally launching the API server before connecting the daemon.

**Parameters:**
- `app` — ToolBoxV2 application instance (must have `.print()`, `.debug`, and `.daemon_app` attributes)
- `api` — Whether to start the API server before daemon connection. Defaults to `True`. When `False`, skips server startup entirely.

**Returns:** `None` (async — awaits `app.daemon_app.connect(app)`)

## Dependencies

No indexed upstream dependencies.

## Used By

No indexed downstream usages.