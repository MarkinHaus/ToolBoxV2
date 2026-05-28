#!/usr/bin/env python3
"""
toolboxv2/utils/workers/fast/tray_api.py - Tray API + Client

The bridge between Python workers and the Tauri tray.

Server side (mounted on the tauri-worker's FastTB app):
    POST /tray/status         — worker pushes status snapshot
    POST /tray/notify         — worker pushes a one-off notification
    GET  /tray/state          — anyone pulls the aggregated state
    GET  /tray/events         — SSE stream for realtime updates (Tauri subscribes)
    POST /tray/command        — Tauri requests an action on a worker

Client side (embedded in any other Python worker):
    from toolboxv2.utils.workers.fast.tray_api import TrayClient
    tray = TrayClient("http_worker", label="HTTP Worker")
    tray.report(running=True, pid=os.getpid())
    # ... later, periodically or on event:
    tray.report(running=True, metric=f"{rps} req/s")
    # on shutdown:
    tray.report(running=False)

The client is best-effort: if TB_TRAY_URL is not set OR the tauri-worker isn't
reachable, every call silently no-ops. Safe to embed everywhere — zero
overhead when no Tauri is running.
"""

import asyncio
import json
import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional


# =============================================================================
# Server-side state
# =============================================================================

class TrayStateStore:
    """In-process aggregated state for the tray. Single instance per process."""

    def __init__(self, max_events: int = 200):
        self.workers: Dict[str, Dict[str, Any]] = {}
        self._events: deque = deque(maxlen=max_events)
        self._subscribers: List[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    def update_status(self, worker_id: str, payload: dict) -> dict:
        with self._lock:
            entry = self.workers.setdefault(worker_id, {})
            entry.update(payload)
            entry["last_seen"] = time.time()
            entry["worker_id"] = worker_id
            ev = {"type": "worker_status", "worker_id": worker_id, **entry, "ts": time.time()}
            self._events.append(ev)
        self._fanout(ev)
        return entry

    def add_notification(self, source: str, message: str, level: str = "info") -> dict:
        ev = {
            "type": "notification",
            "source": source,
            "message": message,
            "level": level,
            "ts": time.time(),
        }
        with self._lock:
            self._events.append(ev)
        self._fanout(ev)
        return ev

    def state(self) -> dict:
        with self._lock:
            workers = dict(self.workers)
        active = [w for w in workers.values() if w.get("running")]
        return {
            "workers": workers,
            "summary": {
                "total": len(workers),
                "running": len(active),
                "tooltip": (
                    f"ToolBox — {len(active)}/{len(workers)} workers up"
                    if workers else "ToolBox — idle"
                ),
            },
            "now": time.time(),
        }

    def recent_events(self, since_ts: Optional[float] = None) -> List[dict]:
        with self._lock:
            evs = list(self._events)
        if since_ts is None:
            return evs
        return [e for e in evs if e.get("ts", 0) > since_ts]

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe():
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def _fanout(self, event: dict) -> None:
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                pass

    @property
    def has_subscribers(self) -> bool:
        with self._lock:
            return len(self._subscribers) > 0

    def emit(self, event: dict) -> None:
        """Push a raw event into the stream — used by emit_open_url and helpers."""
        with self._lock:
            self._events.append(event)
        self._fanout(event)


_default_store: Optional[TrayStateStore] = None


def get_store() -> TrayStateStore:
    """Process-wide singleton store."""
    global _default_store
    if _default_store is None:
        _default_store = TrayStateStore()
    return _default_store


def emit_open_url(url: str, target: str = "main", store: Optional[TrayStateStore] = None) -> dict:
    """Ask Tauri (or any SSE listener) to open / navigate to a URL.

    Tauri subscribes to /tray/events; when it sees an event with type=open_url
    it navigates the window with the given target id (default 'main') to the URL,
    or creates one if it doesn't exist.
    """
    ev = {
        "type": "open_url",
        "url": url,
        "target": target,
        "ts": time.time(),
    }
    (store or get_store()).emit(ev)
    return ev


def has_active_subscribers(store: Optional[TrayStateStore] = None) -> bool:
    """True if at least one client (typically Tauri) is currently SSE-subscribed."""
    return (store or get_store()).has_subscribers


# =============================================================================
# Mount helper — adds /tray/* routes to an existing FastTB app
# =============================================================================

def mount_tray_api(app, store: Optional[TrayStateStore] = None) -> TrayStateStore:
    """Register tray endpoints on a FastTB app. Returns the store used."""
    store = store or get_store()

    def _read_body_as_dict(request) -> dict:
        body = getattr(request, "body", None)
        if isinstance(body, bytes) and body:
            try:
                return json.loads(body.decode("utf-8"))
            except Exception:
                pass
        form = getattr(request, "form", None)
        if form:
            return dict(form)
        return {}

    @app.post("/tray/status")
    async def post_status(request):
        data = _read_body_as_dict(request)
        worker_id = (data.get("worker_id") or "").strip()
        if not worker_id:
            return {"error": "worker_id required"}
        payload = {
            "label": data.get("label", worker_id),
            "running": bool(data.get("running", False)),
            "metric": data.get("metric"),
            "category": data.get("category", "worker"),
            "pid": data.get("pid"),
            "url": data.get("url"),
        }
        # Drop None values for cleaner state
        payload = {k: v for k, v in payload.items() if v is not None}
        store.update_status(worker_id, payload)
        return {"ok": True}

    @app.post("/tray/notify")
    async def post_notify(request):
        data = _read_body_as_dict(request)
        source = (data.get("source") or "unknown").strip()
        message = (data.get("message") or "").strip()
        level = (data.get("level") or "info").strip().lower()
        if not message:
            return {"error": "message required"}
        ev = store.add_notification(source, message, level)
        return {"ok": True, "event": ev}

    @app.get("/tray/state")
    async def get_state(request):
        return store.state()

    @app.sse("/tray/events")
    async def sse_events(request):
        """Tauri (or anything else) subscribes here for live updates."""
        # Emit current state immediately so a fresh subscriber gets oriented
        yield {"event": "state", "data": store.state()}

        # Subscribe and bridge via asyncio.Queue
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        loop = asyncio.get_event_loop()

        def _push(ev: dict):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ev)
            except Exception:
                pass

        unsubscribe = store.subscribe(_push)
        try:
            while True:
                ev = await queue.get()
                yield {"event": ev.get("type", "message"), "data": ev}
        finally:
            unsubscribe()

    @app.post("/tray/command")
    async def post_command(request):
        """Tauri asks for an action. Translates to a registered handler.

        Body: {"command": "start_worker"|"stop_worker"|"open_url"|..., "args": {...}}
        Handlers are added via `register_command_handler(name, callable)`.
        """
        data = _read_body_as_dict(request)
        cmd = (data.get("command") or "").strip()
        args = data.get("args") or {}
        handler = _command_handlers.get(cmd)
        if not handler:
            return {"error": f"unknown command: {cmd}", "available": sorted(_command_handlers.keys())}
        try:
            result = handler(**args) if not asyncio.iscoroutinefunction(handler) else await handler(**args)
            return {"ok": True, "result": result}
        except TypeError as e:
            return {"error": f"bad args for {cmd}: {e}"}
        except Exception as e:
            return {"error": str(e)}

    return store


# Pluggable command handlers — registered by the tauri-worker for Tauri to call.
_command_handlers: Dict[str, Callable[..., Any]] = {}


def register_command_handler(name: str, fn: Callable[..., Any]) -> None:
    """Expose a function to Tauri under /tray/command with `command=<name>`."""
    _command_handlers[name] = fn


# =============================================================================
# Client side — what every other worker embeds
# =============================================================================

class TrayClient:
    """Best-effort publisher to the tray API. Embed in any worker."""

    def __init__(
        self,
        worker_id: str,
        label: Optional[str] = None,
        base_url: Optional[str] = None,
        category: str = "worker",
        timeout: float = 0.6,
    ):
        self.worker_id = worker_id
        self.label = label or worker_id
        self.category = category
        self.timeout = timeout
        self._base = (base_url or os.getenv("TB_TRAY_URL", "")).rstrip("/")
        self._lock = threading.Lock()
        self._last_sent: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._base)

    def _post(self, path: str, body: dict) -> None:
        if not self.enabled:
            return
        try:
            # Use stdlib only to avoid forcing httpx/requests on every worker
            import urllib.request
            req = urllib.request.Request(
                f"{self._base}{path}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception:
            # Silent — tray-API may not be running, that's fine
            pass

    def report(
        self,
        running: bool = True,
        metric: Optional[str] = None,
        url: Optional[str] = None,
        pid: Optional[int] = None,
        **extra,
    ) -> None:
        payload = {
            "worker_id": self.worker_id,
            "label": self.label,
            "category": self.category,
            "running": running,
            "metric": metric,
            "url": url,
            "pid": pid if pid is not None else os.getpid(),
            **extra,
        }
        with self._lock:
            self._last_sent = payload
        self._post("/tray/status", payload)

    def notify(self, message: str, level: str = "info") -> None:
        self._post("/tray/notify", {
            "source": self.worker_id,
            "message": message,
            "level": level,
        })

    def heartbeat(self, interval_s: float = 5.0, get_metric: Optional[Callable[[], str]] = None) -> threading.Thread:
        """Spawn a daemon thread that re-publishes status at intervals.

        Useful to detect dead workers — the tray UI can mark workers stale
        when last_seen is too old.
        """
        stop_event = threading.Event()

        def _loop():
            while not stop_event.wait(interval_s):
                try:
                    metric = get_metric() if get_metric else None
                    last = dict(self._last_sent)
                    last["metric"] = metric or last.get("metric")
                    self.report(**{k: v for k, v in last.items() if k != "worker_id"})
                except Exception:
                    pass

        t = threading.Thread(target=_loop, daemon=True, name=f"tray-heartbeat-{self.worker_id}")
        t.start()
        t.stop = stop_event.set  # type: ignore[attr-defined]
        return t

    def shutdown(self) -> None:
        """Mark this worker as stopped — call from your cleanup path."""
        self.report(running=False)
