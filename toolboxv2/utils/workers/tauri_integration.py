#!/usr/bin/env python3
"""
tauri_integration.py - Tauri Desktop App Integration

Provides seamless integration for running the worker system
inside a Tauri application.

Features:
- Single-process mode for desktop
- Embedded HTTP/WS servers
- IPC via Tauri commands
- Auto-configuration for local use
"""

import asyncio
import json
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class TauriWorkerManager:
    """
    Lightweight worker manager for Tauri desktop apps.

    Runs HTTP and WS workers in the same process,
    optimized for single-user local operation.
    """

    def __init__(self, config=None):
        self._config = config
        self._http_server = None
        self._ws_server = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._app = None

    def _get_config(self):
        """Get or create configuration."""
        if self._config:
            return self._config

        # Set Tauri environment
        os.environ["TAURI_ENV"] = "true"
        os.environ["TB_ENV"] = "tauri"

        try:
            from toolboxv2.utils.workers.config import load_config
            return load_config()
        except ImportError:
            logger.warning("ToolBoxV2 config not available, using defaults")
            raise

    def _init_app(self):
        """Initialize ToolBoxV2 app."""
        if self._app:
            return self._app

        try:
            from toolboxv2.utils.system.getting_and_closing_app import get_app
            self._app = get_app()
            return self._app
        except ImportError:
            logger.warning("ToolBoxV2 not available, running in standalone mode")
            return None

    async def _run_servers(self):
        """Run HTTP and WS servers."""
        config = self._get_config()

        # Initialize app
        self._init_app()

        # Import workers
        from toolboxv2.utils.workers.server_worker import HTTPWorker
        from toolboxv2.utils.workers.ws_worker import WSWorker

        # Create workers
        http_worker = HTTPWorker("tauri_http", config, app=self._app)
        ws_worker = WSWorker("tauri_ws", config)

        # Start WS server
        await ws_worker._init_event_manager()

        # Run HTTP in thread (WSGI is blocking)
        def run_http():
            print(f"RUN HTTP WORKER: {config.http_worker.host}:{config.http_worker.port}")
            http_worker.run(
                host=config.http_worker.host,
                port=config.http_worker.port,
                do_run=False,
            )

        http_thread = threading.Thread(target=run_http, daemon=True)
        http_thread.start()

        # Run WS server
        self._running = True
        await ws_worker.start()

    def start(self):
        """Start workers in background thread."""
        if self._running:
            return

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_servers())

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

        logger.info("Tauri workers started")

    def stop(self):
        """Stop workers."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        logger.info("Tauri workers stopped")

    def get_http_url(self) -> str:
        """Get HTTP server URL."""
        config = self._get_config()
        return f"http://{config.http_worker.host}:{config.http_worker.port}"

    def get_ws_url(self) -> str:
        """Get WebSocket server URL."""
        config = self._get_config()
        return f"ws://{config.ws_worker.host}:{config.ws_worker.port}"


# ============================================================================
# Tauri Command Handlers
# ============================================================================

# Global manager instance
_manager: Optional[TauriWorkerManager] = None


def get_manager() -> TauriWorkerManager:
    """Get or create the global manager."""
    global _manager
    if _manager is None:
        _manager = TauriWorkerManager()
    return _manager


def tauri_start_workers() -> Dict[str, Any]:
    """Start workers (Tauri command)."""
    try:
        manager = get_manager()
        manager.start()
        return {
            "status": "ok",
            "http_url": manager.get_http_url(),
            "ws_url": manager.get_ws_url(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tauri_stop_workers() -> Dict[str, Any]:
    """Stop workers (Tauri command)."""
    try:
        manager = get_manager()
        manager.stop()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tauri_get_status() -> Dict[str, Any]:
    """Get worker status (Tauri command)."""
    manager = get_manager()
    return {
        "running": manager._running,
        "http_url": manager.get_http_url() if manager._running else None,
        "ws_url": manager.get_ws_url() if manager._running else None,
    }


def tauri_call_module(
    module: str,
    function: str,
    args: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Call ToolBoxV2 module function (Tauri command).

    Direct IPC without HTTP for better performance.
    """
    manager = get_manager()

    if not manager._app:
        return {"status": "error", "message": "App not initialized"}

    try:
        result = manager._app.run_any(
            (module, function),
            get_results=True,
            **(args or {}),
        )

        if hasattr(result, "get"):
            return {"status": "ok", "data": result.get()}
        return {"status": "ok", "data": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Run Tauri worker manager standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Tauri Worker Manager")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--ws-port", type=int, default=8001)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Start manager
    result = tauri_start_workers()
    print(f"Started: {json.dumps(result, indent=2)}")

    # Keep running
    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        tauri_stop_workers()
        print("Stopped")


if __name__ == "__main__":
    main()
