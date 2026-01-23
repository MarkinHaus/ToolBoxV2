#!/usr/bin/env python3
"""
tauri_integration.py - Tauri Desktop App Integration

Provides seamless integration for running the worker system
inside a Tauri application.

Features:
- Single-process mode for desktop
- Embedded HTTP/WS servers (unified management)
- IPC via Tauri commands
- Auto-configuration for local use
- WS worker bundled with HTTP worker for production

Architecture:
- HTTP Worker: Handles all API requests, auth, ToolBox module calls
- WS Worker: Handles WebSocket connections for real-time features
- Both share the same ToolBox app instance and communicate via ZMQ
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
    Unified worker manager for Tauri desktop apps.

    Manages both HTTP and WS workers in a single process,
    optimized for single-user local operation.

    The HTTP worker does the "underlying work" (ToolBox calls, auth, etc.)
    while the WS worker provides real-time WebSocket connections.
    Both are bundled together for production builds.
    """

    def __init__(self, config=None):
        self._config = config
        self._http_worker = None
        self._ws_worker = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None
        self._app = None
        self._ws_enabled = True  # Can be disabled via config/env

    def _get_config(self):
        """Get or create configuration."""
        if self._config:
            return self._config

        # Set Tauri environment
        os.environ["TAURI_ENV"] = "true"
        os.environ["TB_ENV"] = "tauri"

        try:
            from toolboxv2.utils.workers.config import load_config
            self._config = load_config()
            return self._config
        except ImportError:
            logger.warning("ToolBoxV2 config not available, using defaults")
            raise

    def _init_app(self):
        """Initialize ToolBoxV2 app (shared between HTTP and WS workers)."""
        if self._app:
            return self._app

        try:
            from toolboxv2.utils.system.getting_and_closing_app import get_app
            self._app = get_app(name="tauri_worker", from_="TauriIntegration")
            logger.info(f"ToolBoxV2 app initialized: {self._app}")
            return self._app
        except ImportError:
            logger.warning("ToolBoxV2 not available, running in standalone mode")
            return None

    async def _run_servers(self):
        """
        Run HTTP and WS servers in unified mode.

        HTTP Worker runs in a separate thread (WSGI is blocking).
        WS Worker runs in the async event loop.
        Both share the same ToolBox app instance.
        """
        config = self._get_config()

        # Check if WS is enabled
        self._ws_enabled = os.environ.get("TB_WS_ENABLED", "true").lower() in ("true", "1", "yes")

        # Initialize shared app instance
        self._init_app()

        # Import workers
        from toolboxv2.utils.workers.server_worker import HTTPWorker

        # Create HTTP worker with shared app
        self._http_worker = HTTPWorker("tauri_http", config, app=self._app)

        # Start HTTP worker in thread (WSGI is blocking)
        def run_http():
            logger.info(f"Starting HTTP worker on {config.http_worker.host}:{config.http_worker.port}")
            try:
                self._http_worker.run(
                    host=config.http_worker.host,
                    port=config.http_worker.port,
                    do_run=True,  # Actually run the server
                )
            except Exception as e:
                logger.error(f"HTTP worker error: {e}")

        self._http_thread = threading.Thread(target=run_http, daemon=True, name="http-worker")
        self._http_thread.start()
        logger.info(f"HTTP worker thread started (PID: {os.getpid()})")

        # Start WS worker if enabled
        if self._ws_enabled:
            from toolboxv2.utils.workers.ws_worker import WSWorker

            self._ws_worker = WSWorker("tauri_ws", config)

            logger.info(f"Starting WS worker on {config.ws_worker.host}:{config.ws_worker.port}")

            # Mark as running
            self._running = True

            # Run WS server (async, blocks until stopped)
            # Note: start() internally calls _init_event_manager() and _init_direct_pull()
            await self._ws_worker.start()
        else:
            logger.info("WS worker disabled, running HTTP-only mode")
            self._running = True

            # Keep running without WS
            while self._running:
                await asyncio.sleep(1)

    def start(self):
        """Start workers in background thread."""
        if self._running:
            logger.warning("Workers already running")
            return

        # Windows: Use SelectorEventLoop for ZMQ compatibility
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run_servers())
            except Exception as e:
                logger.error(f"Worker manager error: {e}")
                import traceback
                traceback.print_exc()

        self._thread = threading.Thread(target=run, daemon=True, name="tauri-worker-manager")
        self._thread.start()

        logger.info("Tauri worker manager started")

    def stop(self):
        """Stop all workers gracefully."""
        logger.info("Stopping Tauri workers...")
        self._running = False

        # Stop WS worker
        if self._ws_worker and self._loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._ws_worker.stop(),
                    self._loop
                )
                future.result(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping WS worker: {e}")

        # Stop event loop
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for threads
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("Tauri workers stopped")

    def get_http_url(self) -> str:
        """Get HTTP server URL."""
        config = self._get_config()
        return f"http://{config.http_worker.host}:{config.http_worker.port}"

    def get_ws_url(self) -> str:
        """Get WebSocket server URL."""
        config = self._get_config()
        if not self._ws_enabled:
            return None
        return f"ws://{config.ws_worker.host}:{config.ws_worker.port}"

    def is_ws_enabled(self) -> bool:
        """Check if WS worker is enabled."""
        return self._ws_enabled


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
        "ws_url": manager.get_ws_url() if manager._running and manager.is_ws_enabled() else None,
        "ws_enabled": manager.is_ws_enabled(),
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
    """
    Run Tauri worker manager standalone.

    This is the entry point for the bundled sidecar binary.
    It starts both HTTP and WS workers in a unified process.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Tauri Worker Manager - Unified HTTP/WS Server",
        prog="tb-worker"
    )
    parser.add_argument("--http-port", type=int, default=5000,
                        help="HTTP server port (default: 5000)")
    parser.add_argument("--ws-port", type=int, default=5001,
                        help="WebSocket server port (default: 5001)")
    parser.add_argument("--mode", default="tauri",
                        help="Run mode (tauri, standalone)")
    parser.add_argument("--no-ws", action="store_true",
                        help="Disable WebSocket server")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("-c", "--config", help="Config file path")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Set environment variables for config
    os.environ["TB_HTTP_PORT"] = str(args.http_port)
    os.environ["TB_WS_PORT"] = str(args.ws_port)
    if args.no_ws:
        os.environ["TB_WS_ENABLED"] = "false"
    if args.verbose:
        os.environ["TB_DEBUG"] = "1"
        os.environ["TOOLBOX_LOGGING_LEVEL"] = "DEBUG"

    logger.info(f"Starting Tauri Worker Manager")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  HTTP Port: {args.http_port}")
    logger.info(f"  WS Port: {args.ws_port}")
    logger.info(f"  WS Enabled: {not args.no_ws}")

    # Start manager
    result = tauri_start_workers()
    print(f"Started: {json.dumps(result, indent=2)}")

    if result.get("status") == "error":
        logger.error(f"Failed to start workers: {result.get('message')}")
        sys.exit(1)

    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        tauri_stop_workers()
        print("Stopped")


if __name__ == "__main__":
    main()
