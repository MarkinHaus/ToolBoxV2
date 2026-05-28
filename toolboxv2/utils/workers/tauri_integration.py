#!/usr/bin/env python3
"""
toolboxv2/utils/workers/tauri_integration.py - Tauri Worker

Owns three pieces in one process, mirroring the original architecture but
replacing HTTPWorker with our FastTB local_ui:
  • FastTB server (local_ui + tray-API) — serves UI on http://127.0.0.1:5000
  • ZMQ broker                          — internal bus, like the original
  • WS worker                            — realtime channel on 127.0.0.1:5001

Other Python workers (tb workers, tb http_worker, …) start as separate
processes and connect to this one via TB_TRAY_URL for tray status updates.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from typing import Any, Dict, Optional

from toolboxv2 import get_logger

logger = get_logger()

DEFAULT_HOST = "127.0.0.1"
# Default 5000 matches worker_manager.rs:DEFAULT_HTTP_PORT and is_healthy() check.
DEFAULT_PORT = int(os.getenv("TB_HTTP_PORT", "5000"))
DEFAULT_WS_PORT = int(os.getenv("TB_WS_PORT", "5001"))


# =============================================================================
# Command handlers — what Tauri's tray menu can ask Python to do
# =============================================================================

def _cmd_start_worker(name: str = "workers", **_) -> dict:
    """Spawn `tb <name>` as a detached background process."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "toolboxv2", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=(sys.platform != "win32"),
        )
        return {"spawned_pid": proc.pid, "command": f"tb {name}"}
    except Exception as e:
        return {"error": str(e)}


def _cmd_stop_worker(worker_id: str, **_) -> dict:
    """Ask the named worker to stop via its pid in the tray state."""
    from .fast.tray_api import get_store
    info = get_store().workers.get(worker_id)
    if not info:
        return {"error": f"unknown worker_id: {worker_id}"}
    pid = info.get("pid")
    if not pid:
        return {"error": "no pid on record"}
    try:
        os.kill(int(pid), signal.SIGTERM)
        return {"sent_signal": "SIGTERM", "pid": pid}
    except Exception as e:
        return {"error": str(e)}


def _cmd_open_url(url: str, target: str = "main", **_) -> dict:
    """Have Tauri navigate to the given URL via the open_url tray event."""
    from .fast.tray_api import emit_open_url
    emit_open_url(url, target=target)
    return {"emitted": url, "target": target}


def _cmd_health(**_) -> dict:
    return {"ok": True, "pid": os.getpid(), "uptime_s": time.time() - _STARTED_AT}


_STARTED_AT = time.time()


# =============================================================================
# Worker
# =============================================================================

class TauriWorker:
    """FastTB UI + ZMQ broker + WS worker, all in one Python process."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        ws_port: int = DEFAULT_WS_PORT,
        ws_enabled: bool = True,
    ):
        self.host = host
        self.port = port
        self.ws_port = ws_port
        self.ws_enabled = ws_enabled

        self._http_server = None              # waitress server (FastTB)
        self._fasttb_thread: Optional[threading.Thread] = None

        self._app = None                       # ToolBox app shared across components
        self._broker = None                    # ZMQEventManager
        self._ws_worker = None                 # WSWorker
        self._async_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._self_tray = None
        self._running = False

    # ------------------------------------------------------------------
    # FastTB (HTTP) — runs in its own thread, blocking
    # ------------------------------------------------------------------

    def _run_fasttb(self):
        from .fast.local_ui import app as local_ui_app
        from .fast.tray_api import (
            mount_tray_api, register_command_handler, TrayClient,
        )
        from .fast_tb_handler import FastTBHandler
        from waitress import create_server

        # 1) Mount tray API onto local_ui's FastTB app
        mount_tray_api(local_ui_app)

        # 2) Register tray commands
        register_command_handler("start_worker", _cmd_start_worker)
        register_command_handler("stop_worker", _cmd_stop_worker)
        register_command_handler("open_url", _cmd_open_url)
        register_command_handler("health", _cmd_health)

        # 3) Publish own URL so embedded TrayClients discover us
        base_url = f"http://{self.host}:{self.port}"
        os.environ["TB_TRAY_URL"] = base_url

        # 4) Self-report into the tray
        self._self_tray = TrayClient(
            worker_id="tauri_worker", label="Tauri / Local UI",
            category="core", base_url=base_url,
        )
        self._self_tray.report(running=True, url=base_url)

        handler = FastTBHandler(local_ui_app)
        wsgi_app = handler.as_wsgi_app(enable_ws=False)

        logger.info(f"[tauri-worker] FastTB on http://{self.host}:{self.port}")
        self._http_server = create_server(
            wsgi_app, host=self.host, port=self.port, ident="ToolBoxTauri"
        )
        try:
            self._http_server.run()
        except Exception as e:
            logger.error(f"[tauri-worker] FastTB error: {e}")
        finally:
            if self._self_tray:
                try: self._self_tray.shutdown()
                except Exception: pass

    # ------------------------------------------------------------------
    # ZMQ broker + WS worker — run on an asyncio loop in another thread
    # ------------------------------------------------------------------

    def _init_toolbox_app(self):
        if self._app is not None:
            return self._app
        from toolboxv2.utils.system.getting_and_closing_app import get_app
        self._app = get_app(name="tauri_worker", from_="TauriIntegration")
        logger.info(f"[tauri-worker] ToolBox app initialized: {self._app}")
        return self._app

    async def _run_broker_and_ws(self):
        """Start ZMQ broker and WS worker. Mirrors the original tauri_integration."""
        try:
            from .config import load_config
            config = load_config()
        except Exception as e:
            logger.error(f"[tauri-worker] config load failed, broker+ws disabled: {e}")
            return

        # ZMQ Broker
        from .event_manager import ZMQEventManager
        logger.info("[tauri-worker] starting ZMQ broker…")
        self._broker = ZMQEventManager(
            worker_id="internal_broker",
            pub_endpoint=config.zmq.pub_endpoint,
            sub_endpoint=config.zmq.sub_endpoint,
            req_endpoint=config.zmq.req_endpoint,
            rep_endpoint=config.zmq.rep_endpoint,
            http_to_ws_endpoint=config.zmq.http_to_ws_endpoint,
            cluster_secret=getattr(config.zmq, "cluster_secret", ""),
            hwm_send=config.zmq.hwm_send,
            hwm_recv=config.zmq.hwm_recv,
        )
        await self._broker.start()
        await asyncio.sleep(0.5)
        logger.info("[tauri-worker] broker started")

        # Toolbox app (shared)
        self._init_toolbox_app()

        # WS Worker
        if self.ws_enabled:
            from .ws_worker import WSWorker
            self._ws_worker = WSWorker("tauri_ws", config)
            logger.info(
                f"[tauri-worker] WS worker on {config.ws_worker.host}:{config.ws_worker.port}"
            )
            # Publish WS into the tray
            try:
                from .fast.tray_api import TrayClient
                ws_tray = TrayClient(
                    worker_id="tauri_ws", label="WS Worker",
                    category="core", base_url=os.environ.get("TB_TRAY_URL", ""),
                )
                ws_tray.report(
                    running=True,
                    url=f"ws://{config.ws_worker.host}:{config.ws_worker.port}",
                )
            except Exception:
                pass
            await self._ws_worker.start()
        else:
            logger.info("[tauri-worker] WS disabled")
            while self._running:
                await asyncio.sleep(1)

    def _run_async_thread(self):
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_broker_and_ws())
        except Exception as e:
            logger.error(f"[tauri-worker] async stack error: {e}")
        finally:
            self._loop.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_blocking(self):
        if self._running:
            logger.warning("[tauri-worker] already running")
            return

        self._running = True

        # 1) FastTB in its own thread
        self._fasttb_thread = threading.Thread(
            target=self._run_fasttb, daemon=True, name="tauri-fasttb",
        )
        self._fasttb_thread.start()
        # Wait briefly so TB_TRAY_URL is set before the WS worker init
        for _ in range(40):
            if os.environ.get("TB_TRAY_URL"):
                break
            time.sleep(0.05)

        # 2) Broker + WS in another thread (blocks on its asyncio loop)
        self._async_thread = threading.Thread(
            target=self._run_async_thread, daemon=True, name="tauri-async",
        )
        self._async_thread.start()

        # 3) Signal handling in main thread
        if threading.current_thread() is threading.main_thread():
            def _shutdown(*_):
                logger.info("[tauri-worker] shutdown signal received")
                self.stop()
            try:
                signal.signal(signal.SIGINT, _shutdown)
                signal.signal(signal.SIGTERM, _shutdown)
            except (ValueError, RuntimeError):
                pass

        # 4) Park main thread until shutdown
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        if not self._running:
            return
        self._running = False
        # Self-tray shutdown
        if self._self_tray:
            try: self._self_tray.shutdown()
            except Exception: pass
        # Stop WS worker
        if self._ws_worker and self._loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._ws_worker.stop(), self._loop)
                fut.result(timeout=2)
            except Exception: pass
        # Stop broker
        if self._broker and self._loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._broker.stop(), self._loop)
                fut.result(timeout=2)
            except Exception: pass
        # Stop loop
        if self._loop:
            try: self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception: pass
        # Stop FastTB
        if self._http_server:
            try: self._http_server.close()
            except Exception: pass
        logger.info("[tauri-worker] stopped")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# =============================================================================
# Entry point — sidecar binary
# =============================================================================

def main():
    """Entry for the bundled `tb-worker` sidecar binary.

    Accepts the legacy sidecar args used by worker_manager.rs:
        --http-port  (FastTB server port; alias: --port)
        --ws-port    (WS worker port)
        --mode       (compat tag, ignored)
    """
    import argparse
    parser = argparse.ArgumentParser(prog="tb-worker")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--http-port", type=int, default=None)
    parser.add_argument("--ws-port", type=int, default=DEFAULT_WS_PORT)
    parser.add_argument("--mode", default="tauri")
    parser.add_argument("--no-ws", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    port = args.http_port if args.http_port is not None else (args.port if args.port is not None else DEFAULT_PORT)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.verbose:
        os.environ["TB_DEBUG"] = "1"
        os.environ["TOOLBOX_LOGGING_LEVEL"] = "DEBUG"
    os.environ["TAURI_ENV"] = "true"
    os.environ["TB_ENV"] = "tauri"

    worker = TauriWorker(
        host=args.host, port=port, ws_port=args.ws_port,
        ws_enabled=not args.no_ws,
    )
    try:
        worker.start_blocking()
    except KeyboardInterrupt:
        worker.stop()


# =============================================================================
# Backcompat — old-style Tauri command handlers
# =============================================================================

_singleton: Optional[TauriWorker] = None


def get_worker() -> TauriWorker:
    global _singleton
    if _singleton is None:
        _singleton = TauriWorker()
    return _singleton


def tauri_start_workers() -> Dict[str, Any]:
    try:
        w = get_worker()
        threading.Thread(target=w.start_blocking, daemon=True, name="tauri-worker-main").start()
        return {"status": "ok", "http_url": w.base_url, "ws_url": f"ws://{w.host}:{w.ws_port}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tauri_stop_workers() -> Dict[str, Any]:
    try:
        get_worker().stop()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def tauri_get_status() -> Dict[str, Any]:
    w = get_worker()
    return {
        "running": w._running,
        "http_url": w.base_url if w._running else None,
        "ws_url": f"ws://{w.host}:{w.ws_port}" if w._running and w.ws_enabled else None,
        "ws_enabled": w.ws_enabled,
    }


def tauri_call_module(module: str, function: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
    w = get_worker()
    if not w._app:
        return {"status": "error", "message": "App not initialized"}
    try:
        result = w._app.run_any((module, function), get_results=True, **(args or {}))
        if hasattr(result, "get"):
            return {"status": "ok", "data": result.get()}
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    main()
