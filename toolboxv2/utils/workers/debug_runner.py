#!/usr/bin/env python3
"""
debug_runner.py - Uvicorn-compatible Debug Server für ToolBoxV2

Simuliert das Nginx-Produktionsrouting lokal auf Windows.
Serviert das statische Frontend (dist) und routet API/Auth Endpoints
zum ToolBoxV2 HTTPWorker auf dem SELBEN Port (:5000). So werden CORS-
und Cookie-Probleme beim Debugging umgangen.

Usage:
    pip install uvicorn a2wsgi
    python debug_runner.py --dist ./dist --port 5000
"""

import argparse
import asyncio
import logging
import mimetypes
import os
import sys
import threading

import uvicorn

try:
    from a2wsgi import WSGIMiddleware
    IS_A2WSGI = True
except ImportError:
    IS_A2WSGI = False
    WSGIMiddleware = None

# ToolBoxV2 Imports (Passe die Pfade an, falls du das Skript verschiebst)
from toolboxv2.utils.workers.config import load_config
from toolboxv2.utils.workers.event_manager import ZMQEventManager
from toolboxv2.utils.workers.server_worker import HTTPWorker, ToolBoxHandler
from toolboxv2.utils.workers.ws_worker import WSWorker
from toolboxv2.utils.system.getting_and_closing_app import get_app

logger = logging.getLogger("DebugRunner")


class DebugASGIDispatcher:
    """
    Ein reiner ASGI Dispatcher. Agiert exakt wie dein Nginx in Produktion.
    Ist absolut "blind" für die ToolBoxV2-Logik und reicht Request strikt durch.
    """

    def __init__(self, api_asgi_app, dist_path):
        self.api_asgi_app = api_asgi_app
        self.dist_path = os.path.abspath(dist_path)

        # Alle Endpunkte, die der HTTPWorker behandelt
        self.api_prefixes = (
            "/api", "/auth", "/validateSession", "/IsValidSession",
            "/web/logoutS", "/api_user_data", "/health", "/metrics"
        )

    async def __call__(self, scope, receive, send):
        # Ignoriere Lifespan-Events, da wir manuell starten
        if scope["type"] not in ("http", "websocket"):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            return

        path = scope.get("path", "/")

        # 1. API Routing (Weiterleitung an den eingepackten WSGI-HTTPWorker)
        if path.startswith(self.api_prefixes):
            await self.api_asgi_app(scope, receive, send)
            return

        # 2. Static File Routing (Frontend / dist)
        if scope["type"] == "http":
            if scope["method"] not in ("GET", "HEAD"):
                await self._send_status(send, 405, b"Method Not Allowed")
                return

            file_path = os.path.join(self.dist_path, path.lstrip("/"))

            # SPA (Single Page Application) Fallback -> wenn File nicht existiert, nutze index.html
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                file_path = os.path.join(self.dist_path, "index.html")

            if os.path.exists(file_path) and os.path.isfile(file_path):
                await self._serve_file(file_path, send)
            else:
                await self._send_status(send, 404, b"Not Found: Dist Ordner fehlt oder ungueltig.")
        else:
            # WebSocket Verbindungen auf Port 5000 verwerfen (WS Worker lauscht auf eigenem Port)
            await send({"type": "websocket.close"})

    async def _serve_file(self, file_path, send):
        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or "application/octet-stream"
        size = os.path.getsize(file_path)

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", content_type.encode()),
                (b"content-length", str(size).encode()),
                (b"cache-control", b"no-cache, no-store, must-revalidate")  # Caching beim Debuggen deaktivieren
            ]
        })
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True
                })
        await send({
            "type": "http.response.body",
            "body": b"",
            "more_body": False
        })

    async def _send_status(self, send, status_code, message):
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"text/plain")]
        })
        await send({
            "type": "http.response.body",
            "body": message,
            "more_body": False
        })


def run_debug_server(dist_path: str, port: int):
    if not IS_A2WSGI:
        print("FEHLER: Bitte a2wsgi installieren: pip install a2wsgi uvicorn")
        return
    # 1. Setup Environment (Windows Event-Loop fix für ZMQ)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    config = load_config()
    config.manager.web_ui_enabled = False  # Konflikte mit anderen Diensten vermeiden

    # 2. Worker Infrastruktur (ZMQ Broker & WS Worker) im Hintergrund starten
    def start_infrastructure():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        logger.info("Starte internen ZMQ Broker...")
        broker = ZMQEventManager(
            worker_id="debug_broker",
            pub_endpoint=config.zmq.pub_endpoint,
            sub_endpoint=config.zmq.sub_endpoint,
            req_endpoint=config.zmq.req_endpoint,
            rep_endpoint=config.zmq.rep_endpoint,
            http_to_ws_endpoint=config.zmq.http_to_ws_endpoint,
            is_broker=True,
        )
        loop.run_until_complete(broker.start())

        if os.environ.get("TB_WS_ENABLED", "true").lower() in ("true", "1", "yes"):
            logger.info(f"Starte WS Worker auf Port {config.ws_worker.port}...")
            ws_worker = WSWorker("debug_ws", config)
            loop.run_until_complete(ws_worker.start())

        loop.run_forever()

    threading.Thread(target=start_infrastructure, daemon=True, name="InfraThread").start()

    # 3. HTTP Worker & App initialisieren (Ohne seinen internen HTTP-Server zu starten)
    logger.info("Initialisiere ToolBoxV2 HTTP Worker Logik...")
    app_instance = get_app(name="debug_instance", from_="DebugRunner")
    http_worker = HTTPWorker("debug_http", config, app=app_instance)

    http_worker._init_session_manager()
    http_worker._init_access_controller()
    http_worker._init_auth_handler()
    http_worker._toolbox_handler = ToolBoxHandler(
        app_instance, config, http_worker._access_controller, config.toolbox.api_prefix
    )

    # HTTP EventManager im Background Loop
    def start_http_em():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(http_worker._init_event_manager())
        loop.run_forever()

    threading.Thread(target=start_http_em, daemon=True, name="HttpEmThread").start()

    # 4. Kombinierten ASGI Server generieren
    # a2wsgi packt den WSGI Worker in eine ASGI konforme Klasse ein, bewahrt aber alle ENV-Pfade!
    api_asgi_app = WSGIMiddleware(http_worker.wsgi_app)
    debug_app = DebugASGIDispatcher(api_asgi_app, dist_path)

    # 5. Uvicorn starten
    logger.info(f"========== DEBUG SERVER READY ==========")
    logger.info(f"Frontend Pfad: {os.path.abspath(dist_path)}")
    logger.info(f"Web Interface: http://localhost:{port}")
    logger.info(f"========================================")

    uvicorn.run(debug_app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToolBoxV2 Unified Debug Server (Uvicorn)")
    parser.add_argument("--dist", default="./dist", help="Pfad zum statischen Frontend (z.B. ./dist)")
    parser.add_argument("--port", type=int, default=5000, help="Port des Servers (Frontend + API)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-Logs aktivieren")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    run_debug_server(args.dist, args.port)
