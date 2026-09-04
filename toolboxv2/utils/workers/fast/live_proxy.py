"""
live_proxy.py - /live sub-mount auf der live-ui (Port 8467).

Die Worker-HTTP-Instanzen servieren das /live Dashboard direkt (Key-Gate via
LIVE_DASHBOARD_KEY env oder manager.live_dashboard_key). Diese Bridge macht es
unter der kanonischen Live-UI-URL erreichbar - als Proxy-Pass-Through, damit
der Browser NICHT auf den internen Worker-Port (8000) umgeleitet wird:

    GET  /live              -> Dashboard-HTML vom Worker (proxied, Bytes 1:1)
    GET  /live/snapshot     -> Topology-JSON (proxied)
    GET+POST /live/mgr/...  -> Manager-Proxy des Workers (proxied)
    GET  /live/health       -> eigener Status (kein Proxy)

Security-Modell:
- Loopback-Guard auf TCP-Peer-Ebene (environ REMOTE_ADDR, XFF wird ignoriert).
- Doppeltes Key-Gate: outer Gate hier (leer = disabled) + Worker-Gate; beide
  lesen dieselbe Quelle (env > manager.live_dashboard_key).
- Header-Whitelist: nur Content-Type zum Upstream; X-Forwarded-For/Host werden
  NIEMALS kopiert (Spoof-Vektor). Response: nur Content-Type + no-store.
- Kein generischer Proxy: strikt /live-Praefix.

Registrierung (in local_ui.py VOR spa_fallback-Catch-all):
    from toolboxv2.utils.workers.fast.live_proxy import register_live_routes
    register_live_routes(app)
"""

from __future__ import annotations

import http.client
import os
import socket
import urllib.parse
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from toolboxv2.utils.workers.fast.endpoint import resolve_local_ui_endpoint

if TYPE_CHECKING:
    from toolboxv2.utils.workers.server_worker import ParsedRequest


def dashboard_key() -> str:
    """Key-Gate: env LIVE_DASHBOARD_KEY gewinnt, sonst manager.live_dashboard_key."""
    key = (os.getenv("LIVE_DASHBOARD_KEY") or "").strip()
    if key:
        return key
    try:
        from toolboxv2.utils.workers.config import load_config
        return str(getattr(load_config().manager, "live_dashboard_key", "") or "")
    except Exception:
        return ""


def _worker_candidates() -> list:
    """Laeufende HTTP-Worker-Ports via Manager-Status (portside truth)."""
    try:
        from toolboxv2.utils.clis.cli_worker_manager import WorkerManager
        status = WorkerManager().get_status() or {}
    except Exception:
        return []
    workers = status.get("workers", {})
    ports = []
    for info in workers.values():
        if not isinstance(info, dict):
            continue
        if info.get("worker_type") == "http" and info.get("state") == "running" and info.get("port"):
            ports.append(int(info["port"]))
    return ports


def _first_alive_worker(key: str) -> Optional[int]:
    """Erster Worker, der /live/snapshot mit dem Key beantwortet."""
    for port in _worker_candidates()[:4]:
        if _proxy_live_raw(
                port, "/live/snapshot", {"key": [key]}, "GET", None, None, timeout=1.0)[0] == 200:
            return port
    return None


def _proxy_live_raw(
    port: int,
    path: str,
    query_params: Any,
    method: str,
    body: Optional[bytes],
    content_type: Optional[str],
    timeout: float = 5.0,
) -> Tuple[int, Dict[str, str], bytes]:
    """Ein Request an den Worker, Antwort 1:1 als (status, headers, bytes).

    Bytes-Passthrough: kein JSON-Re-Encode, kein Charset-Verlust. Nur
    Content-Type/Cache-Control durchreichen - kein Content-Length/Transfer-
    Encoding vom Upstream (FastTB setzt Length selbst).
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        qs = urllib.parse.urlencode(query_params, doseq=True) if query_params else ""
        target = path + ("?" + qs if qs else "")
        conn.request(method, target, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        ct = resp.getheader("Content-Type", "application/octet-stream")
        return resp.status, {"Content-Type": ct, "Cache-Control": "no-store"}, data
    except (socket.timeout, OSError):
        return 504, {"Content-Type": "text/plain; charset=utf-8"}, b"live proxy: upstream timeout"
    finally:
        conn.close()


def _is_loopback(client_ip: str) -> bool:
    return client_ip in ("127.0.0.1", "::1", "localhost")


def _peer_is_local(request: "ParsedRequest") -> bool:
    """True nur wenn die TCP-Peer-IP loopback ist.

    client_ip kann via X-Forwarded-For gespoofed werden (server_worker L308
    vertraut XFF vor REMOTE_ADDR) - darum nur das environ trusten, das die
    echte Peer-Adresse haelt.
    """
    environ = getattr(request, "environ", None) or {}
    peer = str(environ.get("REMOTE_ADDR", "") or getattr(request, "client_ip", "") or "")
    return _is_loopback(peer)


def _ensure_local(request: "ParsedRequest"):
    """Loopback-Guard (Peer-Adresse aus environ, nicht spoofbare Header)."""
    if not _peer_is_local(request):
        return 403, {"error": "local-only endpoint"}
    return None


def register_live_routes(app) -> None:
    from toolboxv2.utils.workers.server_worker import ParsedRequest

    def _guard_and_port(request: "ParsedRequest"):
        """(err, None) oder (None, (key, port))."""
        err = _ensure_local(request)
        if err is not None:
            return err, None
        key = dashboard_key()
        if not key:
            return (403, {"error": "/live disabled - set LIVE_DASHBOARD_KEY or manager.live_dashboard_key"}), None
        return None, (key, _first_alive_worker(key))

    def _proxy(request: "ParsedRequest", method: str = "GET"):
        err, found = _guard_and_port(request)
        if err is not None:
            return err
        if not found:
            return 503, {"error": "no running worker serves /live"}
        key, port = found
        if not port:
            return 503, {"error": "no running worker serves /live"}
        # WS-Upgrade explizit abweisen (Dashboard ist fetch-basiert)
        if (getattr(request, "headers", None) or {}).get("Connection", "").lower() == "upgrade":
            return 501, {"error": "websocket proxying not supported on /live"}
        ct = (getattr(request, "headers", None) or {}).get("Content-Type")
        body = request.body if method == "POST" else None
        status, headers, data = _proxy_live_raw(
            port, request.path, request.query_params, method, body, ct)
        return status, headers, data

    # Reihenfolge: spezifische Pfade vor dem Bare-/live-Handler.
    @app.get("/live/snapshot", auth=False)
    def live_snapshot(request: ParsedRequest):
        return _proxy(request, "GET")

    @app.get("/live/mgr/{rest:path}", auth=False)
    def live_mgr_get(request: ParsedRequest, rest: str = ""):
        return _proxy(request, "GET")

    @app.post("/live/mgr/{rest:path}", auth=False)
    def live_mgr_post(request: ParsedRequest, rest: str = ""):
        return _proxy(request, "POST")

    @app.get("/live", auth=False)
    def live_entry(request: ParsedRequest):
        return _proxy(request, "GET")

    @app.get("/live/health", auth=False)
    def live_health(request: ParsedRequest):
        err = _ensure_local(request)
        if err is not None:
            return err
        key = dashboard_key()
        return {"ok": bool(key), "enabled": bool(key),
                "workers": _worker_candidates() if key else []}


# resolve_local_ui_endpoint import bleibt bewusst: Doku-Kontext fuer Port-Kanon 8467
_ = resolve_local_ui_endpoint
