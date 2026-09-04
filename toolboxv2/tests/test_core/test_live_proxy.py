"""E2E-lite: /live Proxy-Pass-Through auf live-ui (8467).

Proxy statt Redirect: Browser bleibt auf 8467, Worker-Port (8000) bleibt intern.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def _fake_request(path="/live/health", ip="127.0.0.1", query_params=None, method="GET",
                  body=b"", headers=None):
    from toolboxv2.utils.workers.server_worker import ParsedRequest
    return ParsedRequest(
        method=method, path=path, query_params=query_params or {},
        headers=headers or {}, body=body, json_data={}, form_data={},
        client_ip=ip, content_type="", content_length=len(body or b""),
        environ={"REMOTE_ADDR": ip},
    )


def _routes_registered():
    import toolboxv2.utils.workers.fast.local_ui as lu
    return (
        lu.app.has_route("/live", "GET"),
        lu.app.has_route("/live/snapshot", "GET"),
        lu.app.has_route("/live/mgr/{rest:path}", "GET"),
        lu.app.has_route("/live/mgr/{rest:path}", "POST"),
        lu.app.has_route("/live/health", "GET"),
    )


def test_live_routes_registered():
    live, snap, mgr_get, mgr_post, health = _routes_registered()
    assert live and snap and mgr_get and mgr_post and health


class _MockWorker(BaseHTTPRequestHandler):
    """Mini-Worker: /live/snapshot mit Key-Check, /live HTML, sonst 401/404."""

    def do_GET(self):  # noqa: N802
        if self.path == "/live/snapshot?key=secret":
            body = json.dumps({"workers": {"w1": {"running": True}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/live?key=secret":
            body = b"<html><body>LiveDashboard</body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/live/snapshot"):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"Unauthorized")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")

    def log_message(self, *args):  # silence
        pass


class _MockWorkerServer:
    def __init__(self):
        self.server = HTTPServer(("127.0.0.1", 0), _MockWorker)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False


def _patch_workers(monkey_targets):
    from toolboxv2.utils.workers.fast import live_proxy
    live_proxy._worker_candidates = lambda: monkey_targets
    return live_proxy


def _live_key():
    from contextlib import contextmanager

    @contextmanager
    def ctx():
        old = os.environ.get("LIVE_DASHBOARD_KEY")
        os.environ["LIVE_DASHBOARD_KEY"] = "secret"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("LIVE_DASHBOARD_KEY", None)
            else:
                os.environ["LIVE_DASHBOARD_KEY"] = old
    return ctx()


def test_proxy_passthrough_snapshot():
    """Snapshot wird 1:1 durchgereicht, Key-Query kommt am Worker an."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    with _MockWorkerServer() as mock, _live_key():
        _patch_workers([mock.port])
        route, _ = lu.app.resolve_route("/live/snapshot", "GET")
        res = route.handler(_fake_request("/live/snapshot", query_params={"key": ["secret"]}))
        assert isinstance(res, tuple) and res[0] == 200, f"expected 200, got {res!r}"
        body = json.loads(res[2].decode("utf-8"))
        assert body["workers"]["w1"]["running"] is True
        assert "application/json" in res[1]["Content-Type"]


def test_proxy_passthrough_html():
    """Dashboard-HTML kommt durch (Browser bleibt auf 8467, kein Redirect)."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    with _MockWorkerServer() as mock, _live_key():
        _patch_workers([mock.port])
        route, _ = lu.app.resolve_route("/live", "GET")
        res = route.handler(_fake_request("/live", query_params={"key": ["secret"]}))
        assert isinstance(res, tuple) and res[0] == 200
        assert b"LiveDashboard" in res[2]
        assert "text/html" in res[1]["Content-Type"]


def test_proxy_401_passthrough_exact():
    """Falscher Key -> Worker-401 exakt durchgereicht, nicht maskiert."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    with _MockWorkerServer() as mock, _live_key():
        _patch_workers([mock.port])
        route, _ = lu.app.resolve_route("/live/snapshot", "GET")
        res = route.handler(_fake_request("/live/snapshot", query_params={"key": ["wrong"]}))
        assert isinstance(res, tuple) and res[0] == 401


def test_proxy_no_worker_503():
    """Kein lebender Worker -> 503 JSON (kein Passthrough-Versuch)."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    with _live_key():
        _patch_workers([])
        route, _ = lu.app.resolve_route("/live", "GET")
        res = route.handler(_fake_request("/live", query_params={"key": ["secret"]}))
        assert isinstance(res, tuple) and res[0] == 503


def test_live_disabled_without_key():
    """Ohne LIVE_DASHBOARD_KEY: /live -> 403 disabled. Contract: (status, data)."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    old = os.environ.pop("LIVE_DASHBOARD_KEY", None)
    try:
        route, _ = lu.app.resolve_route("/live", "GET")
        res = route.handler(_fake_request("/live"))
        assert isinstance(res, tuple) and res[0] == 403, f"expected 403 tuple, got {res!r}"
    finally:
        if old is not None:
            os.environ["LIVE_DASHBOARD_KEY"] = old


def test_xff_spoof_blocked():
    """XFF-Spoof: client_ip sagt loopback, TCP-Peer (environ REMOTE_ADDR) ist remote -> 403."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    from toolboxv2.utils.workers.server_worker import ParsedRequest
    req = ParsedRequest(
        method="GET", path="/live/health", query_params={}, headers={},
        body=b"", json_data={}, form_data={}, client_ip="127.0.0.1",
        content_type="", content_length=0,
        environ={"REMOTE_ADDR": "203.0.113.5"},  # echter TCP-Peer: remote
    )
    route, _ = lu.app.resolve_route("/live/health", "GET")
    res = route.handler(req)
    assert isinstance(res, tuple) and res[0] == 403, f"spoofed XFF durchgelassen: {res!r}"


def test_live_loopback_guard():
    """Nicht-Loopback-Client -> 403 (local-only). Contract: (status, data)."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    route, _ = lu.app.resolve_route("/live/health", "GET")
    res = route.handler(_fake_request("/live/health", ip="203.0.113.7"))
    assert isinstance(res, tuple) and res[0] == 403


def test_live_health_reports_enabled_state():
    import toolboxv2.utils.workers.fast.local_ui as lu
    route, _ = lu.app.resolve_route("/live/health", "GET")
    res = route.handler(_fake_request("/live/health"))
    assert isinstance(res, dict) and "enabled" in res and "ok" in res
