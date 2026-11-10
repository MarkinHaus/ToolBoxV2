"""
E2E: local_ui (onboarding routes, wizard_api, /live) over a real HTTP stack.

Startet den FastTB local_ui-Server in einem Thread, spreicht die Endpunkte
ueber echtes HTTP an und prueft JSON-Contracts. Danach sauberer Shutdown.
"""
import json
import socket
import threading
import time
import urllib.request


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server:
    def __init__(self):
        import toolboxv2.utils.workers.fast.local_ui as lu
        self.lu = lu
        self.port = _free_port()
        self.thread = threading.Thread(
            target=lu.app.serve, kwargs={"host": "127.0.0.1", "port": self.port},
            daemon=True,
        )
        self.started = False

    def __enter__(self):
        self.thread.start()
        self.started = True
        for _ in range(50):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/live/health", timeout=1)
                break
            except Exception:
                time.sleep(0.2)
        return self

    def __exit__(self, *exc):
        return False

    def get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))

    def post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode("utf-8"))


def test_e2e_wizard_state_over_http():
    with _Server() as srv:
        status, data = srv.get("/api/wizard/state")
        assert status == 200
        assert data["first_run"] in (True, False)
        assert data["installed"] is True


def test_e2e_wizard_profile_over_http():
    with _Server() as srv:
        status, data = srv.post("/api/wizard/profile", {"profile": "developer"})
        assert status == 200 and data == {"ok": True, "profile": "developer"}


def test_browser_onboarding_flow_over_http():
    """installer.html browser path: first_run -> profile -> save -> onboarded."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import toolboxv2.utils.workers.fast.wizard_api as wa

    tmp = Path(tempfile.mkdtemp(prefix="tb_onb_"))
    session = {}
    with patch("toolboxv2.tb_root_dir", tmp), patch.object(wa, "tb_root_dir", tmp), \
         patch.object(wa, "_load_session", lambda: dict(session)), \
         patch.object(wa, "_save_session", lambda s: (session.clear(), session.update(s))), \
         _Server() as srv:
        assert srv.get("/api/wizard/state")[1]["first_run"] is True
        # legacy name from old installer builds is normalized, not rejected
        assert srv.post("/api/wizard/profile", {"profile": "consumer"})[1]["profile"] == "local"
        status, res = srv.post("/api/wizard/save", {"apply": False})
        assert status == 200 and res["ok"] is True
        state = srv.get("/api/wizard/state")[1]
    assert state["first_run"] is False and state["profile"] == "local"


def test_removed_endpoints_are_gone():
    """Dead wizard endpoints (incl. GET /api/wizard/llm, which leaked keys)."""
    import toolboxv2.utils.workers.fast.wizard_api as wa
    import inspect
    src = inspect.getsource(wa)
    for gone in ("/api/wizard/llm", "/api/wizard/step", "/api/wizard/features",
                 "/api/wizard/gui", "/api/wizard/dist", "/api/wizard/profiles"):
        assert gone not in src, gone


def test_e2e_live_health_over_http():
    with _Server() as srv:
        status, data = srv.get("/live/health")
        assert status == 200
        assert data["enabled"] in (True, False)
        assert "workers" in data


def test_e2e_live_redirect_or_disabled():
    """Ohne Key: 403 disabled. Mit Key (kein Worker): 503. Redirect: 302+Location."""
    with _Server() as srv:
        try:
            status, _ = srv.get("/live")
            assert status == 200, f"unexpected direct 200 on /live: {status}"
        except urllib.error.HTTPError as e:
            assert e.code in (302, 403, 503), f"unexpected {e.code}"
            if e.code == 302:
                loc = e.headers.get("Location", "")
                assert loc.startswith("http://127.0.0.1:") and "key=" in loc


def test_e2e_loopback_guard_blocks_remote_environ():
    """Guard-Modell: TCP-Peer (environ REMOTE_ADDR) entscheidet, nicht XFF/Host-Header.

    Echter TCP-Test waere immer 127.0.0.1 (loopback) - darum unit-level:
    environ REMOTE_ADDR=remote + spoofed client_ip/Host -> muss 403 sein.
    (Detailabdeckung: test_live_proxy.py::test_xff_spoof_blocked)
    """
    with _Server() as srv:
        req = urllib.request.Request(
            f"http://127.0.0.1:{srv.port}/live/health",
            headers={"Host": "evil.example.com:1234",
                     "X-Forwarded-For": "10.9.8.7"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                assert r.status in (200, 403), f"unexpected {r.status}"
        except urllib.error.HTTPError as e:
            assert e.code in (400, 403)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """302 nicht folgen - wir wollen den Location-Header sehen."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get_html(srv, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}{path}", timeout=10) as r:
        return r.status, r.read().decode("utf-8")


def test_root_serves_onboarding_on_first_run():
    """first_run (no manifest profile) -> '/' is the one onboarding UI."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    from unittest.mock import patch

    with _Server() as srv, patch.object(
        lu, "get_installation_state",
        return_value={"installed": True, "first_run": True, "profile": None},
    ):
        status, html = _get_html(srv, "/")
    assert status == 200
    assert "ToolBoxV2 Installer" in html
    assert "/api/wizard/state" in html


def test_setup_always_serves_onboarding():
    """/setup re-opens onboarding on an installed TB (replaces /install hub)."""
    status, html = 0, ""
    with _Server() as srv:
        status, html = _get_html(srv, "/setup")
    assert status == 200
    assert "ToolBoxV2 Installer" in html


def test_old_ui_pages_are_gone():
    import toolboxv2.utils.workers.fast.local_ui as lu
    assert not hasattr(lu, "welcome")
    assert not hasattr(lu, "install")
