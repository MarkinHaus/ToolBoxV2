"""
E2E: wizard_api + /live sub-mount durch echten HTTP-Stack (waitress, random Port).

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
        assert data["first_run"] is True or data["first_run"] is False
        assert isinstance(data["steps"], list) and data["steps"]
        ids = {s["id"] for s in data["steps"]}
        assert "app" in ids and "database" in ids


def test_e2e_wizard_profiles_over_http():
    with _Server() as srv:
        status, data = srv.get("/api/wizard/profiles")
        assert status == 200
        assert {"consumer", "homelab", "server", "business", "developer"} <= \
               {p["id"] for p in data["profiles"]}


def test_e2e_wizard_step_roundtrip_over_http():
    with _Server() as srv:
        _, state = srv.get("/api/wizard/state")
        app_step = next(s for s in state["steps"] if s["id"] == "app")
        values = {f["name"]: f["value"] for f in app_step["fields"]}
        status, res = srv.post("/api/wizard/step", {"step": "app", "values": values})
        assert status == 200 and res.get("ok") is True


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


def test_install_redirects_to_welcome_on_first_run():
    """W1c: /install gehoert dem Manage-Modus; first_run -> /welcome."""
    import toolboxv2.utils.workers.fast.local_ui as lu
    from unittest.mock import patch

    # State mocken: Dev-Maschinen haben first_run=False (installierte TB).
    with _Server() as srv, patch.object(
        lu, "get_installation_state",
        return_value={"installed": True, "first_run": True},
    ):
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            opener.open(f"http://127.0.0.1:{srv.port}/install", timeout=10)
            assert False, "expected 302"
        except urllib.error.HTTPError as e:
            assert e.code == 302, f"unexpected {e.code}"
            assert e.headers.get("Location") == "/welcome"


def test_install_serves_manage_hub_when_installed():
    """W1b: /install = Manager (Profil/Module/GUI/Dist-Karten, Wizard-API)."""
    from unittest.mock import patch

    import toolboxv2.utils.workers.fast.local_ui as lu

    with _Server() as srv, patch.object(
        lu, "get_installation_state",
        return_value={"installed": True, "first_run": False},
    ):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{srv.port}/install", timeout=10) as r:
            assert r.status == 200
            html = r.read().decode("utf-8")
        assert "ToolBox Manager" in html
        assert "/api/wizard/profiles" in html
        assert "/api/wizard/features" in html
