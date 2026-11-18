"""
Betriebstests WP3 Auftrag 3 - Konto und Sicherheit (8 Tests, Tabelle Nr. 1-8).

Belegt die 8 Betrieberbefunde aus WP3_ordnung_und_betriebstests_2026-11.md.
Nr. 1, 3, 5, 6 laufen gegen echte Worker-Prozesse (eigener Unterprozess,
eigener TB_DATA_DIR/HOME/TB_MANIFEST_DIR, freie Ports, HTTP/WS von aussen).
Doubles gibt es nur fuer das Netz zu simplecore.app - und die brauchen diese
Tests nicht, weil keiner von ihnen das Netz beruehrt.
"""
import asyncio
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[3]

# Der Unterprozess baut den Worker so wie der Betrieb
# (cli_worker_manager._run_live_ui_process: local_ui.app + mount_tray_api +
# register_collective_commands + app.serve). Zusaetzlich ein Steuer-Socket auf
# Loopback, ueber das der Test den Zustand des Prozesses liest (Routenliste,
# Thread-Namen, Datenordner der App). Der Steuer-Socket aendert nichts am Worker.
_LIVE_UI_RUNNER = r'''
import json, socketserver, sys, threading
host, port, ctl, mount = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "mount"
from toolboxv2.utils.workers.fast import local_ui as lu
from toolboxv2.utils.workers.fast.tray_api import mount_tray_api, register_collective_commands
mount_tray_api(lu.app)
register_collective_commands(lu.app)
if mount:
    from toolboxv2.mods.isaa.extras.live_obs_server import app as obs_app
    lu.app.mount_app(obs_app, "obs", "isaa_obs")
    from toolboxv2.mods.isaa.ui.app import build_app
    lu.app.mount_app(build_app(None), "isaa", "isaa_ui")

class _H(socketserver.StreamRequestHandler):
    def handle(self):
        cmd = self.rfile.readline().decode().strip()
        if cmd == "routes":
            out = [[r.method, r.path] for r in lu.app._routes]
        elif cmd == "threads":
            out = [t.name for t in threading.enumerate()]
        elif cmd == "datadir":
            from toolboxv2 import get_app
            out = get_app("a3.probe").data_dir
        else:
            out = "?"
        self.wfile.write((json.dumps(out) + "\n").encode())

_srv = socketserver.ThreadingTCPServer(("127.0.0.1", ctl), _H)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
lu.app.serve(host=host, port=port, blocking=True, enable_ws=False, allow_specialist=False)
'''

# WS-Worker wie `tb workers start` ihn baut (WSWorker(...).run_sync()), nur mit
# freien Ports fuer WS und ZMQ, damit kein anderer Worker gestoert wird.
_WS_RUNNER = r'''
import sys
port = int(sys.argv[1]); z = [int(p) for p in sys.argv[2:6]]
from toolboxv2.utils.workers.config import load_config
from toolboxv2.utils.workers.ws_worker import WSWorker
cfg = load_config()
cfg.ws_worker.host = "127.0.0.1"
cfg.ws_worker.port = port
cfg.zmq.pub_endpoint = f"tcp://127.0.0.1:{z[0]}"
cfg.zmq.sub_endpoint = f"tcp://127.0.0.1:{z[1]}"
cfg.zmq.req_endpoint = cfg.zmq.rep_endpoint = f"tcp://127.0.0.1:{z[2]}"
cfg.zmq.http_to_ws_endpoint = f"tcp://127.0.0.1:{z[3]}"
WSWorker("a3_ws", cfg).run_sync()
'''


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _non_loopback_address():
    """Eigene Nicht-Loopback-Adresse (gethostbyname, sonst erste aus getaddrinfo)."""
    try:
        addr = socket.gethostbyname(socket.gethostname())
    except OSError:
        addr = "127.0.0.1"
    if not addr.startswith("127."):
        return addr
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if not info[4][0].startswith("127."):
                return info[4][0]
    except OSError:
        pass
    # Hostname loest nur auf Loopback auf: Adresse der Standardroute erfragen
    # (UDP-connect sendet nichts, waehlt nur die Quelladresse).
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))
        addr = s.getsockname()[0]
        return None if addr.startswith("127.") or addr == "0.0.0.0" else addr
    except OSError:
        return None
    finally:
        s.close()


class _Proc:
    """Unterprozess mit eigener, isolierter Umgebung (Protokoll 3.1)."""

    def __init__(self, tag, profile="local"):
        self.tmp = Path(tempfile.mkdtemp(prefix=f"tb_a3_{tag}_"))
        for d in ("home", "data", "cwd", "appdata", "local"):
            (self.tmp / d).mkdir()
        if profile:  # Profil wie `tb -init`: app.profile im Manifest
            (self.tmp / "tb-manifest.yaml").write_text(
                f"app:\n  profile: {profile}\n", encoding="utf-8")
        env = dict(os.environ)
        env.update({
            "HOME": str(self.tmp / "home"), "USERPROFILE": str(self.tmp / "home"),
            "APPDATA": str(self.tmp / "appdata"), "LOCALAPPDATA": str(self.tmp / "local"),
            "TB_DATA_DIR": str(self.tmp / "data"), "TB_MANIFEST_DIR": str(self.tmp),
            "MOCK_EXTERNAL_SERVICES": "true", "IS_OFFLINE_DB": "true",
            "PYTHONPATH": os.pathsep.join(
                [str(_REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
        })
        for k in ("TB_INSTALL_DIR", "TB_CORS_ORIGINS", "TOOLBOXV2_REMOTE_BASE",
                  "LIVE_DASHBOARD_KEY"):
            env.pop(k, None)
        self.env = env
        self.log_path = self.tmp / "proc.log"
        self.proc = None

    def spawn(self, script_name, script, *argv):
        path = self.tmp / script_name
        path.write_text(script, encoding="utf-8")
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [sys.executable, str(path), *map(str, argv)], env=self.env,
            cwd=str(self.tmp / "cwd"), stdout=self._log, stderr=subprocess.STDOUT)

    def log_tail(self, n=2500):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-n:]
        except OSError:
            return ""

    def wait_http(self, port, timeout=150):
        """Wartet, bis der Worker /health beantwortet (oder der Prozess stirbt)."""
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Worker-Prozess beendet ({self.proc.returncode}):\n"
                                   + self.log_tail())
            try:
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.3)
        raise RuntimeError("Worker kam nicht hoch:\n" + self.log_tail())

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        try:
            self._log.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)


class _LiveUI(_Proc):
    """Echter lokaler UI-Worker (local_ui.app) auf freiem Port."""

    def __init__(self, tag, profile="local", host="127.0.0.1", mount=False):
        super().__init__(tag, profile)
        self.host = host
        self.port = _free_port()
        self.ctl = _free_port()
        self._mount = mount

    def start(self):
        self.spawn("live_ui_runner.py", _LIVE_UI_RUNNER, self.host, self.port,
                   self.ctl, "mount" if self._mount else "bare")
        self.wait_http(self.port)
        return self

    def ask(self, cmd):
        s = socket.create_connection(("127.0.0.1", self.ctl), timeout=10)
        try:
            s.sendall((cmd + "\n").encode())
            return json.loads(s.makefile().readline())
        finally:
            s.close()

    def request(self, method, path, body=None, headers=None, addr="127.0.0.1",
                timeout=8):
        """(status, headers, text); folgt keinen Weiterleitungen."""
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdrs.setdefault("Content-Type", "application/json")
        c = http.client.HTTPConnection(addr, self.port, timeout=timeout)
        try:
            c.request(method, path, body=data, headers=hdrs)
            r = c.getresponse()
            return r.status, {k.lower(): v for k, v in r.getheaders()}, \
                r.read().decode("utf-8", errors="replace")
        finally:
            c.close()

    def owner_file(self):
        return Path(self.ask("datadir")) / "device_owner.json"

    def link_account(self):
        f = self.owner_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({
            "user_id": "u-a3", "username": "a3user",
            "base_url": "https://simplecore.app", "linked_at": time.time()}),
            encoding="utf-8")

    def unlink_account(self):
        self.owner_file().unlink()

    def thread_names(self):
        return self.ask("threads")


def _wait_until(predicate, timeout):
    """Wartet bis predicate() truthy; Rueckgabe: letzter Wert."""
    end = time.time() + timeout
    value = predicate()
    while not value and time.time() < end:
        time.sleep(0.5)
        value = predicate()
    return value


def _parsed_request(client_ip="203.0.113.7", headers=None):
    """ParsedRequest mit vollem Feldsatz (dataclass, Pflichtfelder)."""
    from toolboxv2.utils.workers.server_worker import ParsedRequest
    return ParsedRequest(
        method="GET", path="/", query_params={},
        headers=headers or {"host": "127.0.0.1:8467"},
        content_type="", content_length=0, body=b"",
        form_data=None, json_data=None, files=None,
        session=None, client_ip=client_ip, client_port=40000, environ={})


# Ausnahmeliste der Kontopflicht (WP3-05 2.3): zeigen selbst die richtige Seite
# bzw. werden zum Verknuepfen gebraucht.
_ACCOUNT_EXEMPT = {
    ("GET", "/"), ("GET", "/account"), ("GET", "/setup"),
    ("GET", "/api/wizard/state"), ("GET", "/api/account/status"),
    ("POST", "/api/account/start"),
}


def _concrete(path):
    """Pfadvorlage der App -> aufrufbarer Pfad."""
    import re
    path = re.sub(r"\{[^}:]+:path\}", "x/y", path)
    return re.sub(r"\{[^}]+\}", "x", path)


class TestA3Row1_NoAccountRoutesGated(unittest.TestCase):
    """Nr. 1: Profil local, kein device_owner.json: JEDE lokal registrierte
    Route ausser der Ausnahmeliste liefert Kontoseite/403, auch das Setzen
    eines anderen Profils ueber die Wizard-API (Umgehung der Kontopflicht)."""

    @classmethod
    def setUpClass(cls):
        cls.srv = _LiveUI("row1", profile="local", mount=True)
        cls.addClassCleanup(cls.srv.stop)
        cls.srv.start()

    def test_every_local_route_requires_account(self):
        srv = self.srv
        html = {"Accept": "text/html"}
        routes = [tuple(r) for r in srv.ask("routes")]

        # Zustand ist wirklich der der Kontopflicht: App kennt die Routen von
        # Tray, ISAA und OBS, kein Konto, Profil local.
        paths = {p for _, p in routes}
        for must in ("/isaa", "/obs", "/api/chats", "/tray/command", "/api/wizard/profile"):
            self.assertIn(must, paths, f"Route {must} nicht registriert: {sorted(paths)}")
        self.assertFalse(srv.owner_file().exists())

        # Ausnahme '/': Kontoseite (nicht Dashboard).
        code, _, body = srv.request("GET", "/", headers=html)
        self.assertEqual(200, code, body[:200])
        self.assertIn("Link a SimpleCore account", body,
                      "'/' zeigt ohne Konto nicht die Kontoseite")

        # Ausnahme: Wizard-Status meldet needs_account.
        code, _, body = srv.request("GET", "/api/wizard/state")
        self.assertEqual(200, code)
        state = json.loads(body)
        self.assertIs(state.get("needs_account"), True, state)
        self.assertEqual("local", state.get("profile"))

        # Alle uebrigen Routen aus der App (Methode + Pfad aus der Routenliste).
        # Vorlagen mit '{x:path}' (Catch-all, /live/mgr/...) kennt FastTB nicht
        # (_path_to_regex ersetzt nur '{name}'): sie matchen nur als Literal, echte
        # Anfragen laufen daran vorbei in den HTTPWorker (siehe Bericht).
        gated = [(m, p) for m, p in routes
                 if (m, p) not in _ACCOUNT_EXEMPT and ":path}" not in p]
        self.assertGreater(len(gated), 40, gated)
        failures = []
        for method, path in gated:
            url = _concrete(path)
            body_in = {} if method in ("POST", "PUT", "DELETE") else None
            try:
                code, _, body = srv.request(method, url, body_in,
                                            headers={"Accept": "application/json"})
            except OSError as e:
                failures.append(f"{method} {url}: {type(e).__name__} {e}")
                continue
            if code != 403 or "account required" not in body:
                failures.append(f"{method} {url} (JSON): {code} {body[:80]!r}")
            if method == "GET":
                try:
                    code, hdrs, body = srv.request("GET", url, headers=html)
                except OSError as e:
                    failures.append(f"GET {url} (HTML): {type(e).__name__} {e}")
                    continue
                if code != 302 or hdrs.get("location") != "/account":
                    failures.append(f"GET {url} (HTML): {code} {hdrs.get('location')!r}")
        self.assertEqual([], failures, "Routen ohne Kontopflicht:\n" + "\n".join(failures))

        # Umgehung der Kontopflicht: Profil 'server' setzen und speichern.
        code, _, body = srv.request("POST", "/api/wizard/profile", {"profile": "server"})
        self.assertEqual(403, code, f"POST /api/wizard/profile ohne Konto: {code} {body[:200]}")
        code, _, body = srv.request("POST", "/api/wizard/save", {"apply": False})
        self.assertEqual(403, code, f"POST /api/wizard/save ohne Konto: {code} {body[:200]}")
        _, _, body = srv.request("GET", "/api/wizard/state")
        self.assertEqual("local", json.loads(body).get("profile"))

        # Ausnahmen bleiben erreichbar (sonst laesst sich kein Konto verknuepfen).
        code, _, body = srv.request("GET", "/account", headers=html)
        self.assertEqual(200, code)
        self.assertIn("Link a SimpleCore account", body)
        code, _, body = srv.request("GET", "/api/account/status")
        self.assertEqual(200, code, body[:200])
        self.assertIs(json.loads(body).get("linked"), False)
        code, _, body = srv.request("POST", "/api/account/start", {"method": "bogus"})
        self.assertEqual(400, code, "POST /api/account/start muss den Handler erreichen")
        code, _, body = srv.request("GET", "/setup", headers=html)
        self.assertEqual(200, code)
        self.assertIn("ToolBoxV2 Installer", body)
        code, _, body = srv.request("GET", "/favicon.ico")
        self.assertNotIn("account required", body)

        # Mit Konto ist dieselbe Route offen (Gegenstueck: das Gate hangt am Konto).
        srv.link_account()
        code, _, body = srv.request("GET", "/api/vfs/tree",
                                    headers={"Accept": "application/json"})
        self.assertNotIn("account required", body,
                         f"GET /api/vfs/tree mit Konto weiter gesperrt: {code}")

    def test_first_run_wizard_profile_and_save_stay_open(self):
        """Erststart (noch kein Profil): Profil setzen und speichern geht; danach
        ohne Konto nicht mehr (Profilwechsel nur ueber `tb -init`)."""
        srv = _LiveUI("row1fr", profile=None)
        self.addCleanup(srv.stop)
        srv.start()
        code, _, body = srv.request("GET", "/api/wizard/state")
        self.assertTrue(json.loads(body)["first_run"], body)
        code, _, body = srv.request("POST", "/api/wizard/profile", {"profile": "local"})
        self.assertEqual(200, code, body[:200])
        code, _, body = srv.request("POST", "/api/wizard/save", {"apply": False})
        self.assertEqual(200, code, body[:200])
        _, _, body = srv.request("GET", "/api/wizard/state")
        state = json.loads(body)
        self.assertFalse(state["first_run"], state)
        self.assertTrue(state["needs_account"], state)
        code, _, body = srv.request("POST", "/api/wizard/profile", {"profile": "server"})
        self.assertEqual(403, code, body[:200])


class TestA3Row2_HostHeaderSpoof(unittest.TestCase):
    """Nr. 2: Es gibt EINE Lokal-Pruefung (live_proxy._peer_is_local): sie
    entscheidet nach der TCP-Gegenstelle (environ REMOTE_ADDR), nie nach
    Host-Header oder X-Forwarded-For. Fremde Gegenstelle + Host 127.0.0.1 => 403."""

    def test_non_loopback_peer_with_loopback_host_is_403(self):
        from toolboxv2.utils.workers.fast.live_proxy import _ensure_local
        req = _parsed_request(client_ip="203.0.113.7",
                              headers={"host": "127.0.0.1:8467"})
        req.environ = {"REMOTE_ADDR": "203.0.113.7"}
        res = _ensure_local(req)
        self.assertTrue(
            isinstance(res, tuple) and res[0] == 403,
            f"REMOTE-Adresse {req.environ['REMOTE_ADDR']} mit Host-Header 127.0.0.1 "
            f"wird durchgelassen (Guard-Ergebnis: {res!r})")

    def test_missing_remote_addr_is_not_local(self):
        """Ohne REMOTE_ADDR im environ gilt die Anfrage als nicht lokal, auch
        wenn client_ip (aus X-Forwarded-For) Loopback behauptet."""
        from toolboxv2.utils.workers.fast.live_proxy import _ensure_local
        req = _parsed_request(client_ip="127.0.0.1",
                              headers={"host": "127.0.0.1:8467",
                                       "x-forwarded-for": "127.0.0.1"})
        req.environ = {}
        res = _ensure_local(req)
        self.assertTrue(
            isinstance(res, tuple) and res[0] == 403,
            f"client_ip aus X-Forwarded-For ersetzt REMOTE_ADDR (Guard-Ergebnis: {res!r})")

    def test_non_loopback_peer_over_http_is_403(self):
        """Echter Worker auf 0.0.0.0; Anfrage ueber die eigene Nicht-Loopback-
        Adresse mit Host 127.0.0.1 und X-Forwarded-For 127.0.0.1."""
        addr = _non_loopback_address()
        self.assertIsNotNone(
            addr, "Dieser Rechner hat keine Nicht-Loopback-Adresse: Test nicht "
                  "ausfuehrbar (Fehler, kein skip)")
        srv = _LiveUI("row2", profile="server", host="0.0.0.0")
        self.addCleanup(srv.stop)
        srv.start()
        spoof = {"Host": "127.0.0.1", "X-Forwarded-For": "127.0.0.1"}
        # Gegenprobe: von Loopback sind die Routen erreichbar.
        self.assertEqual(200, srv.request("GET", "/api/account/status")[0])
        for path in ("/api/account/status", "/account", "/setup"):
            code, _, body = srv.request("GET", path, headers=spoof, addr=addr)
            self.assertEqual(
                403, code,
                f"GET {path} ueber {addr} mit Host 127.0.0.1 + X-Forwarded-For "
                f"127.0.0.1: {code} {body[:80]!r}")

    def test_rebinding_host_header_over_http_is_403(self):
        """DNS-Rebinding: Peer ist Loopback, der Host-Header (evil.example) nicht."""
        srv = _LiveUI("row2b", profile="server")
        self.addCleanup(srv.stop)
        srv.start()
        evil = {"Host": f"evil.example:{srv.port}"}
        good = {"Host": f"127.0.0.1:{srv.port}"}
        for path in ("/api/account/status", "/account", "/setup", "/api/wizard/state"):
            self.assertEqual(200, srv.request("GET", path, headers=good)[0], path)
            code, _, body = srv.request("GET", path, headers=evil)
            self.assertEqual(403, code, f"GET {path} mit Host evil.example: {code} {body[:80]!r}")
        for path, payload in (("/api/wizard/profile", {"profile": "server"}),
                              ("/api/wizard/save", {"apply": False})):
            code, _, body = srv.request("POST", path, payload, headers=evil)
            self.assertEqual(403, code, f"POST {path} mit Host evil.example: {code} {body[:80]!r}")

    def test_wizard_routes_refuse_foreign_peer(self):
        """wizard state/profile/save haben dieselbe eine Lokal-Pruefung."""
        addr = _non_loopback_address()
        self.assertIsNotNone(addr, "keine Nicht-Loopback-Adresse (Fehler, kein skip)")
        srv = _LiveUI("row2c", profile="server", host="0.0.0.0")
        self.addCleanup(srv.stop)
        srv.start()
        self.assertEqual(200, srv.request("GET", "/api/wizard/state")[0])
        self.assertEqual(403, srv.request("GET", "/api/wizard/state", addr=addr)[0])
        for path, payload in (("/api/wizard/profile", {"profile": "server"}),
                              ("/api/wizard/save", {"apply": False})):
            code, _, body = srv.request("POST", path, payload, addr=addr)
            self.assertEqual(403, code, f"POST {path} von fremder Adresse: {code} {body[:80]!r}")


class TestA3Row3_WsOriginUnchecked(unittest.TestCase):
    """Nr. 3: Echter WS-Worker: Handshake mit fremdem Origin wird mit HTTP 403
    abgelehnt; Origin des eigenen Hosts, Tauri und Clients ohne Origin (CLI)
    werden angenommen."""

    @classmethod
    def setUpClass(cls):
        cls.proc = _Proc("row3", profile=None)
        cls.addClassCleanup(cls.proc.stop)
        cls.port = _free_port()
        cls.proc.spawn("ws_runner.py", _WS_RUNNER, cls.port,
                       *[_free_port() for _ in range(4)])
        cls.proc.wait_http(cls.port, timeout=90)

    def _handshake(self, origin):
        """'accepted' oder HTTP-Status der Ablehnung; echte Bibliothek websockets."""
        import websockets
        from websockets.exceptions import InvalidHandshake

        async def go():
            kw = {"origin": origin} if origin else {}
            try:
                async with websockets.connect(
                        f"ws://127.0.0.1:{self.port}/ws", open_timeout=10, **kw):
                    return "accepted"
            except InvalidHandshake as e:
                resp = getattr(e, "response", None)
                return getattr(resp, "status_code", None) or getattr(e, "status_code", repr(e))

        return asyncio.run(go())

    def test_ws_handshake_rejects_foreign_origin(self):
        self.assertEqual(
            403, self._handshake("http://evil.example"),
            "Upgrade mit Origin http://evil.example wird angenommen")
        self.assertEqual("accepted", self._handshake(f"http://127.0.0.1:{self.port}"))
        self.assertEqual("accepted", self._handshake(f"http://localhost:{self.port}"))
        self.assertEqual("accepted", self._handshake("http://tauri.localhost"))
        self.assertEqual("accepted", self._handshake("tauri://localhost"))
        self.assertEqual("accepted", self._handshake(None),
                         "Client ohne Origin (CLI) muss erlaubt bleiben")


    def _raw_status(self, host, origin):
        """HTTP-Status des Upgrade (roher Handshake, damit der Host-Header frei ist)."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            s.sendall((f"GET /ws HTTP/1.1\r\nHost: {host}\r\nOrigin: {origin}\r\n"
                       "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                       "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                       "Sec-WebSocket-Version: 13\r\n\r\n").encode())
            return int(s.recv(200).split(b" ", 2)[1])
        finally:
            s.close()

    def test_ws_handshake_rejects_rebinding_host(self):
        """DNS-Rebinding: Origin und Host gleich (evil.example:port) = same-origin,
        aber Host nicht Loopback (direkte Verbindung, kein nginx) -> 403."""
        p = self.port
        self.assertEqual(403, self._raw_status(f"evil.example:{p}", f"http://evil.example:{p}"))
        self.assertEqual(101, self._raw_status(f"127.0.0.1:{p}", f"http://127.0.0.1:{p}"))
        self.assertEqual(101, self._raw_status(f"localhost:{p}", f"http://localhost:{p}"))


class TestA3Row4_OnboardingVsAccountState(unittest.TestCase):
    """Nr. 4: Echte App (get_app) mit TB_DATA_DIR in einem Temp-Ordner,
    device_owner.json im app.data_dir, Profil local, Arbeitsordner = ein ANDERER
    Ordner. Aufruf wie im Betrieb OHNE app (local_ui.get_installation_state)."""

    def test_onboarding_state_without_app_sees_account_file(self):
        from toolboxv2.mods.CloudM.auth import device_account as da
        from toolboxv2.tests.a_util import reset_app_singleton

        tmp = Path(tempfile.mkdtemp(prefix="tb_a3_row4_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for d in ("data", "cwd", "manifest"):
            (tmp / d).mkdir()
        (tmp / "manifest" / "tb-manifest.yaml").write_text(
            "app:\n  profile: local\n", encoding="utf-8")
        env = patch.dict(os.environ, {"TB_DATA_DIR": str(tmp / "data"),
                                      "TB_MANIFEST_DIR": str(tmp / "manifest")})
        env.start()
        self.addCleanup(env.stop)
        old_cwd = os.getcwd()
        os.chdir(tmp / "cwd")
        self.addCleanup(os.chdir, old_cwd)
        reset_app_singleton(force_exit=True)
        self.addCleanup(reset_app_singleton, force_exit=True)

        from toolboxv2 import get_app
        app = get_app("a3.row4")
        self.assertTrue(str(app.data_dir).startswith(str(tmp / "data")), app.data_dir)
        owner_file = da.device_owner_file(app)
        self.assertNotEqual(Path(owner_file).parent.resolve(), Path.cwd().resolve())
        owner_file.parent.mkdir(parents=True, exist_ok=True)
        owner_file.write_text(json.dumps(
            {"user_id": "u-row4", "username": "row4",
             "base_url": "https://simplecore.app",
             "linked_at": time.time()}), encoding="utf-8")

        from toolboxv2.utils.workers.fast import local_ui
        st = local_ui.get_installation_state()  # wie local_ui.root: ohne app

        self.assertEqual("local", st.get("profile"), st)
        self.assertIs(
            st.get("needs_account"), False,
            f"onboarding_state ohne app sieht die Kontodatei im Datenordner nicht: {st}")
        self.assertEqual("row4", st.get("account"))


class TestA3Row5_TrustNeedsRestart(unittest.TestCase):
    """Nr. 5: Worker ohne Konto gestartet; Konto verknuepfen bzw. entfernen
    wirkt OHNE Neustart innerhalb von 10 s auf das Geraetevertrauen.
    Beobachtung wie im Betrieb: POST /api_user_data von 127.0.0.1 ohne Token
    (vertraut: nicht 401; unvertraut: 401 'Session invalid')."""

    def test_trust_follows_account_file_without_restart(self):
        srv = _LiveUI("row5", profile="local")
        self.addCleanup(srv.stop)
        srv.start()

        def status():
            return srv.request("POST", "/api_user_data", {})[0]

        self.assertEqual(401, status(), "Vorbedingung: ohne Konto kein Vertrauen")
        srv.link_account()
        trusted = _wait_until(lambda: status() != 401, 10)
        self.assertTrue(
            trusted,
            "Konto verknuepft, 10 s gewartet: Gegenstelle 127.0.0.1 bleibt "
            f"unvertraut (HTTP {status()}) - Vertrauen erst nach Neustart")
        srv.unlink_account()
        untrusted = _wait_until(lambda: status() == 401, 10)
        self.assertTrue(
            untrusted,
            f"Konto entfernt, 10 s gewartet: Vertrauen bleibt (HTTP {status()})")

    def test_first_run_profile_then_account_without_restart(self):
        """Erststart: Worker ohne Profil und ohne Konto; Profil local ueber den
        Wizard-Weg (POST profile + save), danach Konto: innerhalb 10 s vertraut
        und account-sync laeuft - ohne Neustart."""
        srv = _LiveUI("row5b", profile=None)
        self.addCleanup(srv.stop)
        srv.start()

        def status():
            return srv.request("POST", "/api_user_data", {})[0]

        self.assertEqual(401, status())
        self.assertNotIn("account-sync", srv.thread_names())
        self.assertEqual(200, srv.request("POST", "/api/wizard/profile", {"profile": "local"})[0])
        self.assertEqual(200, srv.request("POST", "/api/wizard/save", {"apply": False})[0])
        srv.link_account()
        self.assertTrue(
            _wait_until(lambda: status() != 401, 10),
            f"Profil + Konto nach dem Start: 127.0.0.1 bleibt unvertraut (HTTP {status()})")
        self.assertTrue(_wait_until(lambda: "account-sync" in srv.thread_names(), 10),
                        "account-sync startet nicht")


class TestA3Row6_SyncThreadFollowsAccount(unittest.TestCase):
    """Nr. 6: Thread account-sync laeuft nur, solange ein Konto da ist: nicht
    ohne Konto, startet nach dem Verknuepfen, endet nach dem Entfernen
    (jeweils <= 10 s, ohne Neustart)."""

    def test_account_sync_thread_follows_account_file(self):
        srv = _LiveUI("row6", profile="local")
        self.addCleanup(srv.stop)
        srv.start()

        def running():
            return "account-sync" in srv.thread_names()

        self.assertFalse(running(), "account-sync laeuft ohne Konto")
        srv.link_account()
        self.assertTrue(_wait_until(running, 10),
                        "Konto verknuepft: Thread account-sync startet nicht")
        srv.unlink_account()
        self.assertTrue(_wait_until(lambda: not running(), 10),
                        "Konto entfernt: Thread account-sync laeuft weiter")


class TestA3Row7_LinkAccountLeaksTokens(unittest.TestCase):
    """Nr. 7: link_account setzt session.base und schreibt Tokens in den
    persistenten Store, BEVOR login() geprueft wird. Scheitert der Login,
    bleiben Tokens + base zurueck."""

    def test_failed_login_leaves_no_tokens_and_no_base(self):
        import asyncio

        from toolboxv2.mods.CloudM.auth import device_account as da

        saved = []

        class _FakeSession:
            base = None

            def _save_session_token(self, access, refresh, user_id):
                saved.append({"access": access, "refresh": refresh,
                              "user_id": user_id})

            def _clear_session_token(self):
                # Rollback-Vertrag (WP3 A6e): Store leeren.
                saved.clear()

            async def login(self):
                return False  # Remote-Login scheitert

        class _FakeApp:
            session = _FakeSession()

        app = _FakeApp()

        payload = {"access_token": "LEAK-A", "refresh_token": "LEAK-R",
                   "user_id": "u-row7", "username": "row7"}

        async def _run():
            with patch.object(da, "read_device_owner", return_value=None):
                try:
                    await da.link_account(app, payload)
                except RuntimeError:
                    pass  # erwarteter Fehler - irrelevant fuer die Spuren

        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_run())
        finally:
            loop.close()

        self.assertEqual(
            saved, [],
            f"Token-Schreibzugriffe trotz failed login: {saved!r} - link_"
            "account persistiert VOR der Login-Pruefung")
        self.assertIsNone(
            app.session.base,
            "session.base wurde trotz failed login auf Remote-URL gesetzt")


class TestA3Row8_LocalAdminBaseNone(unittest.TestCase):
    """Nr. 8: CloudM/module.py:_apply_local_admin_to_session setzt
    s.base = None ('insecure base'), statt eine lokale Basis zu setzen."""

    def test_local_admin_session_keeps_a_base(self):
        from types import SimpleNamespace

        from toolboxv2.mods.CloudM.module import Tools

        class _FakeSession:
            username = ""
            user_id = ""
            valid = False
            base = "https://keep-me.example"

        fake_app = SimpleNamespace(session=_FakeSession())
        # Unbound-Call mit Fake-self: umgeht Tools.app-Property (setter
        # blockiert Fremd-Instanzen mit PermissionError).
        fake_self = SimpleNamespace(app=fake_app)

        env = {k: v for k, v in os.environ.items()
               if k != "TOOLBOXV2_REMOTE_BASE"}
        with patch.dict(os.environ, env, clear=True):
            Tools._apply_local_admin_to_session(
                fake_self, {"user_id": "u-row8", "username": "row8"})

        self.assertIsNotNone(
            fake_app.session.base,
            "session.base wurde auf None gesetzt (insecure base) - spaetere "
            "login()-Versuche haben keine lokale Basis mehr")


if __name__ == "__main__":
    unittest.main()
