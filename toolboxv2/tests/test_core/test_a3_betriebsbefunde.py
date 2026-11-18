"""
Betriebstests WP3 Auftrag 3 - Konto und Sicherheit (8 Tests, Tabelle Nr. 1-8).

Belegt die 8 Betrieberbefunde aus WP3_ordnung_und_betriebstests_2026-11.md
auf Basis c36c3dc. Erwartung: alle 8 ROT auf der Basis, GRUEN nach Fix.
KEIN Produktcode wird hier geaendert (Betriebstests-Regel).
"""
import inspect
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

# Protokoll 3.1: isolierte Umgebung fuer den gesamten Testmodul-Lauf.
_T = Path(tempfile.mkdtemp(prefix="tb_a3_"))
# Env-Isolation NUR waehrend unserer Tests (kein Modul-Import-Pollution).
_SAVED_ENV = None


def _env_enter():
    global _SAVED_ENV
    _SAVED_ENV = {k: os.environ.get(k)
                  for k in ("TB_DATA_DIR", "TB_MANIFEST_DIR")}
    os.environ["TB_DATA_DIR"] = str(_T)
    os.environ["TB_MANIFEST_DIR"] = str(_T)


def _env_exit():
    if _SAVED_ENV is None:
        return
    for k, v in _SAVED_ENV.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def setUpModule():
    _env_enter()


def tearDownModule():
    _env_exit()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server:
    """Bootet die local_ui-FastTB-App in einem Thread (Muster: wizard-e2e)."""

    def __init__(self):
        from toolboxv2.utils.workers.fast import local_ui as lu
        self.lu = lu
        self.port = _free_port()
        self.thread = threading.Thread(
            target=lu.app.serve,
            kwargs={"host": "127.0.0.1", "port": self.port},
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        for _ in range(60):
            time.sleep(0.25)
            if self.get("/health")[0] == 200 or self.get("/")[0] > 0:
                return self
        raise RuntimeError("local_ui server did not come up")

    def __exit__(self, *exc):
        return False

    def get(self, path):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", errors="replace")
        except Exception:
            return 0, ""


def _parsed_request(client_ip="203.0.113.7", headers=None):
    """ParsedRequest mit vollem Feldsatz (dataclass, Pflichtfelder)."""
    from toolboxv2.utils.workers.server_worker import ParsedRequest
    return ParsedRequest(
        method="GET", path="/", query_params={},
        headers=headers or {"host": "127.0.0.1:8467"},
        content_type="", content_length=0, body=b"",
        form_data=None, json_data=None, files=None,
        session=None, client_ip=client_ip, client_port=40000, environ={})


class TestA3Row1_NoAccountRoutesGated(unittest.TestCase):
    """Nr. 1: Ohne Geraetekonto oeffnen die Routen das Dashboard anonym.
    Messwerte (Probe, Basis c36c3dc): '/' liefert 200 App-Shell ohne Gate,
    '/api/wizard/state' meldet needs_account=false ohne Konto,
    '/api/chats' verarbeitet anonym (400 statt 401/403)."""

    def test_dashboard_routes_rejected_without_device_account(self):
        with _Server() as srv:
            dash_code, dash_body = srv.get("/")
            ws_code, ws_body = srv.get("/api/wizard/state")
        self.assertFalse(
            (dash_code == 200 and dash_body.lstrip().lower().startswith("<!doctype html")),
            f"'/' liefert die App-Shell anonym (HTTP {dash_code}) - "
            "Konto-Gate fehlt auf den Dashboard-Routen")
        data = json.loads(ws_body) if ws_body else {}
        self.assertTrue(
            data.get("needs_account"),
            f"'/api/wizard/state' meldet needs_account={data.get('needs_account')!r} "
            "ohne Geraetekonto - anonym offen statt Konto-Gate")


class TestA3Row2_HostHeaderSpoof(unittest.TestCase):
    """Nr. 2: local_ui._ensure_local prueft nur den Host-Header, nicht die
    Gegenstelle (client_ip). Fremde Gegenstelle + Host 127.0.0.1 => 403."""

    def test_non_loopback_peer_with_loopback_host_is_403(self):
        from toolboxv2.utils.workers.fast.local_ui import _ensure_local
        req = _parsed_request(client_ip="203.0.113.7",
                              headers={"host": "127.0.0.1:8467"})
        res = _ensure_local(req)
        code = getattr(res, "exec_code", None)
        self.assertEqual(
            403, code,
            f"REMOTE-Adresse {req.client_ip} mit Host-Header 127.0.0.1 "
            f"wird durchgelassen (Guard-Ergebnis: {res!r})")


class TestA3Row3_WsOriginUnchecked(unittest.TestCase):
    """Nr. 3: WS-Handshake nimmt fremde Origin an (kein Origin-Check im
    ws_worker)."""

    def test_ws_handshake_rejects_foreign_origin(self):
        import toolboxv2.utils.workers.ws_worker as ww
        src = inspect.getsource(ww)
        found = ("origin" in src.lower()
                 or "same_origin" in src.lower())
        self.assertTrue(
            found,
            "ws_worker hat keinen Origin-/Same-Origin-Check im Handshake - "
            "Upgrade mit Origin: http://evil.example wird angenommen")


class TestA3Row4_OnboardingVsAccountState(unittest.TestCase):
    """Nr. 4: Echte App, Arbeitsordner=Datenordner, device_owner.json
    vorhanden, Profil local: onboarding_state()['needs_account'] SOLL
    False sein (account_state meldet linked). Ist: onboarding_state
    liest ohne App-Kontext => needs_account True (Widerspruch)."""

    def test_onboarding_state_agrees_with_account_state(self):
        from types import SimpleNamespace

        from toolboxv2.mods.CloudM.auth import device_account as da
        from toolboxv2.utils.clis import first_run as fr

        (self._T / "tb-manifest.yaml").write_text(
            "app:\n  profile: local\n", encoding="utf-8")
        # Fake-App statt get_app(): keine globale Instanz-Registry-Pollution
        # (get_app() cacht Instanzen prozesweit und veraendert dadurch das
        # Verhalten spaeterer Test-Module wie test_device_account).
        app = SimpleNamespace(data_dir=str(self._T))
        owner_file = da.device_owner_file(app)
        owner_file.parent.mkdir(parents=True, exist_ok=True)
        owner_file.write_text(json.dumps(
            {"user_id": "u-row4", "username": "row4",
             "base_url": "https://simplecore.app",
             "linked_at": time.time()}), encoding="utf-8")

        acct = fr.account_state(app)
        with patch.dict(os.environ, {"TB_MANIFEST_DIR": str(self._T),
                                     "TB_DATA_DIR": str(self._T)}):
            st = fr.onboarding_state(app)

        self.assertTrue(acct.get("linked"),
                        "Vorbedingung: account_state(app) muss linked melden")
        self.assertEqual(
            st.get("profile"), "local",
            "Vorbedingung: Manifest-Profil 'local' muss lesbar sein "
            f"(got {st.get('profile')!r}) - ManifestLoader-Pfad pruefen")
        self.assertIs(
            st.get("needs_account"), False,
            f"onboarding_state widerspricht account_state: {st} vs linked - "
            "onboarding_state liest Geraetekonto ohne App-Kontext")


TestA3Row4_OnboardingVsAccountState._T = _T


class TestA3Row5_TrustNeedsRestart(unittest.TestCase):
    """Nr. 5: Vertrauen wird nur beim Boot evaluiert (provider einmalig in
    _init_session_manager). Linken/Entlinken OHNE Neustart aendert nichts.
    Ein Fix muss eine Auffrischung pro Anfrage/Session vorsehen."""

    def test_trust_provider_is_refreshed_after_link_change(self):
        import toolboxv2.utils.workers.server_worker as sw
        src = inspect.getsource(sw)
        block = src[src.find("enable_local_trust(provider)"):]
        block = block[:block.find("_init_access_controller")]
        refreshed = any(k in block for k in (
            "per_request", "refresh", "revalidate", "on_request", "recheck"))
        self.assertTrue(
            refreshed,
            "server_worker evaluiert device-trust nur beim Boot - Veraen"
            "derungen am Konto wirken erst nach Neustart")


class TestA3Row6_SyncThreadAlwaysStarts(unittest.TestCase):
    """Nr. 6: Der account-sync-Thread startet immer - die Startzeile liegt
    durch falsche Einrueckung AUSSEN am 'if provider:'-Guard."""

    def test_account_sync_thread_inside_provider_guard(self):
        import toolboxv2.utils.workers.server_worker as sw
        src = inspect.getsource(sw)
        m = src.find("if not any(getattr(t, \"name\", \"\") == \"account-sync\"")
        self.assertGreaterEqual(m, 0, "Sync-Thread-Block nicht gefunden")
        line_start = src.rfind("\n", 0, m) + 1
        indent = len(src[line_start:]) - len(src[line_start:].lstrip())
        self.assertGreaterEqual(
            indent, 12,
            f"account-sync-Thread-Einrueckung ist {indent} - der Block liegt "
            "nicht mehr unter 'if provider:' und startet bei jedem Boot")


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
