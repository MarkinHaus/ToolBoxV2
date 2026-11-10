"""Origin policy: CORS never '*', CSRF blocks cross-site state-changing requests.

Run: python -m unittest toolboxv2.tests.test_core.test_cors_policy -v
"""
import json
import os
import socket
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock


def _cors(origin=None, **env):
    from toolboxv2.utils.workers.server_worker import HTTPWorker
    environ = {"HTTP_ORIGIN": origin} if origin is not None else {}
    environ.update(env)
    return HTTPWorker._get_cors_headers(None, environ)


def _csrf(method="POST", origin=None, **env):
    from toolboxv2.utils.workers.server_worker import csrf_rejection
    environ = {"REQUEST_METHOD": method, **env}
    if origin is not None:
        environ["HTTP_ORIGIN"] = origin
    return csrf_rejection(environ)


class TestCors(unittest.TestCase):
    def test_foreign_or_missing_origin_gets_no_cors(self):
        self.assertEqual(_cors("https://evil.example"), {})
        self.assertEqual(_cors(), {})
        self.assertEqual(_cors("null"), {})

    def test_allowed_origins_echoed(self):
        for o in ("http://127.0.0.1:8467", "http://localhost:3000", "tauri://localhost",
                  "http://tauri.localhost", "https://simplecore.app"):
            h = _cors(o)
            self.assertEqual(h["Access-Control-Allow-Origin"], o)
            self.assertEqual(h["Access-Control-Allow-Credentials"], "true")

    def test_env_extra_and_remote_base(self):
        with mock.patch.dict(os.environ, {"TB_CORS_ORIGINS": "https://x.y, https://a.b",
                                          "TOOLBOXV2_REMOTE_BASE": "https://my.host/"}):
            self.assertIn("Access-Control-Allow-Origin", _cors("https://a.b"))
            self.assertIn("Access-Control-Allow-Origin", _cors("https://my.host"))
            self.assertEqual(_cors("https://evil.example"), {})

    def test_same_origin_behind_nginx(self):
        self.assertIn("Access-Control-Allow-Origin",
                      _cors("https://box.example", HTTP_X_FORWARDED_HOST="box.example"))


class TestCsrf(unittest.TestCase):
    def test_safe_methods_pass(self):
        self.assertIsNone(_csrf("GET", "https://evil.example"))
        self.assertIsNone(_csrf("OPTIONS", "https://evil.example"))

    def test_cross_site_post_blocked(self):
        for m in ("POST", "PUT", "PATCH", "DELETE"):
            res = _csrf(m, "https://evil.example")
            self.assertEqual(res[0], 403, m)
        self.assertEqual(_csrf("POST", "null")[0], 403)

    def test_cross_site_without_origin_blocked(self):
        self.assertEqual(_csrf("POST", HTTP_SEC_FETCH_SITE="cross-site")[0], 403)

    def test_allowed_and_non_browser_pass(self):
        self.assertIsNone(_csrf("POST", "http://127.0.0.1:8467"))
        self.assertIsNone(_csrf("POST", "https://simplecore.app"))
        self.assertIsNone(_csrf("POST"))  # CLI / curl: no Origin
        self.assertIsNone(_csrf("POST", HTTP_SEC_FETCH_SITE="same-origin"))
        self.assertIsNone(_csrf("POST", "https://box.example", HTTP_HOST="box.example"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestCsrfOverHttp(unittest.TestCase):
    """Real FastTB owner: a cross-site 'simple' POST never reaches the handler."""

    @classmethod
    def setUpClass(cls):
        try:
            import waitress  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("waitress not installed")
        from toolboxv2.utils.workers.fast_tb import FastTB
        cls.calls = []
        app = FastTB(title="csrf-test")
        app.auth = False

        @app.post("/hit", auth=False)
        async def hit(request):
            cls.calls.append(1)
            return {"ok": True}

        cls.port = _free_port()
        cls.server = app.serve(host="127.0.0.1", port=cls.port, blocking=False,
                               enable_ws=False, allow_specialist=False)
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.server.close()
        except Exception:
            pass

    def _post(self, headers):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/hit", data=b'{"x":1}',
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def test_cross_site_text_plain_post_blocked(self):
        before = len(self.calls)
        status = self._post({"Content-Type": "text/plain", "Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(len(self.calls), before)

    def test_local_origin_and_cli_pass(self):
        self.assertEqual(self._post({"Content-Type": "application/json",
                                     "Origin": f"http://127.0.0.1:{self.port}"}), 200)
        self.assertEqual(self._post({"Content-Type": "application/json"}), 200)


if __name__ == "__main__":
    unittest.main()
