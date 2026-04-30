#!/usr/bin/env python3
"""
test_fast_tb.py - Unit Tests for FastTB wrapper

Tests cover:
  - Path-to-regex conversion
  - Route registration (all HTTP methods)
  - Route resolution (exact match, path params, no-match)
  - WebSocket handler registration & export
  - Parameter injection (request, session, path, query, json body, form, defaults)
  - Type coercion (int, float, bool, str)
  - Response formatting (dict, list, str, html, bytes, None, tuples)
  - Async + sync handler execution
  - Edge cases (empty app, duplicate routes, unicode paths, missing params)

Run: python -m unittest test_fast_tb -v
"""
import json
import unittest

from toolboxv2.utils.workers.fast_tb import FastTB, _path_to_regex, Route, WSRoute
from toolboxv2.utils.workers.fast_tb_handler import FastTBHandler
from toolboxv2.utils.workers.server_worker import ParsedRequest
from toolboxv2.utils.workers.session import SessionData


# =============================================================================
# Helpers
# =============================================================================

def make_request(
    method="GET",
    path="/",
    query_params=None,
    json_data=None,
    form_data=None,
    session=None,
    headers=None,
    body=b"",
) -> ParsedRequest:
    """Factory for ParsedRequest with sensible defaults."""
    return ParsedRequest(
        method=method,
        path=path,
        query_params=query_params or {},
        headers=headers or {},
        content_type="application/json" if json_data else "",
        content_length=len(body),
        body=body,
        form_data=form_data,
        json_data=json_data,
        session=session,
    )


def parse_json_body(body: bytes) -> dict:
    """Parse JSON from response body bytes."""
    return json.loads(body.decode("utf-8"))


# =============================================================================
# Tests: _path_to_regex
# =============================================================================

class TestPathToRegex(unittest.TestCase):

    def test_static_path_matches_exact(self):
        pattern, params = _path_to_regex("/health")
        self.assertEqual(params, [])
        self.assertIsNotNone(pattern.match("/health"))

    def test_static_path_rejects_different(self):
        pattern, _ = _path_to_regex("/health")
        self.assertIsNone(pattern.match("/healthz"))

    def test_single_param_extracts_name(self):
        pattern, params = _path_to_regex("/users/{user_id}")
        self.assertEqual(params, ["user_id"])
        m = pattern.match("/users/abc123")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("user_id"), "abc123")

    def test_multiple_params_extracted_in_order(self):
        pattern, params = _path_to_regex("/users/{uid}/posts/{pid}")
        self.assertEqual(params, ["uid", "pid"])
        m = pattern.match("/users/42/posts/99")
        self.assertEqual(m.group("uid"), "42")
        self.assertEqual(m.group("pid"), "99")

    def test_param_does_not_match_slash(self):
        pattern, _ = _path_to_regex("/users/{id}")
        self.assertIsNone(pattern.match("/users/a/b"))

    def test_root_path(self):
        pattern, params = _path_to_regex("/")
        self.assertEqual(params, [])
        self.assertIsNotNone(pattern.match("/"))

    def test_unicode_path(self):
        pattern, _ = _path_to_regex("/grüße/{name}")
        m = pattern.match("/grüße/welt")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "welt")

    def test_url_encoded_chars_in_value(self):
        pattern, _ = _path_to_regex("/files/{filename}")
        m = pattern.match("/files/my%20doc.pdf")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("filename"), "my%20doc.pdf")


# =============================================================================
# Tests: FastTB Route Registration
# =============================================================================

class TestFastTBRegistration(unittest.TestCase):

    def setUp(self):
        self.app = FastTB(title="TestApp")

    def test_get_registers_route(self):
        @self.app.get("/items")
        def get_items():
            return []

        self.assertEqual(len(self.app._routes), 1)
        self.assertEqual(self.app._routes[0].method, "GET")
        self.assertEqual(self.app._routes[0].path, "/items")

    def test_post_registers_route(self):
        @self.app.post("/items")
        def create_item():
            return {}

        self.assertEqual(self.app._routes[0].method, "POST")

    def test_put_registers_route(self):
        @self.app.put("/items/{id}")
        def update_item(id: str):
            return {}

        self.assertEqual(self.app._routes[0].method, "PUT")
        self.assertEqual(self.app._routes[0].param_names, ["id"])

    def test_delete_registers_route(self):
        @self.app.delete("/items/{id}")
        def delete_item(id: str):
            return {}

        self.assertEqual(self.app._routes[0].method, "DELETE")

    def test_patch_registers_route(self):
        @self.app.patch("/items/{id}")
        def patch_item(id: str):
            return {}

        self.assertEqual(self.app._routes[0].method, "PATCH")

    def test_route_multi_method_registers_all(self):
        @self.app.route("/data", methods=["GET", "POST"])
        def data_handler():
            return {}

        methods = [r.method for r in self.app._routes]
        self.assertIn("GET", methods)
        self.assertIn("POST", methods)
        self.assertEqual(len(self.app._routes), 2)

    def test_decorator_returns_original_function(self):
        def my_handler():
            return {}

        decorated = self.app.get("/test")(my_handler)
        self.assertIs(decorated, my_handler)

    def test_custom_name_preserved(self):
        @self.app.get("/test", name="custom_name")
        def handler():
            return {}

        self.assertEqual(self.app._routes[0].name, "custom_name")

    def test_default_name_uses_function_name(self):
        @self.app.get("/test")
        def my_special_handler():
            return {}

        self.assertEqual(self.app._routes[0].name, "my_special_handler")


# =============================================================================
# Tests: FastTB Route Resolution
# =============================================================================

class TestFastTBResolution(unittest.TestCase):

    def setUp(self):
        self.app = FastTB()

        @self.app.get("/health")
        def health():
            return {"status": "ok"}

        @self.app.get("/users/{user_id}")
        def get_user(user_id: str):
            return {"id": user_id}

        @self.app.post("/users")
        def create_user():
            return {}

    def test_resolve_static_route(self):
        result = self.app.resolve_route("/health", "GET")
        self.assertIsNotNone(result)
        route, params = result
        self.assertEqual(route.path, "/health")
        self.assertEqual(params, {})

    def test_resolve_parameterized_route(self):
        result = self.app.resolve_route("/users/42", "GET")
        self.assertIsNotNone(result)
        route, params = result
        self.assertEqual(params, {"user_id": "42"})

    def test_resolve_returns_none_for_wrong_method(self):
        result = self.app.resolve_route("/health", "POST")
        self.assertIsNone(result)

    def test_resolve_returns_none_for_unknown_path(self):
        result = self.app.resolve_route("/nonexistent", "GET")
        self.assertIsNone(result)

    def test_has_route_true(self):
        self.assertTrue(self.app.has_route("/health", "GET"))

    def test_has_route_false(self):
        self.assertFalse(self.app.has_route("/health", "DELETE"))

    def test_method_case_insensitive(self):
        self.assertTrue(self.app.has_route("/health", "get"))

    def test_empty_app_returns_none(self):
        empty = FastTB()
        self.assertIsNone(empty.resolve_route("/anything", "GET"))

    def test_index_rebuilt_after_new_route(self):
        # Force index build
        self.app.resolve_route("/health", "GET")
        self.assertFalse(self.app._index_dirty)

        # Add new route
        @self.app.get("/new")
        def new_route():
            return {}

        self.assertTrue(self.app._index_dirty)
        result = self.app.resolve_route("/new", "GET")
        self.assertIsNotNone(result)


# =============================================================================
# Tests: WebSocket Registration
# =============================================================================

class TestFastTBWebSocket(unittest.TestCase):

    def setUp(self):
        self.app = FastTB()

    def test_websocket_decorator_registers_class(self):
        @self.app.websocket("/ws/chat")
        class ChatHandler:
            async def on_connect(self, conn_id, session):
                pass
            async def on_message(self, payload, conn_id, session, request):
                pass

        self.assertEqual(len(self.app._ws_routes), 1)
        self.assertEqual(self.app._ws_routes[0].path, "/ws/chat")

    def test_websocket_decorator_returns_original_class(self):
        @self.app.websocket("/ws/test")
        class Handler:
            pass

        self.assertEqual(Handler.__name__, "Handler")

    def test_resolve_ws_route(self):
        @self.app.websocket("/ws/room/{room_id}")
        class RoomHandler:
            pass

        result = self.app.resolve_ws_route("/ws/room/lobby")
        self.assertIsNotNone(result)
        ws_route, params = result
        self.assertEqual(params, {"room_id": "lobby"})

    def test_resolve_ws_route_no_match(self):
        result = self.app.resolve_ws_route("/ws/nonexistent")
        self.assertIsNone(result)

    def test_get_websocket_handlers_exports_format(self):
        @self.app.websocket("/ws/chat")
        class ChatHandler:
            async def on_connect(self, conn_id, session):
                pass
            async def on_message(self, payload, conn_id, session, request):
                pass
            async def on_disconnect(self, conn_id, session):
                pass

        handlers = self.app.get_websocket_handlers()
        self.assertIn("ws/chat", handlers)
        entry = handlers["ws/chat"]
        self.assertIn("on_connect", entry)
        self.assertIn("on_message", entry)
        self.assertIn("on_disconnect", entry)

    def test_get_websocket_handlers_skips_missing_methods(self):
        @self.app.websocket("/ws/minimal")
        class MinimalHandler:
            async def on_message(self, payload, conn_id, session, request):
                pass

        handlers = self.app.get_websocket_handlers()
        entry = handlers["ws/minimal"]
        self.assertIn("on_message", entry)
        self.assertNotIn("on_connect", entry)
        self.assertNotIn("on_disconnect", entry)


# =============================================================================
# Tests: list_routes
# =============================================================================

class TestFastTBListRoutes(unittest.TestCase):

    def test_list_routes_includes_http_and_ws(self):
        app = FastTB()

        @app.get("/a")
        def handler_a():
            return {}

        @app.post("/b")
        def handler_b():
            return {}

        @app.websocket("/ws/c")
        class WsC:
            pass

        routes = app.list_routes()
        self.assertEqual(len(routes), 3)
        methods = [r["method"] for r in routes]
        self.assertIn("GET", methods)
        self.assertIn("POST", methods)
        self.assertIn("WS", methods)

    def test_list_routes_empty_app(self):
        app = FastTB()
        self.assertEqual(app.list_routes(), [])


# =============================================================================
# Tests: FastTBHandler — Parameter Injection
# =============================================================================

class TestHandlerParameterInjection(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.ftb = FastTB()
        self.handler = FastTBHandler(self.ftb)

    async def test_inject_request_by_name(self):
        @self.ftb.get("/test")
        async def handler(request):
            return {"path": request.path}

        req = make_request(path="/test")
        status, headers, body = await self.handler.handle_request(req)
        self.assertEqual(status, 200)
        self.assertEqual(parse_json_body(body)["path"], "/test")

    async def test_inject_request_by_type_hint(self):
        @self.ftb.get("/test")
        async def handler(r: ParsedRequest):
            return {"method": r.method}

        req = make_request(path="/test")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["method"], "GET")

    async def test_inject_session_by_name(self):
        @self.ftb.get("/test")
        async def handler(session):
            return {"user": session.user_name}

        session = SessionData(user_name="markin")
        req = make_request(path="/test", session=session)
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["user"], "markin")

    async def test_inject_session_by_type_hint(self):
        @self.ftb.get("/test")
        async def handler(s: SessionData):
            return {"user": s.user_name}

        session = SessionData(user_name="test_user")
        req = make_request(path="/test", session=session)
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["user"], "test_user")

    async def test_inject_session_fallback_to_empty(self):
        @self.ftb.get("/test")
        async def handler(session):
            return {"user": session.user_name}

        req = make_request(path="/test", session=None)
        status, _, body = await self.handler.handle_request(req)
        # SessionData() default user_name is "anonymous"
        self.assertEqual(parse_json_body(body)["user"], "anonymous")

    async def test_inject_path_params(self):
        @self.ftb.get("/users/{user_id}/posts/{post_id}")
        async def handler(user_id: str, post_id: str):
            return {"user": user_id, "post": post_id}

        req = make_request(path="/users/alice/posts/99")
        status, _, body = await self.handler.handle_request(req)
        data = parse_json_body(body)
        self.assertEqual(data["user"], "alice")
        self.assertEqual(data["post"], "99")

    async def test_inject_query_params(self):
        @self.ftb.get("/search")
        async def handler(q: str, limit: int = 10):
            return {"q": q, "limit": limit}

        req = make_request(path="/search", query_params={"q": ["hello"], "limit": ["25"]})
        status, _, body = await self.handler.handle_request(req)
        data = parse_json_body(body)
        self.assertEqual(data["q"], "hello")
        self.assertEqual(data["limit"], 25)

    async def test_inject_json_body_field(self):
        @self.ftb.post("/items")
        async def handler(name: str, price: float):
            return {"name": name, "price": price}

        req = make_request(
            method="POST", path="/items",
            json_data={"name": "Widget", "price": 9.99}
        )
        status, _, body = await self.handler.handle_request(req)
        data = parse_json_body(body)
        self.assertEqual(data["name"], "Widget")
        self.assertAlmostEqual(data["price"], 9.99)

    async def test_inject_form_data_field(self):
        @self.ftb.post("/login")
        async def handler(username: str, password: str):
            return {"user": username}

        req = make_request(
            method="POST", path="/login",
            form_data={"username": "admin", "password": "secret"}
        )
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["user"], "admin")

    async def test_missing_required_param_returns_500(self):
        @self.ftb.get("/need")
        async def handler(required_field: str):
            return {}

        req = make_request(path="/need")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(status, 500)
        self.assertIn("required_field", parse_json_body(body)["message"])

    async def test_default_param_not_required(self):
        @self.ftb.get("/opt")
        async def handler(color: str = "blue"):
            return {"color": color}

        req = make_request(path="/opt")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["color"], "blue")

    async def test_combined_sources_request_session_path_query(self):
        @self.ftb.get("/users/{uid}")
        async def handler(request: ParsedRequest, session: SessionData, uid: str, verbose: bool = False):
            return {
                "path": request.path,
                "user": session.user_name,
                "uid": uid,
                "verbose": verbose,
            }

        session = SessionData(user_name="markin")
        req = make_request(
            path="/users/42",
            query_params={"verbose": ["true"]},
            session=session,
        )
        status, _, body = await self.handler.handle_request(req)
        data = parse_json_body(body)
        self.assertEqual(data["path"], "/users/42")
        self.assertEqual(data["user"], "markin")
        self.assertEqual(data["uid"], "42")
        self.assertTrue(data["verbose"])


# =============================================================================
# Tests: FastTBHandler — Type Coercion
# =============================================================================

class TestHandlerTypeCoercion(unittest.TestCase):

    def test_coerce_int(self):
        self.assertEqual(FastTBHandler._coerce("42", int), 42)

    def test_coerce_float(self):
        self.assertAlmostEqual(FastTBHandler._coerce("3.14", float), 3.14)

    def test_coerce_bool_true(self):
        for val in ("true", "True", "1", "yes"):
            self.assertTrue(FastTBHandler._coerce(val, bool), msg=f"Failed for {val}")

    def test_coerce_bool_false(self):
        for val in ("false", "False", "0", "no"):
            self.assertFalse(FastTBHandler._coerce(val, bool), msg=f"Failed for {val}")

    def test_coerce_str_passthrough(self):
        self.assertEqual(FastTBHandler._coerce("hello", str), "hello")

    def test_coerce_no_annotation_passthrough(self):
        import inspect
        self.assertEqual(FastTBHandler._coerce("hello", inspect.Parameter.empty), "hello")

    def test_coerce_invalid_int_returns_string(self):
        self.assertEqual(FastTBHandler._coerce("not_a_number", int), "not_a_number")

    def test_coerce_invalid_float_returns_string(self):
        self.assertEqual(FastTBHandler._coerce("abc", float), "abc")


# =============================================================================
# Tests: FastTBHandler — Response Formatting
# =============================================================================

class TestHandlerResponseFormatting(unittest.TestCase):

    def test_format_none_returns_ok(self):
        status, headers, body = FastTBHandler._format_result(None)
        self.assertEqual(status, 200)
        self.assertEqual(parse_json_body(body), {"status": "ok"})

    def test_format_dict_returns_json(self):
        status, headers, body = FastTBHandler._format_result({"key": "val"})
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(parse_json_body(body)["key"], "val")

    def test_format_list_returns_json(self):
        status, headers, body = FastTBHandler._format_result([1, 2, 3])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), [1, 2, 3])

    def test_format_html_string(self):
        status, headers, body = FastTBHandler._format_result("<h1>Hi</h1>")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertEqual(body, b"<h1>Hi</h1>")

    def test_format_plain_string_returns_json_wrapped(self):
        status, _, body = FastTBHandler._format_result("hello world")
        self.assertEqual(parse_json_body(body), {"result": "hello world"})

    def test_format_bytes_returns_octet_stream(self):
        raw = b"\x00\x01\x02"
        status, headers, body = FastTBHandler._format_result(raw)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertEqual(body, raw)

    def test_format_tuple_3_passthrough(self):
        original = (201, {"X-Custom": "yes"}, b"created")
        result = FastTBHandler._format_result(original)
        self.assertEqual(result, original)

    def test_format_tuple_2_with_dict(self):
        status, headers, body = FastTBHandler._format_result((201, {"id": "abc"}))
        self.assertEqual(status, 201)
        self.assertEqual(parse_json_body(body)["id"], "abc")

    def test_format_tuple_2_with_string(self):
        status, headers, body = FastTBHandler._format_result((400, "Bad Request"))
        self.assertEqual(status, 400)
        self.assertEqual(body, b"Bad Request")

    def test_format_integer_fallback(self):
        status, _, body = FastTBHandler._format_result(42)
        self.assertEqual(parse_json_body(body), {"result": "42"})


# =============================================================================
# Tests: FastTBHandler — Sync + Async Execution
# =============================================================================

class TestHandlerExecution(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.ftb = FastTB()
        self.handler = FastTBHandler(self.ftb)

    async def test_async_handler_executed(self):
        @self.ftb.get("/async")
        async def async_handler():
            return {"mode": "async"}

        req = make_request(path="/async")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["mode"], "async")

    async def test_sync_handler_executed(self):
        @self.ftb.get("/sync")
        def sync_handler():
            return {"mode": "sync"}

        req = make_request(path="/sync")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["mode"], "sync")

    async def test_unmatched_route_returns_404(self):
        req = make_request(path="/nonexistent")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(status, 404)

    async def test_handler_exception_returns_500(self):
        @self.ftb.get("/boom")
        async def boom():
            raise RuntimeError("kaboom")

        req = make_request(path="/boom")
        status, _, body = await self.handler.handle_request(req)
        self.assertEqual(status, 500)
        self.assertIn("kaboom", parse_json_body(body)["message"])


# =============================================================================
# Tests: FastTBHandler — has_route delegation
# =============================================================================

class TestHandlerHasRoute(unittest.TestCase):

    def test_has_route_delegates_to_app(self):
        ftb = FastTB()

        @ftb.get("/exists")
        def handler():
            return {}

        h = FastTBHandler(ftb)
        self.assertTrue(h.has_route("/exists", "GET"))
        self.assertFalse(h.has_route("/nope", "GET"))


# =============================================================================
# Tests: Static File Mounting
# =============================================================================

class TestStaticMount(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        import tempfile, os
        self.tmpdir = tempfile.mkdtemp()
        # Create test files
        with open(os.path.join(self.tmpdir, "style.css"), "w") as f:
            f.write("body { color: red; }")
        with open(os.path.join(self.tmpdir, "main-5d3f7ed2.js"), "w") as f:
            f.write("console.log('hello');")
        os.makedirs(os.path.join(self.tmpdir, "sub"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "sub", "data.json"), "w") as f:
            f.write('{"ok":true}')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_mount_static_registers(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        self.assertEqual(len(ftb._static_mounts), 1)

    def test_resolve_static_finds_file(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        result = ftb.resolve_static("/dist/style.css")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("style.css"))

    def test_resolve_static_finds_subdir(self):
        ftb = FastTB()
        ftb.mount_static("/assets", self.tmpdir)
        result = ftb.resolve_static("/assets/sub/data.json")
        self.assertIsNotNone(result)

    def test_resolve_static_rejects_traversal(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        result = ftb.resolve_static("/dist/../../../etc/passwd")
        self.assertIsNone(result)

    def test_resolve_static_rejects_nonexistent(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        result = ftb.resolve_static("/dist/nope.txt")
        self.assertIsNone(result)

    def test_resolve_static_rejects_wrong_prefix(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        result = ftb.resolve_static("/other/style.css")
        self.assertIsNone(result)

    def test_has_route_includes_static(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        self.assertTrue(ftb.has_route("/dist/style.css", "GET"))
        self.assertFalse(ftb.has_route("/dist/style.css", "POST"))

    async def test_handler_serves_static_file(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        handler = FastTBHandler(ftb)
        req = make_request(path="/dist/style.css")
        status, headers, body = await handler.handle_request(req)
        self.assertEqual(status, 200)
        self.assertIn("text/css", headers["Content-Type"])
        self.assertEqual(body, b"body { color: red; }")

    async def test_handler_static_hashed_file_immutable_cache(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        handler = FastTBHandler(ftb)
        req = make_request(path="/dist/main-5d3f7ed2.js")
        status, headers, body = await handler.handle_request(req)
        self.assertEqual(status, 200)
        self.assertIn("immutable", headers["Cache-Control"])

    async def test_handler_static_unhashed_file_short_cache(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        handler = FastTBHandler(ftb)
        req = make_request(path="/dist/style.css")
        status, headers, _ = await handler.handle_request(req)
        self.assertNotIn("immutable", headers["Cache-Control"])
        self.assertIn("3600", headers["Cache-Control"])

    async def test_handler_static_traversal_returns_404(self):
        ftb = FastTB()
        ftb.mount_static("/dist", self.tmpdir)
        handler = FastTBHandler(ftb)
        req = make_request(path="/dist/../../etc/passwd")
        status, _, _ = await handler.handle_request(req)
        self.assertEqual(status, 404)


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases(unittest.IsolatedAsyncioTestCase):

    async def test_duplicate_routes_first_wins(self):
        ftb = FastTB()

        @ftb.get("/dup")
        async def first():
            return {"winner": "first"}

        @ftb.get("/dup")
        async def second():
            return {"winner": "second"}

        handler = FastTBHandler(ftb)
        req = make_request(path="/dup")
        status, _, body = await handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["winner"], "first")

    async def test_path_param_with_special_chars(self):
        ftb = FastTB()

        @ftb.get("/files/{name}")
        async def get_file(name: str):
            return {"name": name}

        handler = FastTBHandler(ftb)
        req = make_request(path="/files/hello-world_v2.tar.gz")
        status, _, body = await handler.handle_request(req)
        self.assertEqual(parse_json_body(body)["name"], "hello-world_v2.tar.gz")

    async def test_empty_path_param(self):
        """Path param regex [^/]+ requires at least one char — empty segment won't match."""
        ftb = FastTB()

        @ftb.get("/items/{id}")
        async def handler(id: str):
            return {}

        h = FastTBHandler(ftb)
        # /items/ has empty segment after last slash → no match
        req = make_request(path="/items/")
        status, _, _ = await h.handle_request(req)
        self.assertEqual(status, 404)

    async def test_query_param_multi_value(self):
        ftb = FastTB()

        @ftb.get("/multi")
        async def handler(tags):
            return {"tags": tags}

        h = FastTBHandler(ftb)
        req = make_request(path="/multi", query_params={"tags": ["a", "b", "c"]})
        status, _, body = await h.handle_request(req)
        self.assertEqual(parse_json_body(body)["tags"], ["a", "b", "c"])

    async def test_int_path_param_coerced(self):
        ftb = FastTB()

        @ftb.get("/items/{id}")
        async def handler(id: int):
            return {"id": id, "type": type(id).__name__}

        h = FastTBHandler(ftb)
        req = make_request(path="/items/42")
        status, _, body = await h.handle_request(req)
        data = parse_json_body(body)
        self.assertEqual(data["id"], 42)
        self.assertEqual(data["type"], "int")


    async def test_kwargs_in_handler_skipped(self):
        """**kwargs must be ignored by DI, not treated as required param."""
        ftb = FastTB()

        @ftb.get("/kw")
        async def handler(request: ParsedRequest, format: str = "auto", **kwargs):
            return {"format": format}

        h = FastTBHandler(ftb)
        req = make_request(path="/kw")
        status, _, body = await h.handle_request(req)
        self.assertEqual(status, 200)
        self.assertEqual(parse_json_body(body)["format"], "auto")

    async def test_args_in_handler_skipped(self):
        """*args must be ignored by DI."""
        ftb = FastTB()

        @ftb.get("/va")
        async def handler(*args, name: str = "world"):
            return {"name": name}

        h = FastTBHandler(ftb)
        req = make_request(path="/va")
        status, _, body = await h.handle_request(req)
        self.assertEqual(status, 200)
        self.assertEqual(parse_json_body(body)["name"], "world")


if __name__ == "__main__":
    unittest.main()
