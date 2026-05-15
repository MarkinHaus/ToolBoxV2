"""
Tests for server_worker.py – AuthHandler and HTTPWorker routing.

Tests cover:
- AUTH_ENDPOINTS dictionary completeness (all auth routes registered)
- AuthHandler endpoint dispatch
- Access control integration
- Route matching for auth vs API paths
"""

import unittest
from unittest.mock import MagicMock, AsyncMock, patch


class TestAuthEndpointRegistry(unittest.TestCase):
    """Verify AUTH_ENDPOINTS dict matches all expected custom auth routes."""

    def _get_auth_endpoints(self):
        from toolboxv2.utils.workers.server_worker import HTTPWorker
        return HTTPWorker.AUTH_ENDPOINTS

    def test_auth_endpoints_is_dict(self):
        eps = self._get_auth_endpoints()
        self.assertIsInstance(eps, dict)

    def test_required_legacy_routes(self):
        """Core session routes must be present."""
        eps = self._get_auth_endpoints()
        required = [
            "/validateSession",
            "/IsValidSession",
            "/web/logoutS",
            "/api_user_data",
        ]
        for route in required:
            self.assertIn(route, eps, f"Missing required auth route: {route}")

    def test_oauth_routes(self):
        """OAuth (Discord + Google) routes must be registered."""
        eps = self._get_auth_endpoints()
        oauth_routes = [
            "/auth/discord/url",
            "/auth/discord/callback",
            "/auth/google/url",
            "/auth/google/callback",
        ]
        for route in oauth_routes:
            self.assertIn(route, eps, f"Missing OAuth route: {route}")

    def test_magic_link_route(self):
        """Magic link verification route must be registered."""
        eps = self._get_auth_endpoints()
        self.assertIn("/auth/magic/verify", eps)

    def test_handler_method_names_are_strings(self):
        """All handler values must be non-empty strings."""
        eps = self._get_auth_endpoints()
        for path, handler in eps.items():
            self.assertIsInstance(handler, str, f"{path} handler is not a string")
            self.assertTrue(len(handler) > 0, f"{path} handler is empty")

    def test_all_paths_start_with_slash(self):
        eps = self._get_auth_endpoints()
        for path in eps:
            self.assertTrue(path.startswith("/"), f"Route {path!r} must start with /")

    def test_no_duplicate_handler_names(self):
        """Each handler should map to a unique method."""
        eps = self._get_auth_endpoints()
        handlers = list(eps.values())
        self.assertEqual(len(handlers), len(set(handlers)),
                         f"Duplicate handler names: {handlers}")


class TestAuthHandlerInit(unittest.TestCase):
    """Tests for AuthHandler initialization."""

    def test_auth_handler_reads_config(self):
        from toolboxv2.utils.workers.server_worker import AuthHandler

        mock_session_mgr = MagicMock()
        mock_app = MagicMock()
        mock_config = MagicMock()
        mock_config.toolbox.auth_module = "CloudM.Auth"
        mock_config.toolbox.verify_session_func = "validate_session"

        handler = AuthHandler(mock_session_mgr, mock_app, mock_config)
        self.assertEqual(handler.auth_module, "CloudM.Auth")
        self.assertEqual(handler.verify_func, "validate_session")

    def test_auth_handler_default_config(self):
        from toolboxv2.utils.workers.server_worker import AuthHandler

        mock_session_mgr = MagicMock()
        mock_app = MagicMock()
        mock_config = MagicMock(spec=[])  # No attributes
        # getattr with default should work
        mock_config.toolbox = MagicMock(spec=[])

        handler = AuthHandler(mock_session_mgr, mock_app, mock_config)
        self.assertEqual(handler.auth_module, "CloudM.Auth")
        self.assertEqual(handler.verify_func, "validate_session")


class TestHTTPWorkerRouting(unittest.TestCase):
    """Tests for HTTPWorker route dispatch logic."""

    def test_auth_endpoint_detection(self):
        """Auth paths should be recognized by the worker."""
        from toolboxv2.utils.workers.server_worker import HTTPWorker
        auth_paths = list(HTTPWorker.AUTH_ENDPOINTS.keys())
        non_auth = ["/api/some/func", "/health", "/metrics", "/ws"]

        for path in auth_paths:
            self.assertIn(path, HTTPWorker.AUTH_ENDPOINTS)

        for path in non_auth:
            self.assertNotIn(path, HTTPWorker.AUTH_ENDPOINTS)

    def test_api_path_not_in_auth_endpoints(self):
        """Regular /api/ paths must NOT be in AUTH_ENDPOINTS."""
        from toolboxv2.utils.workers.server_worker import HTTPWorker
        eps = HTTPWorker.AUTH_ENDPOINTS
        api_paths = [
            "/api/CloudM.Auth/validate_session",
            "/api/CloudM.Auth/get_user_data",
            "/api/SomeModule/some_func",
        ]
        for path in api_paths:
            self.assertNotIn(path, eps)


class TestAccessControllerLevels(unittest.TestCase):
    """Tests for AccessController access level definitions."""

    def test_access_levels_defined(self):
        from toolboxv2.utils.workers.session import AccessLevel
        self.assertEqual(AccessLevel.ADMIN, -1)
        self.assertEqual(AccessLevel.NOT_LOGGED_IN, 0)
        self.assertEqual(AccessLevel.LOGGED_IN, 1)
        self.assertEqual(AccessLevel.TRUSTED, 2)

    def test_admin_has_lowest_level(self):
        from toolboxv2.utils.workers.session import AccessLevel
        self.assertLess(AccessLevel.ADMIN, AccessLevel.NOT_LOGGED_IN)


if __name__ == "__main__":
    unittest.main()
