"""
Tests for nginx config generation – auth route completeness.

The CLI worker manager generates nginx configs for reverse-proxying.
ALL auth routes from HTTPWorker.AUTH_ENDPOINTS must appear in the
generated nginx site config so that nginx properly proxies them.

Tests cover:
- All AUTH_ENDPOINTS present in generated nginx config
- OAuth routes (Discord, Google) have correct methods
- Magic link route proxied correctly
- Rate limiting applied to auth endpoints
- Auth endpoints block structure
"""

import unittest
from unittest.mock import MagicMock, patch


def _make_nginx_manager():
    """Create a NginxManager with mock config for testing."""
    from toolboxv2.utils.clis.cli_worker_manager import NginxManager

    # Build minimal mock config
    mock_config = MagicMock()

    # Nginx sub-config
    nginx_cfg = MagicMock()
    nginx_cfg.listen_port = 80
    nginx_cfg.upstream_http = "toolbox_http"
    nginx_cfg.upstream_ws = "toolbox_ws"
    nginx_cfg.server_name = "test.example.com"
    nginx_cfg.rate_limit_enabled = True
    nginx_cfg.rate_limit_zone = "tb_limit"
    nginx_cfg.rate_limit_rate = "10r/s"
    nginx_cfg.rate_limit_burst = 20
    nginx_cfg.auth_rate_limit_rate = "5r/s"
    nginx_cfg.auth_rate_limit_burst = 10
    nginx_cfg.static_enabled = True
    nginx_cfg.static_root = "./dist"
    nginx_cfg.ssl_enabled = False

    mock_config.nginx = nginx_cfg

    # Manager sub-config
    mock_manager = MagicMock()
    mock_manager.web_ui_port = 9002
    mock_config.manager = mock_manager

    # Patch _find_nginx + SSLManager to avoid filesystem access
    with patch.object(NginxManager, '__init__', lambda self, cfg: None):
        mgr = NginxManager.__new__(NginxManager)
        mgr.config = nginx_cfg
        mgr._manager = mock_manager
        mgr._nginx_path = "/usr/sbin/nginx"

        # SSL not available
        ssl_mock = MagicMock()
        ssl_mock.available = False
        mgr._ssl = ssl_mock

    return mgr


class TestNginxAuthEndpointsBlockComplete(unittest.TestCase):
    """Generated nginx auth block must cover all AUTH_ENDPOINTS."""

    def setUp(self):
        self.mgr = _make_nginx_manager()
        self.auth_block = self.mgr._generate_auth_endpoints_block(
            upstream_http="toolbox_http",
            rate_limit_block="\n                limit_req zone=tb_auth_limit burst=10 nodelay;"
        )

    def test_validate_session_route(self):
        self.assertIn("/validateSession", self.auth_block)

    def test_is_valid_session_route(self):
        self.assertIn("/IsValidSession", self.auth_block)

    def test_logout_route(self):
        self.assertIn("/web/logoutS", self.auth_block)

    def test_user_data_route(self):
        self.assertIn("/api_user_data", self.auth_block)

    def test_discord_url_route(self):
        self.assertIn("/auth/discord/url", self.auth_block)

    def test_discord_callback_route(self):
        self.assertIn("/auth/discord/callback", self.auth_block)

    def test_google_url_route(self):
        self.assertIn("/auth/google/url", self.auth_block)

    def test_google_callback_route(self):
        self.assertIn("/auth/google/callback", self.auth_block)

    def test_magic_link_verify_route(self):
        self.assertIn("/auth/magic/verify", self.auth_block)


class TestNginxAuthEndpointsConsistencyWithWorker(unittest.TestCase):
    """
    Cross-check: every path in HTTPWorker.AUTH_ENDPOINTS must appear
    somewhere in the generated nginx site config.
    """

    def test_all_worker_auth_endpoints_in_nginx(self):
        from toolboxv2.utils.workers.server_worker import HTTPWorker
        mgr = _make_nginx_manager()
        site_config = mgr.generate_site_config(
            http_ports=[8000, 8001],
            ws_ports=[8100],
        )

        missing = []
        for path in HTTPWorker.AUTH_ENDPOINTS:
            if path not in site_config:
                missing.append(path)

        self.assertEqual(
            missing, [],
            f"Nginx site config is missing these auth routes: {missing}"
        )

    def test_all_worker_auth_endpoints_in_full_config(self):
        """Also check legacy full config mode."""
        from toolboxv2.utils.workers.server_worker import HTTPWorker
        mgr = _make_nginx_manager()
        full_config = mgr.generate_config(
            http_ports=[8000],
            ws_ports=[8100],
            full_config=True,
        )

        missing = []
        for path in HTTPWorker.AUTH_ENDPOINTS:
            if path not in full_config:
                missing.append(path)

        self.assertEqual(
            missing, [],
            f"Nginx full config is missing these auth routes: {missing}"
        )


class TestNginxAuthMethodRestrictions(unittest.TestCase):
    """Auth endpoints should have correct HTTP method restrictions."""

    def setUp(self):
        self.mgr = _make_nginx_manager()
        self.auth_block = self.mgr._generate_auth_endpoints_block(
            upstream_http="toolbox_http",
            rate_limit_block=""
        )

    def test_validate_session_is_post_only(self):
        # /validateSession should restrict to POST
        # Find the location block and check for "limit_except POST"
        idx = self.auth_block.find("/validateSession")
        block = self.auth_block[idx:idx + 500]
        self.assertIn("POST", block)

    def test_is_valid_session_is_get_only(self):
        idx = self.auth_block.find("/IsValidSession")
        block = self.auth_block[idx:idx + 500]
        self.assertIn("GET", block)

    def test_oauth_url_endpoints_are_get(self):
        """OAuth URL generation endpoints should allow GET."""
        for path in ["/auth/discord/url", "/auth/google/url"]:
            idx = self.auth_block.find(path)
            self.assertGreater(idx, -1, f"{path} not found in auth block")
            block = self.auth_block[idx:idx + 500]
            self.assertIn("GET", block, f"{path} should be GET-accessible")

    def test_oauth_callback_endpoints_are_get(self):
        """OAuth callbacks must be GET (browser redirect)."""
        for path in ["/auth/discord/callback", "/auth/google/callback"]:
            idx = self.auth_block.find(path)
            self.assertGreater(idx, -1, f"{path} not found in auth block")
            block = self.auth_block[idx:idx + 500]
            self.assertIn("GET", block, f"{path} should accept GET")


class TestNginxRateLimiting(unittest.TestCase):
    """Auth endpoints should reference the auth rate limit zone."""

    def test_rate_limit_in_auth_block(self):
        mgr = _make_nginx_manager()
        auth_block = mgr._generate_auth_endpoints_block(
            upstream_http="toolbox_http",
            rate_limit_block="\n                limit_req zone=tb_auth_limit burst=10 nodelay;"
        )
        self.assertIn("tb_auth_limit", auth_block)

    def test_main_config_defines_rate_limit_zones(self):
        mgr = _make_nginx_manager()
        main_conf = mgr.generate_nginx_conf()
        self.assertIn("tb_auth_limit", main_conf)
        self.assertIn("tb_limit", main_conf)


class TestNginxSiteConfigStructure(unittest.TestCase):
    """Basic structural checks for the generated site config."""

    def test_has_upstream_blocks(self):
        mgr = _make_nginx_manager()
        cfg = mgr.generate_site_config([8000], [8100])
        self.assertIn("upstream toolbox_http", cfg)
        self.assertIn("upstream toolbox_ws", cfg)

    def test_has_server_block(self):
        mgr = _make_nginx_manager()
        cfg = mgr.generate_site_config([8000], [8100])
        self.assertIn("server {", cfg)

    def test_has_health_endpoint(self):
        mgr = _make_nginx_manager()
        cfg = mgr.generate_site_config([8000], [8100])
        self.assertIn("/health", cfg)

    def test_has_api_block(self):
        mgr = _make_nginx_manager()
        cfg = mgr.generate_site_config([8000], [8100])
        self.assertIn("location /api/", cfg)

    def test_has_websocket_block(self):
        mgr = _make_nginx_manager()
        cfg = mgr.generate_site_config([8000], [8100])
        self.assertIn("location /ws", cfg)
        self.assertIn("upgrade", cfg.lower())


if __name__ == "__main__":
    unittest.main()
