"""
Tests for the new dashboard-facing credential/share-token API:
  - tb_get_my_minio_credentials
  - tb_generate_share_token
  - tb_get_share_credentials
  - _extract_user_id helper
"""
import unittest
from unittest.mock import MagicMock, patch


def _make_session(user_id="markin", user_name="markin", anonymous=False):
    """Build a Session-like object matching the Session dataclass."""
    sess = MagicMock()
    sess.user_id = user_id
    sess.user_name = user_name
    sess.is_authenticated = not anonymous
    sess.get = lambda key, default=None: {
        "user_id": user_id, "user_name": user_name, "uid": user_id,
    }.get(key, default)
    return sess


def _make_request(user_id="markin", user_name="markin", anonymous=False):
    req = MagicMock()
    req.session = _make_session(user_id, user_name, anonymous)
    return req


class TestExtractUserId(unittest.TestCase):

    def test_extract_user_id_with_attr(self):
        from toolboxv2.mods.CloudM.LiveSync import _extract_user_id
        self.assertEqual(_extract_user_id(_make_request(user_id="alice")), "alice")

    def test_extract_user_id_fallback_to_username(self):
        from toolboxv2.mods.CloudM.LiveSync import _extract_user_id
        req = MagicMock()
        sess = MagicMock(spec=["user_name", "get"])
        sess.user_name = "bob"
        sess.get = lambda k, d=None: None
        req.session = sess
        self.assertEqual(_extract_user_id(req), "bob")

    def test_extract_user_id_anonymous_returns_empty(self):
        from toolboxv2.mods.CloudM.LiveSync import _extract_user_id
        self.assertEqual(
            _extract_user_id(_make_request(anonymous=True, user_id="", user_name="anonymous")),
            "",
        )

    def test_extract_user_id_no_request(self):
        from toolboxv2.mods.CloudM.LiveSync import _extract_user_id
        self.assertEqual(_extract_user_id(None), "")

    def test_extract_user_id_no_session(self):
        from toolboxv2.mods.CloudM.LiveSync import _extract_user_id
        req = MagicMock(spec=[])
        self.assertEqual(_extract_user_id(req), "")


class TestGetMyMinioCredentials(unittest.TestCase):

    def test_success(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_get_my_minio_credentials
        creds = {
            "endpoint": "localhost:9000",
            "access_key": "sa-x",
            "secret_key": "secret",
            "secure": False,
            "buckets": {"private": "tb-users-private", "public": "tb-users-public", "shared": "tb-shared"},
            "user_prefix": "markin",
            "policy_applied": True,
            "expires_in": 86400,
        }
        req = _make_request(user_id="markin")
        with patch("toolboxv2.mods.CloudM.LiveSync.minio_helper.vend_user_credentials_for_user",
                   return_value=creds) as mock_vend:
            result = tb_get_my_minio_credentials(app=None, request=req)
        rd = result.as_dict()
        self.assertEqual(rd["error"], "none")
        self.assertEqual(rd["info"]["exec_code"], 0)
        self.assertEqual(rd["result"]["data"], creds)
        mock_vend.assert_called_once()
        self.assertEqual(mock_vend.call_args[0][0], "markin")

    def test_no_request(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_get_my_minio_credentials
        result = tb_get_my_minio_credentials(app=None, request=None)
        rd = result.as_dict()
        self.assertNotEqual(rd["error"], "none")
        self.assertNotEqual(rd["info"]["exec_code"], 0)
        self.assertIn("request required", str(rd["result"]["data"]))

    def test_no_session_returns_error(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_get_my_minio_credentials
        result = tb_get_my_minio_credentials(
            app=None,
            request=_make_request(anonymous=True, user_id="", user_name="anonymous"),
        )
        rd = result.as_dict()
        self.assertNotEqual(rd["error"], "none")
        self.assertIn("Authentication required", str(rd["result"]["data"]))

    def test_broker_value_error_propagates(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_get_my_minio_credentials
        req = _make_request(user_id="markin")
        with patch("toolboxv2.mods.CloudM.LiveSync.minio_helper.vend_user_credentials_for_user",
                   side_effect=ValueError("env_config.endpoint required")):
            result = tb_get_my_minio_credentials(app=None, request=req)
        rd = result.as_dict()
        self.assertNotEqual(rd["error"], "none")
        self.assertIn("env_config.endpoint", str(rd["result"]["data"]))


class TestGenerateShareToken(unittest.TestCase):

    def test_success_auto_generates_share_id(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_generate_share_token
        result = tb_generate_share_token(app=None, request=_make_request())
        rd = result.as_dict()
        self.assertEqual(rd["error"], "none")
        data = rd["result"]["data"]
        self.assertIn("token", data)
        self.assertIn("share_id", data)
        self.assertTrue(len(data["share_id"]) >= 6)

    def test_success_uses_provided_share_id(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_generate_share_token
        result = tb_generate_share_token(
            app=None,
            request=_make_request(),
            share_id="mycustom",
            ws_host="example.com",
            ws_port=9000,
        )
        rd = result.as_dict()
        self.assertEqual(rd["error"], "none")
        data = rd["result"]["data"]
        self.assertEqual(data["share_id"], "mycustom")
        # Token is now v3 (encrypted) - decode via ShareToken API
        from toolboxv2.mods.CloudM.LiveSync.config import ShareToken
        decoded = ShareToken.decode(data["token"])
        self.assertEqual(decoded.share_id, "mycustom")
        self.assertEqual(decoded.ws_endpoint, "ws://example.com:9000")

    def test_does_not_start_server(self):
        """Critical: generate_share_token must NOT call start_sync."""
        from toolboxv2.mods.CloudM.LiveSync import tb_generate_share_token
        with patch("toolboxv2.mods.CloudM.LiveSync.start_sync") as mock_start:
            result = tb_generate_share_token(app=None, request=_make_request())
        self.assertEqual(result.as_dict()["error"], "none")
        mock_start.assert_not_called()


class TestGetShareCredentials(unittest.TestCase):

    def test_success(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_get_share_credentials
        creds = {
            "endpoint": "localhost:9000",
            "access_key": "sa-share",
            "secret_key": "secret",
            "secure": False,
            "bucket": "livesync",
            "prefix": "share123",
            "policy_applied": True,
        }
        with patch("toolboxv2.mods.CloudM.LiveSync.minio_helper.vend_credentials_for_share",
                   return_value=creds) as mock_vend:
            result = tb_get_share_credentials(
                app=None, request=_make_request(), share_id="share123"
            )
        rd = result.as_dict()
        self.assertEqual(rd["error"], "none")
        self.assertEqual(rd["result"]["data"], creds)
        mock_vend.assert_called_once()
        self.assertEqual(mock_vend.call_args[0][0], "share123")

    def test_missing_share_id(self):
        from toolboxv2.mods.CloudM.LiveSync import tb_get_share_credentials
        result = tb_get_share_credentials(app=None, request=_make_request(), share_id="")
        rd = result.as_dict()
        self.assertNotEqual(rd["error"], "none")
        self.assertIn("share_id required", str(rd["result"]["data"]))


if __name__ == "__main__":
    unittest.main()
