"""WP1 K2: local device trust - unauthenticated loopback requests run as
the linked device owner (level -1) without a token. Remote bases and
server profiles never enable it (guarded at the call sites).
"""
import asyncio
import unittest

from toolboxv2.utils.workers.session import (
    AccessLevel, SessionData, SessionManager,
)


def _manager():
    return SessionManager(cookie_secret="unit-test-secret-0123456789abcdef")


class LocalTrustTest(unittest.TestCase):
    def test_enable_local_trust_stores_provider(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        self.assertEqual(m._device_trust["user_id"], "usr_a")

    def test_enable_local_trust_none_disables(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a"})
        m.enable_local_trust(None)
        self.assertIsNone(m._device_trust)

    def test_device_trust_session_shape(self):
        m = _manager()
        s = m._device_trust_session({"user_id": "usr_a", "username": "markin"})
        self.assertEqual(s.user_id, "usr_a")
        self.assertEqual(s.user_name, "markin")
        self.assertEqual(s.level, -1)
        self.assertTrue(s.validated)
        self.assertFalse(s.anonymous)
        self.assertTrue(s.is_authenticated)
        self.assertTrue(s.extra.get("trusted_device"))

    def test_async_request_without_trust_stays_anonymous(self):
        m = _manager()
        s = asyncio.run(m.get_session_from_request({}))
        self.assertTrue(s.anonymous)
        self.assertEqual(s.level, AccessLevel.NOT_LOGGED_IN)

    def test_async_request_with_trust_is_owner(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertEqual(s.user_id, "usr_a")
        self.assertEqual(s.level, -1)
        self.assertFalse(s.anonymous)

    def test_sync_request_without_trust_stays_anonymous(self):
        m = _manager()
        s = m.get_session_from_request_sync({})
        self.assertTrue(s.anonymous)

    def test_sync_request_with_trust_is_owner(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertEqual(s.user_id, "usr_a")
        self.assertEqual(s.level, -1)

    def test_trust_does_not_shadow_api_key_auth(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a"})
        key_session = SessionData.authenticated_session(
            "usr_key", "keyuser", AccessLevel.LOGGED_IN)
        m.register_api_key("k-123", key_session)
        s = m.get_session_from_request_sync({}, headers={"X-API-Key": "k-123"})
        self.assertEqual(s.user_id, "usr_key")



    def test_async_loopback_v4(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertFalse(s.anonymous)
        self.assertEqual(s.user_id, "usr_a")
    def test_sync_loopback_v4(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertFalse(s.anonymous)
        self.assertEqual(s.user_id, "usr_a")
    def test_async_loopback_v6(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "::1", "HTTP_HOST": "[::1]:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertFalse(s.anonymous)
        self.assertEqual(s.user_id, "usr_a")
    def test_sync_loopback_v6(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "::1", "HTTP_HOST": "[::1]:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertFalse(s.anonymous)
        self.assertEqual(s.user_id, "usr_a")
    def test_async_remote_peer(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "192.168.1.5", "HTTP_HOST": "127.0.0.1:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertTrue(s.anonymous)
    def test_sync_remote_peer(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "192.168.1.5", "HTTP_HOST": "127.0.0.1:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertTrue(s.anonymous)
    def test_async_proxy_header(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467", "HTTP_X_FORWARDED_FOR": "1.2.3.4"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertTrue(s.anonymous)
    def test_sync_proxy_header(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467", "HTTP_X_FORWARDED_FOR": "1.2.3.4"}
        s = m.get_session_from_request_sync(env)
        self.assertTrue(s.anonymous)
    def test_async_evil_host(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "evil.example"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertTrue(s.anonymous)
    def test_sync_evil_host(self):
        m = _manager()
        m.enable_local_trust({"user_id": "usr_a", "username": "markin"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "evil.example"}
        s = m.get_session_from_request_sync(env)
        self.assertTrue(s.anonymous)
    def test_async_provider_none(self):
        m = _manager()
        m.enable_local_trust(None)
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertTrue(s.anonymous)
    def test_sync_provider_none(self):
        m = _manager()
        m.enable_local_trust(None)
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertTrue(s.anonymous)
    def test_async_provider_no_uid(self):
        m = _manager()
        m.enable_local_trust({"username": "x"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertTrue(s.anonymous)
    def test_sync_provider_no_uid(self):
        m = _manager()
        m.enable_local_trust({"username": "x"})
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertTrue(s.anonymous)
    def test_async_no_trust(self):
        m = _manager()
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = asyncio.run(m.get_session_from_request(env))
        self.assertTrue(s.anonymous)
    def test_sync_no_trust(self):
        m = _manager()
        env = {"REMOTE_ADDR": "127.0.0.1", "HTTP_HOST": "127.0.0.1:8467"}
        s = m.get_session_from_request_sync(env)
        self.assertTrue(s.anonymous)

if __name__ == "__main__":
    unittest.main()
