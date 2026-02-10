"""
Tests for CLI session management (toolboxv2/utils/system/session.py).

Tests cover:
- Token storage (JSON file based)
- Token loading & clearing
- Path generation & safety
- Backwards-compat aliases (clerk_user_id, clerk_session_token)
- _get_auth_headers()
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from toolboxv2.utils.singelton_class import Singleton


def _reset_session_singleton():
    """Reset Session singleton for test isolation."""
    try:
        from toolboxv2.utils.system.session import Session
        if Session in Singleton._instances:
            del Singleton._instances[Session]
            Singleton._args.pop(Session, None)
            Singleton._kwargs.pop(Session, None)
    except (ImportError, KeyError):
        pass


class TestSessionTokenStorage(unittest.TestCase):
    """Tests for local JSON file based token storage."""

    def setUp(self):
        _reset_session_singleton()
        self.tmpdir = tempfile.mkdtemp()
        self._env_patcher = patch.dict(os.environ, {"TB_DATA_DIR": self.tmpdir})
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()
        _reset_session_singleton()
        # Clean up temp files
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_session(self, username="testuser"):
        from toolboxv2.utils.system.session import Session
        _reset_session_singleton()
        return Session(username=username, base="http://localhost:8000")

    def test_save_and_load_token(self):
        sess = self._make_session()
        ok = sess._save_session_token("at_123", "rt_456", "u_789")
        self.assertTrue(ok)

        # Verify file exists
        path = sess._get_token_path()
        self.assertTrue(path.exists())

        # Load back
        _reset_session_singleton()
        sess2 = self._make_session()
        data = sess2._load_session_token()
        self.assertIsNotNone(data)
        self.assertEqual(data["access_token"], "at_123")
        self.assertEqual(data["refresh_token"], "rt_456")
        self.assertEqual(data["user_id"], "u_789")

    def test_save_updates_instance_attributes(self):
        sess = self._make_session()
        sess._save_session_token("at_x", "rt_y", "u_z")
        self.assertEqual(sess.access_token, "at_x")
        self.assertEqual(sess.refresh_token, "rt_y")
        self.assertEqual(sess.user_id, "u_z")
        # Backwards compat aliases
        self.assertEqual(sess.clerk_session_token, "at_x")
        self.assertEqual(sess.clerk_user_id, "u_z")

    def test_load_updates_instance_attributes(self):
        sess = self._make_session()
        sess._save_session_token("at_a", "rt_b", "u_c")
        _reset_session_singleton()
        sess2 = self._make_session()
        sess2._load_session_token()
        self.assertEqual(sess2.access_token, "at_a")
        self.assertEqual(sess2.clerk_session_token, "at_a")

    def test_load_nonexistent_returns_none(self):
        sess = self._make_session("no_such_user")
        data = sess._load_session_token()
        self.assertIsNone(data)

    def test_clear_session_token(self):
        sess = self._make_session()
        sess._save_session_token("at_x", "rt_y", "u_z")
        path = sess._get_token_path()
        self.assertTrue(path.exists())

        ok = sess._clear_session_token()
        self.assertTrue(ok)
        self.assertFalse(path.exists())
        self.assertIsNone(sess.access_token)
        self.assertIsNone(sess.refresh_token)
        self.assertIsNone(sess.user_id)
        self.assertIsNone(sess.clerk_session_token)
        self.assertIsNone(sess.clerk_user_id)


class TestSessionPathSafety(unittest.TestCase):
    """Token path generation should be safe with unusual usernames."""

    def setUp(self):
        _reset_session_singleton()
        self.tmpdir = tempfile.mkdtemp()
        self._env_patcher = patch.dict(os.environ, {"TB_DATA_DIR": self.tmpdir})
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()
        _reset_session_singleton()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_session(self, username):
        from toolboxv2.utils.system.session import Session
        _reset_session_singleton()
        return Session(username=username, base="http://localhost:8000")

    def test_special_chars_sanitized(self):
        sess = self._make_session("user@evil/../../../etc/passwd")
        path = sess._get_token_path()
        # Path should be inside the token dir
        self.assertTrue(str(path).startswith(str(sess._get_token_dir())))
        # Filename should not contain path separators
        self.assertNotIn("/", path.name.replace("_session.json", ""))
        self.assertNotIn("\\", path.name.replace("_session.json", ""))

    def test_empty_username_uses_default(self):
        sess = self._make_session(None)
        path = sess._get_token_path()
        self.assertIn("default", path.name)

    def test_long_username_truncated(self):
        sess = self._make_session("a" * 200)
        path = sess._get_token_path()
        # Filename part (without extension) should be <= 32 + len("_session.json")
        name_part = path.name.replace("_session.json", "")
        self.assertLessEqual(len(name_part), 32)

    def test_token_dir_created(self):
        sess = self._make_session("newuser")
        token_dir = sess._get_token_dir()
        self.assertTrue(token_dir.is_dir())


class TestSessionBaseURL(unittest.TestCase):
    """Base URL normalization."""

    def setUp(self):
        _reset_session_singleton()

    def tearDown(self):
        _reset_session_singleton()

    def test_strips_trailing_slash(self):
        from toolboxv2.utils.system.session import Session
        sess = Session(username="test", base="http://localhost:8000/")
        self.assertEqual(sess.base, "http://localhost:8000")

    def test_removes_api_suffix(self):
        _reset_session_singleton()
        from toolboxv2.utils.system.session import Session
        sess = Session(username="test", base="http://localhost:8000/api/")
        self.assertEqual(sess.base, "http://localhost:8000")


class TestSessionAuthHeaders(unittest.TestCase):
    """Tests for _get_auth_headers()."""

    def setUp(self):
        _reset_session_singleton()
        self.tmpdir = tempfile.mkdtemp()
        self._env_patcher = patch.dict(os.environ, {"TB_DATA_DIR": self.tmpdir})
        self._env_patcher.start()

    def tearDown(self):
        self._env_patcher.stop()
        _reset_session_singleton()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_headers_with_access_token(self):
        from toolboxv2.utils.system.session import Session
        sess = Session(username="test", base="http://localhost:8000")
        sess.access_token = "at_xyz"
        headers = sess._get_auth_headers()
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], "Bearer at_xyz")

    def test_headers_fallback_to_clerk_session_token(self):
        from toolboxv2.utils.system.session import Session
        _reset_session_singleton()
        sess = Session(username="test2", base="http://localhost:8000")
        sess.access_token = None
        sess.clerk_session_token = "legacy_tok"
        headers = sess._get_auth_headers()
        self.assertIn("Authorization", headers)
        self.assertEqual(headers["Authorization"], "Bearer legacy_tok")

    def test_headers_without_any_token(self):
        from toolboxv2.utils.system.session import Session
        _reset_session_singleton()
        sess = Session(username="test3", base="http://localhost:8000")
        sess.access_token = None
        sess.clerk_session_token = None
        headers = sess._get_auth_headers()
        # Should still return a dict, possibly without Authorization
        self.assertIsInstance(headers, dict)


if __name__ == "__main__":
    unittest.main()
