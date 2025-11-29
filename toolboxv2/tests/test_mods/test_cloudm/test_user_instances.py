# file: toolboxv2/tests/test_mods/test_cloudm/test_user_instances.py
"""
Tests for CloudM UserInstances module.

Tests user instance management including:
- User instance creation and retrieval
- Session ID generation (SI, VT, WebSocket, CLI)
- CLI session management
- Instance persistence and hydration
- Instance deletion
"""

import unittest
import json
import time
from unittest.mock import patch, MagicMock

from toolboxv2.mods.CloudM.UserInstances import (
    UserInstances,
    get_user_instance,
    save_user_instances,
    delete_user_instance,
    get_instance_si_id,
    register_cli_session,
    get_user_cli_sessions
)
from toolboxv2 import Result, Code


class TestUserInstances(unittest.TestCase):
    """Tests for UserInstances singleton class"""

    def setUp(self):
        """Clear user instances before each test"""
        UserInstances().live_user_instances.clear()
        UserInstances().user_instances.clear()
        UserInstances().cli_sessions.clear()

        # Mock app globally for UserInstances
        self.app_patcher = patch('toolboxv2.mods.CloudM.UserInstances.app')
        self.mock_app = self.app_patcher.start()
        self.mock_app.id = "test-app-id"

    def tearDown(self):
        """Stop patches"""
        self.app_patcher.stop()

    def test_user_instances_singleton(self):
        """Test that UserInstances is a singleton"""
        instance1 = UserInstances()
        instance2 = UserInstances()

        self.assertIs(instance1, instance2)

    def test_user_instances_has_required_attributes(self):
        """Test that UserInstances has all required attributes"""
        instance = UserInstances()

        self.assertTrue(hasattr(instance, 'live_user_instances'))
        self.assertTrue(hasattr(instance, 'user_instances'))
        self.assertTrue(hasattr(instance, 'cli_sessions'))

        self.assertIsInstance(instance.live_user_instances, dict)
        self.assertIsInstance(instance.user_instances, dict)
        self.assertIsInstance(instance.cli_sessions, dict)

    def test_get_si_id_generates_consistent_hash(self):
        """Test that get_si_id generates consistent hash for same UID"""
        uid = "test_user_123"

        si_id_1 = UserInstances.get_si_id(uid)
        si_id_2 = UserInstances.get_si_id(uid)

        # Should return Result objects
        self.assertIsInstance(si_id_1, Result)
        self.assertIsInstance(si_id_2, Result)

        # Should be consistent
        self.assertEqual(si_id_1.get(), si_id_2.get())

    def test_get_vt_id_generates_consistent_hash(self):
        """Test that get_vt_id generates consistent hash"""
        uid = "test_user_123"

        vt_id_1 = UserInstances.get_vt_id(uid)
        vt_id_2 = UserInstances.get_vt_id(uid)

        self.assertEqual(vt_id_1.get(), vt_id_2.get())

    def test_get_web_socket_id_generates_consistent_hash(self):
        """Test that get_web_socket_id generates consistent hash"""
        uid = "test_user_123"

        ws_id_1 = UserInstances.get_web_socket_id(uid)
        ws_id_2 = UserInstances.get_web_socket_id(uid)

        self.assertEqual(ws_id_1.get(), ws_id_2.get())

    def test_get_cli_session_id_generates_consistent_hash(self):
        """Test that get_cli_session_id generates consistent hash"""
        uid = "test_user_123"

        cli_id_1 = UserInstances.get_cli_session_id(uid)
        cli_id_2 = UserInstances.get_cli_session_id(uid)

        self.assertEqual(cli_id_1.get(), cli_id_2.get())

    def test_different_id_types_generate_different_hashes(self):
        """Test that different ID types generate different hashes for same UID"""
        uid = "test_user_123"

        si_id = UserInstances.get_si_id(uid).get()
        vt_id = UserInstances.get_vt_id(uid).get()
        ws_id = UserInstances.get_web_socket_id(uid).get()
        cli_id = UserInstances.get_cli_session_id(uid).get()

        # All should be different
        ids = [si_id, vt_id, ws_id, cli_id]
        self.assertEqual(len(ids), len(set(ids)))


class TestGetUserInstance(unittest.TestCase):
    """Tests for get_user_instance function"""

    def setUp(self):
        """Clear instances before each test"""
        UserInstances().live_user_instances.clear()
        UserInstances().user_instances.clear()

        # Mock app
        self.app_patcher = patch('toolboxv2.mods.CloudM.UserInstances.app')
        self.mock_app = self.app_patcher.start()
        self.mock_app.id = "test-app-id"

    def tearDown(self):
        """Stop patches"""
        self.app_patcher.stop()

    def test_get_user_instance_creates_new_instance(self):
        """Test creating a new user instance"""
        uid = "new_user_123"

        # Mock DB to return no existing data
        self.mock_app.run_any.return_value = Result.default_internal_error("No data")

        instance = get_user_instance(uid, hydrate=False)

        self.assertIsNotNone(instance)
        self.assertEqual(instance['save']['uid'], uid)
        self.assertIn('SiID', instance)
        self.assertIn('VtID', instance)
        self.assertIn('webSocketID', instance)

    def test_get_user_instance_returns_from_live_cache(self):
        """Test retrieving instance from live cache"""
        uid = "cached_user"
        si_id = UserInstances.get_si_id(uid).get()

        # Add to live cache
        cached_instance = {
            'save': {'uid': uid, 'mods': ['mod1']},
            'live': {'some': 'data'},
            'SiID': si_id,
            'VtID': 'vt_123',
            'webSocketID': 'ws_123'
        }
        UserInstances().live_user_instances[si_id] = cached_instance

        instance = get_user_instance(uid)

        # Should return cached instance without DB call
        self.assertEqual(instance, cached_instance)
        self.mock_app.run_any.assert_not_called()

    def test_get_user_instance_with_none_uid(self):
        """Test that None UID returns None"""
        instance = get_user_instance(None)
        self.assertIsNone(instance)


class TestSaveUserInstances(unittest.TestCase):
    """Tests for save_user_instances function"""

    def setUp(self):
        """Clear instances"""
        UserInstances().live_user_instances.clear()
        UserInstances().user_instances.clear()

    @patch('toolboxv2.mods.CloudM.UserInstances.app')
    def test_save_user_instances_stores_instance(self, mock_app):
        """Test saving a user instance"""
        instance = {
            'SiID': 'si_123',
            'webSocketID': 'ws_123',
            'save': {'uid': 'user_123'}
        }

        save_user_instances(instance)

        # Check it's stored in both dicts
        self.assertIn('si_123', UserInstances().user_instances)
        self.assertIn('si_123', UserInstances().live_user_instances)
        self.assertEqual(UserInstances().user_instances['si_123'], 'ws_123')
        self.assertEqual(UserInstances().live_user_instances['si_123'], instance)

    def test_save_user_instances_with_none(self):
        """Test that None instance is handled gracefully"""
        result = save_user_instances(None)
        self.assertIsNone(result)


class TestDeleteUserInstance(unittest.TestCase):
    """Tests for delete_user_instance function"""

    def setUp(self):
        """Clear instances"""
        UserInstances().live_user_instances.clear()
        UserInstances().user_instances.clear()

        # Mock app
        self.app_patcher = patch('toolboxv2.mods.CloudM.UserInstances.app')
        self.mock_app = self.app_patcher.start()
        self.mock_app.id = "test-app-id"

    def tearDown(self):
        """Stop patches"""
        self.app_patcher.stop()

    def test_delete_user_instance_removes_from_cache(self):
        """Test deleting a user instance"""
        uid = "user_to_delete"
        si_id = UserInstances.get_si_id(uid).get()

        # Add instance to caches
        UserInstances().user_instances[si_id] = "ws_123"
        UserInstances().live_user_instances[si_id] = {'data': 'test'}

        result = delete_user_instance(uid)

        # Should be removed from both caches
        self.assertNotIn(si_id, UserInstances().user_instances)
        self.assertNotIn(si_id, UserInstances().live_user_instances)
        self.assertEqual(result, "Instance deleted successfully")

    def test_delete_user_instance_not_found(self):
        """Test deleting non-existent instance"""
        uid = "nonexistent_user"

        result = delete_user_instance(uid)

        self.assertEqual(result, "User instance not found")

    def test_delete_user_instance_with_none(self):
        """Test that None UID is handled"""
        result = delete_user_instance(None)
        self.assertEqual(result, "UID required")


class TestGetInstanceSiId(unittest.TestCase):
    """Tests for get_instance_si_id function"""

    def setUp(self):
        """Clear instances"""
        UserInstances().live_user_instances.clear()

    def test_get_instance_si_id_found(self):
        """Test retrieving instance by SI ID"""
        si_id = "si_789"
        instance_data = {'uid': 'user_789', 'data': 'test'}

        UserInstances().live_user_instances[si_id] = instance_data

        result = get_instance_si_id(si_id)

        self.assertEqual(result, instance_data)

    def test_get_instance_si_id_not_found(self):
        """Test retrieving non-existent instance"""
        result = get_instance_si_id("nonexistent_si_id")

        self.assertFalse(result)


class TestCLISessionManagement(unittest.TestCase):
    """Tests for CLI session management functions"""

    def setUp(self):
        """Clear CLI sessions"""
        UserInstances().cli_sessions.clear()

        # Mock app
        self.app_patcher = patch('toolboxv2.mods.CloudM.UserInstances.app')
        self.mock_app = self.app_patcher.start()
        self.mock_app.id = "test-app-id"

    def tearDown(self):
        """Stop patches"""
        self.app_patcher.stop()

    def test_register_cli_session(self):
        """Test registering a new CLI session"""
        uid = "cli_user_123"
        session_token = "test_jwt_token"
        session_info = {'ip': '127.0.0.1', 'device': 'test'}
        clerk_user_id = None

        result = register_cli_session(uid, session_token, session_info, clerk_user_id)

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_ok())

        # Check session is stored
        cli_session_id = UserInstances.get_cli_session_id(uid).get()
        self.assertIn(cli_session_id, UserInstances().cli_sessions)

        session = UserInstances().cli_sessions[cli_session_id]

        session_data = {
            'uid': uid,
            'cli_session_id': cli_session_id,
            'session_token': session_token,
            'clerk_user_id': clerk_user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'status': 'active',
            'session_info': session_info or {}
        }
        self.assertEqual(session['uid'], uid)
        self.assertEqual(session['cli_session_id'], cli_session_id)
        self.assertEqual(session['session_token'], session_token)
        self.assertEqual(session['clerk_user_id'], clerk_user_id)
        self.assertAlmostEqual(session['created_at'], session['last_activity'], delta=4)
        self.assertEqual(session['status'], 'active')
        self.assertEqual(session['session_info'], session_info)

    def test_register_cli_session_with_none_uid(self):
        """Test that None UID returns error"""
        result = register_cli_session(None, "token")

        self.assertIsInstance(result, Result)
        self.assertTrue(result.is_error())

    def test_get_user_cli_sessions(self):
        """Test retrieving all CLI sessions for a user"""
        uid = "user_with_sessions"
        cli_session_id = UserInstances.get_cli_session_id(uid).get()

        # Add session
        session_data = {
            'uid': uid,
            'cli_session_id': cli_session_id,
            'jwt_token': 'token',
            'status': 'active',
            'created_at': time.time()
        }
        UserInstances().cli_sessions[cli_session_id] = session_data

        sessions = get_user_cli_sessions(uid)

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]['uid'], uid)

    def test_get_user_cli_sessions_empty(self):
        """Test retrieving CLI sessions when none exist"""
        uid = "user_no_sessions"

        sessions = get_user_cli_sessions(uid)

        self.assertEqual(len(sessions), 0)


class TestUserInstancesIntegration(unittest.TestCase):
    """Integration tests for user instance management"""

    def setUp(self):
        """Clear all instances"""
        UserInstances().live_user_instances.clear()
        UserInstances().user_instances.clear()
        UserInstances().cli_sessions.clear()

        # Mock app
        self.app_patcher = patch('toolboxv2.mods.CloudM.UserInstances.app')
        self.mock_app = self.app_patcher.start()
        self.mock_app.id = "test-app-id"

    def tearDown(self):
        """Stop patches"""
        self.app_patcher.stop()

    def test_full_user_lifecycle(self):
        """Test complete user instance lifecycle"""
        uid = "lifecycle_user"

        # Mock DB responses
        self.mock_app.run_any.return_value = Result.default_internal_error("No data")

        # 1. Create instance
        instance = get_user_instance(uid, hydrate=False)
        self.assertIsNotNone(instance)

        # 2. Save instance
        save_user_instances(instance)
        si_id = instance['SiID']
        self.assertIn(si_id, UserInstances().user_instances)

        # 3. Register CLI session
        result = register_cli_session(uid, "test_token")
        self.assertTrue(result.is_ok())

        # 4. Retrieve instance
        retrieved = get_instance_si_id(si_id)
        self.assertEqual(retrieved, instance)

        # 5. Delete instance
        delete_result = delete_user_instance(uid)
        self.assertEqual(delete_result, "Instance deleted successfully")
        self.assertNotIn(si_id, UserInstances().user_instances)


if __name__ == '__main__':
    unittest.main(verbosity=2)

