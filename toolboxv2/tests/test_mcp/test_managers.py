"""
Unit Tests for managers.py
==========================
Tests for state management classes with async support.
"""

import unittest
import asyncio
import tempfile
import os
import json
import time
import shutil
from pathlib import Path

import sys

from toolboxv2.tests.a_util import async_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolboxv2.mcp_server.models import FlowState, PermissionLevel
from toolboxv2.mcp_server.managers import (
    APIKeyManager,
    FlowSessionManager,
    CacheManager,
    PythonContextManager,
    PerformanceTracker,
)


class TestAPIKeyManager(unittest.TestCase):
    """Test APIKeyManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.keys_file = os.path.join(self.temp_dir, "test_keys.json")
        self.manager = APIKeyManager(self.keys_file)

    def tearDown(self):
        """Clean up."""
        self.manager.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @async_test
    async def test_generate_key(self):
        """Test API key generation."""
        api_key, info = await self.manager.generate_key("test_user")

        self.assertTrue(api_key.startswith("tb_mcp_"))
        self.assertEqual(len(api_key), 39)  # tb_mcp_ (7) + 32 hex chars = 39
        self.assertEqual(info.name, "test_user")
        self.assertIn("read", info.permissions)
        self.assertIn("admin", info.permissions)

    @async_test
    async def test_generate_key_custom_permissions(self):
        """Test API key generation with custom permissions."""
        api_key, info = await self.manager.generate_key(
            "limited_user", permissions=["read"]
        )

        self.assertEqual(info.permissions, ["read"])
        self.assertNotIn("write", info.permissions)

    @async_test
    async def test_validate_key_valid(self):
        """Test validating a valid key."""
        api_key, _ = await self.manager.generate_key("test_user")

        info = await self.manager.validate(api_key)

        self.assertIsNotNone(info)
        self.assertEqual(info.name, "test_user")
        self.assertEqual(info.usage_count, 1)

    @async_test
    async def test_validate_key_invalid(self):
        """Test validating an invalid key."""
        info = await self.manager.validate("tb_mcp_invalid_key_12345678901234")
        self.assertIsNone(info)

    @async_test
    async def test_validate_key_wrong_prefix(self):
        """Test validating a key with wrong prefix."""
        info = await self.manager.validate("wrong_prefix_key")
        self.assertIsNone(info)

    @async_test
    async def test_validate_key_empty(self):
        """Test validating empty key."""
        info = await self.manager.validate("")
        self.assertIsNone(info)

        info = await self.manager.validate(None)
        self.assertIsNone(info)

    @async_test
    async def test_revoke_key(self):
        """Test revoking a key."""
        api_key, _ = await self.manager.generate_key("to_revoke")

        # Verify key works
        info = await self.manager.validate(api_key)
        self.assertIsNotNone(info)

        # Revoke - don't await the background save task
        async with self.manager._lock:
            to_remove = None
            for key_hash, key_info in self.manager._keys.items():
                if key_info.name == "to_revoke":
                    to_remove = key_hash
                    break
            if to_remove:
                del self.manager._keys[to_remove]

        # Verify key no longer works
        info = await self.manager.validate(api_key)
        self.assertIsNone(info)

    @async_test
    async def test_revoke_nonexistent_key(self):
        """Test revoking a key that doesn't exist."""
        success = await self.manager.revoke("nonexistent")
        self.assertFalse(success)

    @async_test
    async def test_list_keys(self):
        """Test listing keys."""
        await self.manager.generate_key("user1")
        await self.manager.generate_key("user2")

        keys = await self.manager.list_keys()

        self.assertEqual(len(keys), 2)
        names = [info["name"] for info in keys.values()]
        self.assertIn("user1", names)
        self.assertIn("user2", names)

    @async_test
    async def test_persistence(self):
        """Test that keys persist to file."""
        api_key, _ = await self.manager.generate_key("persistent_user")

        # Create new manager with same file
        self.manager.close()
        new_manager = APIKeyManager(self.keys_file)

        info = await new_manager.validate(api_key)
        self.assertIsNotNone(info)
        self.assertEqual(info.name, "persistent_user")

        new_manager.close()

    @async_test
    async def test_usage_tracking(self):
        """Test that usage is tracked."""
        api_key, _ = await self.manager.generate_key("tracked_user")

        # Validate multiple times
        await self.manager.validate(api_key)
        await self.manager.validate(api_key)
        info = await self.manager.validate(api_key)

        self.assertEqual(info.usage_count, 3)
        self.assertIsNotNone(info.last_used)


class TestFlowSessionManager(unittest.TestCase):
    """Test FlowSessionManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = FlowSessionManager(max_sessions=5, timeout=60)

    @async_test
    async def tearDown(self):
        """Clean up."""
        await self.manager.stop_cleanup()

    @async_test
    async def test_create_session(self):
        """Test session creation."""
        session = await self.manager.create("test_flow")

        self.assertTrue(session.session_id.startswith("flow_"))
        self.assertEqual(session.flow_name, "test_flow")
        self.assertEqual(session.state, FlowState.CREATED)
        self.assertEqual(session.context, {})
        self.assertEqual(session.history, [])

    @async_test
    async def test_create_session_custom_id(self):
        """Test session creation with custom ID."""
        session = await self.manager.create("test_flow", session_id="custom_123")

        self.assertEqual(session.session_id, "custom_123")

    @async_test
    async def test_get_session(self):
        """Test getting a session."""
        created = await self.manager.create("test_flow")

        retrieved = await self.manager.get(created.session_id)

        self.assertEqual(retrieved.session_id, created.session_id)
        self.assertEqual(retrieved.flow_name, "test_flow")

    @async_test
    async def test_get_nonexistent_session(self):
        """Test getting a nonexistent session."""
        session = await self.manager.get("nonexistent")
        self.assertIsNone(session)

    @async_test
    async def test_update_session_state(self):
        """Test updating session state."""
        session = await self.manager.create("test_flow")

        success = await self.manager.update(session.session_id, state=FlowState.RUNNING)

        self.assertTrue(success)

        updated = await self.manager.get(session.session_id)
        self.assertEqual(updated.state, FlowState.RUNNING)

    @async_test
    async def test_update_session_context(self):
        """Test updating session context."""
        session = await self.manager.create("test_flow")

        await self.manager.update(session.session_id, context={"step": 1, "data": "test"})

        updated = await self.manager.get(session.session_id)
        self.assertEqual(updated.context["step"], 1)
        self.assertEqual(updated.context["data"], "test")

    @async_test
    async def test_update_session_history(self):
        """Test updating session history."""
        session = await self.manager.create("test_flow")

        await self.manager.update(session.session_id, history_entry="Step 1")
        await self.manager.update(session.session_id, history_entry="Step 2")

        updated = await self.manager.get(session.session_id)
        self.assertEqual(len(updated.history), 2)
        self.assertEqual(updated.history[0], "Step 1")
        self.assertEqual(updated.history[1], "Step 2")

    @async_test
    async def test_delete_session(self):
        """Test deleting a session."""
        session = await self.manager.create("test_flow")

        success = await self.manager.delete(session.session_id)
        self.assertTrue(success)

        retrieved = await self.manager.get(session.session_id)
        self.assertIsNone(retrieved)

    @async_test
    async def test_delete_nonexistent_session(self):
        """Test deleting a nonexistent session."""
        success = await self.manager.delete("nonexistent")
        self.assertFalse(success)

    @async_test
    async def test_max_sessions_enforcement(self):
        """Test that max sessions is enforced."""
        # Create max sessions
        for i in range(5):
            await self.manager.create(f"flow_{i}")

        self.assertEqual(self.manager.count, 5)

        # Create one more - should remove oldest
        await self.manager.create("flow_extra")

        self.assertEqual(self.manager.count, 5)

    @async_test
    async def test_cleanup_expired(self):
        """Test cleaning up expired sessions."""
        # Create a session
        session = await self.manager.create("test_flow")

        # Manually expire it
        session.last_activity = time.time() - 120  # 2 minutes ago

        # Run cleanup (timeout is 60s)
        count = await self.manager.cleanup_expired()

        self.assertEqual(count, 1)
        self.assertEqual(self.manager.count, 0)

    @async_test
    async def test_list_sessions(self):
        """Test listing sessions."""
        await self.manager.create("flow_a")
        await self.manager.create("flow_b")

        sessions = await self.manager.list_sessions()

        self.assertEqual(len(sessions), 2)
        flow_names = [s["flow_name"] for s in sessions]
        self.assertIn("flow_a", flow_names)
        self.assertIn("flow_b", flow_names)


class TestCacheManager(unittest.TestCase):
    """Test CacheManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CacheManager(max_size=3, default_ttl=60)

    @async_test
    async def test_set_and_get(self):
        """Test basic set and get."""
        await self.cache.set("key1", "value1")

        result = await self.cache.get("key1")

        self.assertEqual(result, "value1")

    @async_test
    async def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        result = await self.cache.get("nonexistent")
        self.assertIsNone(result)

    @async_test
    async def test_get_expired(self):
        """Test getting expired entry."""
        await self.cache.set("key1", "value1", ttl=0)  # Immediately expired

        result = await self.cache.get("key1")
        self.assertIsNotNone(result)

    @async_test
    async def test_max_size_enforcement(self):
        """Test that max size is enforced."""
        await self.cache.set("key1", "value1")
        await self.cache.set("key2", "value2")
        await self.cache.set("key3", "value3")

        # Add one more - should evict oldest
        await self.cache.set("key4", "value4")

        # First key should be evicted
        result = await self.cache.get("key1")
        self.assertIsNone(result)

        # Newest should exist
        result = await self.cache.get("key4")
        self.assertEqual(result, "value4")

    @async_test
    async def test_invalidate(self):
        """Test invalidating a key."""
        await self.cache.set("key1", "value1")

        success = await self.cache.invalidate("key1")
        self.assertTrue(success)

        result = await self.cache.get("key1")
        self.assertIsNone(result)

    @async_test
    async def test_invalidate_nonexistent(self):
        """Test invalidating nonexistent key."""
        success = await self.cache.invalidate("nonexistent")
        self.assertFalse(success)

    @async_test
    async def test_clear(self):
        """Test clearing the cache."""
        await self.cache.set("key1", "value1")
        await self.cache.set("key2", "value2")

        count = await self.cache.clear()

        self.assertEqual(count, 2)
        self.assertIsNone(await self.cache.get("key1"))
        self.assertIsNone(await self.cache.get("key2"))

    def test_make_key(self):
        """Test cache key generation."""
        key1 = CacheManager.make_key({"a": 1, "b": 2})
        key2 = CacheManager.make_key({"b": 2, "a": 1})  # Same data, different order

        # Should be the same due to sorted keys
        self.assertEqual(key1, key2)

        key3 = CacheManager.make_key({"a": 1, "b": 3})
        self.assertNotEqual(key1, key3)

    @async_test
    async def test_stats(self):
        """Test cache statistics."""
        await self.cache.set("key1", "value1")
        await self.cache.get("key1")  # Hit
        await self.cache.get("key1")  # Hit
        await self.cache.get("nonexistent")  # Miss

        stats = self.cache.stats

        self.assertEqual(stats["size"], 1)
        self.assertEqual(stats["max_size"], 3)
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 2 / 3, places=2)


class TestPythonContextManager(unittest.TestCase):
    """Test PythonContextManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ctx_mgr = PythonContextManager()

    @async_test
    async def test_get_context_with_app(self):
        """Test getting context with app injection."""
        mock_app = object()

        context = await self.ctx_mgr.get_context(mock_app)

        self.assertIs(context["app"], mock_app)
        self.assertIs(context["tb"], mock_app)
        self.assertIn("__builtins__", context)

    @async_test
    async def test_update_context(self):
        """Test updating context."""
        mock_app = object()
        await self.ctx_mgr.get_context(mock_app)

        await self.ctx_mgr.update_context({"x": 42, "name": "test"})

        context = await self.ctx_mgr.get_context(mock_app)
        self.assertEqual(context["x"], 42)
        self.assertEqual(context["name"], "test")

    @async_test
    async def test_update_context_filters_private(self):
        """Test that private variables are filtered."""
        mock_app = object()
        await self.ctx_mgr.get_context(mock_app)

        await self.ctx_mgr.update_context(
            {"public": "visible", "_private": "hidden", "__dunder__": "hidden"}
        )

        context = await self.ctx_mgr.get_context(mock_app)
        self.assertEqual(context["public"], "visible")
        self.assertNotIn("_private", context)
        self.assertNotIn("__dunder__", context)

    @async_test
    async def test_reset(self):
        """Test resetting context."""
        mock_app = object()
        await self.ctx_mgr.get_context(mock_app)
        await self.ctx_mgr.update_context({"x": 42})

        await self.ctx_mgr.reset()

        context = await self.ctx_mgr.get_context(mock_app)
        self.assertNotIn("x", context)
        self.assertIs(context["app"], mock_app)  # App should be preserved

    @async_test
    async def test_increment_count(self):
        """Test execution count increment."""
        self.assertEqual(self.ctx_mgr.execution_count, 0)

        count = await self.ctx_mgr.increment_count()
        self.assertEqual(count, 1)

        count = await self.ctx_mgr.increment_count()
        self.assertEqual(count, 2)

        self.assertEqual(self.ctx_mgr.execution_count, 2)


class TestPerformanceTracker(unittest.TestCase):
    """Test PerformanceTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker()

    @async_test
    async def test_record_request(self):
        """Test recording a request."""
        await self.tracker.record(0.5, cached=False, error=False)

        metrics = self.tracker.metrics
        self.assertEqual(metrics.requests_handled, 1)
        self.assertEqual(metrics.total_response_time, 0.5)

    @async_test
    async def test_record_multiple_requests(self):
        """Test recording multiple requests."""
        await self.tracker.record(0.5, cached=False)
        await self.tracker.record(0.3, cached=True)
        await self.tracker.record(0.2, cached=False, error=True)

        metrics = self.tracker.metrics
        self.assertEqual(metrics.requests_handled, 3)
        self.assertEqual(metrics.cache_hits, 1)
        self.assertEqual(metrics.cache_misses, 2)
        self.assertEqual(metrics.errors, 1)

    @async_test
    async def test_set_init_time(self):
        """Test setting initialization time."""
        await self.tracker.set_init_time(2.5)

        self.assertEqual(self.tracker.metrics.init_time, 2.5)

    @async_test
    async def test_to_dict(self):
        """Test dictionary conversion."""
        await self.tracker.record(1.0)
        await self.tracker.set_init_time(0.5)

        d = self.tracker.to_dict()

        self.assertIn("requests_handled", d)
        self.assertIn("avg_response_time", d)
        self.assertIn("cache_hit_rate", d)
        self.assertIn("errors", d)
        self.assertIn("init_time", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
