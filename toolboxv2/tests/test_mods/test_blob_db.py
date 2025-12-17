"""
Tests for BlobDB with automatic offline fallback when MinIO is unavailable.
Tests the refactored BlobDB that uses Environment-based configuration
and SQLite fallback for offline/mobile modes.
"""

import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import json
import time


class TestBlobDBOfflineFallback(unittest.TestCase):
    """
    Tests for BlobDB automatic offline fallback when MinIO is not available.
    """

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_env = os.environ.copy()

        # Force offline mode for testing
        os.environ["IS_OFFLINE_DB"] = "true"
        os.environ["SERVER_ID"] = "test_server"
        os.environ["DB_CACHE_TTL"] = "60"

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_initialize_offline_mode(self):
        """Test BlobDB initializes in offline mode when IS_OFFLINE_DB=true."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        result = db.initialize(db_path="test_db")

        self.assertFalse(result.is_error())
        self.assertIsNotNone(db._sqlite)
        self.assertIsNone(db._local_minio)
        self.assertIsNone(db._cloud_minio)

        db.exit()

    def test_initialize_fallback_when_minio_unavailable(self):
        """Test BlobDB falls back to SQLite when MinIO credentials are missing."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        # Clear MinIO credentials
        os.environ["IS_OFFLINE_DB"] = "false"
        os.environ["MINIO_SECRET_KEY"] = ""

        Config.reload()
        db = BlobDB()
        result = db.initialize(db_path="test_db")

        self.assertFalse(result.is_error())
        # Should fall back to SQLite
        self.assertIsNotNone(db._sqlite)

        db.exit()

    def test_set_and_get_offline(self):
        """Test set and get operations in offline mode."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        key = "users/john/profile"
        value = {"name": "John Doe", "age": 30}

        # Set value
        result = db.set(key, value)
        self.assertFalse(result.is_error())

        # Get value
        result = db.get(key)
        self.assertFalse(result.is_error())
        self.assertEqual(result.get(), value)

        db.exit()

    def test_key_to_path_conversion(self):
        """Test key to path conversion."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        # Test path conversion
        key = "users/alice/profile"
        expected_path = "test_db/users/alice/profile.json"
        actual_path = db._key_to_path(key)

        self.assertEqual(actual_path, expected_path)

        db.exit()

    def test_multiple_keys_different_paths(self):
        """Test storing multiple keys creates proper paths."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        keys_values = [
            ("users/alice/profile", {"name": "Alice"}),
            ("users/bob/profile", {"name": "Bob"}),
            ("admin/settings", {"theme": "dark"}),
        ]

        for key, value in keys_values:
            result = db.set(key, value)
            self.assertFalse(result.is_error())

        # Verify all keys exist in manifest
        for key, _ in keys_values:
            self.assertTrue(db.if_exist(key))

        db.exit()

    def test_delete_key_offline(self):
        """Test deleting a key in offline mode."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        key = "users/alice/profile"
        db.set(key, {"name": "Alice"})

        # Verify exists
        self.assertTrue(db.if_exist(key))

        # Delete
        result = db.delete(key)
        self.assertFalse(result.is_error())

        # Verify deleted
        self.assertFalse(db.if_exist(key))

        db.exit()

    def test_append_on_set_offline(self):
        """Test append_on_set creates and appends to lists."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        key = "users/alice/tags"

        # First append creates list
        result = db.append_on_set(key, ["tag1", "tag2"])
        self.assertFalse(result.is_error())

        # Second append adds to list (no duplicates)
        result = db.append_on_set(key, ["tag3", "tag1"])
        self.assertFalse(result.is_error())

        # Verify list
        result = db.get(key)
        self.assertFalse(result.is_error())
        tags = result.get()
        self.assertEqual(len(tags), 3)
        self.assertIn("tag1", tags)
        self.assertIn("tag2", tags)
        self.assertIn("tag3", tags)

        db.exit()

    def test_pattern_matching_get(self):
        """Test wildcard pattern matching for get."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        # Add test data
        db.set("users/alice/profile", {"name": "Alice"})
        db.set("users/bob/profile", {"name": "Bob"})
        db.set("admin/settings", {"theme": "dark"})

        # Pattern match
        result = db.get("users/*")
        self.assertFalse(result.is_error())

        users = result.get()
        self.assertEqual(len(users), 2)

        db.exit()

    def test_get_all_keys(self):
        """Test getting all keys."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        db.set("key1", "value1")
        db.set("key2", "value2")

        result = db.get("all-k")
        self.assertFalse(result.is_error())

        keys = result.get()
        self.assertIn("key1", keys)
        self.assertIn("key2", keys)

        db.exit()

    def test_cache_ttl(self):
        """Test cache TTL behavior."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        os.environ["DB_CACHE_TTL"] = "1"  # 1 second TTL
        Config.reload()

        db = BlobDB()
        db.initialize(db_path="test_db")

        key = "test/cache"
        value = {"data": "test"}

        db.set(key, value)

        # Should be in cache
        found, cached = db._cache_get(key)
        self.assertTrue(found)
        self.assertEqual(cached, value)

        # Wait for TTL
        time.sleep(1.5)

        # Should be expired
        found, cached = db._cache_get(key)
        self.assertFalse(found)

        db.exit()

    def test_stats(self):
        """Test get_stats returns correct info."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        db.delete('*')

        db.set("key1", "value1")
        db.set("key2", "value2")

        stats = db.get_stats()

        self.assertTrue(stats["initialized"])
        self.assertEqual(stats["server_id"], "test_db")
        self.assertEqual(stats["keys_count"], 2)
        self.assertTrue(stats["has_sqlite"])
        self.assertFalse(stats["has_local_minio"])
        self.assertTrue(stats["is_offline"])

        db.exit()

    def test_clear_cache(self):
        """Test cache clearing."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        db.set("key1", "value1")
        db.set("key2", "value2")

        # Cache should have entries
        self.assertGreater(len(db._cache), 0)

        db.clear_cache()

        # Cache should be empty
        self.assertEqual(len(db._cache), 0)

        db.exit()

    def test_reload_manifest(self):
        """Test manifest reloading."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        db.set("key1", "value1")

        # Clear manifest
        db._manifest = set()
        db._manifest_loaded = False

        # Reload
        db.reload_manifest()

        # Should have key1 back
        self.assertIn("key1", db._manifest)

        db.exit()

    def test_exit_closes_connections(self):
        """Test exit properly closes all connections."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        self.assertTrue(db._initialized)
        self.assertIsNotNone(db._sqlite)

        result = db.exit()
        self.assertFalse(result.is_error())

        self.assertFalse(db._initialized)
        self.assertIsNone(db._sqlite)

    def test_delete_with_pattern(self):
        """Test delete with pattern matching."""
        from toolboxv2.mods.DB.blob_instance import BlobDB, Config

        Config.reload()
        db = BlobDB()
        db.initialize(db_path="test_db")

        db.set("users/alice", {"name": "Alice"})
        db.set("users/bob", {"name": "Bob"})
        db.set("admin/settings", {"theme": "dark"})

        # Delete all users
        result = db.delete("users/*", matching=True)
        self.assertFalse(result.is_error())

        # Users should be gone
        self.assertFalse(db.if_exist("users/alice"))
        self.assertFalse(db.if_exist("users/bob"))

        # Admin should still exist
        self.assertTrue(db.if_exist("admin/settings"))

        db.exit()

class TestConfigReload(unittest.TestCase):
    """Tests for Config class environment variable handling."""

    def setUp(self):
        self.original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_defaults(self):
        """Test Config uses defaults when env vars not set."""
        from toolboxv2.mods.DB.blob_instance import Config

        # Clear relevant env vars
        for key in ["MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "IS_OFFLINE_DB"]:
            os.environ.pop(key, None)

        Config.reload()

        self.assertEqual(Config.MINIO_ENDPOINT, "127.0.0.1:9000")
        self.assertEqual(Config.MINIO_ACCESS_KEY, "admin")
        self.assertFalse(Config.IS_OFFLINE_DB)

    def test_config_from_env(self):
        """Test Config reads from environment variables."""
        from toolboxv2.mods.DB.blob_instance import Config

        os.environ["MINIO_ENDPOINT"] = "minio.example.com:9000"
        os.environ["MINIO_ACCESS_KEY"] = "mykey"
        os.environ["MINIO_SECRET_KEY"] = "mysecret"
        os.environ["IS_OFFLINE_DB"] = "true"
        os.environ["DB_CACHE_TTL"] = "120"

        Config.reload()

        self.assertEqual(Config.MINIO_ENDPOINT, "minio.example.com:9000")
        self.assertEqual(Config.MINIO_ACCESS_KEY, "mykey")
        self.assertEqual(Config.MINIO_SECRET_KEY, "mysecret")
        self.assertTrue(Config.IS_OFFLINE_DB)
        self.assertEqual(Config.DB_CACHE_TTL, 120)

    def test_has_local_minio(self):
        """Test has_local_minio check."""
        from toolboxv2.mods.DB.blob_instance import Config

        os.environ["MINIO_ENDPOINT"] = "localhost:9000"
        os.environ["MINIO_ACCESS_KEY"] = "admin"
        os.environ["MINIO_SECRET_KEY"] = ""

        Config.reload()
        self.assertFalse(Config.has_local_minio())

        os.environ["MINIO_SECRET_KEY"] = "secret"
        Config.reload()
        self.assertTrue(Config.has_local_minio())

    def test_has_cloud_minio(self):
        """Test has_cloud_minio check."""
        from toolboxv2.mods.DB.blob_instance import Config

        os.environ["CLOUD_ENDPOINT"] = ""
        Config.reload()
        self.assertFalse(Config.has_cloud_minio())

        os.environ["CLOUD_ENDPOINT"] = "cloud.example.com"
        os.environ["CLOUD_ACCESS_KEY"] = "key"
        os.environ["CLOUD_SECRET_KEY"] = "secret"
        Config.reload()
        self.assertTrue(Config.has_cloud_minio())


class TestSQLiteCache(unittest.TestCase):
    """Tests for SQLiteCache offline storage."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sqlite_put_get(self):
        """Test SQLite put and get operations."""
        from toolboxv2.mods.DB.blob_instance import SQLiteCache

        cache = SQLiteCache(self.db_path)

        path = "test/path.json"
        data = b'{"key": "value"}'

        self.assertTrue(cache.put(path, data))

        result = cache.get(path)
        self.assertEqual(result, data)

        cache.close()

    def test_sqlite_exists(self):
        """Test SQLite exists check."""
        from toolboxv2.mods.DB.blob_instance import SQLiteCache

        cache = SQLiteCache(self.db_path)

        path = "test/path.json"
        self.assertFalse(cache.exists(path))

        cache.put(path, b"data")
        self.assertTrue(cache.exists(path))

        cache.close()

    def test_sqlite_delete(self):
        """Test SQLite delete operation."""
        from toolboxv2.mods.DB.blob_instance import SQLiteCache

        cache = SQLiteCache(self.db_path)

        path = "test/path.json"
        cache.put(path, b"data")

        self.assertTrue(cache.exists(path))
        cache.delete(path)
        self.assertFalse(cache.exists(path))

        cache.close()

    def test_sqlite_list(self):
        """Test SQLite list with prefix."""
        from toolboxv2.mods.DB.blob_instance import SQLiteCache

        cache = SQLiteCache(self.db_path)

        cache.put("users/alice.json", b"data1")
        cache.put("users/bob.json", b"data2")
        cache.put("admin/settings.json", b"data3")

        users = cache.list("users/")
        self.assertEqual(len(users), 2)
        self.assertIn("users/alice.json", users)
        self.assertIn("users/bob.json", users)

        cache.close()

    def test_sqlite_manifest(self):
        """Test SQLite manifest operations."""
        from toolboxv2.mods.DB.blob_instance import SQLiteCache

        cache = SQLiteCache(self.db_path)

        cache.add_to_manifest("key1")
        cache.add_to_manifest("key2")

        manifest = cache.get_manifest()
        self.assertIn("key1", manifest)
        self.assertIn("key2", manifest)

        cache.remove_from_manifest("key1")
        manifest = cache.get_manifest()
        self.assertNotIn("key1", manifest)
        self.assertIn("key2", manifest)

        cache.close()

    def test_sqlite_dirty_tracking(self):
        """Test SQLite dirty/sync status tracking."""
        from toolboxv2.mods.DB.blob_instance import SQLiteCache

        cache = SQLiteCache(self.db_path)

        cache.put("path1", b"data1")
        cache.put("path2", b"data2")

        # Both should be dirty
        dirty = cache.get_dirty()
        self.assertEqual(len(dirty), 2)

        # Mark one as synced
        cache.mark_synced("path1")

        dirty = cache.get_dirty()
        self.assertEqual(len(dirty), 1)
        self.assertIn("path2", dirty)

        cache.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
