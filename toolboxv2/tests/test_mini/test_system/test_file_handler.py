import os
import tempfile
import unittest
from unittest.mock import patch

from toolboxv2 import setup_logging
from toolboxv2.utils.system.file_handler import (
    FileHandlerV2,
    FileHandler,  # Legacy alias
    StorageScope,
    StorageBackend,
    UserContext,
    set_current_context,
    get_current_context,
    create_config_handler,
    create_data_handler,
    LocalStorageBackend
)


class TestFileHandler(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        setup_logging(10)
        self.test_dir = tempfile.mkdtemp()

        # Patch the file prefix to use the temp directory
        self.patcher = patch.object(FileHandler, 'file_handler_file_prefix',
                                    f"{self.test_dir}/.test/")
        #self.mock_prefix = self.patcher.start()

    def tearDown(self):
        # Stop the patcher
        self.patcher.stop()

        # Clean up temporary directory
        for root, dirs, files in os.walk(self.test_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_dir)

    def test_initialization(self):
        # Test valid filename extensions
        FileHandler('test.config')
        FileHandler('test.data')

        # Test invalid filename extension
        with self.assertRaises(ValueError):
            FileHandler('test.txt')

    def test_add_and_get_file_handler(self):
        # Create FileHandler instance
        file_handler = FileHandler('test.config')

        # Add items to save
        self.assertTrue(file_handler.add_to_save_file_handler('1234567890', 'test_value'))
        self.assertTrue(file_handler.add_to_save_file_handler('0987654321', 'another_value'))

        # Test invalid key length
        # self.assertFalse(file_handler.add_to_save_file_handler('short', 'value')) all length ar allowed

        # Save and reload
        file_handler.save_file_handler()
        reloaded_handler = FileHandler('test.config')
        reloaded_handler.load_file_handler()

        # Retrieve and verify values
        self.assertEqual(reloaded_handler.get_file_handler('1234567890'), 'test_value')
        self.assertEqual(reloaded_handler.get_file_handler('0987654321'), 'another_value')

    def test_default_keys(self):
        # Test setting default keys
        keys = {
            'test_key1': '1234567890',
            'test_key2': '0987654321'
        }
        defaults = {
            'test_key1': 'default_value1',
            'test_key2': 'default_value2'
        }

        file_handler = FileHandler('test.config', keys=keys, defaults=defaults)
        # file_handler.set_defaults_keys_file_handler(keys, defaults)

        # Verify default values
        self.assertEqual(file_handler.get_file_handler('test_key1'), 'default_value1')
        self.assertEqual(file_handler.get_file_handler('test_key2'), 'default_value2')

    def test_key_mapping(self):
        # Test key mapping functionality
        keys = {
            'short_key1': '1234567890',
            'short_key2': '0987654321'
        }
        defaults = {
            'short_key1': 'value1',
            'short_key2': 'value2'
        }

        file_handler = FileHandler('test.config')
        file_handler.set_defaults_keys_file_handler(keys, defaults)

        # Add values using mapped keys
        file_handler.add_to_save_file_handler('short_key1', 'mapped_value1')
        file_handler.add_to_save_file_handler('0987654321', 'mapped_value2')

        # Verify retrieving values works with both original and mapped keys
        self.assertEqual(file_handler.get_file_handler('short_key1'), 'mapped_value1')
        self.assertEqual(file_handler.get_file_handler('0987654321'), 'mapped_value2')
        self.assertEqual(file_handler.get_file_handler('short_key2'), 'mapped_value2')
        self.assertEqual(file_handler.get_file_handler('1234567890'), 'mapped_value1')

    def test_remove_key(self):
        # Create FileHandler and add some keys
        file_handler = FileHandler('test.config')
        file_handler.add_to_save_file_handler('1234567890', 'test_value')
        file_handler.add_to_save_file_handler('0987654321', 'another_value')

        # Remove a key
        file_handler.remove_key_file_handler('1234567890')

        # Verify key is removed
        self.assertIsNone(file_handler.get_file_handler('1234567890'))

    def test_delete_file(self):
        # Create FileHandler and add some data
        file_handler = FileHandler('test.config')
        file_handler.add_to_save_file_handler('1234567890', 'test_value')
        file_handler.save_file_handler()

        # Delete the file
        file_handler.delete_file()

        # Verify file is deleted
        self.assertFalse(os.path.exists(
            os.path.join(self.test_dir, '.test/test.config')
        ))

    def test_file_handler_error_handling(self):
        # Test various error scenarios
        file_handler = FileHandler('test.config')

        # Add a complex value that might cause evaluation issues
        file_handler.add_to_save_file_handler('1234567890', {"key": "value"})
        file_handler.save_file_handler()

        # Reload and verify
        reloaded_handler = FileHandler('test.config')
        reloaded_handler.load_file_handler()

        # Check retrieval of complex value
        retrieved_value = reloaded_handler.get_file_handler('1234567890')
        self.assertEqual(retrieved_value, {'key': 'value'})

    def test_multiple_file_operations(self):
        # Simulate multiple file operations
        file_handler = FileHandler('test.config')

        # Add initial data
        file_handler.add_to_save_file_handler('1234567890', 'initial_value')
        file_handler.save_file_handler()

        # Reload and update
        file_handler.load_file_handler()
        file_handler.add_to_save_file_handler('1234567890', 'updated_value')
        file_handler.save_file_handler()

        # Reload and verify
        reloaded_handler = FileHandler('test.config')
        reloaded_handler.load_file_handler()

        self.assertEqual(reloaded_handler.get_file_handler('1234567890'), 'updated_value')


"""
Tests for FileHandlerV2
=======================

Run with: python -m unittest test_file_handler_v2
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


class TestStorageScope(unittest.TestCase):
    """Test scope detection from filenames."""

    def test_config_file_scope(self):
        """Config files should have CONFIG scope."""
        fh = FileHandlerV2("test.config", name="test")
        self.assertEqual(fh.scope, StorageScope.CONFIG)

    def test_private_data_scope(self):
        """private.data should have USER_PRIVATE scope."""
        fh = FileHandlerV2("private.data", name="test")
        self.assertEqual(fh.scope, StorageScope.USER_PRIVATE)

    def test_public_data_scope(self):
        """public.data should have USER_PUBLIC scope."""
        fh = FileHandlerV2("public.data", name="test")
        self.assertEqual(fh.scope, StorageScope.USER_PUBLIC)

    def test_shared_data_scope(self):
        """shared.data should have PUBLIC_RW scope."""
        fh = FileHandlerV2("shared.data", name="test")
        self.assertEqual(fh.scope, StorageScope.PUBLIC_RW)

    def test_server_data_scope(self):
        """server.data should have SERVER_SCOPE scope."""
        fh = FileHandlerV2("server.data", name="test")
        self.assertEqual(fh.scope, StorageScope.SERVER_SCOPE)

    def test_default_data_scope(self):
        """Regular .data files should have MOD_DATA scope."""
        fh = FileHandlerV2("mymodule.data", name="test")
        self.assertEqual(fh.scope, StorageScope.MOD_DATA)

    def test_explicit_scope_override(self):
        """Explicit scope should override filename detection."""
        fh = FileHandlerV2("test.data", name="test", scope=StorageScope.USER_PRIVATE)
        self.assertEqual(fh.scope, StorageScope.USER_PRIVATE)

    def test_invalid_filename_raises(self):
        """Invalid filename should raise ValueError."""
        with self.assertRaises(ValueError):
            FileHandlerV2("test.txt", name="test")


class TestUserContext(unittest.TestCase):
    """Test UserContext creation and management."""

    def test_system_context(self):
        """System context should be admin."""
        ctx = UserContext.system()
        self.assertEqual(ctx.user_id, "system")
        self.assertTrue(ctx.is_admin)
        self.assertTrue(ctx.is_authenticated)

    def test_anonymous_context(self):
        """Anonymous context should not be authenticated."""
        ctx = UserContext.anonymous()
        self.assertEqual(ctx.user_id, "anonymous")
        self.assertFalse(ctx.is_authenticated)
        self.assertFalse(ctx.is_admin)

    def test_global_context(self):
        """Global context should be settable and gettable."""
        ctx = UserContext(user_id="test_user", is_authenticated=True)
        set_current_context(ctx)

        retrieved = get_current_context()
        self.assertEqual(retrieved.user_id, "test_user")

        # Clean up
        set_current_context(None)

    def test_default_context_is_system(self):
        """Default context should be system when not set."""
        set_current_context(None)
        ctx = get_current_context()
        self.assertEqual(ctx.user_id, "system")


class TestLocalStorageBackend(unittest.TestCase):
    """Test LocalStorageBackend."""

    def setUp(self):
        """Create temp directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.backend = LocalStorageBackend(Path(self.temp_dir))

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load(self):
        """Test basic save and load."""
        data = {"key": "value", "number": 42}
        self.assertTrue(self.backend.save("test", data))

        loaded = self.backend.load("test")
        self.assertEqual(loaded, data)

    def test_load_nonexistent(self):
        """Loading nonexistent key should return None."""
        self.assertIsNone(self.backend.load("nonexistent"))

    def test_exists(self):
        """Test exists check."""
        self.assertFalse(self.backend.exists("test"))
        self.backend.save("test", {"data": True})
        self.assertTrue(self.backend.exists("test"))

    def test_delete(self):
        """Test delete."""
        self.backend.save("test", {"data": True})
        self.assertTrue(self.backend.exists("test"))

        self.backend.delete("test")
        self.assertFalse(self.backend.exists("test"))

    def test_list_keys(self):
        """Test key listing."""
        self.backend.save("key1", {})
        self.backend.save("key2", {})
        self.backend.save("other", {})

        keys = self.backend.list_keys()
        self.assertEqual(set(keys), {"key1", "key2", "other"})

        # With prefix
        self.backend.save("prefix_a", {})
        self.backend.save("prefix_b", {})
        keys = self.backend.list_keys("prefix_")
        self.assertEqual(set(keys), {"prefix_a", "prefix_b"})

    def test_complex_data(self):
        """Test saving complex data structures."""
        data = {
            "string": "hello",
            "number": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": {"b": {"c": 1}}},
        }
        self.backend.save("complex", data)
        loaded = self.backend.load("complex")
        self.assertEqual(loaded, data)


class TestFileHandlerV2Sync(unittest.TestCase):
    """Test FileHandlerV2 synchronous API."""

    def setUp(self):
        """Create temp directory and handler."""
        self.temp_dir = tempfile.mkdtemp()
        self.fh = FileHandlerV2(
            "test.data",
            name="test_mod",
            backend=StorageBackend.LOCAL,
            base_path=Path(self.temp_dir),
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_set_and_get(self):
        """Test basic set and get."""
        self.fh.set("key1", "value1")
        self.assertEqual(self.fh.get("key1"), "value1")

    def test_get_default(self):
        """Test get with default."""
        self.assertEqual(self.fh.get("nonexistent", "default"), "default")

    def test_save_and_load(self):
        """Test save and load persistence."""
        self.fh.set("persistent", "data")
        self.fh.save()

        # Create new handler
        fh2 = FileHandlerV2(
            "test.data",
            name="test_mod",
            backend=StorageBackend.LOCAL,
            base_path=Path(self.temp_dir),
        )
        fh2.load()
        self.assertEqual(fh2.get("persistent"), "data")

    def test_context_manager(self):
        """Test context manager usage."""
        with FileHandlerV2(
            "context.data",
            name="test_mod",
            backend=StorageBackend.LOCAL,
            base_path=Path(self.temp_dir),
        ) as fh:
            fh.set("context_key", "context_value")

        # Data should be saved
        fh2 = FileHandlerV2(
            "context.data",
            name="test_mod",
            backend=StorageBackend.LOCAL,
            base_path=Path(self.temp_dir),
        )
        fh2.load()
        self.assertEqual(fh2.get("context_key"), "context_value")

    def test_dict_like_access(self):
        """Test dict-like access."""
        self.fh["key"] = "value"
        self.assertEqual(self.fh["key"], "value")
        self.assertIn("key", self.fh)

        del self.fh["key"]
        self.assertNotIn("key", self.fh)

    def test_update(self):
        """Test bulk update."""
        self.fh.update({"a": 1, "b": 2, "c": 3})
        self.assertEqual(self.fh.to_dict(), {"a": 1, "b": 2, "c": 3})

    def test_keys_and_items(self):
        """Test keys() and items()."""
        self.fh.update({"x": 1, "y": 2})
        self.assertEqual(set(self.fh.keys()), {"x", "y"})
        self.assertEqual(set(self.fh.items()), {("x", 1), ("y", 2)})


class TestFileHandlerV2Async(unittest.TestCase):
    """Test FileHandlerV2 async API."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_async_save_and_load(self):
        """Test async save and load."""
        async def run_test():
            fh = FileHandlerV2(
                "async.data",
                name="test_mod",
                backend=StorageBackend.LOCAL,
                base_path=Path(self.temp_dir),
            )

            await fh.aset("async_key", "async_value")
            await fh.asave()

            # New handler
            fh2 = FileHandlerV2(
                "async.data",
                name="test_mod",
                backend=StorageBackend.LOCAL,
                base_path=Path(self.temp_dir),
            )
            await fh2.aload()
            value = await fh2.aget("async_key")
            self.assertEqual(value, "async_value")

        asyncio.run(run_test())

    def test_async_context_manager(self):
        """Test async context manager."""
        async def run_test():
            async with FileHandlerV2(
                "async_ctx.data",
                name="test_mod",
                backend=StorageBackend.LOCAL,
                base_path=Path(self.temp_dir),
            ) as fh:
                await fh.aset("key", "value")

            # Verify saved
            fh2 = FileHandlerV2(
                "async_ctx.data",
                name="test_mod",
                backend=StorageBackend.LOCAL,
                base_path=Path(self.temp_dir),
            )
            await fh2.aload()
            self.assertEqual(await fh2.aget("key"), "value")

        asyncio.run(run_test())


class TestLegacyCompatibility(unittest.TestCase):
    """Test backward compatibility with legacy FileHandler API."""

    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_legacy_alias(self):
        """FileHandler should be alias for FileHandlerV2."""
        self.assertIs(FileHandler, FileHandlerV2)

    def test_legacy_methods(self):
        """Test legacy method names work."""
        fh = FileHandler(
            "legacy.config",
            name="legacy_test",
            base_path=Path(self.temp_dir),
        )

        # Legacy methods
        fh.add_to_save_file_handler("key", "value")
        self.assertEqual(fh.get_file_handler("key"), "value")

        fh.save_file_handler()

        # Load in new handler
        fh2 = FileHandler(
            "legacy.config",
            name="legacy_test",
            base_path=Path(self.temp_dir),
        )
        fh2.load_file_handler()
        self.assertEqual(fh2.get_file_handler("key"), "value")

    def test_key_mapping(self):
        """Test legacy key mapping."""
        fh = FileHandler(
            "mapping.config",
            name="test",
            keys={"short": "full_key_name"},
            defaults={"short": "default_value"},
            base_path=Path(self.temp_dir),
        )

        # Access via short key
        self.assertEqual(fh.get("short"), "default_value")
        self.assertEqual(fh.get("full_key_name"), "default_value")


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""

    def test_create_config_handler(self):
        """Test create_config_handler."""
        fh = create_config_handler("test_mod")
        self.assertEqual(fh.scope, StorageScope.CONFIG)
        self.assertTrue(fh.encrypt)

    def test_create_data_handler(self):
        """Test create_data_handler."""
        fh = create_data_handler("test_mod", scope=StorageScope.USER_PRIVATE)
        self.assertEqual(fh.scope, StorageScope.USER_PRIVATE)


if __name__ == "__main__":
    unittest.main()
