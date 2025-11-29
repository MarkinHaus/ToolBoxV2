# file: toolboxv2/tests/test_mods/test_blob_db.py
import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, patch, call
import pickle
import json

from toolboxv2.mods.DB.blob_instance import BlobDB
from toolboxv2.utils.extras.blobs import BlobStorage, BlobFile
from toolboxv2.utils.security.cryp import Code


class TestBlobDBMultiFile(unittest.TestCase):
    """
    Tests for the refactored BlobDB that uses multiple BlobFiles instead of a single file.
    """

    def setUp(self):
        """Set up a mock BlobStorage and BlobDB instance."""
        self.test_dir = tempfile.mkdtemp()
        self.mock_storage = MagicMock(spec=BlobStorage)
        self.db = BlobDB()
        self.key = Code.generate_symmetric_key()
        self.db_path = "test_db_blob"

        # Track created blobs for verification
        self.created_blobs = {}

        # Mock BlobFile behavior
        def mock_blob_file_init(filename, mode='r', storage=None, key=None, servers=None, use_cache=True):
            """Mock BlobFile initialization"""
            instance = MagicMock()
            instance.filename = filename
            instance.mode = mode
            instance.storage = storage or self.mock_storage
            instance.key = key
            instance.data_buffer = b""

            # Parse the filename
            parts = filename.lstrip('/\\').split('/')
            instance.blob_id = parts[0] if parts else ""
            instance.datei = parts[-1] if len(parts) > 1 else ""
            instance.folder = '|'.join(parts[1:-1]) if len(parts) > 2 else ""

            # Mock exists method
            def exists():
                return filename in self.created_blobs
            instance.exists.return_value = exists()
            instance.exists = exists

            # Mock create method
            def create():
                self.created_blobs[filename] = {}
                return instance
            instance.create = create

            # Mock context manager
            def enter(self_arg):
                if 'r' in mode and filename in self.created_blobs:
                    instance.data_buffer = json.dumps(self.created_blobs[filename]).encode()
                return instance

            def exit(self_arg, exc_type, exc_value, traceback):
                if 'w' in mode:
                    # Parse the written data
                    try:
                        data = json.loads(instance.data_buffer.decode())
                        self.created_blobs[filename] = data
                    except:
                        pass

            instance.__enter__ = enter
            instance.__exit__ = exit

            # Mock read_json
            def read_json():
                if filename in self.created_blobs:
                    return self.created_blobs[filename]
                return {}
            instance.read_json = read_json

            # Mock write_json
            def write_json(data):
                instance.data_buffer = json.dumps(data).encode()
            instance.write_json = write_json

            return instance

        self.blob_file_patcher = patch('toolboxv2.mods.DB.blob_instance.BlobFile', side_effect=mock_blob_file_init)
        self.mock_blob_file = self.blob_file_patcher.start()

    def tearDown(self):
        """Clean up the temporary directory and stop patches."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.blob_file_patcher.stop()

    def test_initialize(self):
        """Test that BlobDB initializes correctly with multi-file storage."""
        result = self.db.initialize(self.db_path, self.key, self.mock_storage)

        self.assertTrue(result.is_ok())
        self.assertEqual(self.db.db_base_path, self.db_path)
        self.assertEqual(self.db.enc_key, self.key)
        self.assertEqual(self.db.storage_client, self.mock_storage)

    def test_key_to_blob_path_conversion(self):
        """Test that keys are correctly converted to blob file paths."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        # Test various key formats
        test_cases = [
            ("USER::XYZ::", f"{self.db_path}/USER/XYZ.json"),
            ("MANAGER::SPACE::DATA::XYZ", f"{self.db_path}/MANAGER/SPACE/DATA/XYZ.json"),
            ("SIMPLE::KEY", f"{self.db_path}/SIMPLE/KEY.json"),
        ]

        for key, expected_path in test_cases:
            actual_path = self.db._key_to_path(key)
            self.assertEqual(actual_path, expected_path, f"Failed for key: {key}")

    def test_set_and_get_single_key(self):
        """Test setting and getting a single key."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        key = "USER::john::profile"
        value = {"name": "John Doe", "age": 30}

        # Set the value
        result = self.db.set(key, value)
        self.assertTrue(result.is_ok())

        # Get the value
        result = self.db.get(key)
        self.assertTrue(result.is_ok())
        print(result.result.data)
        self.assertEqual(value, result.get())

    def test_set_multiple_keys_different_paths(self):
        """Test that keys with different prefixes are stored in different blob files."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        # Set keys with different prefixes
        keys_values = [
            ("USER::alice::profile", {"name": "Alice"}),
            ("USER::bob::profile", {"name": "Bob"}),
            ("MANAGER::team1::data", {"team": "Team 1"}),
            ("MANAGER::team2::data", {"team": "Team 2"}),
        ]

        for key, value in keys_values:
            result = self.db.set(key, value)
            self.assertTrue(result.is_ok())

        # Exit to save all data
        result = self.db.exit()
        self.assertTrue(result.is_ok())

        # Verify that multiple blob files were created
        expected_paths = [
            f"{self.db_path}/USER/alice/profile.json",
            f"{self.db_path}/USER/bob/profile.json",
            f"{self.db_path}/MANAGER/team1/data.json",
            f"{self.db_path}/MANAGER/team2/data.json",
        ]

        for path in expected_paths:
            self.assertIn(path, self.created_blobs, f"Expected blob file not created: {path}")

    def test_exit_saves_to_multiple_files(self):
        """Test that exit() saves data to multiple blob files based on key structure."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        # Add data with different key prefixes
        self.db.set("USER::alice::name", "Alice")
        self.db.set("USER::bob::name", "Bob")
        self.db.set("ADMIN::settings::theme", "dark")

        # Exit should save to multiple files
        result = self.db.exit()
        self.assertTrue(result.is_ok())

        # Verify multiple blob files were created
        self.assertGreater(len(self.created_blobs), 0)

    def test_scan_iter_with_prefix(self):
        """Test scanning for keys with a specific prefix."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        # Add multiple keys with same prefix
        self.db.set("USER::alice::profile", {"name": "Alice"})
        self.db.set("USER::bob::profile", {"name": "Bob"})
        self.db.set("ADMIN::settings", {"theme": "dark"})

        # Scan for USER keys
        user_keys = self.db.get("USER::*").get()
        self.assertEqual(len(user_keys), 2)
        self.assertTrue(all(k.startswith("USER::") for k in user_keys))

    def test_delete_key(self):
        """Test deleting a key."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        key = "USER::alice::profile"
        self.db.set(key, {"name": "Alice"})

        # Verify key exists
        print(self.db.if_exist(key), self.db._manifest_cache)
        self.assertTrue(self.db.if_exist(key))

        # Delete the key
        result = self.db.delete(key)
        self.assertTrue(result.is_ok())

        # Verify key is gone
        self.assertEqual(self.db.if_exist(key), 0)

    def test_append_on_set(self):
        """Test appending to a list value."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        key = "USER::alice::tags"

        # First append creates the list
        result = self.db.append_on_set(key, ["tag1", "tag2"])
        self.assertTrue(result.is_ok())

        # Second append adds to existing list
        result = self.db.append_on_set(key, ["tag3", "tag1"])  # tag1 is duplicate
        self.assertTrue(result.is_ok())

        # Verify the list contains unique items
        result = self.db.get(key)
        self.assertTrue(result.is_ok())
        result.print()
        tags = result.get()
        self.assertEqual(len(tags), 3)  # tag1, tag2, tag3 (no duplicate)

    def test_isolation_between_blob_files(self):
        """Test that a problem with one blob file doesn't affect others."""
        self.db.initialize(self.db_path, self.key, self.mock_storage)

        # Set data in different blob files
        self.db.set("USER::alice::data", {"value": 1})
        self.db.set("ADMIN::settings", {"value": 2})

        # Simulate corruption in one blob file
        corrupted_path = f"{self.db_path}/USER/alice/data.json"
        self.created_blobs[corrupted_path] = "CORRUPTED_DATA"

        # The ADMIN data should still be accessible
        result = self.db.get("ADMIN::settings")
        self.assertTrue(result.is_ok())


if __name__ == '__main__':
    unittest.main(verbosity=2)

