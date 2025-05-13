import os
import tempfile
import unittest
from unittest.mock import patch

from toolboxv2 import setup_logging
from toolboxv2.utils.system.file_handler import FileHandler


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
        with self.assertRaises(AssertionError):
            FileHandler('test.txt')

    def test_add_and_get_file_handler(self):
        # Create FileHandler instance
        file_handler = FileHandler('test.config')

        # Add items to save
        self.assertTrue(file_handler.add_to_save_file_handler('1234567890', 'test_value'))
        self.assertTrue(file_handler.add_to_save_file_handler('0987654321', 'another_value'))

        # Test invalid key length
        self.assertFalse(file_handler.add_to_save_file_handler('short', 'value'))

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

        # Attempt to remove root key
        with patch('builtins.print') as mock_print:
            file_handler.remove_key_file_handler('Pka7237327')
            mock_print.assert_called_with("Cant remove Root Key")

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
        file_handler.add_to_save_file_handler('1234567890', '{"key": "value"}')
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


if __name__ == '__main__':
    unittest.main()
